$scriptPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'tools/galaxeavla/stage_g05_base_remote.ps1'

Describe 'Pinned G0.5 remote LFS stager' {
    $source = Get-Content -LiteralPath $scriptPath -Raw

    It 'is valid PowerShell syntax' {
        $tokens = $null
        $parseErrors = $null
        [Management.Automation.Language.Parser]::ParseFile(
            $scriptPath,
            [ref] $tokens,
            [ref] $parseErrors
        ) | Out-Null
        $parseErrors.Count | Should Be 0
    }

    It 'pins the official repository, revision, and three LFS objects' {
        $source | Should Match "RepoId = 'OpenGalaxea/G05'"
        $source | Should Match "Revision = 'aca2916ba2adaba3ce05bfa279cdc6b98da35cad'"
        $source | Should Match "OfficialHost = 'huggingface.co'"
        $source | Should Match ([Regex]::Escape("'action_tokenizer.pt'"))
        $source | Should Match ([Regex]::Escape("'g05-base/checkpoints/model_state_dict.pt'"))
        $source | Should Match ([Regex]::Escape("'qwen3_5_2b_base_processor/tokenizer.json'"))

        $entryLines = @(
            $source -split "`n" |
                Where-Object { $_ -match "^\s*'[^']+'\s+=\s+\[Int64\]" }
        )
        $entryLines.Count | Should Be 3
    }

    It 'uses the local SSH tunnel and requires an explicit verified host alias' {
        $source | Should Match "SshHost = '127\.0\.0\.1'"
        $source | Should Match 'SshPort = 22482'
        $source | Should Match '(?s)\[Parameter\(Mandatory = \$true\)\]\s+\[ValidateNotNullOrEmpty\(\)\]\s+\[string\] \$HostKeyAlias,'
        $source | Should Not Match '\$HostKeyAlias\s*='
        $source | Should Match "RemoteRoot = '/root/work/models/galaxeavla-g05'"
    }

    It 'reads the general token interactively and limits it to the official host' {
        $source | Should Match 'Read-Host.+-AsSecureString'
        $source | Should Match 'ZeroFreeBSTR'
        $source | Should Match 'general Hugging Face token may only be sent'
        $source | Should Match 'DnsSafeHost -ne \$Script:OfficialHost'
    }

    It 'accepts only HTTPS official signed CDN redirects' {
        $source | Should Match 'Test-TrustedSignedCdnUri'
        $source | Should Match "EndsWith\('\.huggingface\.co'"
        $source | Should Match "EndsWith\('\.hf\.co'"
        $source | Should Match 'official LFS redirect was not HTTPS'
    }

    It 'passes the scoped URL only through SSH and curl stdin' {
        $source | Should Match 'StandardInput\.Write\(\$ScriptText\)'
        $source | Should Match 'RedirectStandardInput = \$true'
        $source | Should Match 'curl --config -'
        $source | Should Match 'SIGNED_URL_BASE64'
        $source | Should Match 'never.+placed in argv'
        $source | Should Not Match 'sshArguments.+SignedUri'
    }

    It 'supports resumable, stalled-transfer recovery and URL refresh' {
        $source | Should Match 'continue-at = \"-\"'
        $source | Should Match 'retry-all-errors'
        $source | Should Match 'speed-limit = 65536'
        $source | Should Match 'speed-time = 90'
        $source | Should Match 'requesting a fresh scoped URL'
    }

    It 'offers bounded parallel segments while preserving the single-stream default' {
        $source | Should Match '\[ValidateRange\(1, 16\)\]'
        $source | Should Match '\[int\] \$Segments = 1'
        $source | Should Match 'for \(\(segment_index = 1; segment_index <= segments; segment_index\+\+\)\)'
        $source | Should Match 'download_segment \"\$segment_index\" &'
        $source | Should Match 'segment_pids\+='
    }

    It 'keeps each fixed range resumable in its own segment part' {
        $source | Should Match '\.segment-\$\{idx\}\.part'
        $source | Should Match 'next_start=\"\$\(\(start \+ current_length\)\)\"'
        $source | Should Match 'range = \"%s-%s\"'
        $source | Should Match 'SEGMENT_ALREADY_COMPLETE'
        $source | Should Match 'SEGMENT_FINAL_SIZE_INVALID'
    }

    It 'uses in-memory headers and FIFO-only curl configuration' {
        $source | Should Match 'mkfifo -m 600'
        $source | Should Match 'headers=\"\$\(cat -- \"\$header_fifo\"\)\"'
        $source | Should Match 'curl --config \"\$cfg_fifo\"'
        $source | Should Match 'dump-header = \"%s\"'
        $source | Should Not Match 'signed_url.+>.+\.conf'
    }

    It 'strictly validates partial content before appending segment bytes' {
        $source | Should Match "http_status.*!= '206'"
        $source | Should Match 'content-range:'
        $source | Should Match 'cr_start.*next_start'
        $source | Should Match 'cr_end.*end'
        $source | Should Match 'cr_total.*object_size'
        $source | Should Match 'cat -- \"\$tail_path\" >> \"\$segment_path\"'
    }

    It 'refuses an existing non-segment partial without deleting it' {
        $source | Should Match 'SEGMENTED_EXISTING_PART_REFUSED'
        $source | Should Not Match 'rm -[^\r\n]+\$formal_part'
    }

    It 'merges in segment order and verifies the official object hash' {
        $source | Should Match 'cat -- \"\$segment_path\" >> \"\$merge_path\"'
        $source | Should Match 'merged_sha256=\"\$\(sha256sum'
        $source | Should Match 'merged_sha256.*expected_sha256'
        $source | Should Match 'mv -f -- \"\$formal_part\" \"\$final_target\"'
        $source | Should Match 'SEGMENTED_REMOTE_OK'
    }

    It 'verifies pinned size and official LFS SHA-256 before atomic promotion' {
        $source | Should Match '\^\[0-9a-fA-F\]\{64\}\$'
        $source | Should Match 'sha256sum'
        $source | Should Match 'REMOTE_HASH_MISMATCH'
        $source | Should Match 'mv -f -- \"\$candidate\" \"\$target\"'
    }

    It 'keeps the probe separate from the formal partial file' {
        $source | Should Match 'ProbeBytes'
        $source | Should Match 'probe-\$\{probe_bytes\}\.bin'
        $source | Should Match 'range = \"0-%s\"'
        $source | Should Match 'formal \.part file was not touched'
        $source | Should Match 'probe-\$\{transfer_size\}-s\$\{segments\}\.bin'
        $source | Should Match 'SEGMENT_AGGREGATE'
        $source | Should Match 'PROBE_SEGMENTED_OK'
    }
}
