$scriptPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'tools/galaxeavla/download_g05_base.ps1'

Describe 'Pinned G0.5 base downloader' {
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

    It 'pins the official repository and immutable revision' {
        $source | Should Match "RepoId = 'OpenGalaxea/G05'"
        $source | Should Match "Revision = 'aca2916ba2adaba3ce05bfa279cdc6b98da35cad'"
        $source | Should Match "OfficialHost = 'huggingface.co'"
    }

    It 'contains exactly the intended file entries' {
        $expectedPaths = @(
            'action_tokenizer.pt',
            'g05-base/.hydra/config.yaml',
            'g05-base/checkpoints/model_state_dict.pt',
            'g05-base/dataset_stats.json',
            'qwen3_5_2b_base_processor/config.json',
            'qwen3_5_2b_base_processor/configuration.json',
            'qwen3_5_2b_base_processor/merges.txt',
            'qwen3_5_2b_base_processor/preprocessor_config.json',
            'qwen3_5_2b_base_processor/tokenizer.json',
            'qwen3_5_2b_base_processor/tokenizer_config.json',
            'qwen3_5_2b_base_processor/video_preprocessor_config.json',
            'qwen3_5_2b_base_processor/vocab.json'
        )
        foreach ($path in $expectedPaths) {
            $source | Should Match ([Regex]::Escape("'$path'"))
        }

        $entryLines = @(
            $source -split "`n" |
                Where-Object { $_ -match "^\s*'[^']+'\s+=\s+\[Int64\]" }
        )
        $entryLines.Count | Should Be $expectedPaths.Count
    }

    It 'can fetch only non-LFS sidecars for remote staging' {
        $source | Should Match '\[switch\] \$SmallFilesOnly'
        $source | Should Match '\$SmallFilesOnly -and \$Script:ExpectedSha256\.ContainsKey'
    }

    It 'keeps credentials in memory and prompts securely' {
        $source | Should Match 'HUGGING_FACE_HUB_TOKEN'
        $source | Should Match 'Read-Host.+-AsSecureString'
        $source | Should Match 'ZeroFreeBSTR'
        $source | Should Not Match 'Set-Content.+Token'
        $source | Should Not Match 'Out-File.+Token'
    }

    It 'supports range resume and atomic partial files' {
        $source | Should Match 'RangeHeaderValue'
        $source | Should Match "\.part'"
        $source | Should Match '\[IO\.FileMode\]::Append'
        $source | Should Match 'statusCode -ne 200.+statusCode -ne 206'
        $source | Should Match 'Copy-StreamWithIdleTimeout'
        $source | Should Match 'IdleTimeoutSeconds 90'
    }

    It 'writes a SHA-256 manifest after size validation' {
        $source | Should Match 'Get-FileHash.+SHA256'
        $source | Should Match "model_state_dict\.pt' = '072211e5b2f5ef036729bae673f3f44da40adbea5c0044af55fe2fb8af654327'"
        $source | Should Match 'wrong official SHA-256'
        $source | Should Match 'g05-base-download-manifest\.json'
        $source | Should Match 'Downloaded file has the wrong size'
        $source | Should Match 'Move-Item.+manifestPartPath.+manifestPath'
    }

    It 'preserves a full manifest when subsequently writing only the small-file subset' {
        # Load only the two local manifest helpers; do not execute token or network code.
        $tokens = $null
        $parseErrors = $null
        $ast = [Management.Automation.Language.Parser]::ParseFile(
            $scriptPath,
            [ref] $tokens,
            [ref] $parseErrors
        )
        foreach ($functionName in @('Get-SafeTargetPath', 'Write-DownloadManifest')) {
            $functionAst = $ast.Find({
                param($node)
                $node -is [Management.Automation.Language.FunctionDefinitionAst] -and
                    $node.Name -eq $functionName
            }, $true)
            . ([scriptblock]::Create($functionAst.Extent.Text))
        }
        $Script:RepoId = 'OpenGalaxea/G05'
        $Script:Revision = 'aca2916ba2adaba3ce05bfa279cdc6b98da35cad'
        $Script:OfficialBaseUri = [Uri] 'https://huggingface.co/'
        $manifestRoot = Join-Path $TestDrive 'manifest-regression'
        New-Item -ItemType Directory -Path $manifestRoot | Out-Null
        $smallFile = [pscustomobject] @{ path = 'config.json'; size = 2; sha256 = 'sidecar-hash' }
        $weightFile = [pscustomobject] @{ path = 'weights.pt'; size = 8; sha256 = 'weight-hash' }

        $fullPath = Write-DownloadManifest -Root $manifestRoot -Files @($smallFile, $weightFile)
        $originalFullManifest = [IO.File]::ReadAllText($fullPath)
        $smallPath = Write-DownloadManifest -Root $manifestRoot -Files @($smallFile) -SmallFilesOnly

        [IO.Path]::GetFileName($fullPath) | Should Be 'g05-base-download-manifest.json'
        [IO.Path]::GetFileName($smallPath) | Should Be 'g05-base-small-files-download-manifest.json'
        [IO.File]::ReadAllText($fullPath) | Should Be $originalFullManifest
        $fullManifest = $originalFullManifest | ConvertFrom-Json
        $smallManifest = Get-Content -LiteralPath $smallPath -Raw | ConvertFrom-Json
        $fullManifest.scope | Should Be 'g05-base'
        $fullManifest.file_count | Should Be 2
        $smallManifest.scope | Should Be 'small-files-only'
        $smallManifest.file_count | Should Be 1
        @($smallManifest.files).Count | Should Be 1
        $smallManifest.files[0].path | Should Be $smallFile.path
    }

    It 'rejects destinations inside Git worktrees and untrusted redirects' {
        $source | Should Match 'Assert-DestinationOutsideGitWorktree'
        $source | Should Match 'redirected to an untrusted host'
        $source | Should Match "EndsWith\('\.huggingface\.co'"
        $source | Should Match "EndsWith\('\.hf\.co'"
    }
}
