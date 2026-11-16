<#
.SYNOPSIS
Stages the pinned G0.5 base or LIBERO Git-LFS objects on the Ascend host.

.DESCRIPTION
The long-lived Hugging Face token stays on the local machine. The script asks
the official endpoint for one short-lived, file-scoped signed URL at a time and
sends that URL only through SSH stdin. Completed files are promoted only after
their size and official LFS SHA-256 match.

.EXAMPLE
.\stage_g05_base_remote.ps1 -HostKeyAlias ascend-910b-host -ProbeBytes 1MB -Segments 4

Downloads an independent four-segment 1 MiB checkpoint probe without touching
the formal checkpoint or its .part file.

.EXAMPLE
.\stage_g05_base_remote.ps1 -HostKeyAlias ascend-910b-host

Resumably stages and verifies the three large LFS objects in g05-base.
Replace ascend-910b-host with the verified target alias in your known_hosts file.

.EXAMPLE
.\stage_g05_base_remote.ps1 -HostKeyAlias ascend-910b-host -CheckpointVariant libero -Segments 8

Stages the LIBERO checkpoint and shared codec/tokenizer. Configuration and
dataset_stats.json are separate small files from the same pinned revision.
Reads HUGGING_FACE_HUB_TOKEN when set, otherwise prompts securely.
#>
[CmdletBinding()]
param(
    [Parameter()]
    [ValidateNotNullOrEmpty()]
    [string] $RemoteRoot = '/root/work/models/galaxeavla-g05',

    [Parameter()]
    [ValidateNotNullOrEmpty()]
    [string] $SshHost = '127.0.0.1',

    [Parameter()]
    [ValidateRange(1, 65535)]
    [int] $SshPort = 22482,

    [Parameter()]
    [ValidateNotNullOrEmpty()]
    [string] $SshUser = 'root',

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string] $HostKeyAlias,

    [Parameter()]
    [ValidateRange(0, [Int64]::MaxValue)]
    [Int64] $ProbeBytes = 0,

    [Parameter()]
    [ValidateNotNullOrEmpty()]
    [string] $ProbeRelativePath = 'g05-base/checkpoints/model_state_dict.pt',

    [Parameter()]
    [ValidateRange(1, 20)]
    [int] $MaxUrlRefreshes = 5,

    [Parameter()]
    [ValidateRange(1, 16)]
    [int] $Segments = 1,

    [Parameter()]
    [ValidateSet('base', 'libero')]
    [string] $CheckpointVariant = 'base'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# This helper stages only the three Git-LFS objects in the selected subset. The
# remaining small configuration/tokenizer files can use the regular pinned
# downloader. Keeping the high-volume path LFS-only lets this script validate
# every completed object against the official LFS SHA-256.
$Script:RepoId = 'OpenGalaxea/G05'
$Script:Revision = 'aca2916ba2adaba3ce05bfa279cdc6b98da35cad'
$Script:OfficialHost = 'huggingface.co'
$Script:OfficialBaseUri = [Uri] 'https://huggingface.co/'
$Script:OfficialTreeUri = [Uri] (
    'https://huggingface.co/api/models/OpenGalaxea/G05/tree/' +
    'aca2916ba2adaba3ce05bfa279cdc6b98da35cad?recursive=true&expand=false&limit=1000'
)

$Script:ExpectedLfsFiles = [ordered] @{
    'action_tokenizer.pt'                              = [Int64] 506886775
    'g05-base/checkpoints/model_state_dict.pt'         = [Int64] 11440519964
    'qwen3_5_2b_base_processor/tokenizer.json'         = [Int64] 12807196
}
if ($CheckpointVariant -eq 'libero') {
    $Script:ExpectedLfsFiles.Remove('g05-base/checkpoints/model_state_dict.pt')
    $Script:ExpectedLfsFiles['g05-libero/model.pt'] = [Int64] 11440381065
}

function Get-HuggingFaceTokenInteractive {
    $environmentToken = [Environment]::GetEnvironmentVariable('HUGGING_FACE_HUB_TOKEN')
    if (-not [String]::IsNullOrWhiteSpace($environmentToken)) {
        return $environmentToken.Trim()
    }
    $secureToken = Read-Host 'Hugging Face access token' -AsSecureString
    $tokenPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
    try {
        $plainToken = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($tokenPointer)
        if ([String]::IsNullOrWhiteSpace($plainToken)) {
            throw [InvalidOperationException]::new('The Hugging Face access token is empty.')
        }
        return $plainToken
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($tokenPointer)
        $secureToken.Dispose()
    }
}

function New-OfficialHuggingFaceClient {
    $handler = [Net.Http.HttpClientHandler]::new()
    $handler.AllowAutoRedirect = $false
    $handler.UseProxy = $true
    $handler.Proxy = [Net.WebRequest]::DefaultWebProxy
    if ($null -ne $handler.Proxy) {
        $handler.Proxy.Credentials = [Net.CredentialCache]::DefaultCredentials
    }

    $client = [Net.Http.HttpClient]::new($handler, $true)
    $client.Timeout = [TimeSpan]::FromMinutes(5)
    $client.DefaultRequestHeaders.UserAgent.ParseAdd('gac-g05-remote-stager/1.0')
    return $client
}

function New-AuthenticatedRequest {
    param(
        [Parameter(Mandatory = $true)][Net.Http.HttpMethod] $Method,
        [Parameter(Mandatory = $true)][Uri] $Uri,
        [Parameter(Mandatory = $true)][string] $Token
    )

    if ($Uri.Scheme -ne [Uri]::UriSchemeHttps -or $Uri.DnsSafeHost -ne $Script:OfficialHost) {
        throw [InvalidOperationException]::new(
            'The general Hugging Face token may only be sent to https://huggingface.co.'
        )
    }
    $request = [Net.Http.HttpRequestMessage]::new($Method, $Uri)
    $request.Headers.Authorization = [Net.Http.Headers.AuthenticationHeaderValue]::new(
        'Bearer',
        $Token
    )
    return $request
}

function Assert-SuccessfulAuthenticatedResponse {
    param([Parameter(Mandatory = $true)][Net.Http.HttpResponseMessage] $Response)

    $statusCode = [int] $Response.StatusCode
    if ($statusCode -eq 401 -or $statusCode -eq 403) {
        throw [UnauthorizedAccessException]::new(
            "Hugging Face returned HTTP $statusCode. Verify the token and accept the OpenGalaxea/G05 access terms."
        )
    }
    if (-not $Response.IsSuccessStatusCode) {
        throw [Net.Http.HttpRequestException]::new(
            "Official Hugging Face request failed with HTTP $statusCode."
        )
    }
}

function Get-OfficialLfsMetadata {
    param(
        [Parameter(Mandatory = $true)][Net.Http.HttpClient] $Client,
        [Parameter(Mandatory = $true)][string] $Token
    )

    $request = New-AuthenticatedRequest `
        -Method ([Net.Http.HttpMethod]::Get) `
        -Uri $Script:OfficialTreeUri `
        -Token $Token
    try {
        $response = $Client.SendAsync($request).GetAwaiter().GetResult()
    }
    finally {
        $request.Dispose()
    }
    try {
        Assert-SuccessfulAuthenticatedResponse -Response $response
        $json = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    }
    finally {
        $response.Dispose()
    }

    $tree = @($json | ConvertFrom-Json)
    $metadata = [ordered] @{}
    foreach ($entry in $Script:ExpectedLfsFiles.GetEnumerator()) {
        $matches = @($tree | Where-Object { $_.path -eq $entry.Key })
        if ($matches.Count -ne 1) {
            throw [IO.InvalidDataException]::new(
                "Official tree metadata did not contain exactly one entry for $($entry.Key)."
            )
        }
        $item = $matches[0]
        if ([Int64] $item.size -ne [Int64] $entry.Value) {
            throw [IO.InvalidDataException]::new(
                "Official size changed for $($entry.Key); pinned $($entry.Value), received $($item.size)."
            )
        }
        if ($null -eq $item.lfs -or [string] $item.lfs.oid -notmatch '^[0-9a-fA-F]{64}$') {
            throw [IO.InvalidDataException]::new(
                "Official LFS SHA-256 metadata is missing for $($entry.Key)."
            )
        }
        $metadata[$entry.Key] = [pscustomobject] @{
            path = $entry.Key
            size = [Int64] $entry.Value
            sha256 = ([string] $item.lfs.oid).ToLowerInvariant()
        }
    }
    return $metadata
}

function ConvertTo-HuggingFacePath {
    param([Parameter(Mandatory = $true)][string] $RelativePath)

    $segments = foreach ($segment in ($RelativePath -split '/')) {
        [Uri]::EscapeDataString($segment)
    }
    return ($segments -join '/')
}

function Test-TrustedSignedCdnUri {
    param([Parameter(Mandatory = $true)][Uri] $Uri)

    if ($Uri.Scheme -ne [Uri]::UriSchemeHttps) {
        return $false
    }
    $hostName = $Uri.DnsSafeHost.ToLowerInvariant()
    return (
        $hostName.EndsWith('.huggingface.co', [StringComparison]::Ordinal) -or
        $hostName.EndsWith('.hf.co', [StringComparison]::Ordinal)
    )
}

function Get-OfficialSignedLfsUri {
    param(
        [Parameter(Mandatory = $true)][Net.Http.HttpClient] $Client,
        [Parameter(Mandatory = $true)][string] $Token,
        [Parameter(Mandatory = $true)][string] $RelativePath
    )

    $encodedPath = ConvertTo-HuggingFacePath -RelativePath $RelativePath
    $currentUri = [Uri]::new(
        $Script:OfficialBaseUri,
        "$($Script:RepoId)/resolve/$($Script:Revision)/$encodedPath`?download=true"
    )

    for ($redirectCount = 0; $redirectCount -le 10; $redirectCount++) {
        $request = New-AuthenticatedRequest `
            -Method ([Net.Http.HttpMethod]::Get) `
            -Uri $currentUri `
            -Token $Token
        try {
            $response = $Client.SendAsync(
                $request,
                [Net.Http.HttpCompletionOption]::ResponseHeadersRead
            ).GetAwaiter().GetResult()
        }
        finally {
            $request.Dispose()
        }

        try {
            $statusCode = [int] $response.StatusCode
            if ($statusCode -eq 401 -or $statusCode -eq 403) {
                throw [UnauthorizedAccessException]::new(
                    "Hugging Face returned HTTP $statusCode for $RelativePath. Verify the token and model access terms."
                )
            }
            if ($statusCode -notin 301, 302, 303, 307, 308) {
                throw [Net.Http.HttpRequestException]::new(
                    "Expected an official LFS redirect for $RelativePath, received HTTP $statusCode."
                )
            }

            $location = $response.Headers.Location
            if ($null -eq $location) {
                throw [Net.Http.HttpRequestException]::new(
                    "Official LFS redirect for $RelativePath had no Location header."
                )
            }
            if (-not $location.IsAbsoluteUri) {
                $location = [Uri]::new($currentUri, $location)
            }

            if ($location.Scheme -ne [Uri]::UriSchemeHttps) {
                throw [InvalidOperationException]::new('The official LFS redirect was not HTTPS.')
            }
            if ($location.DnsSafeHost -eq $Script:OfficialHost) {
                $currentUri = $location
                continue
            }
            if (-not (Test-TrustedSignedCdnUri -Uri $location)) {
                throw [InvalidOperationException]::new(
                    "The official endpoint redirected to an untrusted host: $($location.DnsSafeHost)"
                )
            }
            if ($location.AbsoluteUri -match '["\\\r\n]') {
                throw [InvalidOperationException]::new('The signed LFS URL contained unsafe characters.')
            }
            return $location
        }
        finally {
            $response.Dispose()
        }
    }

    throw [Net.Http.HttpRequestException]::new(
        "Official LFS resolution exceeded 10 redirects for $RelativePath."
    )
}

function Assert-SshParameters {
    if ($SshHost -notmatch '^[A-Za-z0-9.-]+$') {
        throw [ArgumentException]::new('SshHost contains unsupported characters.')
    }
    if ($SshUser -notmatch '^[A-Za-z0-9._-]+$') {
        throw [ArgumentException]::new('SshUser contains unsupported characters.')
    }
    if ($HostKeyAlias -notmatch '^[A-Za-z0-9.-]+$') {
        throw [ArgumentException]::new('HostKeyAlias contains unsupported characters.')
    }
    if ($RemoteRoot -notmatch '^/[A-Za-z0-9._/-]+$' -or $RemoteRoot -match '(^|/)\.\.(/|$)') {
        throw [ArgumentException]::new(
            'RemoteRoot must be an absolute path containing only letters, numbers, dot, underscore, dash, and slash.'
        )
    }
}

function Get-RemoteTargetPath {
    param([Parameter(Mandatory = $true)][string] $RelativePath)

    if (-not $Script:ExpectedLfsFiles.Contains($RelativePath)) {
        throw [ArgumentException]::new("The path is not in the pinned LFS allowlist: $RelativePath")
    }
    return $RemoteRoot.TrimEnd('/') + '/' + $RelativePath
}

function ConvertTo-Base64Utf8 {
    param([Parameter(Mandatory = $true)][string] $Value)
    return [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($Value))
}

function New-RemoteDownloadScript {
    param(
        [Parameter(Mandatory = $true)][Uri] $SignedUri,
        [Parameter(Mandatory = $true)][string] $TargetPath,
        [Parameter(Mandatory = $true)][Int64] $ExpectedSize,
        [Parameter(Mandatory = $true)][string] $ExpectedSha256
    )

    $template = @'
set -euo pipefail
umask 077
signed_url="$(printf '%s' '__SIGNED_URL_BASE64__' | base64 --decode)"
target="$(printf '%s' '__TARGET_BASE64__' | base64 --decode)"
expected_size='__EXPECTED_SIZE__'
expected_sha256='__EXPECTED_SHA256__'
part="${target}.part"
mkdir -p -- "$(dirname -- "$target")"

verify_and_promote() {
    local candidate="$1"
    local actual_size actual_sha256
    actual_size="$(stat -c '%s' -- "$candidate")"
    if [ "$actual_size" != "$expected_size" ]; then
        printf 'REMOTE_SIZE_MISMATCH %s %s\n' "$actual_size" "$expected_size"
        return 42
    fi
    actual_sha256="$(sha256sum -- "$candidate" | awk '{print $1}')"
    if [ "$actual_sha256" != "$expected_sha256" ]; then
        printf 'REMOTE_HASH_MISMATCH %s\n' "$actual_sha256"
        return 43
    fi
    if [ "$candidate" != "$target" ]; then
        mv -f -- "$candidate" "$target"
    fi
    printf 'REMOTE_OK %s %s\n' "$actual_size" "$actual_sha256"
}

if [ -f "$target" ]; then
    verify_and_promote "$target"
    exit $?
fi
if [ -e "$target" ]; then
    printf 'REMOTE_TARGET_NOT_FILE\n'
    exit 44
fi
if [ -f "$part" ]; then
    part_size="$(stat -c '%s' -- "$part")"
    if [ "$part_size" -gt "$expected_size" ]; then
        printf 'REMOTE_PART_TOO_LARGE %s %s\n' "$part_size" "$expected_size"
        exit 45
    fi
    if [ "$part_size" -eq "$expected_size" ]; then
        verify_and_promote "$part"
        exit $?
    fi
elif [ -e "$part" ]; then
    printf 'REMOTE_PART_NOT_FILE\n'
    exit 46
fi

set +e
{
    printf 'url = "%s"\n' "$signed_url"
    printf 'output = "%s"\n' "$part"
    if [ -f "$part" ]; then
        printf '%s\n' 'continue-at = "-"'
    fi
    printf '%s\n' 'fail'
    printf '%s\n' 'silent'
    printf '%s\n' 'show-error'
    printf '%s\n' 'retry = 8'
    printf '%s\n' 'retry-all-errors'
    printf '%s\n' 'retry-delay = 2'
    printf '%s\n' 'connect-timeout = 30'
    printf '%s\n' 'speed-limit = 65536'
    printf '%s\n' 'speed-time = 90'
    printf '%s\n' 'proto = "=https"'
    printf '%s\n' 'write-out = "REMOTE_TRANSFER %{http_code} %{size_download} %{speed_download} %{time_total}\n"'
} | curl --config -
curl_status=$?
set -e
if [ "$curl_status" -ne 0 ]; then
    printf 'REMOTE_CURL_FAILED %s\n' "$curl_status"
    exit 50
fi

verify_and_promote "$part"
'@

    return $template.Replace(
        '__SIGNED_URL_BASE64__',
        (ConvertTo-Base64Utf8 -Value $SignedUri.AbsoluteUri)
    ).Replace(
        '__TARGET_BASE64__',
        (ConvertTo-Base64Utf8 -Value $TargetPath)
    ).Replace(
        '__EXPECTED_SIZE__',
        $ExpectedSize.ToString([Globalization.CultureInfo]::InvariantCulture)
    ).Replace(
        '__EXPECTED_SHA256__',
        $ExpectedSha256
    )
}

function New-RemoteProbeScript {
    param(
        [Parameter(Mandatory = $true)][Uri] $SignedUri,
        [Parameter(Mandatory = $true)][string] $TargetPath,
        [Parameter(Mandatory = $true)][Int64] $ByteCount
    )

    $template = @'
set -euo pipefail
umask 077
signed_url="$(printf '%s' '__SIGNED_URL_BASE64__' | base64 --decode)"
target="$(printf '%s' '__TARGET_BASE64__' | base64 --decode)"
probe_bytes='__PROBE_BYTES__'
range_end="$((probe_bytes - 1))"
probe_target="${target}.probe-${probe_bytes}.bin"
probe_part="${probe_target}.part"
mkdir -p -- "$(dirname -- "$probe_target")"

set +e
{
    printf 'url = "%s"\n' "$signed_url"
    printf 'output = "%s"\n' "$probe_part"
    printf 'range = "0-%s"\n' "$range_end"
    printf 'max-filesize = "%s"\n' "$probe_bytes"
    printf '%s\n' 'fail'
    printf '%s\n' 'silent'
    printf '%s\n' 'show-error'
    printf '%s\n' 'retry = 5'
    printf '%s\n' 'retry-all-errors'
    printf '%s\n' 'retry-delay = 2'
    printf '%s\n' 'connect-timeout = 30'
    printf '%s\n' 'speed-limit = 1024'
    printf '%s\n' 'speed-time = 60'
    printf '%s\n' 'proto = "=https"'
    printf '%s\n' 'write-out = "PROBE_TRANSFER %{http_code} %{size_download} %{speed_download} %{time_total}\n"'
} | curl --config -
curl_status=$?
set -e
if [ "$curl_status" -ne 0 ]; then
    printf 'PROBE_CURL_FAILED %s\n' "$curl_status"
    exit 60
fi

actual_size="$(stat -c '%s' -- "$probe_part")"
if [ "$actual_size" != "$probe_bytes" ]; then
    printf 'PROBE_SIZE_MISMATCH %s %s\n' "$actual_size" "$probe_bytes"
    exit 61
fi
probe_sha256="$(sha256sum -- "$probe_part" | awk '{print $1}')"
mv -f -- "$probe_part" "$probe_target"
printf 'PROBE_OK %s %s %s\n' "$probe_target" "$actual_size" "$probe_sha256"
'@

    return $template.Replace(
        '__SIGNED_URL_BASE64__',
        (ConvertTo-Base64Utf8 -Value $SignedUri.AbsoluteUri)
    ).Replace(
        '__TARGET_BASE64__',
        (ConvertTo-Base64Utf8 -Value $TargetPath)
    ).Replace(
        '__PROBE_BYTES__',
        $ByteCount.ToString([Globalization.CultureInfo]::InvariantCulture)
    )
}

function New-RemoteSegmentedTransferScript {
    param(
        [Parameter(Mandatory = $true)][Uri] $SignedUri,
        [Parameter(Mandatory = $true)][string] $TargetPath,
        [Parameter(Mandatory = $true)][Int64] $ObjectSize,
        [Parameter(Mandatory = $true)][Int64] $TransferSize,
        [Parameter(Mandatory = $true)][string] $ExpectedSha256,
        [Parameter(Mandatory = $true)][ValidateRange(2, 16)][int] $SegmentCount,
        [Parameter()][switch] $Probe
    )

    $mode = if ($Probe) { 'probe' } else { 'formal' }
    $template = @'
set -euo pipefail
umask 077
signed_url="$(printf '%s' '__SIGNED_URL_BASE64__' | base64 --decode)"
target="$(printf '%s' '__TARGET_BASE64__' | base64 --decode)"
mode='__MODE__'
segments='__SEGMENTS__'
object_size='__OBJECT_SIZE__'
transfer_size='__TRANSFER_SIZE__'
expected_sha256='__EXPECTED_SHA256__'
mkdir -p -- "$(dirname -- "$target")"

if [ "$mode" = 'formal' ]; then
    final_target="$target"
    formal_part="${target}.part"
    merge_path="${formal_part}.merge"
    segment_prefix="$target"

    if [ -f "$final_target" ]; then
        existing_size="$(stat -c '%s' -- "$final_target")"
        existing_sha256="$(sha256sum -- "$final_target" | awk '{print $1}')"
        if [ "$existing_size" = "$object_size" ] && [ "$existing_sha256" = "$expected_sha256" ]; then
            printf 'SEGMENTED_REMOTE_ALREADY_OK %s %s\n' "$existing_size" "$existing_sha256"
            exit 0
        fi
        printf 'SEGMENTED_REMOTE_TARGET_MISMATCH %s %s\n' "$existing_size" "$existing_sha256"
        exit 70
    fi
    if [ -e "$final_target" ]; then
        printf 'SEGMENTED_REMOTE_TARGET_NOT_FILE\n'
        exit 71
    fi
    if [ -e "$formal_part" ]; then
        if [ -f "$formal_part" ]; then
            printf 'SEGMENTED_EXISTING_PART_REFUSED %s\n' "$(stat -c '%s' -- "$formal_part")"
        else
            printf 'SEGMENTED_EXISTING_PART_REFUSED not-a-file\n'
        fi
        exit 72
    fi
else
    final_target="${target}.probe-${transfer_size}-s${segments}.bin"
    formal_part=''
    merge_path="${final_target}.merge"
    segment_prefix="$final_target"
    if [ -f "$final_target" ]; then
        printf 'PROBE_SEGMENTED_ALREADY_EXISTS %s %s\n' \
            "$(stat -c '%s' -- "$final_target")" \
            "$(sha256sum -- "$final_target" | awk '{print $1}')"
        exit 0
    fi
    if [ -e "$final_target" ]; then
        printf 'PROBE_SEGMENTED_TARGET_NOT_FILE\n'
        exit 73
    fi
fi

if [ -L "$merge_path" ] || { [ -e "$merge_path" ] && [ ! -f "$merge_path" ]; }; then
    printf 'SEGMENTED_MERGE_PATH_UNSAFE\n'
    exit 74
fi

fifo_dir="$(mktemp -d "$(dirname -- "$target")/.g05-curl-fifo.XXXXXX")"
cleanup_fifos() {
    local fifo_path
    if [ -d "$fifo_dir" ]; then
        for fifo_path in "$fifo_dir"/*; do
            if [ -p "$fifo_path" ]; then
                rm -f -- "$fifo_path"
            fi
        done
        rmdir -- "$fifo_dir" 2>/dev/null || true
    fi
}
trap cleanup_fifos EXIT

segment_bounds() {
    local idx="$1" zero base remainder start length end
    zero="$((idx - 1))"
    base="$((transfer_size / segments))"
    remainder="$((transfer_size % segments))"
    if [ "$zero" -lt "$remainder" ]; then
        length="$((base + 1))"
        start="$((zero * length))"
    else
        length="$base"
        start="$((remainder * (base + 1) + (zero - remainder) * base))"
    fi
    end="$((start + length - 1))"
    printf '%s %s %s\n' "$start" "$end" "$length"
}

download_segment() {
    local idx="$1" start end segment_length segment_path current_length next_start
    local expected_tail_length tail_path cfg_fifo header_fifo writer_pid curl_pid
    local headers curl_status writer_status http_status content_range content_range_lower
    local cr_start cr_end cr_total tail_length final_segment_length
    read -r start end segment_length < <(segment_bounds "$idx")
    segment_path="${segment_prefix}.segment-${idx}.part"

    if [ -L "$segment_path" ] || { [ -e "$segment_path" ] && [ ! -f "$segment_path" ]; }; then
        printf 'SEGMENT_PATH_UNSAFE %s\n' "$idx"
        return 75
    fi
    current_length=0
    if [ -f "$segment_path" ]; then
        current_length="$(stat -c '%s' -- "$segment_path")"
    fi
    if [ "$current_length" -gt "$segment_length" ]; then
        printf 'SEGMENT_TOO_LARGE %s %s %s\n' "$idx" "$current_length" "$segment_length"
        return 76
    fi
    if [ "$current_length" -eq "$segment_length" ]; then
        printf 'SEGMENT_ALREADY_COMPLETE %s %s\n' "$idx" "$segment_length"
        return 0
    fi

    next_start="$((start + current_length))"
    expected_tail_length="$((end - next_start + 1))"
    tail_path="${segment_path}.tail"
    if [ -L "$tail_path" ] || { [ -e "$tail_path" ] && [ ! -f "$tail_path" ]; }; then
        printf 'SEGMENT_TAIL_PATH_UNSAFE %s\n' "$idx"
        return 77
    fi
    cfg_fifo="${fifo_dir}/config-${idx}"
    header_fifo="${fifo_dir}/headers-${idx}"
    mkfifo -m 600 -- "$cfg_fifo" "$header_fifo"
    : > "$tail_path"

    {
        printf 'url = "%s"\n' "$signed_url"
        printf 'output = "%s"\n' "$tail_path"
        printf 'range = "%s-%s"\n' "$next_start" "$end"
        printf 'dump-header = "%s"\n' "$header_fifo"
        printf '%s\n' 'fail'
        printf '%s\n' 'silent'
        printf '%s\n' 'show-error'
        printf '%s\n' 'connect-timeout = 30'
        printf '%s\n' 'speed-limit = 65536'
        printf '%s\n' 'speed-time = 90'
        printf '%s\n' 'proto = "=https"'
        printf 'write-out = "SEGMENT_TRANSFER %s %%{http_code} %%{size_download} %%{speed_download} %%{time_total}\\n"\n' "$idx"
    } > "$cfg_fifo" &
    writer_pid=$!

    set +e
    curl --config "$cfg_fifo" &
    curl_pid=$!
    headers="$(cat -- "$header_fifo")"
    wait "$curl_pid"
    curl_status=$?
    wait "$writer_pid"
    writer_status=$?
    set -e
    rm -f -- "$cfg_fifo" "$header_fifo"

    if [ "$writer_status" -ne 0 ]; then
        printf 'SEGMENT_CONFIG_FAILED %s %s\n' "$idx" "$writer_status"
        return 78
    fi
    if [ "$curl_status" -ne 0 ]; then
        printf 'SEGMENT_CURL_FAILED %s %s\n' "$idx" "$curl_status"
        return 79
    fi

    http_status="$(printf '%s\n' "$headers" | awk '/^HTTP\// {gsub(/\r/, "", $2); code=$2} END {print code}')"
    content_range="$(
        printf '%s\n' "$headers" |
            awk 'tolower($0) ~ /^content-range:/ {line=$0} END {sub(/^[^:]*:[[:space:]]*/, "", line); gsub(/\r/, "", line); print line}'
    )"
    content_range_lower="${content_range,,}"
    if [ "$http_status" != '206' ]; then
        printf 'SEGMENT_HTTP_STATUS_INVALID %s %s\n' "$idx" "$http_status"
        return 80
    fi
    if [[ ! "$content_range_lower" =~ ^bytes[[:space:]]+([0-9]+)-([0-9]+)/([0-9]+)$ ]]; then
        printf 'SEGMENT_CONTENT_RANGE_MISSING %s\n' "$idx"
        return 81
    fi
    cr_start="${BASH_REMATCH[1]}"
    cr_end="${BASH_REMATCH[2]}"
    cr_total="${BASH_REMATCH[3]}"
    if [ "$cr_start" != "$next_start" ] || [ "$cr_end" != "$end" ] || [ "$cr_total" != "$object_size" ]; then
        printf 'SEGMENT_CONTENT_RANGE_INVALID %s %s %s %s\n' "$idx" "$cr_start" "$cr_end" "$cr_total"
        return 82
    fi
    if [ ! -f "$tail_path" ]; then
        printf 'SEGMENT_TAIL_MISSING %s\n' "$idx"
        return 83
    fi
    tail_length="$(stat -c '%s' -- "$tail_path")"
    if [ "$tail_length" != "$expected_tail_length" ]; then
        printf 'SEGMENT_TAIL_SIZE_INVALID %s %s %s\n' "$idx" "$tail_length" "$expected_tail_length"
        return 84
    fi

    cat -- "$tail_path" >> "$segment_path"
    rm -f -- "$tail_path"
    final_segment_length="$(stat -c '%s' -- "$segment_path")"
    if [ "$final_segment_length" != "$segment_length" ]; then
        printf 'SEGMENT_FINAL_SIZE_INVALID %s %s %s\n' "$idx" "$final_segment_length" "$segment_length"
        return 85
    fi
    printf 'SEGMENT_OK %s %s-%s %s\n' "$idx" "$start" "$end" "$final_segment_length"
}

transfer_start_ns="$(date +%s%N)"
segment_pids=()
for ((segment_index = 1; segment_index <= segments; segment_index++)); do
    download_segment "$segment_index" &
    segment_pids+=("$!")
done

transfer_failed=0
for segment_pid in "${segment_pids[@]}"; do
    if ! wait "$segment_pid"; then
        transfer_failed=1
    fi
done
transfer_end_ns="$(date +%s%N)"
cleanup_fifos
trap - EXIT

elapsed_ns="$((transfer_end_ns - transfer_start_ns))"
if [ "$elapsed_ns" -le 0 ]; then
    elapsed_ns=1
fi
aggregate_bps="$(awk -v bytes="$transfer_size" -v ns="$elapsed_ns" 'BEGIN {printf "%.0f", bytes * 1000000000 / ns}')"
elapsed_seconds="$(awk -v ns="$elapsed_ns" 'BEGIN {printf "%.6f", ns / 1000000000}')"
printf 'SEGMENT_AGGREGATE %s %s %s %s\n' "$segments" "$transfer_size" "$aggregate_bps" "$elapsed_seconds"

if [ "$transfer_failed" -ne 0 ]; then
    printf 'SEGMENTED_TRANSFER_INCOMPLETE\n'
    exit 86
fi

: > "$merge_path"
for ((segment_index = 1; segment_index <= segments; segment_index++)); do
    read -r unused_start unused_end required_length < <(segment_bounds "$segment_index")
    segment_path="${segment_prefix}.segment-${segment_index}.part"
    actual_length="$(stat -c '%s' -- "$segment_path")"
    if [ "$actual_length" != "$required_length" ]; then
        printf 'SEGMENT_MERGE_SIZE_INVALID %s %s %s\n' "$segment_index" "$actual_length" "$required_length"
        exit 87
    fi
    cat -- "$segment_path" >> "$merge_path"
done

merged_size="$(stat -c '%s' -- "$merge_path")"
if [ "$merged_size" != "$transfer_size" ]; then
    printf 'SEGMENTED_MERGE_SIZE_MISMATCH %s %s\n' "$merged_size" "$transfer_size"
    exit 88
fi
merged_sha256="$(sha256sum -- "$merge_path" | awk '{print $1}')"

if [ "$mode" = 'formal' ]; then
    mv -f -- "$merge_path" "$formal_part"
    if [ "$merged_size" != "$object_size" ] || [ "$merged_sha256" != "$expected_sha256" ]; then
        printf 'REMOTE_HASH_MISMATCH %s\n' "$merged_sha256"
        exit 43
    fi
    mv -f -- "$formal_part" "$final_target"
    printf 'SEGMENTED_REMOTE_OK %s %s %s\n' "$segments" "$merged_size" "$merged_sha256"
else
    mv -f -- "$merge_path" "$final_target"
    printf 'PROBE_SEGMENTED_OK %s %s %s %s\n' "$final_target" "$segments" "$merged_size" "$merged_sha256"
fi
'@

    return $template.Replace(
        '__SIGNED_URL_BASE64__',
        (ConvertTo-Base64Utf8 -Value $SignedUri.AbsoluteUri)
    ).Replace(
        '__TARGET_BASE64__',
        (ConvertTo-Base64Utf8 -Value $TargetPath)
    ).Replace(
        '__MODE__',
        $mode
    ).Replace(
        '__SEGMENTS__',
        $SegmentCount.ToString([Globalization.CultureInfo]::InvariantCulture)
    ).Replace(
        '__OBJECT_SIZE__',
        $ObjectSize.ToString([Globalization.CultureInfo]::InvariantCulture)
    ).Replace(
        '__TRANSFER_SIZE__',
        $TransferSize.ToString([Globalization.CultureInfo]::InvariantCulture)
    ).Replace(
        '__EXPECTED_SHA256__',
        $ExpectedSha256
    )
}

function Protect-RemoteOutput {
    param([Parameter(Mandatory = $true)][object[]] $Lines)

    return @($Lines | ForEach-Object {
        ([string] $_) -replace 'https://\S+', '[redacted-signed-url]'
    })
}

function Invoke-RemoteScriptViaSsh {
    param([Parameter(Mandatory = $true)][string] $ScriptText)

    $sshCommand = Get-Command ssh -ErrorAction Stop
    $sshArguments = (
        '-p {0} -o BatchMode=yes -o HostKeyAlias={1} ' +
        '-o StrictHostKeyChecking=yes -o LogLevel=ERROR {2}@{3} "bash -s"'
    ) -f @(
        $SshPort.ToString([Globalization.CultureInfo]::InvariantCulture),
        $HostKeyAlias,
        $SshUser,
        $SshHost
    )

    $startInfo = [Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $sshCommand.Source
    $startInfo.Arguments = $sshArguments
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardInput = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $startInfo

    try {
        # Signed URLs use SSH stdin and are never placed in argv, shell history,
        # a remote config file, or console output.
        if (-not $process.Start()) {
            throw [InvalidOperationException]::new('Failed to start ssh.')
        }
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        $process.StandardInput.Write($ScriptText)
        $process.StandardInput.Write("`n")
        $process.StandardInput.Close()
        $process.WaitForExit()
        $stdout = $stdoutTask.GetAwaiter().GetResult()
        $stderr = $stderrTask.GetAwaiter().GetResult()
        $exitCode = $process.ExitCode
    }
    finally {
        $process.Dispose()
    }

    $output = @(
        @($stdout -split '\r?\n') +
        @($stderr -split '\r?\n') |
            Where-Object { -not [String]::IsNullOrWhiteSpace($_) }
    )

    return [pscustomobject] @{
        exit_code = [int] $exitCode
        output = @(Protect-RemoteOutput -Lines $output)
    }
}

function Invoke-StagedLfsTransfer {
    param(
        [Parameter(Mandatory = $true)][Net.Http.HttpClient] $Client,
        [Parameter(Mandatory = $true)][string] $Token,
        [Parameter(Mandatory = $true)][object] $Metadata,
        [Parameter(Mandatory = $true)][Int64] $ProbeSize,
        [Parameter(Mandatory = $true)][int] $SegmentCount
    )

    $targetPath = Get-RemoteTargetPath -RelativePath $Metadata.path
    for ($attempt = 1; $attempt -le $MaxUrlRefreshes; $attempt++) {
        $signedUri = $null
        try {
            $signedUri = Get-OfficialSignedLfsUri `
                -Client $Client `
                -Token $Token `
                -RelativePath $Metadata.path
            if ($ProbeSize -gt 0 -and $SegmentCount -gt 1) {
                $remoteScript = New-RemoteSegmentedTransferScript `
                    -SignedUri $signedUri `
                    -TargetPath $targetPath `
                    -ObjectSize $Metadata.size `
                    -TransferSize $ProbeSize `
                    -ExpectedSha256 $Metadata.sha256 `
                    -SegmentCount $SegmentCount `
                    -Probe
            }
            elseif ($ProbeSize -gt 0) {
                $remoteScript = New-RemoteProbeScript `
                    -SignedUri $signedUri `
                    -TargetPath $targetPath `
                    -ByteCount $ProbeSize
            }
            elseif ($SegmentCount -gt 1) {
                $remoteScript = New-RemoteSegmentedTransferScript `
                    -SignedUri $signedUri `
                    -TargetPath $targetPath `
                    -ObjectSize $Metadata.size `
                    -TransferSize $Metadata.size `
                    -ExpectedSha256 $Metadata.sha256 `
                    -SegmentCount $SegmentCount
            }
            else {
                $remoteScript = New-RemoteDownloadScript `
                    -SignedUri $signedUri `
                    -TargetPath $targetPath `
                    -ExpectedSize $Metadata.size `
                    -ExpectedSha256 $Metadata.sha256
            }

            $result = Invoke-RemoteScriptViaSsh -ScriptText $remoteScript
            foreach ($line in $result.output) {
                Write-Host $line
            }
            if ($result.exit_code -eq 0) {
                return
            }
            if ($result.output -match (
                'REMOTE_HASH_MISMATCH|REMOTE_PART_TOO_LARGE|REMOTE_TARGET_NOT_FILE|' +
                'SEGMENTED_EXISTING_PART_REFUSED|SEGMENTED_REMOTE_TARGET_MISMATCH|' +
                'SEGMENTED_REMOTE_TARGET_NOT_FILE|SEGMENTED_MERGE_PATH_UNSAFE|' +
                'SEGMENT_PATH_UNSAFE|SEGMENT_TOO_LARGE|SEGMENT_TAIL_PATH_UNSAFE'
            )) {
                throw [IO.InvalidDataException]::new(
                    "Remote validation failed for $($Metadata.path); the existing artifact was preserved."
                )
            }
            if ($attempt -ge $MaxUrlRefreshes) {
                throw [InvalidOperationException]::new(
                    "Remote transfer failed after $attempt signed-URL attempts for $($Metadata.path)."
                )
            }
            Write-Warning "Remote transfer attempt $attempt failed for $($Metadata.path); requesting a fresh scoped URL."
        }
        finally {
            $remoteScript = $null
            $signedUri = $null
        }
    }
}

Assert-SshParameters
if ($ProbeBytes -lt 0) {
    throw [ArgumentOutOfRangeException]::new('ProbeBytes cannot be negative.')
}
if ($ProbeBytes -gt 0 -and -not $Script:ExpectedLfsFiles.Contains($ProbeRelativePath)) {
    throw [ArgumentException]::new(
        "ProbeRelativePath is not in the pinned LFS allowlist: $ProbeRelativePath"
    )
}
if ($ProbeBytes -gt 0 -and $ProbeBytes -gt $Script:ExpectedLfsFiles[$ProbeRelativePath]) {
    throw [ArgumentOutOfRangeException]::new(
        'ProbeBytes cannot exceed the pinned LFS object size.'
    )
}
if ($ProbeBytes -gt 0 -and $ProbeBytes -lt $Segments) {
    throw [ArgumentOutOfRangeException]::new(
        'ProbeBytes must be at least as large as Segments.'
    )
}

$token = $null
$client = $null
try {
    $token = Get-HuggingFaceTokenInteractive
    $client = New-OfficialHuggingFaceClient
    $metadata = Get-OfficialLfsMetadata -Client $client -Token $token

    if ($ProbeBytes -gt 0) {
        Invoke-StagedLfsTransfer `
            -Client $client `
            -Token $token `
            -Metadata $metadata[$ProbeRelativePath] `
            -ProbeSize $ProbeBytes `
            -SegmentCount $Segments
        Write-Host 'Probe complete. The formal .part file was not touched.'
    }
    else {
        foreach ($entry in $metadata.GetEnumerator()) {
            Invoke-StagedLfsTransfer `
                -Client $client `
                -Token $token `
                -Metadata $entry.Value `
                -ProbeSize 0 `
                -SegmentCount $Segments
        }
        Write-Host "All pinned g05-$CheckpointVariant LFS objects were staged and SHA-256 verified."
    }
}
finally {
    if ($null -ne $client) {
        $client.Dispose()
    }
    $token = $null
}
