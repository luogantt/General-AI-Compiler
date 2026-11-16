[CmdletBinding()]
param(
    [Parameter()]
    [ValidateNotNullOrEmpty()]
    [string] $Destination = (Join-Path ([Environment]::GetFolderPath('UserProfile')) '.cache/galaxeavla/OpenGalaxea-G05'),

    [Parameter()]
    [ValidateRange(1, 20)]
    [int] $MaxRetries = 5,

    [Parameter()]
    [ValidateRange(1, 64)]
    [int] $BufferSizeMiB = 8,

    [Parameter()]
    [switch] $NonInteractive,

    [Parameter()]
    # Writes a separate small-files manifest without replacing the full manifest.
    [switch] $SmallFilesOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# This downloader is intentionally pinned. Changing any of these values requires a
# source review because G0.5 is a gated model and its large weights use pickle.
$Script:RepoId = 'OpenGalaxea/G05'
$Script:Revision = 'aca2916ba2adaba3ce05bfa279cdc6b98da35cad'
$Script:OfficialHost = 'huggingface.co'
$Script:OfficialBaseUri = [Uri] 'https://huggingface.co/'

# Files and byte sizes reported by the official Hugging Face tree API for the
# pinned revision. Directories are omitted. No caller-provided path reaches the
# network or filesystem path builder.
$Script:ExpectedFiles = [ordered] @{
    'action_tokenizer.pt'                                      = [Int64] 506886775
    'g05-base/.hydra/config.yaml'                              = [Int64] 29481
    'g05-base/checkpoints/model_state_dict.pt'                 = [Int64] 11440519964
    'g05-base/dataset_stats.json'                              = [Int64] 5258626
    'qwen3_5_2b_base_processor/config.json'                    = [Int64] 2908
    'qwen3_5_2b_base_processor/configuration.json'             = [Int64] 51
    'qwen3_5_2b_base_processor/merges.txt'                     = [Int64] 3353259
    'qwen3_5_2b_base_processor/preprocessor_config.json'       = [Int64] 390
    'qwen3_5_2b_base_processor/tokenizer.json'                 = [Int64] 12807196
    'qwen3_5_2b_base_processor/tokenizer_config.json'          = [Int64] 16712
    'qwen3_5_2b_base_processor/video_preprocessor_config.json' = [Int64] 386
    'qwen3_5_2b_base_processor/vocab.json'                     = [Int64] 6722759
}

# SHA-256 values published by the official Hugging Face LFS metadata.  The
# checkpoint and tokenizer artifacts must match these values before they can be
# promoted from their .part files.  Small Git blobs are still hashed into the
# generated manifest, but Hugging Face exposes their immutable Git blob IDs
# rather than SHA-256 values.
$Script:ExpectedSha256 = @{
    'action_tokenizer.pt' = '5088f64a5452a60bbc8cac90ee7d79c156f1d14540bc3139aa7060f862dddace'
    'g05-base/checkpoints/model_state_dict.pt' = '072211e5b2f5ef036729bae673f3f44da40adbea5c0044af55fe2fb8af654327'
    'qwen3_5_2b_base_processor/tokenizer.json' = 'fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927'
}

function Get-HuggingFaceToken {
    param([switch] $DisablePrompt)

    $fromEnvironment = [Environment]::GetEnvironmentVariable(
        'HUGGING_FACE_HUB_TOKEN',
        [EnvironmentVariableTarget]::Process
    )
    if (-not [String]::IsNullOrWhiteSpace($fromEnvironment)) {
        return $fromEnvironment.Trim()
    }

    if ($DisablePrompt) {
        throw [InvalidOperationException]::new(
            'HUGGING_FACE_HUB_TOKEN is not set and interactive input is disabled.'
        )
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

function Assert-DestinationOutsideGitWorktree {
    param([Parameter(Mandatory = $true)][string] $Path)

    $cursor = [IO.DirectoryInfo]::new([IO.Path]::GetFullPath($Path))
    while ($null -ne $cursor) {
        if (Test-Path -LiteralPath (Join-Path $cursor.FullName '.git')) {
            throw [InvalidOperationException]::new(
                "Refusing to place model files inside a Git worktree: $($cursor.FullName)"
            )
        }
        $cursor = $cursor.Parent
    }
}

function Get-SafeTargetPath {
    param(
        [Parameter(Mandatory = $true)][string] $Root,
        [Parameter(Mandatory = $true)][string] $RelativePath
    )

    if ([IO.Path]::IsPathRooted($RelativePath) -or $RelativePath -match '(^|/|\\)\.\.(/|\\|$)') {
        throw [InvalidOperationException]::new("Unsafe model path: $RelativePath")
    }

    $rootPath = [IO.Path]::GetFullPath($Root).TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar
    )
    $platformRelativePath = $RelativePath.Replace('/', [IO.Path]::DirectorySeparatorChar)
    $targetPath = [IO.Path]::GetFullPath((Join-Path $rootPath $platformRelativePath))
    $rootPrefix = $rootPath + [IO.Path]::DirectorySeparatorChar
    if (-not $targetPath.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw [InvalidOperationException]::new("Model path escaped the destination: $RelativePath")
    }
    return $targetPath
}

function ConvertTo-HuggingFacePath {
    param([Parameter(Mandatory = $true)][string] $RelativePath)

    $encodedSegments = foreach ($segment in ($RelativePath -split '/')) {
        [Uri]::EscapeDataString($segment)
    }
    return ($encodedSegments -join '/')
}

function Test-TrustedDownloadUri {
    param([Parameter(Mandatory = $true)][Uri] $Uri)

    if ($Uri.Scheme -ne [Uri]::UriSchemeHttps) {
        return $false
    }

    $hostName = $Uri.DnsSafeHost.ToLowerInvariant()
    return (
        $hostName -eq $Script:OfficialHost -or
        $hostName.EndsWith('.huggingface.co', [StringComparison]::Ordinal) -or
        $hostName.EndsWith('.hf.co', [StringComparison]::Ordinal)
    )
}

function New-DownloadClient {
    $handler = [Net.Http.HttpClientHandler]::new()
    $handler.AllowAutoRedirect = $false
    $handler.UseProxy = $true
    $handler.Proxy = [Net.WebRequest]::DefaultWebProxy
    if ($null -ne $handler.Proxy) {
        $handler.Proxy.Credentials = [Net.CredentialCache]::DefaultCredentials
    }

    $client = [Net.Http.HttpClient]::new($handler, $true)
    $client.Timeout = [TimeSpan]::FromHours(12)
    $client.DefaultRequestHeaders.UserAgent.ParseAdd('gac-g05-pinned-downloader/1.0')
    return $client
}

function Get-DownloadResponse {
    param(
        [Parameter(Mandatory = $true)][Net.Http.HttpClient] $Client,
        [Parameter(Mandatory = $true)][Uri] $Uri,
        [Parameter(Mandatory = $true)][string] $Token,
        [Parameter(Mandatory = $true)][Int64] $RangeStart
    )

    $currentUri = $Uri
    for ($redirectCount = 0; $redirectCount -le 10; $redirectCount++) {
        if (-not (Test-TrustedDownloadUri -Uri $currentUri)) {
            throw [InvalidOperationException]::new(
                "The official endpoint redirected to an untrusted host: $($currentUri.DnsSafeHost)"
            )
        }

        $request = [Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::Get, $currentUri)
        if ($currentUri.DnsSafeHost -eq $Script:OfficialHost) {
            # The bearer token is never sent to a redirect/CDN host.
            $request.Headers.Authorization = [Net.Http.Headers.AuthenticationHeaderValue]::new(
                'Bearer',
                $Token
            )
        }
        if ($RangeStart -gt 0) {
            $request.Headers.Range = [Net.Http.Headers.RangeHeaderValue]::new($RangeStart, $null)
        }

        try {
            $response = $Client.SendAsync(
                $request,
                [Net.Http.HttpCompletionOption]::ResponseHeadersRead
            ).GetAwaiter().GetResult()
        }
        finally {
            $request.Dispose()
        }

        $statusCode = [int] $response.StatusCode
        if ($statusCode -eq 401 -or $statusCode -eq 403) {
            $response.Dispose()
            throw [UnauthorizedAccessException]::new(
                "Hugging Face returned HTTP $statusCode. Verify that the token is valid and its account has accepted the OpenGalaxea/G05 access terms."
            )
        }

        if ($statusCode -in 301, 302, 303, 307, 308) {
            $location = $response.Headers.Location
            $response.Dispose()
            if ($null -eq $location) {
                throw [Net.Http.HttpRequestException]::new('The download redirect had no Location header.')
            }
            if (-not $location.IsAbsoluteUri) {
                $location = [Uri]::new($currentUri, $location)
            }
            if (-not (Test-TrustedDownloadUri -Uri $location)) {
                throw [InvalidOperationException]::new(
                    "The official endpoint redirected to an untrusted host: $($location.DnsSafeHost)"
                )
            }
            $currentUri = $location
            continue
        }

        return $response
    }

    throw [Net.Http.HttpRequestException]::new('The download exceeded 10 redirects.')
}

function Copy-StreamWithIdleTimeout {
    param(
        [Parameter(Mandatory = $true)][IO.Stream] $Source,
        [Parameter(Mandatory = $true)][IO.Stream] $Destination,
        [Parameter(Mandatory = $true)][int] $BufferSize,
        [Parameter(Mandatory = $true)][int] $IdleTimeoutSeconds
    )

    $buffer = [byte[]]::new($BufferSize)
    while ($true) {
        $cancellation = [Threading.CancellationTokenSource]::new(
            [TimeSpan]::FromSeconds($IdleTimeoutSeconds)
        )
        try {
            $bytesRead = $Source.ReadAsync(
                $buffer,
                0,
                $buffer.Length,
                $cancellation.Token
            ).GetAwaiter().GetResult()
        }
        catch [OperationCanceledException] {
            throw [TimeoutException]::new(
                "The download produced no data for $IdleTimeoutSeconds seconds."
            )
        }
        finally {
            $cancellation.Dispose()
        }

        if ($bytesRead -eq 0) {
            break
        }
        $Destination.Write($buffer, 0, $bytesRead)
    }
}

function Save-ExpectedFile {
    param(
        [Parameter(Mandatory = $true)][Net.Http.HttpClient] $Client,
        [Parameter(Mandatory = $true)][string] $Token,
        [Parameter(Mandatory = $true)][string] $Root,
        [Parameter(Mandatory = $true)][string] $RelativePath,
        [Parameter(Mandatory = $true)][Int64] $ExpectedSize,
        [Parameter()][AllowNull()][string] $ExpectedSha256,
        [Parameter(Mandatory = $true)][int] $RetryCount,
        [Parameter(Mandatory = $true)][int] $CopyBufferSize
    )

    $targetPath = Get-SafeTargetPath -Root $Root -RelativePath $RelativePath
    $partPath = $targetPath + '.part'
    $targetDirectory = Split-Path -Parent $targetPath
    [IO.Directory]::CreateDirectory($targetDirectory) | Out-Null

    if (Test-Path -LiteralPath $targetPath -PathType Leaf) {
        $existingSize = (Get-Item -LiteralPath $targetPath).Length
        if ($existingSize -eq $ExpectedSize) {
            $existingHash = (Get-FileHash -LiteralPath $targetPath -Algorithm SHA256).Hash.ToLowerInvariant()
            if (-not [String]::IsNullOrEmpty($ExpectedSha256) -and $existingHash -ne $ExpectedSha256) {
                throw [IO.InvalidDataException]::new(
                    "Existing final file has the wrong SHA-256: $RelativePath"
                )
            }
            Write-Host "Already complete and verified: $RelativePath ($existingSize bytes)"
            return [pscustomobject] @{
                path = $RelativePath
                size = [Int64] $existingSize
                sha256 = $existingHash
            }
        }
        throw [IO.InvalidDataException]::new(
            "Existing final file has the wrong size: $RelativePath ($existingSize, expected $ExpectedSize). Move it aside before retrying."
        )
    }

    $encodedPath = ConvertTo-HuggingFacePath -RelativePath $RelativePath
    $downloadUri = [Uri]::new(
        $Script:OfficialBaseUri,
        "$($Script:RepoId)/resolve/$($Script:Revision)/$encodedPath`?download=true"
    )
    if ($downloadUri.DnsSafeHost -ne $Script:OfficialHost) {
        throw [InvalidOperationException]::new('The constructed download URI is not official Hugging Face.')
    }

    for ($attempt = 1; $attempt -le $RetryCount; $attempt++) {
        $rangeStart = [Int64] 0
        if (Test-Path -LiteralPath $partPath -PathType Leaf) {
            $rangeStart = (Get-Item -LiteralPath $partPath).Length
        }
        if ($rangeStart -gt $ExpectedSize) {
            throw [IO.InvalidDataException]::new(
                "Partial file is larger than expected: $RelativePath ($rangeStart, expected $ExpectedSize)."
            )
        }

        if ($rangeStart -eq $ExpectedSize) {
            Write-Host "Verifying completed partial file: $RelativePath"
            break
        }

        $response = $null
        try {
            $response = Get-DownloadResponse `
                -Client $Client `
                -Uri $downloadUri `
                -Token $Token `
                -RangeStart $rangeStart

            $statusCode = [int] $response.StatusCode
            if ($statusCode -eq 416 -and $rangeStart -eq $ExpectedSize) {
                break
            }
            if ($statusCode -ne 200 -and $statusCode -ne 206) {
                throw [Net.Http.HttpRequestException]::new(
                    "Download failed with HTTP $statusCode for $RelativePath."
                )
            }

            if ($statusCode -eq 206) {
                $contentRange = $response.Content.Headers.ContentRange
                if ($null -eq $contentRange -or $contentRange.From -ne $rangeStart) {
                    throw [IO.InvalidDataException]::new(
                        "Server returned an unexpected byte range for $RelativePath."
                    )
                }
            }

            $append = $rangeStart -gt 0 -and $statusCode -eq 206
            if ($rangeStart -gt 0 -and -not $append) {
                Write-Host "Server did not honor the byte range; restarting: $RelativePath"
                $rangeStart = 0
            }

            $fileMode = if ($append) { [IO.FileMode]::Append } else { [IO.FileMode]::Create }
            Write-Host "Downloading: $RelativePath (offset $rangeStart of $ExpectedSize)"
            $sourceStream = $response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
            $targetStream = [IO.FileStream]::new(
                $partPath,
                $fileMode,
                [IO.FileAccess]::Write,
                [IO.FileShare]::None,
                $CopyBufferSize,
                [IO.FileOptions]::SequentialScan
            )
            try {
                Copy-StreamWithIdleTimeout `
                    -Source $sourceStream `
                    -Destination $targetStream `
                    -BufferSize $CopyBufferSize `
                    -IdleTimeoutSeconds 90
                $targetStream.Flush($true)
            }
            finally {
                $targetStream.Dispose()
                $sourceStream.Dispose()
            }

            $downloadedSize = (Get-Item -LiteralPath $partPath).Length
            if ($downloadedSize -eq $ExpectedSize) {
                break
            }
            if ($downloadedSize -gt $ExpectedSize) {
                throw [IO.InvalidDataException]::new(
                    "Downloaded file is larger than expected: $RelativePath ($downloadedSize, expected $ExpectedSize)."
                )
            }
            throw [IO.EndOfStreamException]::new(
                "Download ended early: $RelativePath ($downloadedSize of $ExpectedSize bytes)."
            )
        }
        catch [UnauthorizedAccessException] {
            throw
        }
        catch {
            if ($attempt -ge $RetryCount) {
                throw
            }
            $delaySeconds = [Math]::Min(30, [Math]::Pow(2, $attempt))
            Write-Warning "Attempt $attempt failed for $RelativePath; preserving .part and retrying in $delaySeconds seconds."
            Start-Sleep -Seconds $delaySeconds
        }
        finally {
            if ($null -ne $response) {
                $response.Dispose()
            }
        }
    }

    if (-not (Test-Path -LiteralPath $partPath -PathType Leaf)) {
        throw [IO.FileNotFoundException]::new("Partial download is missing: $RelativePath")
    }
    $finalSize = (Get-Item -LiteralPath $partPath).Length
    if ($finalSize -ne $ExpectedSize) {
        throw [IO.InvalidDataException]::new(
            "Downloaded file has the wrong size: $RelativePath ($finalSize, expected $ExpectedSize)."
        )
    }

    Write-Host "Computing SHA-256: $RelativePath"
    $hash = (Get-FileHash -LiteralPath $partPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if (-not [String]::IsNullOrEmpty($ExpectedSha256) -and $hash -ne $ExpectedSha256) {
        throw [IO.InvalidDataException]::new(
            "Downloaded file has the wrong official SHA-256: $RelativePath"
        )
    }
    Move-Item -LiteralPath $partPath -Destination $targetPath -Force
    return [pscustomobject] @{
        path = $RelativePath
        size = [Int64] $finalSize
        sha256 = $hash
    }
}

function Write-DownloadManifest {
    param(
        [Parameter(Mandatory = $true)][string] $Root,
        [Parameter(Mandatory = $true)][object[]] $Files,
        [Parameter()][switch] $SmallFilesOnly
    )

    $manifestName = if ($SmallFilesOnly) {
        'g05-base-small-files-download-manifest.json'
    }
    else {
        'g05-base-download-manifest.json'
    }
    $manifestPath = Get-SafeTargetPath -Root $Root -RelativePath $manifestName
    $manifestPartPath = $manifestPath + '.part'
    $manifest = [ordered] @{
        schema_version = 1
        repository = $Script:RepoId
        revision = $Script:Revision
        source = $Script:OfficialBaseUri.AbsoluteUri
        generated_utc = [DateTime]::UtcNow.ToString('o')
        scope = if ($SmallFilesOnly) { 'small-files-only' } else { 'g05-base' }
        file_count = $Files.Count
        files = $Files
    }
    $json = $manifest | ConvertTo-Json -Depth 6
    $utf8WithoutBom = [Text.UTF8Encoding]::new($false)
    [IO.File]::WriteAllText($manifestPartPath, $json + [Environment]::NewLine, $utf8WithoutBom)
    Move-Item -LiteralPath $manifestPartPath -Destination $manifestPath -Force
    return $manifestPath
}

$destinationPath = [IO.Path]::GetFullPath($Destination)
[IO.Directory]::CreateDirectory($destinationPath) | Out-Null
Assert-DestinationOutsideGitWorktree -Path $destinationPath

$token = $null
$client = $null
try {
    $token = Get-HuggingFaceToken -DisablePrompt:$NonInteractive
    $client = New-DownloadClient
    $results = [Collections.Generic.List[object]]::new()
    $bufferSize = $BufferSizeMiB * 1MB

    foreach ($entry in $Script:ExpectedFiles.GetEnumerator()) {
        if ($SmallFilesOnly -and $Script:ExpectedSha256.ContainsKey($entry.Key)) {
            Write-Host "Staged separately as an LFS object: $($entry.Key)"
            continue
        }
        $result = Save-ExpectedFile `
            -Client $client `
            -Token $token `
            -Root $destinationPath `
            -RelativePath $entry.Key `
            -ExpectedSize $entry.Value `
            -ExpectedSha256 $Script:ExpectedSha256[$entry.Key] `
            -RetryCount $MaxRetries `
            -CopyBufferSize $bufferSize
        $results.Add($result)
    }

    $manifestPath = Write-DownloadManifest `
        -Root $destinationPath `
        -Files $results.ToArray() `
        -SmallFilesOnly:$SmallFilesOnly
    if ($SmallFilesOnly) {
        Write-Host "Small-file download complete; LFS objects were excluded. Integrity manifest: $manifestPath"
    }
    else {
        Write-Host "Download complete. Integrity manifest: $manifestPath"
    }
}
finally {
    if ($null -ne $client) {
        $client.Dispose()
    }
    $token = $null
}
