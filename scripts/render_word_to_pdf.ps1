param(
    [Parameter(Mandatory = $true)]
    [string[]]$InputPath,

    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory
)

$ErrorActionPreference = "Stop"
$outputRoot = [System.IO.Path]::GetFullPath($OutputDirectory)
[System.IO.Directory]::CreateDirectory($outputRoot) | Out-Null

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0

try {
    foreach ($item in $InputPath) {
        $source = [System.IO.Path]::GetFullPath($item)
        $pdfName = [System.IO.Path]::GetFileNameWithoutExtension($source) + ".pdf"
        $destination = [System.IO.Path]::Combine($outputRoot, $pdfName)

        $document = $word.Documents.Open($source, $false, $true)
        try {
            $document.ExportAsFixedFormat($destination, 17)
        }
        finally {
            $document.Close($false)
        }
    }
}
finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($word) | Out-Null
}
