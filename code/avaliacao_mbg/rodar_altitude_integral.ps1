param([Parameter(Mandatory=$true)][string]$Python)

$ErrorActionPreference = 'Stop'
$ensaio = Join-Path $PSScriptRoot 'ensaio_altitude_integral.py'
for ($tentativa = 1; $tentativa -le 3; $tentativa++) {
    Write-Output ("{0:o} Iniciando ensaio, tentativa {1}" -f [DateTimeOffset]::Now, $tentativa)
    & $Python -u $ensaio
    $codigo = $LASTEXITCODE
    if ($codigo -eq 0) {
        & $Python -u (Join-Path $PSScriptRoot 'resumir_altitude_integral.py')
        exit $LASTEXITCODE
    }
    Write-Output ("{0:o} Ensaio interrompido, código {1}" -f [DateTimeOffset]::Now, $codigo)
    if ($tentativa -lt 3) { Start-Sleep -Seconds 30 }
}
exit $codigo
