<#
.SYNOPSIS
  Ejecuta un script del proyecto y junta la evidencia de uso de recursos (PLAN.md, seccion 5).

.DESCRIPTION
  Mientras corre el comando deja en resultados\evidencia\<Version>\:
    log.txt          salida completa del script (lo que antes era log_<version>.txt)
    cpu.csv          typeperf: % de uso de CADA procesador logico, cada 1 s
    gpu.csv          nvidia-smi: uso, memoria, temperatura, reloj y potencia de la GPU, cada 200 ms
    captura_*.jpg    capturas de pantalla completas cada -CadaSegundos (la primera a los
                     -PrimeraCaptura s), y una al final
    resumen.txt      estado de reposo antes de empezar y promedios durante la ejecucion

  Antes de correrlo, dejar en primer plano lo que se quiere capturar (Administrador de
  tareas en la vista correcta, y la ventana de nvidia-smi si se pidio con -VentanaNvidiaSmi).

.EXAMPLE
  .\evidencia.ps1 -Version optimizado -Comando "optimizado.py"
  .\evidencia.ps1 -Version gpu_30epocas -Comando "optimizado.py --epocas 30 --corridas 1 --no-registrar" -CadaSegundos 5 -VentanaNvidiaSmi
#>
param(
    [Parameter(Mandatory = $true)] [string]$Version,
    [Parameter(Mandatory = $true)] [string]$Comando,
    [int]$CadaSegundos = 30,
    [int]$PrimeraCaptura = 10,
    [switch]$VentanaNvidiaSmi,
    [string]$Python = "$env:USERPROFILE\.venvs\asl-cuda\Scripts\python.exe"
)
$ErrorActionPreference = "Stop"
$raiz = $PSScriptRoot
$dir = Join-Path $raiz "resultados\evidencia\$Version"
New-Item -ItemType Directory -Force $dir | Out-Null
if (-not (Test-Path $Python)) { throw "No existe el Python del venv: $Python" }

# ---- Estado de reposo antes de empezar (chequeo de sanidad del PLAN) ----
$cpuReposo = [math]::Round(((Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 1 -MaxSamples 3).CounterSamples |
    Measure-Object CookedValue -Average).Average, 1)
$gpuReposo = (& nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader) -join " "
$plan = (powercfg /getactivescheme) -join " "
$overlay = (Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes" -ErrorAction SilentlyContinue).ActiveOverlayAcPowerScheme
$modo = if ($overlay -eq "ded574b5-45a0-4f42-8737-46345c09c238") { "Best performance" } elseif ($overlay) { $overlay } else { "Balanced" }
$enchufe = (Get-CimInstance -ClassName BatteryStatus -Namespace root/wmi -ErrorAction SilentlyContinue | Select-Object -First 1).PowerOnline
$inicio = Get-Date
$resumen = Join-Path $dir "resumen.txt"
# PowerShell 5.1 escribe UTF-16 con Tee-Object/Out-File por defecto; todo va en UTF-8.
function Anotar([string]$texto, [switch]$Agregar) {
    Write-Host $texto
    if ($Agregar) { Add-Content -Path $resumen -Value $texto -Encoding UTF8 }
    else { Set-Content -Path $resumen -Value $texto -Encoding UTF8 }
}
Anotar @"
version:            $Version
comando:            python $Comando
inicio:             $($inicio.ToString("s"))
CPU en reposo:      $cpuReposo %   (PLAN: < 5 %)
GPU en reposo:      $gpuReposo   (uso %, memoria, temperatura)
modo de energia:    $modo   ($plan)
conectada a la red: $enchufe
"@
if ($cpuReposo -gt 10) { Write-Warning "La CPU no esta en reposo ($cpuReposo %). Revisar el Administrador de tareas antes de aceptar los numeros." }

# ---- Monitores en segundo plano ----
$cpuCsv = Join-Path $dir "cpu.csv"
$gpuCsv = Join-Path $dir "gpu.csv"
$typeperf = Start-Process typeperf -PassThru -WindowStyle Hidden -ArgumentList @(
    '"\Processor(*)\% Processor Time"', "-si", "1", "-o", "`"$cpuCsv`"", "-y")
$ventana = $null
if ($VentanaNvidiaSmi) {
    $ventana = Start-Process (Join-Path $PSHOME "powershell.exe") -PassThru -ArgumentList @(
        "-NoExit", "-Command", "`$Host.UI.RawUI.WindowTitle='nvidia-smi'; nvidia-smi -l 1")
}

# Capturas: un job aparte que fotografia la pantalla mientras exista la bandera.
$bandera = Join-Path $dir ".capturando"
New-Item -ItemType File -Force $bandera | Out-Null
$capturador = Start-Job -ArgumentList $dir, $bandera, $CadaSegundos, $PrimeraCaptura -ScriptBlock {
    param($dir, $bandera, $cada, $primera)
    Add-Type -AssemblyName System.Windows.Forms, System.Drawing
    Add-Type -Namespace W -Name U -MemberDefinition '[DllImport("user32.dll")] public static extern bool SetProcessDPIAware();'
    [void][W.U]::SetProcessDPIAware()      # resolucion real, sin escalado de Windows
    $jpg = [System.Drawing.Imaging.ImageCodecInfo]::GetImageEncoders() | Where-Object { $_.MimeType -eq "image/jpeg" }
    $par = New-Object System.Drawing.Imaging.EncoderParameters 1
    $par.Param[0] = New-Object System.Drawing.Imaging.EncoderParameter ([System.Drawing.Imaging.Encoder]::Quality), 85L
    function Capturar($nombre) {
        $b = [System.Windows.Forms.SystemInformation]::VirtualScreen
        $bmp = New-Object System.Drawing.Bitmap $b.Width, $b.Height
        $g = [System.Drawing.Graphics]::FromImage($bmp)
        $g.CopyFromScreen($b.Left, $b.Top, 0, 0, $bmp.Size)
        $bmp.Save((Join-Path $dir $nombre), $jpg, $par)
        $g.Dispose(); $bmp.Dispose()
    }
    $espera = $primera
    while (Test-Path $bandera) {
        for ($i = 0; $i -lt $espera * 4 -and (Test-Path $bandera); $i++) { Start-Sleep -Milliseconds 250 }
        if (Test-Path $bandera) { Capturar ("captura_" + (Get-Date -Format "HHmmss") + ".jpg") }
        $espera = $cada
    }
    Capturar ("captura_" + (Get-Date -Format "HHmmss") + "_final.jpg")
}

# GPU: una consulta a nvidia-smi por muestra (~5 por segundo), escrita al archivo al
# instante. NO se usa "nvidia-smi -lms 200 -f archivo": escribe en bloques y, al cerrarlo,
# se perdian hasta ~20 s del final de la ejecucion (justo la ultima corrida).
$muestreador = Start-Job -ArgumentList $gpuCsv, $bandera -ScriptBlock {
    param($csv, $bandera)
    $q = "timestamp,utilization.gpu,utilization.memory,memory.used,temperature.gpu,clocks.sm,power.draw"
    Set-Content -Path $csv -Encoding UTF8 -Value ((& nvidia-smi "--query-gpu=$q" --format=csv) | Select-Object -First 1)
    $finTras = $null
    while ($true) {
        $linea = & nvidia-smi "--query-gpu=$q" --format=csv,noheader
        Add-Content -Path $csv -Encoding UTF8 -Value $linea
        # tras quitar la bandera, 3 s mas de muestras para ver la bajada en las graficas
        if (-not (Test-Path $bandera)) {
            if (-not $finTras) { $finTras = (Get-Date).AddSeconds(3) }
            elseif ((Get-Date) -gt $finTras) { break }
        }
        Start-Sleep -Milliseconds 100
    }
}

# ---- El comando ----
# cmd /c junta stdout y stderr sin que PowerShell 5.1 convierta stderr en errores;
# -u: salida sin buffer, para que el log avance en vivo.
$log = Join-Path $dir "log.txt"
Push-Location $raiz
try {
    if (Test-Path $log) { Remove-Item $log }
    cmd /c "`"$Python`" -u -W ignore $Comando 2>&1" | ForEach-Object {
        $_
        Add-Content -Path $log -Value $_ -Encoding UTF8
    }
    $codigo = $LASTEXITCODE
}
finally {
    Pop-Location
    Remove-Item $bandera -ErrorAction SilentlyContinue
    Wait-Job $capturador, $muestreador -Timeout 30 | Out-Null
    Remove-Job $capturador, $muestreador -Force
    foreach ($p in @($typeperf, $ventana)) {
        # taskkill /T cierra tambien los hijos (el nvidia-smi -l 1 dentro de la ventana);
        # con Stop-Process solo, la ventana quedaba abierta refrescando para siempre.
        if ($p -and -not $p.HasExited) { taskkill /PID $p.Id /T /F 2>&1 | Out-Null }
    }
}

# ---- Resumen de la ejecucion ----
$fin = Get-Date
# Encabezado propio: los de nvidia-smi traen espacios y unidades.
$filas = Get-Content $gpuCsv | Select-Object -Skip 1 |
    ConvertFrom-Csv -Header hora, uso, uso_mem, mem, temp, reloj, potencia
function Num($v) { [double](($v -replace "[^\d.]", "")) }
$gpu = $filas | ForEach-Object { Num $_.uso }
$gpuMax = ($gpu | Measure-Object -Maximum).Maximum
$gpuMedia = [math]::Round(($gpu | Measure-Object -Average).Average, 1)
$memMax = ($filas | ForEach-Object { Num $_.mem } | Measure-Object -Maximum).Maximum
$tempMax = ($filas | ForEach-Object { Num $_.temp } | Measure-Object -Maximum).Maximum
$capturas = (Get-ChildItem $dir -Filter "captura_*.jpg").Count
Anotar -Agregar @"
fin:                $($fin.ToString("s"))  (duracion $([math]::Round(($fin - $inicio).TotalMinutes, 1)) min)
codigo de salida:   $codigo
GPU uso medio/max:  $gpuMedia % / $gpuMax %
GPU memoria max:    $memMax MiB
GPU temperatura max: $tempMax C
capturas:           $capturas
"@
exit $codigo
