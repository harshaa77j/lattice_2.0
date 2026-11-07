# Lattice one-step installer for Windows. Run it from PowerShell with:
#   irm "<server>/install.ps1?code=<invite code>" | iex
# It installs into %LOCALAPPDATA%\Lattice, needs no admin rights, and can be run
# again safely: finished steps are reused, and the app is updated to the latest version.

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"          # makes downloads much faster in Windows PowerShell
$Server = "__LATTICE_SERVER__"
$Code = "__LATTICE_CODE__"
$Dir = Join-Path $env:LOCALAPPDATA "Lattice"

function Step($text) { Write-Host "`n-> $text" -ForegroundColor Cyan }

function Find-Uv {
    foreach ($p in @("$env:USERPROFILE\.local\bin\uv.exe", "$env:USERPROFILE\.cargo\bin\uv.exe")) {
        if (Test-Path $p) { return $p }
    }
    $cmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

try {
    Write-Host "Lattice volunteer setup" -ForegroundColor White
    Write-Host "Installing into $Dir"

    Step "Downloading Lattice"
    $zip = Join-Path $env:TEMP "lattice-volunteer.zip"
    $tmp = Join-Path $env:TEMP "lattice-extract"
    Invoke-WebRequest -Uri "$Server/download/windows?code=$Code" -OutFile $zip -UseBasicParsing
    if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
    Expand-Archive -Path $zip -DestinationPath $tmp -Force
    New-Item -ItemType Directory -Force -Path $Dir | Out-Null
    # Copy the new code over any earlier install; .venv and data are kept.
    Copy-Item -Path (Join-Path $tmp "lattice-volunteer\*") -Destination $Dir -Recurse -Force
    Remove-Item $zip, $tmp -Recurse -Force

    Step "Getting uv (it installs a private Python; no admin rights needed)"
    $uv = Find-Uv
    if (-not $uv) {
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        $uv = Find-Uv
    }
    if (-not $uv) { throw "uv could not be installed." }

    Set-Location $Dir
    Step "Creating a private Python 3.10 environment"
    if (-not (Test-Path ".venv\Scripts\python.exe")) {
        & $uv venv --python 3.10 .venv
        if ($LASTEXITCODE) { throw "Creating the Python environment failed." }
    }

    Step "Installing PyTorch and other packages (several hundred MB the first time)"
    & $uv pip install --python ".venv\Scripts\python.exe" -r requirements-node.txt
    if ($LASTEXITCODE) { throw "Installing the packages failed." }

    Step "Checking everything and downloading the training data"
    & ".venv\Scripts\python.exe" -m node.prepare
    if ($LASTEXITCODE) { throw "The setup check failed (see the message above)." }

    Step "Creating the Lattice shortcuts"
    $shell = New-Object -ComObject WScript.Shell
    $places = @([Environment]::GetFolderPath("Desktop"), (Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"))
    foreach ($folder in $places) {
        $link = $shell.CreateShortcut((Join-Path $folder "Lattice.lnk"))
        $link.TargetPath = Join-Path $Dir ".venv\Scripts\pythonw.exe"
        $link.Arguments = "-m node.app"
        $link.WorkingDirectory = $Dir
        $link.Description = "Lattice volunteer"
        $link.Save()
    }

    Write-Host "`nDone! Lattice is installed and opening now." -ForegroundColor Green
    Write-Host "Next time, start it from the Lattice icon on your Desktop or in the Start menu."
    Start-Process -FilePath (Join-Path $Dir ".venv\Scripts\pythonw.exe") -ArgumentList "-m node.app" -WorkingDirectory $Dir
}
catch {
    Write-Host "`nSetup did not finish: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Check your internet connection and run the same command again; finished steps are reused."
}