# game-lens: first-time setup on a new computer. Run setup.bat (or this file) once; afterwards open game-lens
# with start.bat or the Desktop icon it makes.
#   .\setup.ps1            install everything (the translation model is downloaded by game-lens itself)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$LlamaUrl = "https://github.com/ggml-org/Llama-Windows/releases"

function Step($t) { Write-Host ""; Write-Host "== $t" -ForegroundColor Cyan }

Step "1/4 Python 3.14"
$ok = $false
try { & py -3.14 -c "import sys" 2>$null; $ok = ($LASTEXITCODE -eq 0) } catch { }
if (-not $ok) {
    Write-Host "ไม่พบ Python 3.14" -ForegroundColor Yellow
    Write-Host "ติดตั้งจาก https://www.python.org/downloads/ (เลือก 3.14) หรือพิมพ์:  winget install Python.Python.3.14"
    Write-Host "แล้วรัน setup.bat อีกครั้ง"
    exit 1
}
Write-Host "พบแล้ว"

Step "2/4 ติดตั้งส่วนประกอบของโปรแกรม (ครั้งแรกใช้เวลาหลายนาที)"
if (-not (Test-Path ".venv\Scripts\python.exe")) { & py -3.14 -m venv .venv }
& .venv\Scripts\python.exe -m pip install --disable-pip-version-check -q --upgrade pip
& .venv\Scripts\python.exe -m pip install --disable-pip-version-check -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Host "ติดตั้งไม่สำเร็จ ดูข้อความด้านบน" -ForegroundColor Red; exit 1 }

Step "3/4 llama (ตัวรันโมเดลแปลในเครื่อง)"
if (Get-Command llama -ErrorAction SilentlyContinue) {
    Write-Host "พบแล้ว"
} else {
    Write-Host "ยังไม่มีคำสั่ง llama — ถ้าจะใช้โมเดลแปลในเครื่อง (Hy-MT2) ให้ติดตั้ง Llama for Windows" -ForegroundColor Yellow
    Write-Host "  ดาวน์โหลด .msixbundle ล่าสุดจาก $LlamaUrl แล้วดับเบิลคลิกติดตั้ง"
    Write-Host "  (ไม่ติดตั้งก็ได้ ถ้าจะใช้ Google / DeepL / Azure แทน)"
}

Step "4/4 โมเดลแปล"
Write-Host "game-lens ดาวน์โหลดเองตอนเปิดครั้งแรก: Index-Translate 9B (5.8 GB) ถ้าการ์ดจอมี 10 GB ขึ้นไป ไม่อย่างนั้น Hy-MT2 1.8B (1.1 GB)"

Write-Host ""
Write-Host "ติดตั้งเสร็จ — เปิดโปรแกรมด้วย start.bat แล้วกดปุ่ม 'สร้างไอคอนบน Desktop'" -ForegroundColor Green
