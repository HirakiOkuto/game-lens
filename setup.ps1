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
    Write-Host "กรุณาติดตั้งจาก https://www.python.org/downloads/ (เลือกรุ่น 3.14) หรือใช้คำสั่ง:  winget install Python.Python.3.14"
    Write-Host "จากนั้นเรียกใช้ setup.bat อีกครั้ง"
    exit 1
}
Write-Host "พบ Python 3.14"

Step "2/4 ติดตั้งส่วนประกอบของโปรแกรม (การติดตั้งครั้งแรกอาจใช้เวลาหลายนาที)"
if (-not (Test-Path ".venv\Scripts\python.exe")) { & py -3.14 -m venv .venv }
& .venv\Scripts\python.exe -m pip install --disable-pip-version-check -q --upgrade pip
& .venv\Scripts\python.exe -m pip install --disable-pip-version-check -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Host "การติดตั้งไม่สำเร็จ กรุณาตรวจสอบข้อความด้านบน" -ForegroundColor Red; exit 1 }

Step "3/4 llama (โปรแกรมสำหรับเรียกใช้โมเดลแปลภาษาในเครื่อง)"
if (Get-Command llama -ErrorAction SilentlyContinue) {
    Write-Host "พบคำสั่ง llama"
} else {
    Write-Host "ไม่พบคำสั่ง llama — หากต้องการใช้โมเดลแปลภาษาในเครื่อง กรุณาติดตั้ง Llama for Windows" -ForegroundColor Yellow
    Write-Host "  โดยดาวน์โหลดไฟล์ .msixbundle รุ่นล่าสุดจาก $LlamaUrl แล้วดับเบิลคลิกเพื่อติดตั้ง"
    Write-Host "  (ไม่จำเป็นต้องติดตั้ง หากใช้ระบบแปลภาษาออนไลน์ เช่น DeepL หรือ Azure)"
}

Step "4/4 โมเดล"
Write-Host "game-lens จะดาวน์โหลดไฟล์ต่อไปนี้โดยอัตโนมัติเมื่อเปิดใช้งานครั้งแรก: ระบบอ่านข้อความ (ประมาณ 87 MB) และโมเดลแปลภาษา"
Write-Host "Index-Translate 9B (5.8 GB) สำหรับการ์ดแสดงผลที่มีหน่วยความจำ 10 GB ขึ้นไป หรือ Hy-MT2 1.8B (1.1 GB)"

Write-Host ""
Write-Host "การติดตั้งเสร็จสมบูรณ์ — เปิดโปรแกรมด้วย start.bat แล้วคลิกปุ่ม 'สร้างไอคอนบนเดสก์ท็อป'" -ForegroundColor Green
