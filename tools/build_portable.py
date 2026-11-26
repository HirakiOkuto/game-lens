r"""Build the portable game-lens: one folder (and a zip of it) that runs on a computer with nothing installed.
Unzip, double-click game-lens.exe. The text reader's models (~87 MB) and the translation model (1.1-5.8 GB, by
graphics card) are downloaded by the app the first time.

    .venv\Scripts\python.exe tools\build_portable.py --llama path\to\llama-bXXXX-bin-win-vulkan-x64.zip

What goes in: the app as committed (no tests), a copy of this computer's Python 3.14 (without its docs, tests,
tkinter), the packages of .venv without the parts of Qt the app does not use, llama.cpp (Vulkan: works on NVIDIA,
AMD and Intel cards with no extra runtime) and a small launcher, game-lens.exe. Nothing of this computer's use
(keys, token, cache, logs) goes in.
"""
from __future__ import annotations

import argparse
import fnmatch
import io
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from public import MARKER, MARKER_TEXT, has_key, package_licenses, public_file  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "game-lens"
APP_LEAVE_OUT = ("tests/", "server/tests/", "tools/", "experiments/", "setup.ps1", "setup.bat", "start.ps1", "start.bat")
PY_LEAVE_OUT = ["Doc", "include", "libs", "Scripts", "tcl", "Tools", "Lib/test", "Lib/idlelib", "Lib/tkinter",
                "Lib/turtledemo", "Lib/site-packages", "DLLs/_tkinter.pyd", "DLLs/tcl*.dll", "DLLs/tk*.dll",
                "NEWS.txt", "python.pdb"]
SITE_LEAVE_OUT = ["pip", "pip-*", "_pytest", "pytest", "pytest-*", "py.py", "pluggy", "pluggy-*", "iniconfig",
                  "iniconfig-*", "pygments", "pygments-*", "__pycache__",
                  "cv2/opencv_videoio_ffmpeg*",  # reading video files: only the replay tool does that
                  "PySide6/opengl32sw.dll",  # software OpenGL: the app draws with QPainter
                  "PySide6/av*.dll", "PySide6/sw*.dll",  # FFmpeg for Qt Multimedia
                  "PySide6/*.exe", "*.pyi", "PySide6/QtAxContainer*", "PySide6/Qt6PrintSupport*", "PySide6/QtPrintSupport*",
                  # the text reader's models: game-lens fetches the ones the game's language needs the first time
                  "rapidocr/models/*.onnx",
                  # only onnxruntime's model-converting tools use these (symbolic shapes, quantizing): reading does not
                  "sympy", "sympy-*", "isympy.py", "mpmath", "mpmath-*", "onnxruntime/transformers", "onnxruntime/quantization",
                  "onnxruntime/tools", "onnxruntime/datasets",
                  "onnxruntime/capi/onnxruntime.dll",  # the C API for other languages: Python has its own (.pyd)
                  "PySide6/translations",  # Qt's own dialogs in other languages: game-lens has none of them
                  "PIL/_avif*"]  # AVIF pictures
# Qt modules game-lens never loads (it uses Core, Gui, Widgets, Network): their DLLs, Python modules and tools
QT_UNUSED = ["WebEngine", "WebChannel", "WebSockets", "WebView", "Quick", "Qml", "3D", "Pdf", "Multimedia",
             "Designer", "Charts", "DataVisualization", "Graphs", "Location", "Bluetooth", "Sensors", "ShaderTools",
             "SpatialAudio", "Scxml", "StateMachine", "Positioning", "RemoteObjects", "HttpServer", "TextToSpeech",
             "VirtualKeyboard", "Nfc", "SerialBus", "SerialPort", "Test", "Sql", "Svg", "UiTools", "Help",
             "OpenGL", "Concurrent", "DBus", "Xml", "NetworkAuth", "Canvas", "Lottie", "Labs"]
QT_TOOLS = ["designer.exe", "linguist.exe", "assistant.exe", "lrelease.exe", "lupdate.exe", "qmlls.exe",
            "QtWebEngineProcess.exe", "qml", "resources", "examples", "glue", "include", "typesystems", "scripts",
            "doc"]

LAUNCHER = r'''
using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

static class Launcher {
    [STAThread]
    static int Main(string[] args) {
        string dir = AppDomain.CurrentDomain.BaseDirectory;
        string py = Path.Combine(dir, "python", "pythonw.exe");
        if (!File.Exists(py)) {
            MessageBox.Show("ไม่พบ " + py + "\nแตกไฟล์ zip ให้ครบทั้งโฟลเดอร์ก่อนเปิด", "game-lens");
            return 1;
        }
        string rest = "";
        foreach (string a in args) rest += " \"" + a.Replace("\"", "") + "\"";
        var si = new ProcessStartInfo(py, "-m gamelens" + rest);
        si.WorkingDirectory = dir;
        si.UseShellExecute = false;
        si.EnvironmentVariables["PYTHONNOUSERSITE"] = "1";
        si.EnvironmentVariables.Remove("PYTHONHOME");
        si.EnvironmentVariables.Remove("PYTHONPATH");
        Process.Start(si);
        return 0;
    }
}
'''


def leave_out(rel: str, patterns: list[str]) -> bool:
    rel = rel.replace("\\", "/")
    return any(fnmatch.fnmatch(rel, p) or rel.startswith(p.rstrip("*") + "/") for p in patterns)


def copy_tree(src: Path, dst: Path, skip) -> None:
    for p in src.rglob("*"):
        rel = p.relative_to(src).as_posix()
        if p.is_dir() or skip(rel):
            continue
        t = dst / rel
        t.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, t)


def qt_unused(rel: str) -> bool:
    """rel is a path inside site-packages."""
    if not rel.startswith("PySide6/"):
        return False
    name = rel.split("/", 1)[1]
    if name.split("/")[0] in QT_TOOLS:
        return True
    top = name.split("/")[-1]
    for m in QT_UNUSED:
        if fnmatch.fnmatch(top, f"Qt6{m}*") or fnmatch.fnmatch(top, f"Qt{m}*") or fnmatch.fnmatch(top, f"qt{m.lower()}*"):
            return True
    return name.startswith("translations/qtwebengine") or name.startswith("plugins/") and any(
        k in name.lower() for k in ("webengine", "qml", "quick", "multimedia", "sceneparser", "geometryloaders",
                                    "renderers", "position", "sensors", "sqldrivers", "designer", "texttospeech",
                                    "canbus", "assetimporters", "renderplugins"))


def build_app(personal: bool) -> None:
    raw = subprocess.run(["git", "archive", "--format=zip", "HEAD"], cwd=ROOT, capture_output=True, check=True).stdout
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for info in z.infolist():
            name = info.filename
            if info.is_dir() or name.startswith(APP_LEAVE_OUT):
                continue
            data = z.read(info)
            if name == "settings.json":
                s = json.loads(data)
                s["models_dir"] = "models"  # the app's own folder, not this computer's shared one
                data = json.dumps(s, ensure_ascii=False, indent=2).encode("utf-8")
            if not personal:
                data = public_file(name, data)
            t = OUT / name
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_bytes(data)
    if not personal:
        (OUT / MARKER).write_bytes(MARKER_TEXT)
    (OUT / "models").mkdir(exist_ok=True)
    (OUT / "models" / "วางไฟล์โมเดล .gguf ไว้ที่นี่.txt").write_text(
        "game-lens ดาวน์โหลดโมเดลแปลภาษามาไว้ที่นี่เองเมื่อเปิดใช้ครั้งแรก\r\n", encoding="utf-8")


def build_python() -> None:
    base = Path(sys.base_prefix)
    copy_tree(base, OUT / "python", lambda rel: leave_out(rel, PY_LEAVE_OUT) or "__pycache__" in rel)
    site = Path(sys.prefix) / "Lib" / "site-packages"  # this .venv
    dst = OUT / "python" / "Lib" / "site-packages"
    copy_tree(site, dst, lambda rel: leave_out(rel, SITE_LEAVE_OUT) or "/__pycache__/" in f"/{rel}" or qt_unused(rel))
    # the licenses that go along: Python's own, and a list of every package's (their texts are in their dist-info)
    (OUT / "licenses").mkdir(exist_ok=True)
    shutil.copyfile(base / "LICENSE.txt", OUT / "licenses" / "Python-LICENSE.txt")
    (OUT / "licenses" / "python-packages.txt").write_bytes(package_licenses(dst))


def build_llama(src_zip: Path) -> None:
    with zipfile.ZipFile(src_zip) as z:
        for info in z.infolist():
            n = info.filename
            keep = n.lower().endswith(".dll") or n in ("llama.exe",) or n.upper().startswith("LICENSE")
            if keep and not info.is_dir():
                (OUT / "llama").mkdir(exist_ok=True)
                (OUT / "llama" / Path(n).name).write_bytes(z.read(info))


def build_launcher(icon: Path | None) -> None:
    csc = Path(r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe")
    src = ROOT / "build" / "launcher.cs"
    src.write_text(LAUNCHER, encoding="utf-8-sig")
    args = [str(csc), "/nologo", "/target:winexe", "/optimize", "/codepage:65001", "/r:System.Windows.Forms.dll",
            f"/out:{OUT / 'game-lens.exe'}", str(src)]
    if icon and icon.exists():
        args.insert(-1, f"/win32icon:{icon}")
    subprocess.run(args, check=True)


def readme() -> None:
    (OUT / "อ่านก่อน.txt").write_text(
        "game-lens (portable) — แปลข้อความในเกมเป็นภาษาไทยหรืออังกฤษ แสดงทับบนหน้าจอเกม\r\n\r\n"
        "1. แตกไฟล์ zip ไว้ในโฟลเดอร์ที่เขียนไฟล์ได้ เช่น D:\\game-lens (ไม่ควรไว้ใน Program Files)\r\n"
        "2. ดับเบิลคลิก game-lens.exe\r\n"
        "   หาก Windows แสดง 'Windows protected your PC' ให้กด More info แล้วกด Run anyway (โปรแกรมไม่ได้ลงลายเซ็นดิจิทัล)\r\n"
        "   ครั้งแรกโปรแกรมจะดาวน์โหลดตัวอ่านข้อความ (~87 MB) และโมเดลแปลภาษา (1.1-5.8 GB ตามการ์ดจอ) เอง\r\n"
        "   โดยแสดงความคืบหน้าในหน้าต่าง game-lens\r\n"
        "3. ตั้งเกมเป็นแบบ Borderless หรือ Windowed แล้วดำเนินการตาม 3 ขั้นตอนในหน้าต่าง game-lens\r\n\r\n"
        "หน้าต่างของโปรแกรมเปลี่ยนเป็นภาษาไทยได้ที่มุมขวาบน (🌐)\r\n"
        "แก้คำแปลที่ผิดได้เอง: กด F7 แล้วกด ✎ ท้ายคำแปล • เมนูหน้าใดไม่ถูกแปล กด F9\r\n"
        "รายละเอียดทั้งหมดอยู่ใน README.md\r\n\r\n"
        "⚠️ game-lens ไม่ใช่โปรแกรมของผู้พัฒนาเกมใด และไม่เกี่ยวข้องกับผู้พัฒนาเกม แม้ไม่แตะต้องตัวเกม แต่หน้าต่างคำแปลที่วางทับเกม\r\n"
        "อาจถูกระบบกันโกงบางระบบมองว่าเป็นโปรแกรมภายนอก และข้อตกลงของบางเกมห้ามใช้โปรแกรมภายนอก ไม่มีการรับประกันว่า\r\n"
        "บัญชีเกมจะปลอดภัย ผู้ใช้รับความเสี่ยงเอง ควรอ่านข้อตกลงการใช้งานของเกมก่อนใช้\r\n"
        "ซอฟต์แวร์และสัญญาอนุญาตของผู้อื่นที่ใช้: THIRD_PARTY_NOTICES.md และโฟลเดอร์ licenses\r\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama", required=True, help="llama.cpp Windows Vulkan x64 release zip")
    ap.add_argument("--icon", default=str(ROOT / "logs" / "game-lens-icon.ico"))
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--personal", action="store_true", help="keep \"Google free\" (Chrome's key): for your own computers only")
    a = ap.parse_args()
    if Path(sys.prefix).name != ".venv":
        sys.exit("run it with the .venv's python (its packages are what is copied)")
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    build_app(a.personal)
    build_python()
    build_llama(Path(a.llama))
    build_launcher(Path(a.icon))
    readme()
    for bad in ("token.txt", "keys.json", "cache.sqlite"):
        assert not list(OUT.rglob(bad)), bad
    if not a.personal:
        assert not has_key((OUT / "server" / "translate" / "web_engines.py").read_bytes()), "Chrome's key left in"
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"{OUT}  {size / 1e6:.0f} MB")
    if not a.no_zip:
        kind = "portable-personal" if a.personal else "portable"  # (a personal one is not to be passed on)
        out = ROOT / "dist" / f"game-lens-{kind}-{date.today():%Y%m%d}.zip"
        out.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for p in sorted(OUT.rglob("*")):
                if p.is_file():
                    z.write(p, "game-lens/" + p.relative_to(OUT).as_posix())
        print(f"{out}  {out.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
