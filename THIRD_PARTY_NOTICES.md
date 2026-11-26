# Third-party software in game-lens

game-lens ใช้ซอฟต์แวร์และไฟล์ของผู้อื่นตามรายการด้านล่าง แต่ละรายการอยู่ภายใต้สัญญาอนุญาต (license) ของตัวเอง
game-lens uses the software and files below, each under its own license.

game-lens is not made by, endorsed by or connected with the developer or publisher of any game it is used with.
Game names in its profiles and glossaries only say which game a profile is for.

## Bundled in the portable build

| Component | License | Where the license text is |
|---|---|---|
| Python 3.14 | PSF License | `licenses/Python-LICENSE.txt` |
| Python packages (PySide6, OpenCV, NumPy, ONNX Runtime, RapidOCR, FastAPI, httpx, …) | see `licenses/python-packages.txt` | each package's `python/Lib/site-packages/<name>.dist-info` folder |
| llama.cpp / ggml (`llama/`) | MIT | `licenses/llama.cpp-LICENSE.txt` |
| LLVM OpenMP runtime (`llama/libomp.dll`) | Apache-2.0 with LLVM exception | `llama/LICENSE-LLVM-OpenMP` |
| Fonts: Bai Jamjuree, Kanit, Mitr, Sarabun, Noto Sans Thai Looped | SIL Open Font License 1.1 | `fonts/<family>/OFL.txt` |
| PP-OCR text detection and recognition models (downloaded by RapidOCR on first use) | Apache-2.0 | `python/Lib/site-packages/rapidocr-*.dist-info/licenses/MODEL_LICENSES.md` |

### Qt for Python (PySide6, Shiboken6)

game-lens uses Qt for Python (PySide6 6.12) under the **GNU Lesser General Public License v3** (`licenses/LGPL-3.0.txt`,
which refers to `licenses/GPL-3.0.txt`). The Qt and PySide6 libraries are included unmodified, as separate files in
`python/Lib/site-packages/PySide6` and `.../shiboken6`; you may replace them with other compatible builds of the same
version. Their source code is available from the Qt Project: https://code.qt.io and https://download.qt.io/official_releases/QtForPython/.

## Downloaded on first use (not included)

game-lens downloads one translation model from its publisher's own Hugging Face page. By using it you accept its
license:

| Model | Used when | License |
|---|---|---|
| Index-Translate 9B (IndexTeam / bilibili) | graphics card with 10 GB or more | Apache-2.0 — https://huggingface.co/IndexTeam/Index-Translate-9B-GGUF |
| Hy-MT2 7B / 1.8B (Tencent) | smaller or no graphics card, or chosen in settings | Tencent HY Community License — https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF ; read its terms, which (per copies of the license text) do not cover use in the European Union |

## Online translation services (optional)

DeepL, Microsoft Azure, Google Cloud Translation and Anthropic Claude are used only when you choose them and enter your
own key; their terms of service apply. Copies of game-lens built to be given to others leave out the unofficial
"Google free" engine.
