# lens-server (Rust prototype)

A prototype of game-lens's server in Rust, written on 2026-11-26 to measure what a port would bring. It is **not
used by game-lens**; the Python server in `server/` is.

It has the three routes the app uses (`/health`, `/ocr/lines`, `/translate/lines`): the same text-reader models
(PP-OCRv6 through ONNX Runtime 1.31 with DirectML, `ort` 2.0.0-rc.15), translation through the local llama.cpp
server with Index-Translate's prompt, four menu labels at a time, a cache in SQLite. It leaves out names, dialogue
re-reads, the glossary, number patterns and the other engines.

    cargo build --release
    target\release\lens-server.exe ..\..\.venv\Lib\site-packages\rapidocr\models 9961

## Measured (RTX 5060 Ti, Windows 10, the same 19 screenshots / texts)

| | Python server | Rust prototype |
|---|---|---|
| files | onnxruntime, rapidocr, FastAPI ... inside the 408 MB portable folder | 26 MB exe + 18.5 MB DirectML.dll |
| launch to first screen read | 2.5-2.9 s | 1.3-1.4 s |
| memory | 529 MB | 465 MB (the models on the graphics card dominate) |
| reading a screenshot (request to answer) | 287 ms | 219 ms |
| 24 new menu texts / a dialogue line | 2.3-3.7 s / 0.45-0.50 s | 3.8 s / 0.52 s (the model's own time: no difference) |
| lines read the same | - | 82% (it crops and resizes text a little differently; a real port has to match that) |

Conclusion: translation speed does not change, start-up gains about 1.3 s once, memory about 60 MB. Not worth
rewriting the tuned Python code for; trimming the portable build (479 -> 408 MB) brought more for an hour's work.
