//! Prototype of game-lens's server in Rust: the same text-reader models (PP-OCRv6 through ONNX Runtime, on the
//! graphics card with DirectML), translation through the local llama.cpp server, a cache in SQLite, and the same
//! three routes the app uses (/health, /ocr/lines, /translate/lines). It is here to measure size, start-up time
//! and memory against the Python server; it leaves out what the real one does besides (names, dialogue re-reads,
//! glossary, the other engines).
//!
//!   lens-server <folder with the .onnx models> [port] [--cpu]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use axum::extract::{Multipart, State};
use axum::routing::{get, post};
use axum::{Json, Router};
use image::imageops::{self, FilterType};
use image::RgbImage;
use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;
use serde::Deserialize;
use serde_json::{json, Value};

const LLM: &str = "http://127.0.0.1:9944";
const PAD: u32 = 32;

type Res<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

struct Ocr {
    det: Session,
    rec: Session,
    chars: Vec<String>,
}

struct Line {
    bbox: [i32; 4], // x, y, w, h
    text: String,
    conf: f32,
}

fn open(env: &Environment, path: &Path, gpu: bool) -> Res<Session> {
    let text = |e: ort::Error<ort::session::builder::SessionBuilder>| e.to_string();
    let mut b = Session::builder(env)?;
    if gpu {
        // (DirectML wants its sessions without memory patterns and run one node at a time)
        b = b
            .with_execution_providers([ort::ep::DirectML::default().build()])
            .map_err(text)?
            .with_memory_pattern(false)
            .map_err(text)?
            .with_parallel_execution(false)
            .map_err(text)?;
    }
    Ok(b.with_intra_threads(4).map_err(text)?.commit_from_file(path)?)
}

/// Python's round(): halves go to the even number (720 / 32 = 22.5 -> 22).
fn round_even(v: f32) -> i32 {
    let f = v.floor();
    let d = v - f;
    if (d - 0.5).abs() < 1e-6 {
        if (f as i32) % 2 == 0 { f as i32 } else { f as i32 + 1 }
    } else {
        v.round() as i32
    }
}

/// CHW floats, (v / 255 - 0.5) / 0.5, into `out` at row/column 0 of a canvas `cw` wide.
fn fill_chw(img: &RgbImage, out: &mut [f32], cw: usize, ch: usize) {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let raw = img.as_raw();
    for y in 0..h {
        for x in 0..w {
            let p = (y * w + x) * 3;
            for c in 0..3 {
                out[c * cw * ch + y * cw + x] = raw[p + c] as f32 / 127.5 - 1.0;
            }
        }
    }
}

impl Ocr {
    fn load(dir: &Path, gpu: bool) -> Res<Ocr> {
        let env = ort::init().build()?;
        let det = open(&env, &dir.join("PP-OCRv6_det_small.onnx"), gpu)?;
        let rec = open(&env, &dir.join("PP-OCRv6_rec_medium.onnx"), gpu)?;
        // the recogniser's letters are in the model file itself
        let listed = rec.metadata()?.custom("character").unwrap_or_default();
        let mut chars: Vec<String> = vec!["blank".into()];
        chars.extend(listed.split('\n').map(|s| s.to_string()));
        chars.push(" ".into());
        Ok(Ocr { det, rec, chars })
    }

    /// Text boxes (x0, y0, x1, y1 in the picture's pixels), in reading order.
    fn detect(&mut self, img: &RgbImage) -> Res<Vec<[i32; 4]>> {
        let (w, h) = (img.width() as f32, img.height() as f32);
        let ratio = if w.max(h) > 2000.0 { 2000.0 / w.max(h) } else { 1.0 };
        let rh = (round_even((h * ratio).floor() / 32.0) * 32).max(32) as u32;
        let rw = (round_even((w * ratio).floor() / 32.0) * 32).max(32) as u32;
        let small = imageops::resize(img, rw, rh, FilterType::Triangle);
        let (cw, ch) = (rw as usize, rh as usize);
        let mut data = vec![0f32; 3 * cw * ch];
        fill_chw(&small, &mut data, cw, ch);
        let input = Tensor::from_array(([1usize, 3, ch, cw], data))?;
        let outputs = self.det.run(ort::inputs![input])?;
        let (_, pred) = outputs[0].try_extract_tensor::<f32>()?;

        // text pixels, grown by one (a 2x2 dilation), then every connected patch is one text box
        let mut mask = vec![false; cw * ch];
        for y in 0..ch {
            for x in 0..cw {
                if pred[y * cw + x] > 0.3 {
                    for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                        if y + dy < ch && x + dx < cw {
                            mask[(y + dy) * cw + x + dx] = true;
                        }
                    }
                }
            }
        }
        let mut boxes = Vec::new();
        let mut stack = Vec::new();
        for start in 0..cw * ch {
            if !mask[start] {
                continue;
            }
            mask[start] = false;
            stack.push(start);
            let (mut x0, mut y0, mut x1, mut y1) = (cw, ch, 0usize, 0usize);
            while let Some(p) = stack.pop() {
                let (x, y) = (p % cw, p / cw);
                x0 = x0.min(x);
                x1 = x1.max(x);
                y0 = y0.min(y);
                y1 = y1.max(y);
                for ny in y.saturating_sub(1)..=(y + 1).min(ch - 1) {
                    for nx in x.saturating_sub(1)..=(x + 1).min(cw - 1) {
                        if mask[ny * cw + nx] {
                            mask[ny * cw + nx] = false;
                            stack.push(ny * cw + nx);
                        }
                    }
                }
            }
            let (bw, bh) = ((x1 - x0) as f32, (y1 - y0) as f32);
            if bw.min(bh) < 3.0 {
                continue;
            }
            let mut sum = 0f32;
            for y in y0..=y1 {
                for x in x0..=x1 {
                    sum += pred[y * cw + x];
                }
            }
            if (sum / ((x1 - x0 + 1) * (y1 - y0 + 1)) as f32) < 0.5 {
                continue;
            }
            // the detector marks the middle of the letters: grow the box back out ("unclip", ratio 1.6)
            let d = bw * bh * 1.6 / (2.0 * (bw + bh));
            if bw.min(bh) + 2.0 * d < 5.0 {
                continue;
            }
            let sx = |v: f32| (v / cw as f32 * w).round().clamp(0.0, w - 1.0) as i32;
            let sy = |v: f32| (v / ch as f32 * h).round().clamp(0.0, h - 1.0) as i32;
            let b = [sx(x0 as f32 - d), sy(y0 as f32 - d), sx(x1 as f32 + d), sy(y1 as f32 + d)];
            if b[2] - b[0] > 3 && b[3] - b[1] > 3 {
                boxes.push(b);
            }
        }
        // rows top to bottom (boxes less than 10 px apart in height are one row), left to right in a row
        boxes.sort_by_key(|b| b[1]);
        let mut row = 0;
        let mut keyed: Vec<(i32, [i32; 4])> = Vec::new();
        for (i, b) in boxes.iter().enumerate() {
            if i > 0 && b[1] - boxes[i - 1][1] >= 10 {
                row += 1;
            }
            keyed.push((row, *b));
        }
        keyed.sort_by_key(|(r, b)| (*r, b[0]));
        Ok(keyed.into_iter().map(|(_, b)| b).collect())
    }

    /// The text of each crop, with how sure the model is.
    fn recognise(&mut self, crops: &[RgbImage]) -> Res<Vec<(String, f32)>> {
        let mut order: Vec<usize> = (0..crops.len()).collect();
        let ratio = |i: usize| crops[i].width() as f32 / crops[i].height() as f32;
        order.sort_by(|&a, &b| ratio(a).partial_cmp(&ratio(b)).unwrap());
        let mut out = vec![(String::new(), 0f32); crops.len()];
        for batch in order.chunks(6) {
            let widest = batch.iter().map(|&i| ratio(i)).fold(320.0 / 48.0, f32::max);
            let (bw, bh) = ((48.0 * widest) as usize, 48usize);
            let mut data = vec![0f32; batch.len() * 3 * bw * bh];
            for (n, &i) in batch.iter().enumerate() {
                let rw = ((48.0 * ratio(i)).ceil() as usize).min(bw).max(1);
                let line = imageops::resize(&crops[i], rw as u32, 48, FilterType::Triangle);
                fill_chw(&line, &mut data[n * 3 * bw * bh..(n + 1) * 3 * bw * bh], bw, bh);
            }
            let input = Tensor::from_array(([batch.len(), 3, bh, bw], data))?;
            let outputs = self.rec.run(ort::inputs![input])?;
            let (shape, probs) = outputs[0].try_extract_tensor::<f32>()?;
            let (steps, classes) = (shape[1] as usize, shape[2] as usize);
            for (n, &i) in batch.iter().enumerate() {
                let (mut text, mut confs, mut last) = (String::new(), Vec::new(), 0usize);
                for t in 0..steps {
                    let row = &probs[(n * steps + t) * classes..(n * steps + t + 1) * classes];
                    let (best, p) = row.iter().enumerate().fold((0, f32::MIN), |m, (k, &v)| if v > m.1 { (k, v) } else { m });
                    if best != 0 && !(t > 0 && best == last) {
                        text.push_str(self.chars.get(best).map(|s| s.as_str()).unwrap_or(""));
                        confs.push(p);
                    }
                    last = best;
                }
                let conf = if confs.is_empty() { 0.0 } else { confs.iter().sum::<f32>() / confs.len() as f32 };
                out[i] = (text, conf);
            }
        }
        Ok(out)
    }

    fn read(&mut self, img: &RgbImage, min_score: f32) -> Res<Vec<Line>> {
        // game text sits at the very edge of dark boxes too: a border of the edge's own colour around the picture
        let (w, h) = img.dimensions();
        let padded = RgbImage::from_fn(w + 2 * PAD, h + 2 * PAD, |x, y| {
            *img.get_pixel(x.saturating_sub(PAD).min(w - 1), y.saturating_sub(PAD).min(h - 1))
        });
        let boxes = self.detect(&padded)?;
        let crops: Vec<RgbImage> = boxes
            .iter()
            .map(|b| {
                let c = imageops::crop_imm(&padded, b[0] as u32, b[1] as u32, (b[2] - b[0]) as u32, (b[3] - b[1]) as u32).to_image();
                if c.height() as f32 / c.width() as f32 >= 1.5 { imageops::rotate270(&c) } else { c }
            })
            .collect();
        let texts = self.recognise(&crops)?;
        let mut lines = Vec::new();
        for (b, (text, conf)) in boxes.iter().zip(texts) {
            let text = text.trim().to_string();
            if conf < min_score.max(0.5) || text.is_empty() {
                continue;
            }
            let (x0, y0) = ((b[0] - PAD as i32).max(0), (b[1] - PAD as i32).max(0));
            let (x1, y1) = ((b[2] - PAD as i32).min(w as i32), (b[3] - PAD as i32).min(h as i32));
            if x1 - x0 >= 2 && y1 - y0 >= 2 {
                lines.push(Line { bbox: [x0, y0, x1 - x0, y1 - y0], text, conf });
            }
        }
        Ok(lines)
    }
}

struct App {
    ocr: Mutex<Ocr>,
    cache: Mutex<rusqlite::Connection>,
    http: reqwest::Client,
}

async fn health(State(app): State<Arc<App>>) -> Json<Value> {
    let up = app.http.get(format!("{LLM}/health")).send().await.map(|r| r.status().is_success()).unwrap_or(false);
    Json(json!({"ok": true, "app": "game-lens", "server": "rust prototype", "index": up, "targets": ["th", "en"]}))
}

async fn ocr_lines(State(app): State<Arc<App>>, mut form: Multipart) -> Result<Json<Value>, String> {
    let mut bytes = Vec::new();
    let mut min_score = 0.5f32;
    while let Some(field) = form.next_field().await.map_err(|e| e.to_string())? {
        match field.name().unwrap_or("") {
            "image" => bytes = field.bytes().await.map_err(|e| e.to_string())?.to_vec(),
            "min_score" => min_score = field.text().await.unwrap_or_default().parse().unwrap_or(0.5),
            _ => {}
        }
    }
    let t0 = Instant::now();
    let (lines, w, h) = tokio::task::spawn_blocking(move || -> Res<(Vec<Line>, u32, u32)> {
        let img = image::load_from_memory(&bytes)?.to_rgb8();
        let lines = app.ocr.lock().unwrap().read(&img, min_score)?;
        Ok((lines, img.width(), img.height()))
    })
    .await
    .map_err(|e| e.to_string())?
    .map_err(|e| e.to_string())?;
    let text: String = lines.iter().map(|l| l.text.as_str()).collect();
    let lines: Vec<Value> = lines
        .iter()
        .enumerate()
        .map(|(i, l)| json!({"id": format!("l{i}"), "bbox": l.bbox, "dbox": l.bbox, "text": l.text, "conf": (l.conf * 1000.0).round() / 1000.0}))
        .collect();
    Ok(Json(json!({"w": w, "h": h, "srcLang": "ja", "lines": lines, "text": text, "ms": t0.elapsed().as_millis() as u64})))
}

#[derive(Deserialize)]
struct GameLine {
    id: String,
    src: String,
    #[serde(default)]
    kind: String,
}

#[derive(Deserialize)]
struct Context {
    src: String,
    tl: String,
}

#[derive(Deserialize)]
struct TranslateReq {
    lines: Vec<GameLine>,
    #[serde(default = "thai")]
    target: String,
    #[serde(default)]
    context: Vec<Context>,
}

fn thai() -> String {
    "th".into()
}

/// Index-Translate's own prompt form, as the Python server writes it.
fn prompt(text: &str, history: &[(String, String)], target: &str) -> String {
    let lang = if target == "en" { "英语" } else { "泰语" };
    let mut rules = vec!["【注意】这是电子游戏中的对白、角色名或菜单文字，译文要口语化、自然，像官方本地化一样符合角色说话的语气".to_string()];
    if !history.is_empty() {
        let from = history.len().saturating_sub(4);
        let earlier: Vec<String> = history[from..].iter().map(|(a, b)| format!("{a} => {b}")).collect();
        rules.push(format!("【注意】前文（仅供理解语境，不要翻译）：{}", earlier.join(" / ")));
    }
    let rules: Vec<String> = rules.iter().enumerate().map(|(i, r)| format!("{}. {r}", i + 1)).collect();
    format!(
        "请将以下文本翻译成{lang}，并且严格遵循所有约束要求。\n\n【源文】\n{}\n\n【约束要求】\n{}\n\n只输出译文，不要有任何额外说明。",
        text.trim(),
        rules.join("\n")
    )
}

async fn ask(http: &reqwest::Client, prompt: String) -> Result<String, String> {
    let body = json!({"messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 512,
                      "cache_prompt": true, "chat_template_kwargs": {"enable_thinking": false}});
    let r: Value = http
        .post(format!("{LLM}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .map_err(|e| e.to_string())?
        .json()
        .await
        .map_err(|e| e.to_string())?;
    Ok(r["choices"][0]["message"]["content"].as_str().unwrap_or("").trim().to_string())
}

async fn translate_lines(State(app): State<Arc<App>>, Json(req): Json<TranslateReq>) -> Result<Json<Value>, String> {
    let t0 = Instant::now();
    let key = |l: &GameLine| format!("{}|{}|{}", req.target, l.kind, l.src.trim());
    let mut out: HashMap<String, String> = HashMap::new();
    let mut cached = Vec::new();
    let mut todo = Vec::new();
    {
        let db = app.cache.lock().unwrap();
        for l in &req.lines {
            let hit: Option<String> = db.query_row("SELECT tl FROM line_tl WHERE key = ?1", [key(l)], |r| r.get(0)).ok();
            match hit {
                Some(tl) => {
                    out.insert(l.id.clone(), tl);
                    cached.push(l.id.clone());
                }
                None if l.src.trim().is_empty() => {
                    out.insert(l.id.clone(), String::new());
                }
                None => todo.push(l),
            }
        }
    }
    let t1 = Instant::now();
    // menu labels need no context: four at a time (the model's slots); dialogue in order, each line with those before it
    let labels: Vec<&&GameLine> = todo.iter().filter(|l| l.kind == "ui").collect();
    for part in labels.chunks(4) {
        let mut set = tokio::task::JoinSet::new();
        for l in part {
            let (http, id, p) = (app.http.clone(), l.id.clone(), prompt(&l.src, &[], &req.target));
            set.spawn(async move { (id, ask(&http, p).await) });
        }
        while let Some(done) = set.join_next().await {
            let (id, tl) = done.map_err(|e| e.to_string())?;
            out.insert(id, tl?);
        }
    }
    let mut history: Vec<(String, String)> = req.context.iter().map(|c| (c.src.clone(), c.tl.clone())).collect();
    for l in todo.iter().filter(|l| l.kind != "ui") {
        let tl = ask(&app.http, prompt(&l.src, &history, &req.target)).await?;
        history.push((l.src.clone(), tl.clone()));
        out.insert(l.id.clone(), tl);
    }
    let model_ms = t1.elapsed().as_millis() as u64;
    {
        let db = app.cache.lock().unwrap();
        for l in &todo {
            if let Some(tl) = out.get(&l.id).filter(|t| !t.is_empty()) {
                let _ = db.execute("INSERT OR REPLACE INTO line_tl VALUES (?1, ?2)", [key(l), tl.clone()]);
            }
        }
    }
    Ok(Json(json!({"lines": out, "engine": "index", "engineRequested": "index", "cached": cached,
                   "metrics": {"ms": model_ms, "totalMs": t0.elapsed().as_millis() as u64}})))
}

#[tokio::main]
async fn main() -> Res<()> {
    let t0 = Instant::now();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = PathBuf::from(args.first().expect("usage: lens-server <models folder> [port] [--cpu]"));
    let port: u16 = args.get(1).and_then(|p| p.parse().ok()).unwrap_or(9961);
    let gpu = !args.iter().any(|a| a == "--cpu");
    let ocr = Ocr::load(&dir, gpu)?;
    let cache = rusqlite::Connection::open(std::env::var("LENS_CACHE").unwrap_or_else(|_| "lens-cache.sqlite".into()))?;
    cache.execute("CREATE TABLE IF NOT EXISTS line_tl (key TEXT PRIMARY KEY, tl TEXT NOT NULL)", [])?;
    let app = Arc::new(App { ocr: Mutex::new(ocr), cache: Mutex::new(cache), http: reqwest::Client::new() });
    let routes = Router::new()
        .route("/health", get(health))
        .route("/ocr/lines", post(ocr_lines))
        .route("/translate/lines", post(translate_lines))
        .layer(axum::extract::DefaultBodyLimit::max(64 << 20))
        .with_state(app);
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
    println!("listening on http://127.0.0.1:{port} after {} ms ({})", t0.elapsed().as_millis(), if gpu { "graphics card" } else { "processor" });
    axum::serve(listener, routes).await?;
    Ok(())
}
