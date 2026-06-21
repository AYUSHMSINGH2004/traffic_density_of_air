import os
import tempfile
from collections import Counter
from pathlib import Path

# Avoid Ultralytics config warnings on Hugging Face Spaces
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

import cv2
import gradio as gr
from ultralytics import YOLO

MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")

EMISSION_FACTORS = {
    "motorcycle": 80,
    "car": 120,
    "bus": 800,
    "truck": 1000,
}


def load_model():
    try:
        model = YOLO(MODEL_PATH)
        return model
    except Exception as exc:
        print(f"Model load failed: {exc}")
        return None


MODEL = load_model()


def format_report(frames, fps, totals, peak, co2_proxy_grams, model_ready=True):
    vehicles = sorted(set(list(EMISSION_FACTORS.keys()) + list(totals.keys())))
    total_detections = sum(totals.values())
    avg_per_frame = total_detections / max(frames, 1)
    duration = frames / max(fps, 1.0)

    lines = [
        "## 📊 Analysis Report",
        "",
        "### Model Status",
    ]

    if model_ready:
        lines.append("✅ YOLOv8 model loaded successfully.")
    else:
        lines.append("⚠️ Model could not be loaded, so the app returned the original video without detection.")

    lines += [
        "",
        "### Video Summary",
        f"- Frames processed: **{frames}**",
        f"- Video duration: **{duration:.2f} s**",
        f"- Average detected vehicles per frame: **{avg_per_frame:.2f}**",
        f"- Estimated CO₂ footprint proxy: **{co2_proxy_grams:.2f} g**",
        "",
        "### Vehicle Breakdown",
    ]

    any_vehicle = False
    for name in vehicles:
        if totals.get(name, 0) > 0 or peak.get(name, 0) > 0:
            any_vehicle = True
            lines.append(
                f"- **{name.title()}**: total detections `{totals.get(name, 0)}`, peak in one frame `{peak.get(name, 0)}`"
            )

    if not any_vehicle:
        lines.append("- No supported vehicles were detected in this video.")

    lines += [
        "",
        "### Note",
        "The CO₂ figure is a model-based estimate derived from detected traffic classes. It is best used as a relative environmental indicator rather than a literal emissions audit.",
    ]

    return "\n".join(lines)


def process_video(video_input):
    if video_input is None:
        return None, "## ⚠️ Upload a video first."

    # Gradio may return a string path, a dict, or another file-like structure depending on version/settings.
    if isinstance(video_input, dict):
        video_path = video_input.get("path") or video_input.get("name")
    else:
        video_path = video_input

    if not video_path:
        return None, "## ⚠️ Invalid video input."

    video_path = str(video_path)

    if not Path(video_path).exists():
        return None, "## ⚠️ The uploaded video file was not found."

    # If model is unavailable, return the original video gracefully.
    if MODEL is None:
        return video_path, (
            "## ⚠️ Detection model unavailable\n\n"
            "The app is running, but the YOLO model could not be loaded. "
            "Please make sure `yolov8n.pt` is present in the Space repository root "
            "or set `MODEL_PATH` correctly."
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "## ⚠️ Could not open the uploaded video."

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if width <= 0 or height <= 0:
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            return None, "## ⚠️ The video appears to be empty or unreadable."
        height, width = frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    out_dir = Path(tempfile.mkdtemp(prefix="ecovision_"))
    out_path = out_dir / "processed.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        return None, "## ⚠️ Could not create the output video file."

    totals = Counter()
    peak = Counter()
    co2_proxy_grams = 0.0
    frames = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frames += 1

            # Force CPU-friendly inference
            result = MODEL.predict(frame, imgsz=640, conf=0.25, verbose=False, device="cpu")[0]
            annotated = result.plot()

            if annotated.shape[1] != width or annotated.shape[0] != height:
                annotated = cv2.resize(annotated, (width, height))

            writer.write(annotated)

            frame_counts = Counter()
            if result.boxes is not None and len(result.boxes) > 0:
                class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
                for cls_id in class_ids:
                    label = result.names.get(int(cls_id), str(cls_id))
                    if label in EMISSION_FACTORS:
                        frame_counts[label] += 1

            for label, count in frame_counts.items():
                totals[label] += count
                if count > peak[label]:
                    peak[label] = count

            co2_proxy_grams += (
                sum(EMISSION_FACTORS.get(label, 0) * count for label, count in frame_counts.items())
                / max(fps, 1.0)
            )

    except Exception as exc:
        cap.release()
        writer.release()
        return None, f"## ❌ Processing failed\n\n`{exc}`"
    finally:
        cap.release()
        writer.release()

    report = format_report(frames, fps, totals, peak, co2_proxy_grams, model_ready=True)
    return str(out_path), report


css = """
:root{
    --bg1:#03140d;
    --bg2:#071b2d;
    --bg3:#081f16;
    --border:rgba(255,255,255,0.13);
    --text:#eaf5ef;
    --muted:#b8c8c0;
    --shadow:0 18px 60px rgba(0,0,0,0.32);
}

body{
    background:
        radial-gradient(circle at top left, rgba(0,230,118,.20), transparent 28%),
        radial-gradient(circle at top right, rgba(0,212,255,.16), transparent 25%),
        radial-gradient(circle at bottom left, rgba(255,209,102,.10), transparent 28%),
        linear-gradient(135deg, var(--bg1), var(--bg2) 52%, var(--bg3));
    color:var(--text);
}

.gradio-container{
    max-width: 1500px !important;
    margin: 0 auto !important;
    padding: 32px !important;
    background: rgba(255,255,255,0.03) !important;
    backdrop-filter: blur(18px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 30px !important;
    box-shadow: var(--shadow);
}

h1, h2, h3, h4, p, li, label, .markdown, .prose{
    color: var(--text) !important;
}

.card{
    background: linear-gradient(180deg, rgba(255,255,255,0.10), rgba(255,255,255,0.05));
    border: 1px solid var(--border);
    border-radius: 26px;
    padding: 24px;
    box-shadow: var(--shadow);
}

.hero{
    text-align:center;
    padding: 26px 18px 8px;
}

.hero h1{
    font-size: clamp(42px, 6vw, 68px);
    line-height: 1.05;
    margin-bottom: 12px;
    background: linear-gradient(90deg, #00ff87, #00d4ff, #ffd166);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent !important;
    letter-spacing: -0.03em;
}

.hero .subtitle{
    max-width: 920px;
    margin: 0 auto;
    color: var(--muted);
    font-size: 18px;
}

.badges{
    display:flex;
    gap:10px;
    justify-content:center;
    flex-wrap:wrap;
    margin-top: 18px;
}

.badge{
    padding: 8px 14px;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.16);
    background: rgba(255,255,255,0.08);
    color: var(--text);
    font-size: 13px;
}

.metric{
    display:flex;
    flex-direction:column;
    gap:8px;
    padding: 18px;
    border-radius: 22px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    min-height: 110px;
}

.metric .k{
    font-size: 32px;
    font-weight: 800;
    line-height: 1;
    background: linear-gradient(90deg, #00e676, #00d4ff);
    -webkit-background-clip:text;
    background-clip:text;
    color: transparent !important;
}

.metric .l{
    color: var(--muted);
    font-size: 14px;
}

button.primary{
    background: linear-gradient(90deg, #00e676, #00d4ff) !important;
    color: #02130d !important;
    border: none !important;
    font-weight: 800 !important;
    border-radius: 16px !important;
    box-shadow: 0 14px 30px rgba(0, 230, 118, 0.25);
}

button.primary:hover{
    transform: translateY(-1px);
}

footer{
    display:none !important;
}
"""

theme = gr.themes.Soft(primary_hue="green")

with gr.Blocks(theme=theme, css=css, title="EcoVision AI") as demo:
    gr.HTML(
        """
        <div class="hero">
            <h1>EcoVision AI</h1>
            <div class="subtitle">
                A premium traffic intelligence dashboard that detects vehicles, processes video in real time,
                and generates a clean CO₂ impact summary for every upload.
            </div>
            <div class="badges">
                <span class="badge">YOLOv8 vehicle detection</span>
                <span class="badge">OpenCV video processing</span>
                <span class="badge">CO₂ analytics</span>
                <span class="badge">Hugging Face Spaces ready</span>
            </div>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=2, elem_classes="card"):
            gr.Markdown(
                """
## Project overview
EcoVision AI turns traffic footage into a polished environmental dashboard. It detects common road vehicles, draws annotated bounding boxes, and produces an emission-style report you can share instantly.
                """
            )

        with gr.Column(scale=2, elem_classes="card"):
            gr.Markdown(
                """
## AI pipeline
Upload video → YOLOv8 detection → frame-by-frame annotation → vehicle statistics → CO₂ estimate → downloadable result
                """
            )

    gr.Markdown("## Intelligent features")

    with gr.Row():
        with gr.Column(elem_classes="card"):
            gr.Markdown("### 🚗 Vehicle detection\nAccurate bounding boxes and class labels for traffic objects.")
        with gr.Column(elem_classes="card"):
            gr.Markdown("### 🌍 CO₂ analytics\nReadable footprint-style estimates based on detected vehicle classes.")
        with gr.Column(elem_classes="card"):
            gr.Markdown("### 📊 Premium dashboard\nGlassmorphism cards, gradient accents, and a high-end visual finish.")

    with gr.Row():
        with gr.Column(elem_classes="metric"):
            gr.HTML('<div class="k">4</div><div class="l">Supported vehicle classes</div>')
        with gr.Column(elem_classes="metric"):
            gr.HTML('<div class="k">YOLOv8</div><div class="l">Detection engine</div>')
        with gr.Column(elem_classes="metric"):
            gr.HTML('<div class="k">OpenCV</div><div class="l">Video rendering</div>')
        with gr.Column(elem_classes="metric"):
            gr.HTML('<div class="k">HF</div><div class="l">Deployment ready</div>')

    gr.Markdown("## Vehicle emission metrics")
    gr.Markdown(
        """
| Vehicle | CO₂ emission proxy |
|---|---:|
| Motorcycle | 80 g/km |
| Car | 120 g/km |
| Bus | 800 g/km |
| Truck | 1000 g/km |
        """
    )

    gr.Markdown("## Live traffic intelligence console")

    with gr.Row():
        with gr.Column(scale=1, elem_classes="card"):
            video = gr.Video(
                label="Upload traffic video",
                sources=["upload"],
            )

            button = gr.Button(
                "Start AI analysis",
                variant="primary",
            )

        with gr.Column(scale=1, elem_classes="card"):
            output_video = gr.Video(
                label="Processed video",
            )
            report = gr.Markdown("Your analysis report will appear here.")

    button.click(
        fn=process_video,
        inputs=video,
        outputs=[output_video, report],
    )

    gr.Markdown(
        """
## Future-ready vision
Web dashboard expansion, live CCTV support, richer emission models, and cloud-connected reporting.
        """
    )

    gr.HTML(
        """
        <div style="text-align:center; padding: 22px 8px 8px; color: rgba(255,255,255,0.72);">
            Built with YOLOv8, OpenCV, and Gradio.
        </div>
        """
    )

demo.queue(max_size=8)

if __name__ == "__main__":
    demo.launch(show_error=True)
