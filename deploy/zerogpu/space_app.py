"""Gradio entry point; staged with src/ and trusted trained heads by build_zerogpu.py."""
import spaces  # must precede torch imports: installs ZeroGPU CUDA emulation
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import gradio as gr
import numpy as np
from PIL import Image
from predict import Predictor, load_image
from dimension import Dimensioner, draw_overlay, focal_35mm, fuse_site, measure_ranked
from app import INFO, cut_edges
import overlap

Image.MAX_IMAGE_PIXELS = 40_000_000
warnings.simplefilter("error", Image.DecompressionBombWarning)

# Explicit CUDA placement at module scope is required by ZeroGPU. Never load these
# lazily in GPU callbacks. Ordinary laptop/CPU use remains in src/app.py.
predictor = Predictor(device="cuda")
dimensioner = Dimensioner(device="cuda", offload=False)


def positive(value):
    if value is None or value == 0:
        return None
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise gr.Error("Enter a positive distance, or leave it blank.")
    return value


def read_photo(path):
    if not path or Path(path).stat().st_size > 25_000_000:
        raise gr.Error("Choose a photo smaller than 25 MB.")
    try:
        return load_image(path), focal_35mm(path)
    except Exception as exc:
        raise gr.Error("Could not read this photo. Use JPEG or PNG under 40 megapixels.") from exc


def video_frames(path):
    import cv2
    cap = cv2.VideoCapture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or count < 2 or count / fps > 60:
            raise gr.Error("Choose a valid sweep video of at most 60 seconds.")
        frames = []
        for index in np.linspace(0, count - 1, min(8, count), dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if ok:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image.thumbnail((1024, 1024))
                frames.append((image, None))
        if len(frames) < 2:
            raise gr.Error("Could not extract enough video frames.")
        return frames
    finally:
        cap.release()


def clean(value):
    """Convert numpy scalars for Gradio JSON and downloadable reports."""
    return json.loads(json.dumps(value, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))


def save_report(report):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix="slopelens-",
                                     encoding="utf-8", delete=False, dir=download.GRADIO_CACHE) as out:
        json.dump(clean(report), out, indent=2, allow_nan=False)
        return out.name


def describe(pred):
    info = INFO[pred["label"]]
    text = f"### {info['title']} — {pred['confidence']:.0%} confidence\n\n"
    if pred["status"] == "bad_photo":
        text += "**Poor photo suitability: this prediction and any size estimate are unreliable. Retake the photo.**\n\n"
    elif pred["status"] == "uncertain":
        text += "**Uncertain:** verify the top two classes shown in the report.\n\n"
    text += "**Verify on site:**\n\n" + "\n".join(f"- {item}" for item in info["check"])
    return text + "\n\n**Safety:** " + info["safety"]


def prepare(photo, files, video, mode, motion, baseline):
    """CPU preprocessing is outside the GPU reservation."""
    if mode == "Single photo":
        shots = [read_photo(photo)] if photo else []
    elif mode == "Video sweep":
        shots = video_frames(video) if video else []
    else:
        limit = 4 if mode == "Site survey" else 8
        if len(files or []) > limit:
            raise gr.Error(f"Choose at most {limit} photos.")
        shots = [read_photo(path) for path in (files or [])]
    if not shots:
        raise gr.Error("Add photos or a video for the selected input mode.")
    if mode in ("Overlapping photos", "Video sweep") and len(shots) < 2:
        raise gr.Error("An overlapping set needs at least two photos.")
    images = [shot[0] for shot in shots]
    state = {"shots": shots, "mode": mode, "measure": []}
    if mode in ("Overlapping photos", "Video sweep"):
        report, ex = overlap.analyse(images, kind=motion, f35=[s[1] for s in shots])
        if not report["joinable"]:
            raise gr.Error("These frames do not overlap enough. Retake with more overlap.")
        state.update(overlap=report, extra=ex)
        kw = {}
        if ex["mosaic"] is not None:
            shot = (ex["mosaic"], ex["mosaic_f35"])
            kw = {"focal_sd": report["focal_sd"], "focal_src": report["focal_source"]}
        else:
            center = ex["central"]
            shot = shots[center]
            baseline = positive(baseline)
            if motion == "walk" and baseline:
                # User supplies metres between adjacent frames, not total walking distance.
                neighbors = [j for j in ex["main"] if abs(j - center) == 1]
                f35 = shot[1] or overlap.DEFAULT_F35
                points = [overlap.stereo_points(shot[0], images[j],
                          f35 / 36 * max(shot[0].size), baseline) for j in neighbors]
                points = [p for p in points if p]
                if points:
                    kw = {"metric_points": {"xy": sum([p["xy"] for p in points], []),
                          "z": sum([p["z"] for p in points], []),
                          "log_sd": max(p["log_sd"] for p in points)}}
        state["measure"] = [(shot[0], shot[1], kw)]
    else:
        state["measure"] = [(im, focal, {}) for im, focal in shots]
    return state


@spaces.GPU(duration=60)
def classify(state):
    if not state:
        raise gr.Error("Add photos first.")
    shots = state["shots"]
    if "extra" in state:
        ex = state["extra"]
        batch = [shots[i][0] for i in ex["main"]]
        if ex["mosaic"] is not None:
            batch.append(ex["mosaic"])
        results = predictor(batch)
        singles = results[:len(ex["main"])]
        joined = results[-1] if ex["mosaic"] is not None else None
        probs, usable = overlap.combine(joined, singles)
        pred = predictor.decide(usable, probs)
        measure_preds = [pred]
    else:
        results = predictor([im for im, _ in shots])
        probs, usable = overlap.combine(None, results)
        pred = predictor.decide(usable, probs)
        measure_preds = results
    report = {"prediction": pred, "photos": results, "mode": state["mode"],
              "caveat": "Research prototype; trained/evaluated on AI labels. Sizes are estimates."}
    if "overlap" in state:
        report["overlap"] = {k: v for k, v in state["overlap"].items() if k != "preview"}
    state["predictions"] = measure_preds
    state["report"] = report
    return describe(pred), clean(report), state, [(s[0], f"Photo {i+1}") for i, s in enumerate(shots)], save_report(report)


@spaces.GPU(duration=180)
def measure(state, distance):
    if not state or "predictions" not in state:
        raise gr.Error("Classify your photos first.")
    distance = positive(distance)
    dimensions, overlays = [], []
    for (image, focal, kw), pred in zip(state["measure"], state["predictions"]):
        result = measure_ranked(dimensioner, image, pred, f35=focal,
                                ref_distance_m=distance, **kw)
        if "overlay" in result:
            mask = np.asarray(result.pop("overlay"), bool)
            result["cut_edges"] = cut_edges(mask)
            overlays.append((draw_overlay(image, mask), "Estimated landslide outline"))
        dimensions.append(result)
    report = dict(state["report"], dimensions=dimensions)
    if len(dimensions) > 1:
        report["site_dimensions"] = fuse_site(dimensions)
    report["model_availability"] = {name: getattr(dimensioner, name, None) is not None
                                    for name in ("seg", "sam", "da", "pro", "owl", "head")}
    return clean(report), overlays, save_report(report)


with gr.Blocks(title="SlopeLens", delete_cache=(3600, 3600)) as demo:
    gr.Markdown("# SlopeLens\nLandslide classification and size estimates from ground photos. "
                "Observe from a safe location. This research prototype does not determine whether a slope is safe.")
    gr.Markdown("Shared cloud GPU • queues and daily quotas apply • no laptop GPU needed. "
                "Upload an overview with the entire slope visible. Temporary uploads/reports are cleaned periodically.")
    state = gr.State(time_to_live=1800)
    pending = gr.State(time_to_live=1800)
    mode = gr.Radio(["Single photo", "Site survey", "Overlapping photos", "Video sweep"],
                    value="Single photo", label="Input mode")
    with gr.Row():
        photo = gr.Image(type="filepath", sources=["upload", "webcam"], label="Single photo")
        files = gr.File(file_count="multiple", file_types=["image"], label="Survey or overlapping photos, in order")
        video = gr.Video(sources=["upload"], label="Sweep video (max 60 seconds)", max_length=60)
    with gr.Row():
        motion = gr.Radio([("Turn on the spot", "turn"), ("Walk along", "walk")], value="turn", label="Photo set motion")
        baseline = gr.Number(label="Metres between adjacent photos/frames (optional, walking only)", minimum=0)
        distance = gr.Number(label="Camera-to-slope distance in metres (optional)", minimum=0)
    run = gr.Button("1. Classify", variant="primary")
    sizes = gr.Button("2. Estimate sizes (uses additional GPU quota)")
    summary = gr.Markdown()
    gallery = gr.Gallery(label="Analyzed photos", columns=2)
    outlines = gr.Gallery(label="Measurement outlines", columns=2)
    report = gr.JSON(label="Probabilities, measurements and uncertainty")
    download = gr.File(label="Download JSON report", interactive=False)
    # All GPU events share one concurrency group, protecting the resident models.
    start = run.click(lambda: (None, None, None, None, [], [], None),
                      outputs=[state, pending, summary, report, gallery, outlines, download], queue=False)
    prepared = start.then(prepare, [photo, files, video, mode, motion, baseline], pending)
    prepared.success(classify, pending, [summary, report, state, gallery, download],
                     concurrency_id="inference", concurrency_limit=1, api_visibility="private")
    sizes.click(measure, [state, distance], [report, outlines, download],
                concurrency_id="inference", concurrency_limit=1, api_visibility="private")
    gr.Markdown("Site surveys must show **one landslide**. A single supplied distance applies to all survey shots; "
                "leave it blank if camera positions differ. Video frames are evenly sampled; "
                "walking scale is valid only when adjacent samples have the specified baseline. "
                "The original phone interface remains available in the repository.")

if __name__ == "__main__":
    demo.queue(max_size=8).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")),
                                max_file_size="100mb", show_error=False, ssr_mode=False)
