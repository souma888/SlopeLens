# IC202P: Landslide type classifier (ground photos)

To build a **new** model without the original artifacts, see [fresh model workflow](FRESH_MODEL.md).
It requires newly collected/reviewed photos, trains new heads, calibrates on validation only,
and reports new test scores. The historical report below does not describe this new experiment.

## Cloud deployment without a laptop GPU

The Gradio/ZeroGPU edition includes classification, optional size estimation, surveys,
photo sets, video sampling and JSON reports. See [deployment setup](deploy/zerogpu/SETUP.md).
Deploy with `python tools/build_zerogpu.py --push YOUR_HF_USERNAME/SlopeLens`, or the manual
GitHub Actions workflow. It requests only free ZeroGPU hardware and waits for startup.
Free hosting needs an eligible Hugging Face account (verified, over 30 days old), with
visitor queues/quotas. The original phone interface remains available below.

**This source checkout lacks the trained heads and dataset.** Restore the original `models/`
artifacts or retrain from labeled data before deploying; the script fails clearly if missing.
The ZeroGPU edition has not yet been tested with the original trained artifacts.

**Use case:** a junior engineer standing on a road or in a valley takes a phone photo looking up
at a slope. The app tells them which landslide type it is (Varnes classification, Hungr et al.
2014), how confident it is, what to verify on site and what to stay away from. Aerial, satellite,
map or close-up photos are rejected with tips for retaking the photo.

```
.venv\Scripts\python src/app.py          # then open the printed http://<laptop-ip>:8000 on a phone
```

Results and caveats: **`report.md`**. Everything else in this repo builds and tests that app:
a training set of ground-level photos, labels, trained models, and a comparison with a
zero-shot vision-language model (Qwen-VL).

**Status: working prototype on this laptop, not a public website.** The phone opens the laptop's
address, so it must be on the same Wi-Fi, or the laptop on the phone's hotspot (works in the
field without internet). Use `--token SECRET` on shared networks (phones then open
`http://<laptop-ip>:8000/?t=SECRET`). Uploads are saved to `data/app_uploads/` for later
labeling (`--no-save` to disable; capped by `--max-saved`).
For real use the app would be hosted on a server reachable from anywhere. It runs on CPU
(three backbones + linear heads, a few seconds per photo, ~3 GB RAM; the backbones download from
Hugging Face on first start). Copy `src/`, `models/probe_*.joblib`, `models/selected.json`,
install `requirements.txt` (CPU torch), run `python src/app.py --port 8000 --token ...` behind
HTTPS. Hosting is a deliberate decision for the team: it publishes the service and its uploads.

Classes: `fall_topple`, `rotational_slide`, `translational_slide`, `debris_flow`,
`earth_mud_flow`, `no_landslide` (+ `reject` for unusable photos). Definitions and Hungr 2014
types: `src/common.py` / `labeling_guide.md`. Background: `research_notes.md`, `papers/`
(`design.md` is the original plan).

**One GPU job at a time.** Qwen-VL uses ~6 GB of the 8 GB; running DINOv2 or ConvNeXt
alongside it fails (the probe falls back to CPU; CNN training does not). With little free RAM,
run long GPU jobs in the foreground in chunks (`train_cnn.py --max-minutes 8 --resume`).

## Setup (done once)

```
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements.txt
```
Models download to `.cache/huggingface/` in this folder (C: is too full).
All commands below run from the project folder with `.venv\Scripts\python`.

## Scenarios

- **Scenario B (default, what was done): we build the dataset** with steps 1–5 below.
- **Scenario A: a labeled dataset is given.** Instead of steps 1–5:
  ```
  .venv\Scripts\python src/import_dataset.py <dataset_folder> --name given
  ```
  Folder-per-class (`<label>/*.jpg`) or `<train|val|test>/<label>/*.jpg` both work; a CSV with
  `--csv`, `--image-col`, `--label-col` (and `--group-col`, `--split-col` if present) too.
  The first run writes `data/label_map_given.json` with guessed mappings to our classes:
  check every line, then rerun with `--map data/label_map_given.json`. The dataset's own
  split is kept. **If it has no train/val/test split, run `src/split.py` afterwards** (pass
  `--group-col` at import if there is a site/event column, otherwise photos of one site may land
  in different splits). A dataset without `reject` labels trains no photo check (every upload is
  treated as usable). Then continue at step 6.

## Workflow (Scenario B)

| Step | Command | What it does |
|---|---|---|
| 1a. Collect | `src/collect_commons.py --search-only` | Wikimedia Commons ground photos (title searches; many Geograph field photos) + licence info -> `data/raw/images`, `data/metadata.csv` |
| 1b. | `src/collect_commons.py --match "India\|Nepal\|Himachal\|Uttarakhand\|Kerala\|Wayanad\|Sikkim\|Bhutan\|Pakistan\|Sri Lanka\|Bangladesh\|Road cuts"` | South-Asian landslide/road categories (category trees are otherwise mostly aerial/glacier/art) |
| 1c. | `src/collect_openverse.py` | CC photos from Flickr via Openverse (Indian/Nepali hill roads); skips Commons duplicates |
| 1d. Own photos | `src/import_local.py field_photos` | one subfolder per site: `field_photos/<site>/*.jpg` |
| 1e. Camera EXIF | `tools/camera_info.py` | flags phone photos (`is_phone`) for the phone tests |
| 2a. Link sites | `tools/link_sites.py` | merges photos of the same place/event (titles, photo series, visual similarity) into one site |
| 2b. Split | `src/split.py` | train/val/test **by site**, before labeling (review.py also does this for new photos) |
| 3. VLM | `src/vlm_label.py` (all) or `--split test` (baseline only) | Qwen2.5-VL-7B (4-bit) zero-shot probabilities -> `data/vlm_predictions.csv` |
| 4. Label | `src/review.py --reviewer <name>` | browser tool: keys 1-6 / R, V = viewpoint; test split shown blind |
| 4b. AI labels | `tools/contact_sheet.py OUT --cols 4 --rows 3 --tile 320 --skip-labeled`, then `tools/ai_labels.py OUT codes.txt` | what was used here (reviewer `claude_ai`; instructions `tools/ai_labeling_prompt.md`; audit trail `data/labels/ai_annotation/`) |
| 4c. Label noise | `tools/find_label_issues.py OUT` | flags doubtful train/val labels for a second look (never test) |
| 5. Features | `tools/precompute_embeddings.py --backbone <bb>` for `facebook/dinov2-base`, `facebook/dinov2-large`, `google/siglip2-base-patch16-224` | caches features of every photo + phone-style copies (resumable) |
| 6a. Probes | `src/probe.py --backbone <bb> --name probe_base` (and `probe_large`, `probe_siglip`) | photo-check head + type head per backbone; C by grouped CV on train+val |
| 6b. Choose | `tools/robustness_eval.py` | train+val only: site-grouped CV + 3 unseen-source tests for every backbone/ensemble -> `models/selected.json` (the app's model + threshold), `results/robustness.md` |
| 6c. CNN | `src/train_cnn.py --epochs 25 --max-minutes 8 --resume` (repeat until done) | fine-tuned ConvNeXt-Tiny, for comparison |
| 7a. Evaluate | `src/eval.py` | every model on the untouched test split (+ phone-degraded copies, uphill views, South Asia, phone photos) -> `results/summary.md`; writes `pred_selected*.csv` |
| 7b. System test | `tools/system_test.py` | the real app pipeline on every test photo (photo check + type + uncertain) -> `results/system_test.md` |
| 7c. Hard cases | `tools/hard_case_eval.py` | app behaviour on ambiguous photos the labels rejected -> `results/hard_cases.md` |
| 8. Check | `tools/check_integrity.py` | everything complete and consistent, no train/test leakage |
| Credits | `tools/credits.py` | per-photo attribution `data/credits.csv`, licence summary `results/licences.md` |
| App / CLI | `src/app.py`, `src/predict.py photo.jpg` | phone web app / command line, same model |

Collection results change over time (Commons/Openverse are live); `data/` is the frozen dataset
to share (with `data/credits.csv`).

### Labeling rules (details: `labeling_guide.md`)
- Label what is **visible**. Unusable photo (aerial, map, close-up, no slope, …): **R**.
  Unsure between two landslide types: also **R** (the rule all current labels follow; the
  hardest cases are therefore evaluated separately, see 7c).
- Human labels always override the AI annotator's. Two teammates labeling the same images
  (`--relabel-others`) gives inter-annotator agreement (eval.py); ties: `--conflicts`.
- Never label the test split with VLM suggestions visible (the tool enforces this; eval.py
  reports the count, which must be 0).
- Expert check: `review.py --reviewer faculty --split test --relabel-others --skip-ai-rejects --sample 50`.

### Photo licences
Every photo keeps licence, author and source URL in `data/metadata.csv` (`tools/credits.py`
writes `data/credits.csv`). 753 photos are non-commercial (NC) and 411 no-derivatives (ND): fine
for this course project, but credit every photo you show ("Title" by Author, Licence, URL), do not
publish modified versions of ND photos, and remove NC photos before any commercial use.
See `results/licences.md`.

## Online demo (Hugging Face Space, free)
`tools/build_hf_space.py --push USER/landslide-classifier` stages `deploy/hf_space/` (Dockerfile, Space header),
`src/` and the trained heads from `models/`, then uploads them as a Docker Space (needs `hf auth login` once with a
write token). The free Space has 2 CPU cores and 16 GB RAM and serves over https, so the phone's live camera
guide works. Timing on this laptop with CUDA off and 2 threads (3 train photos; the Space CPU may differ): startup
27-46 s, landslide type 6-19 s per photo, one size estimate ~8 min. So the Space runs with `--no-dims` (no sizes)
and `--no-save` (photos are not kept). For sizes, run the app on a GPU laptop. Accuracy is the same model as above.
Note (Nov 2026): Hugging Face now requires a PRO subscription for Docker Spaces, so the free route is the one below.

## Public link from this laptop (free, with sizes)
`python tools/serve_public.py` starts the app on 127.0.0.1 and opens a Cloudflare quick tunnel; it prints
`https://<random words>.trycloudflare.com`, which works on any phone (https, so the live camera guide works) while
the window stays open. `cloudflared.exe` is a general tool and lives outside the project in `E:\cloudflared\`
(or set `CLOUDFLARED`); the tunnel uses HTTP/2 because this network blocks QUIC. Options: `--save` keeps uploads,
`--no-dims` drops sizes. Checked through the public link: page 2.4 s, full result with sizes 40 s (one train photo).

## User guidance in the app

`src/guide.py` adds:
- a first-run tour and a per-session safety checklist;
- single-photo or 4-shot site-survey mode;
- a live framing and level viewfinder with a photo-quality meter (Good / Fair / Poor / Very poor), one
  instruction at a time and a direction arrow;
- an instant on-phone photo check, each problem graded minor or serious;
- results that are always shown, even for a photo that fails the photo check: a quality band above the
  result says how reliable it is, and nothing is adjusted to look better;
- staged progress, "next steps" after each result, tickable on-site signs and a help sheet with a glossary.

It also adds:
- type first, then sizes;
- a cut-off-edge warning and a mostly-sky check;
- yes/no decision questions when the model is unsure;
- a combined site result for surveys;
- notes, GPS and saved or shared reports;
- an offline queue, a practice quiz and Hindi.

Details and tests are in `report.md` §5.1–5.2.
- **Measuring whether the guidance helps:** `python tools/guidance_report.py` (after field use without `--no-save`).
- **Live camera guide on phones** needs https: `python src/app.py --https`. This makes a self-signed certificate once in `.cache/tls`; accept the browser's warning on the phone. Without it the phone's camera app is used.
- **Photo-check thresholds:** recalibrate after the dataset changes with `python tools/calibrate_photo_check.py` (train split only).

## Photo sets and video (too big for one photo)

In the app, choose **photo set**, then:
- **Turn on the spot:** the photos are joined into one picture.
- **Walk along:** the photos are analysed together. Type the normal steps between photos to get real
  sizes from triangulation.
- **Auto sweep:** switch "Auto" on in the camera view, take the first photo, then turn or walk slowly.
  The next photos are taken by themselves.
- **Video:** pick one slow sweep from the gallery, and the phone picks sharp, overlapping frames from it.
  For a walk video, type the total steps walked during it.

The tools behind this (simulated sets from train photos; the test split is never read):
- `tools/overlap_eval.py`: thresholds and the joined-picture weight (`src/overlap_check.json`).
- `tools/stereo_eval.py`: the walk-along scale.
- `tools/focal_eval.py`: the focal length of a joined picture (`src/focal_check.json`).
- `tools/accuracy_eval.py`: out-of-fold accuracy experiments.
- `tools/object_prior_eval.py`: refits the known-object scale prior (`models/dimension_object_prior.json`).

All of them run in resumable chunks (`--budget 540`). The results are in report.md §4.6 and §5.3.

## Dimensions (after classification)

`src/dimension.py` (deep-learning models and orchestration) and `src/dim_stats.py` (pixel statistics)
measure the landslide once it is classified. The app does this automatically; skip it with
`--no-dims`. From the command line:
`python src/dimension.py photo.jpg [--label debris_flow] [--distance 40] [--save-overlay] [--json]`.
The architecture table and the assumed values are in report.md, section 6b. In short:

- **Outline:** CLIPSeg, the self-trained head and the jittered SAM soft mask are fused, damped by a
  vegetation/sky prior, and cut by a graph cut with colour and depth edges.
- **Depth:** Depth Anything V2 and Depth Pro, each with test-time augmentation, fused per
  pixel by inverse variance, then guided-filtered. ZoeDepth was tested and dropped; the evidence
  is in report.md, section 6b.
- **Scale**, best first: your typed distance; OWLv2 known-height objects (weighted median); the
  field calibration; the ensemble, corrected by the known-object prior
  (`models/dimension_object_prior.json`).
- **Measurement:** RANSAC plane geometry and per-class measures. Volume comes from published laws
  combined with a reconstructed pre-failure surface. Grains are SAM instances at two zooms, fitted
  with a truncated log-normal.
- **Ranges:** Monte Carlo P5-P95 ranges and size-class probabilities.
- **Class doubt:** when the classifier is unsure, the landslide is also measured as the runner-up
  class.

Training and calibration (both leakage-free):
- `python tools/train_outline_head.py` self-trains the outline head on the train split and selects
  it on val. Test is never used.
- `python tools/calibrate_dimensions.py` fits the scale from field measurements in
  `data/field/measurements.csv`, validated leave-one-site-out.

`python src/dimension.py a.jpg b.jpg c.jpg --site` fuses several photos of ONE landslide: the
median of the photos, with the between-photo spread added to the range. Different photos of one
site typically differ by about 1.5-2x, so use this for important sites.

`python tools/dimension_eval.py [--tag v4] [--budget-min 3]` is resumable: on low-RAM machines,
run it in chunks, and exit code 3 means rerun to continue. It reports availability, stability under phone-like edits,
the depth checks, interval widths and same-site consistency. It also writes the known-object
scale evidence.

## Why these choices (for the report)
- **Zero-shot VLM is a baseline, not the main model.** LandslideBench (2026, overhead imagery)
  measured Qwen3-VL-8B zero-shot at 19% vs 79% for a fine-tuned ConvNeXt; on our ground photos
  Qwen2.5-VL is similarly far behind the trained models (report.md).
- **Frozen features + linear heads:** with ~40–400 photos per class, frozen foundation-model
  features (DINOv2, SigLIP2) + logistic regression generalise better than fine-tuning a whole
  network (confirmed: the fine-tuned ConvNeXt scores lower).
- **Split by site, and sites linked across categories:** photos of the same landslide in train
  and test would inflate accuracy.
- **Model chosen on train+val only** by site-balanced CV + distribution-shift tests; the test
  split is only used to report.
- **No vertical flips in augmentation:** the downhill viewpoint is part of the signal.
