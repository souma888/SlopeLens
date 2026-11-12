---
title: SlopeLens
emoji: ⛰️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.9.0
python_version: '3.12'
app_file: space_app.py
pinned: false
short_description: Ground-photo landslide classification and size estimates
---

# SlopeLens online demo

Classification, photo checks, segmentation, depth, dimensions, volume and grain estimates
run on shared Hugging Face GPUs. Your laptop can be switched off after deployment.
This is a research prototype; its scores measure agreement with AI annotations.
Measurements are estimates, not a substitute for field measurements or professional assessment.

Take an overview from a safe position, with the full slope visible. Results include
confidence, photo suitability, checks and safety guidance. Classification runs first;
size estimation is a separate, optional request to conserve your GPU quota.

Upload one photo, up to four survey photos of ONE site, up to eight overlapping photos,
or a short sweep video (up to 60 seconds). Survey estimates fuse the individual shots;
photo sets can stitch rotating views and triangulate walking views with a supplied baseline.
Video samples are evenly spaced; inspect their quality before trusting the result.

GPU jobs queue and use the visitor's daily quota. This demo does not provide unlimited
usage or guaranteed availability. Uploads and generated reports use temporary storage,
cleaned periodically; there is no permanent upload archive.

This Gradio edition provides upload/webcam capture, surveys, sets, video and JSON reports.
The original phone interface remains in `src/app.py`; its live arrows, auto capture,
offline queue, Hindi interface and GPS/share flows have not been ported to this edition.

Source and deployment instructions: https://github.com/souma888/SlopeLens
