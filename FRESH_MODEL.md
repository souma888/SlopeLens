# Rebuilding SlopeLens without the original data or models

The new dataset/model is a separate experiment. The original report's scores do not apply.
Public pretrained DINOv2/SigLIP backbones provide image features; NEW logistic regression
heads learn the six landslide labels and a usable-photo/reject gate. This is transfer learning,
not random initialization of the entire vision backbone.

## Started

A first small collection is stored under ignored `data/`, with per-image licence, author,
source URL, category and description. Search/category labels are HINTS, never training labels.
Collection is resumable. Increase coverage and independent sites before expecting a useful model.
The first batch has 65 unreviewed candidate images; its metadata is versioned in
`datasets/fresh_seed_manifest.csv`. That manifest is provenance, not a labeled dataset.
The six contact sheets are under `data/labels/codex_seed_review/`; no labels have been
assigned and no trained model or new accuracy result exists yet.

```powershell
python src/collect_commons.py --search-only --max-per-search 60
python src/collect_openverse.py --pages 2
python tools/precompute_embeddings.py --backbone facebook/dinov2-base
python src/split.py
python tools/link_sites.py
python src/review.py --reviewer YOUR_NAME
```

The visual site-linking step needs DINOv2 features, and must happen before blind labeling.
Check linked sites manually; similar generic scenes are not necessarily the same event.
Run `tools/credits.py` to export attribution. Respect per-image licence restrictions.
Review all seven categories (six types plus unusable/reject); include negatives and
hard cases. Do not infer a landslide type solely from the filename, title or search query.

AI-assisted visual annotation is possible using `tools/contact_sheet.py` and
`tools/ai_labels.py --reviewer codex_ai`. Retain contact sheets, codes, image IDs and reviewer
names as the audit trail. Human review overrides AI labels. A test set reviewed by a
geotechnical expert is needed before presenting the model as field-validated.

## Train a new model

Use Python 3.12 and the project requirements (CPU torch is supported, though slow).
Run on a cloud notebook/server if local GPU use is unwanted. Do not install inference
dependencies in the local Python 3.15 environment just to run the deployment uploader.

```powershell
python tools/train_fresh.py --check-only
python tools/train_fresh.py
```

The readiness check requires all six types and reject: at least 10 train, 5 validation
and 5 test photos per label, with at least 3 independent train sites and 2 sites in each
held-out split. These are small PILOT minima, not an accuracy guarantee; substantially more
data and external field testing are needed. It rejects sites crossing splits and tests
labeled with model suggestions visible. Near-duplicates must also be checked before training.

This fresh baseline trains each head ONLY on train, choosing regularization by grouped CV.
The ensemble membership is fixed up front. Validation alone sets the uncertainty threshold;
if no threshold meets the target accuracy and coverage, the threshold is 1.0 and the report
records failure. Test is used only for final reporting. Test metrics have no confidence
intervals yet; this is a baseline, not a replacement for the original full evaluation.

Outputs go to `models/fresh_TIMESTAMP/`, including a machine-readable report with label
provenance. Original heads/selection are never overwritten. Start with `--members probe_base`
for a single-backbone pilot if compute is limited; the default trains all three.
The fresh baseline has no phone augmentations; the original `src/probe.py` supports these
when extending experiments after the baseline is evaluated.

## Measurements and deployment

The existing public segmentation/depth models can still estimate dimensions, volume and
grain sizes after classification. The original self-trained outline head and field
calibration are absent. Recover them or train/calibrate anew; classification photos alone
do not establish metric accuracy. Collect measured distances/dimensions and validate the
ranges against field ground truth before claiming comparable measurement accuracy.

When the new experiment has been reviewed:

```powershell
python tools/build_zerogpu.py --models models/fresh_TIMESTAMP --push YOUR_ACCOUNT/SlopeLens
```

The original free ZeroGPU account-age/usage restrictions still apply. A newly trained model
does not remove hosting eligibility limits. Never carry over the original 80.6% score to
the new model or silently substitute zero-shot predictions.
