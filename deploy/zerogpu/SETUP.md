# Deploy to free ZeroGPU

## Current prerequisites

The GitHub source checkout does **not** contain the trained classifier heads or dataset.
The demo cannot perform the project's trained classification until those files are restored.
Do not replace them with random heads or report the published scores for a different model.

Required: `models/selected.json` (members + calibrated confident_min), and the selected
`probe_base.joblib`, `probe_large.joblib`, `probe_siglip.joblib` bundles. Recover the original
files from the original training machine/author, or obtain labeled images and rerun the
training/selection workflow in the main README. Training is separate from deployment.
The large public backbones download automatically at Space startup.

Restore `outline_head.joblib`, `dimension_calibration.json` and `dimension_object_prior.json`
when available. Without them measurements run with the remaining model components/priors;
the report lists availability. Joblib files execute Python when loaded: use trusted artifacts.

## Authenticate

1. Sign up at https://huggingface.co/join and verify your email.
2. Free ZeroGPU hosting requires an account in good standing older than 30 days.
   A brand-new free account cannot deploy this today. Do not select a paid GPU as a fallback.
3. Create a Write token at https://huggingface.co/settings/tokens.
4. In PowerShell (do not put the token in chat or a Git file):

```powershell
python -m pip install --user huggingface_hub==2.2.0
python -c "from huggingface_hub import login; login()"
```

## Deploy from your computer (no local GPU or inference dependencies needed)

With the trained files restored under `models/`, from the project root:

```powershell
python -m unittest discover -s tests -v
python tools/build_zerogpu.py --push YOUR_HF_USERNAME/SlopeLens
```

This validates the selected heads, stages only application code and allowlisted artifacts,
creates a Gradio Space with `zero-a10g` hardware (the API's legacy name for ZeroGPU),
uploads it, and waits up to 20 minutes for RUNNING. It never requests paid hardware.
Account/eligibility errors must be resolved before deployment can continue.
Optional CPU/GPU measurement components that fail to load are visible in the JSON report.

For staging only: `python tools/build_zerogpu.py`. Use `--models PATH` for artifacts elsewhere.
The staging script itself needs only Python's standard library. The Space runs Python 3.12;
your local Python 3.15 can stage/upload without installing torch locally.

## Deploy through GitHub Actions

1. Upload the trusted model artifacts to a Hugging Face **model** repository (root files,
   including `selected.json`). Record its full 40-character commit SHA.
2. In GitHub SlopeLens → Settings → Secrets and variables → Actions, add `HF_TOKEN`.
   Use the same Write token, with read access to the model repo and write access to the Space.
3. Actions → Deploy SlopeLens to free ZeroGPU → Run workflow. Supply your destination
   Space ID, model repository ID and immutable model commit SHA.

Deployments run only when manually requested. They are not triggered by untrusted PRs.
The deployment stages model bytes without loading joblib on the GitHub runner.

## Verify after deployment

Open the returned HTTPS Space URL. Confirm it says ZeroGPU, then:

1. Upload a ground-level slope photo. Classify and check probabilities/photo suitability.
2. Request sizes. Confirm depth/segmentation models are available, inspect the outline,
   dimension ranges and any cut-off-edge warnings. GPU quotas may prevent long jobs.
3. Try an unusable photo and confirm its warning. Try a four-shot same-site survey and an
   overlapping set; inspect frames and the overlap report before trusting a video result.
4. Download the JSON report. Test in a new browser session to confirm reports are separate.

RUNNING confirms startup only. It does **not** validate model accuracy or measurement quality.
The free service queues jobs, has visitor quotas and can sleep. Batch measurement requests
reserve up to 180 seconds; classification up to 60 seconds. The original phone interface
and its camera arrows/Hindi/offline features remain local in `src/app.py`.

References: https://huggingface.co/docs/hub/spaces-zerogpu
