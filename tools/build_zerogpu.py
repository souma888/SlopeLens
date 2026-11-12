"""Validate, stage and deploy the complete Gradio edition to FREE ZeroGPU only.

Staging uses only the standard library. Upload additionally needs huggingface_hub.
Never substitutes random/untrained heads for missing artifacts or requests paid hardware.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
MEMBERS = {"probe_base", "probe_large", "probe_siglip"}
OPTIONAL = ("outline_head.joblib", "dimension_calibration.json", "dimension_object_prior.json")


def validate_models(model_dir):
    selection = model_dir / "selected.json"
    if not selection.is_file():
        raise ValueError(f"Missing trained selection: {selection}. Restore the original models/ folder "
                         "or retrain using labeled data. Public backbones alone are insufficient.")
    selected = json.loads(selection.read_text(encoding="utf-8"))
    members = selected.get("members")
    if not isinstance(members, list) or not members or any(m not in MEMBERS for m in members):
        raise ValueError("selected.json must select trained probe_base/probe_large/probe_siglip members.")
    threshold = selected.get("confident_min")
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("selected.json needs a calibrated confident_min in (0, 1].")
    files = [selection] + [model_dir / f"{member}.joblib" for member in sorted(set(members))]
    missing = [f.name for f in files if not f.is_file() or not f.stat().st_size]
    if missing:
        raise ValueError(f"Missing/empty trained heads: {', '.join(missing)}")
    files += [model_dir / name for name in OPTIONAL if (model_dir / name).is_file()]
    return files


def stage(model_dir, destination):
    files = validate_models(model_dir)  # fail before modifying any staging folder
    destination = destination.resolve()
    if destination.exists():
        raise ValueError(f"Staging destination already exists: {destination}")
    destination.mkdir(parents=True)
    shutil.copytree(ROOT / "deploy" / "zerogpu", destination,
                    ignore=shutil.ignore_patterns("__pycache__"), dirs_exist_ok=True)
    (destination / "src").mkdir()
    for source in (ROOT / "src").iterdir():
        if source.suffix in (".py", ".json"):
            shutil.copy2(source, destination / "src" / source.name)
    (destination / "models").mkdir()
    for source in files:
        shutil.copy2(source, destination / "models" / source.name)
    manifest = {p.relative_to(destination).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(destination.rglob("*")) if p.is_file()}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Staged {len(manifest)} files at {destination}")
    absent = [name for name in OPTIONAL if not (model_dir / name).is_file()]
    if absent:
        print("Optional measurement artifacts unavailable (reported in results): " + ", ".join(absent))
    return destination


def deploy(api, repo, folder, wait_minutes):
    identity = api.whoami()
    if "/" not in repo:
        repo = f"{identity['name']}/{repo}"
    # No cpu-basic or paid-GPU fallback: a free-account eligibility error stays an error.
    api.create_repo(repo, repo_type="space", space_sdk="gradio", space_hardware="zero-a10g", exist_ok=True)
    info = api.space_info(repo)
    if (info.card_data or {}).get("sdk") != "gradio":
        raise ValueError("Existing destination is not a Gradio Space. Choose a new Space name.")
    runtime = api.get_space_runtime(repo)
    hardware = runtime.requested_hardware or runtime.hardware
    if hardware != "zero-a10g":
        raise ValueError(f"Destination hardware is {hardware!r}, not free ZeroGPU. "
                         "Select ZeroGPU in the Space settings before retrying.")
    commit = api.upload_folder(repo_id=repo, repo_type="space", folder_path=str(folder),
                               commit_message="Deploy SlopeLens ZeroGPU",
                               delete_patterns=["src/*", "models/*"])
    print(f"Uploaded: {commit.commit_url}")
    deadline = time.monotonic() + wait_minutes * 60
    previous = None
    while time.monotonic() < deadline:
        info = api.space_info(repo)
        runtime = api.get_space_runtime(repo)
        status = getattr(runtime.stage, "value", runtime.stage)
        if status != previous:
            print(f"Space status: {status}", flush=True)
            previous = status
        if status == "RUNNING":
            print(f"Running: https://huggingface.co/spaces/{repo}")
            print("Still required: upload a field photo and verify classification AND a size estimate.")
            return
        if "ERROR" in status or status in ("STOPPED", "PAUSED"):
            raise RuntimeError(f"Deployment stopped: {status}. Inspect the Space build/runtime logs.")
        time.sleep(10)
    raise RuntimeError(f"Deployment not confirmed running within {wait_minutes} minutes. "
                       f"Check https://huggingface.co/spaces/{repo}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--output", type=Path, help="New staging directory (must not exist)")
    parser.add_argument("--model-repo", help="Trusted Hugging Face model repository containing trained heads")
    parser.add_argument("--model-revision", help="Immutable 40-character commit SHA for --model-repo")
    parser.add_argument("--push", metavar="OWNER/SPACE", help="Create/update a ZeroGPU Space")
    parser.add_argument("--wait-minutes", type=int, default=20)
    args = parser.parse_args()
    if args.wait_minutes < 1:
        parser.error("--wait-minutes must be positive")
    try:
        if args.model_repo:
            if not args.model_revision or len(args.model_revision) != 40 or any(
                    c not in "0123456789abcdef" for c in args.model_revision.lower()):
                raise ValueError("--model-repo requires an immutable 40-character --model-revision commit SHA")
            from huggingface_hub import snapshot_download
            args.models = Path(snapshot_download(args.model_repo, revision=args.model_revision,
                allow_patterns=["selected.json", "probe_*.joblib", *OPTIONAL]))
        output = args.output or Path(tempfile.mkdtemp(prefix="slopelens-stage-")) / "space"
        folder = stage(args.models, output)
        if args.push:
            from huggingface_hub import HfApi
            deploy(HfApi(), args.push, folder, args.wait_minutes)
    except (ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"Deployment blocked: {exc}\n")


if __name__ == "__main__":
    main()
