"""Classify new phone photos with the trained model (command line; the web app uses the same code).

Every photo goes through two steps:
  1. photo check: is this a usable ground photo of a slope? (rejects aerial/satellite/map/close-up)
  2. landslide type: probabilities for the 6 classes; low confidence is reported as uncertain

Which model is used: `--model auto` (default) follows models/selected.json, written by
tools/robustness_eval.py (chosen on train+val by cross-validation and distribution-shift tests;
a single probe or an average of several). An explicit name (probe_base, probe_large,
probe_siglip, cnn) overrides it; the confidence threshold stays the calibrated one from
selected.json when present. The photo check always comes from probe bundles (the CNN has none).

Usage:
  python src/predict.py photo1.jpg photo2.jpg
  python src/predict.py --model cnn photo.jpg
"""
import common  # noqa: F401  (sets HF cache dir; must precede transformers/timm)
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image, ImageOps

from common import CLASS_NAMES, MODELS

try:                         # iPhone HEIC photos, if pillow-heif is installed
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

USABLE_MIN = 0.5             # photo-check probability below this -> ask for a better photo
CONFIDENT_MIN = 0.5          # top class probability below this -> "uncertain" (selected.json overrides)
# When unsure between two related types, the family is still a useful, more reliable answer.
FAMILIES = {"flow": ("debris_flow", "earth_mud_flow"), "slide": ("rotational_slide", "translational_slide")}


def load_image(src):
    """Path or file-like -> upright RGB, long side <= 1024 (phones store rotation in EXIF)."""
    img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    img.thumbnail((1024, 1024))
    return img


def selection():
    sel = MODELS / "selected.json"
    return json.loads(sel.read_text(encoding="utf-8")) if sel.exists() else {}


def resolve_members(model):
    """-> (members, confidence threshold, expected backbones). auto follows models/selected.json."""
    s = selection()
    conf = s.get("confident_min", CONFIDENT_MIN)
    if model == "auto":
        return (s["members"], conf, s.get("backbones", {})) if s else (["probe"], conf, {})
    return [model], conf, {}


class Predictor:
    def __init__(self, model="auto", device=None):
        from probe import Embedder
        self.members, self.confident_min, expected = resolve_members(model)
        probe_names = [m for m in self.members if m != "cnn"]
        if not probe_names:   # the photo check always comes from a probe bundle
            probe_names_gate = ["probe_base"] if (MODELS / "probe_base.joblib").exists() else ["probe"]
        else:
            probe_names_gate = probe_names
        self.bundles = {}
        for name in set(probe_names) | set(probe_names_gate):
            path = MODELS / f"{name}.joblib"
            if not path.exists():
                hint = expected.get(name, "<backbone>")
                raise SystemExit(f"{path} missing; run src/probe.py --backbone {hint} --name {name}")
            self.bundles[name] = joblib.load(path)
            want = expected.get(name)
            if want and self.bundles[name]["backbone"] != want:
                raise SystemExit(f"{path.name} was trained on {self.bundles[name]['backbone']}, but the "
                                 f"selection expects {want}; retrain it: src/probe.py --backbone {want} --name {name}")
        self.embedders = {b["backbone"]: None for b in self.bundles.values()}
        for bb in self.embedders:
            self.embedders[bb] = Embedder(bb, device=device)
        self.type_probes, self.gate_probes = probe_names, probe_names_gate
        self.cnn = self._load_cnn() if "cnn" in self.members else None
        self.model_name = " + ".join(self.members)
        self.train_ids = set().union(*[set(b.get("train_ids", [])) for b in self.bundles.values()])
        if self.cnn:
            self.train_ids |= set(self.cnn_train_ids)

    def _load_cnn(self):
        import timm
        from train_cnn import transforms
        # finished checkpoints only (not *.best.pt / *.state.pt of a run in progress)
        ckpts = sorted((p for p in MODELS.glob("cnn_*.pt") if not p.name.endswith((".best.pt", ".state.pt"))),
                       key=lambda p: p.stat().st_mtime)
        if not ckpts:
            raise SystemExit("no finished CNN checkpoint in models/; run src/train_cnn.py first")
        ck = torch.load(ckpts[-1], map_location="cpu")
        self.cnn_train_ids = ck.get("train_ids", [])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = timm.create_model(ck["model"], pretrained=False, num_classes=len(ck["classes"]))
        net.load_state_dict(ck["state_dict"])
        net.to(device).eval()
        _, tf = transforms(timm.data.resolve_data_config({}, model=net), ck["size"])
        return net, tf, ck["classes"], device

    @torch.inference_mode()
    def _cnn_probs(self, images):
        net, tf, classes, device = self.cnn
        x = torch.stack([tf(im) for im in images]).to(device)
        probs = ((net(x).softmax(-1) + net(x.flip(-1)).softmax(-1)) / 2).float().cpu().numpy()
        return self._align(probs, classes)

    @staticmethod
    def _align(probs, classes):
        out = np.zeros((len(probs), len(CLASS_NAMES)))
        for j, c in enumerate(classes):
            out[:, CLASS_NAMES.index(c)] = probs[:, j]
        return out

    def __call__(self, images):
        """List of PIL images -> list of result dicts."""
        feats = {bb: emb(images) for bb, emb in self.embedders.items()}
        gates = [b for b in (self.bundles[n] for n in self.gate_probes) if b.get("gate") is not None]
        usable = (np.mean([b["gate"].predict_proba(feats[b["backbone"]])[:, list(b["gate"].classes_).index("usable")]
                           for b in gates], axis=0) if gates else np.ones(len(images)))
        parts = [self._align(self.bundles[n]["clf"].predict_proba(feats[self.bundles[n]["backbone"]]),
                             self.bundles[n]["clf"].classes_) for n in self.type_probes]
        if self.cnn:
            parts.append(self._cnn_probs(images))
        probs = np.mean(parts, axis=0)
        return [self.decide(u, dict(zip(CLASS_NAMES, p))) for u, p in zip(usable, probs)]

    def decide(self, u, probs):
        """Photo-check probability + class probabilities -> result dict (also used for combined photo sets)."""
        ranked = sorted(((c, float(probs.get(c, 0.0))) for c in CLASS_NAMES), key=lambda kv: -kv[1])
        if u < USABLE_MIN:
            status = "bad_photo"
        elif ranked[0][1] < self.confident_min:
            status = "uncertain"
        else:
            status = "ok"
        family = None
        if status == "uncertain":
            top2 = {ranked[0][0], ranked[1][0]}
            for fam, members in FAMILIES.items():
                if top2 == set(members) and ranked[0][1] + ranked[1][1] >= self.confident_min:
                    family = fam
        return {"status": status, "usable": float(u), "label": ranked[0][0], "family": family,
                "confidence": ranked[0][1], "ranked": ranked, "model": self.model_name}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model", default="auto", help="auto (models/selected.json), probe, probe_base, probe_large, cnn")
    args = ap.parse_args()
    pred = Predictor(args.model)
    results = pred([load_image(p) for p in args.images])
    print(f"model: {pred.model_name}")
    for path, r in zip(args.images, results):
        print(f"\n{Path(path).name}: photo check {r['usable']:.0%} usable -> {r['status']}")
        for c, p in r["ranked"][:3]:
            print(f"     {c:20s} {p:.0%}")


if __name__ == "__main__":
    main()
