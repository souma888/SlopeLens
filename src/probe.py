"""Small-data model: frozen image features (DINOv2 or SigLIP2) + logistic regression ("linear probe").

Works with ~30-150 labeled images per class: the backbone is never trained, so it cannot
overfit; only a linear classifier on top is fit. Embeddings are cached per image, so adding
labels and refitting takes seconds.

Features = [CLS token, mean of patch tokens], averaged over the image and its horizontal flip.
Two heads share the features:
  - photo check: usable ground photo vs `reject` (aerial/satellite/map/close-up/...)
  - landslide type: the 6 classes, trained on usable photos only
Training images are also embedded as `--aug` phone-style variants (src/phone_aug.py) so the
model sees framing, tilt, blur and JPEG artifacts like the photos engineers will send.
C (regularization) is chosen by site-grouped, class-stratified 3-fold cross-validation over
train+val (choose_C: the same rule tools/robustness_eval.py uses, so the deployed probes are the
models that were evaluated), then the head is refit on all train+val. The test split is never
used for fitting or choosing; the bundle stores the ids it was trained on (`train_ids`) so
eval.py / system_test.py / check_integrity.py can assert they are disjoint from test.
Test predictions are written for the original photos and for phone-degraded copies
(`<name>_phonesim`), so eval.py shows how much accuracy drops on phone-quality input.

Usage:  python src/probe.py --backbone facebook/dinov2-base --name probe_base [--aug 3]
Output: models/<name>.joblib, results/pred_<name>.csv, results/pred_<name>_phonesim.csv
        (the app uses the members listed in models/selected.json, see tools/robustness_eval.py)
"""
import common  # noqa: F401  (sets HF cache dir; must precede transformers/timm)
import argparse
import os
import random
import time
import zlib

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from common import CLASS_NAMES, DATA, MODELS, REJECT, RESULTS, data_fingerprint, labeled_dataset, write_csv
from phone_aug import phone_like


class Embedder:
    def __init__(self, name, device=None):
        # An explicit device also supports ZeroGPU's CUDA emulation during startup.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()

    def __call__(self, images):
        try:
            return self._embed(images)
        except torch.OutOfMemoryError:
            # GPU busy (e.g. the VLM is running): finish on CPU instead of crashing
            print("GPU out of memory; continuing on CPU")
            torch.cuda.empty_cache()
            self.device = "cpu"
            self.model.to("cpu")
            return self._embed(images)

    @torch.inference_mode()
    def _embed(self, images):
        x = self.proc(images=images, return_tensors="pt")["pixel_values"].to(self.device)
        feats = []
        for xx in (x, x.flip(-1)):
            if hasattr(self.model, "get_image_features"):   # CLIP / SigLIP: pooled image embedding
                f = self.model.get_image_features(pixel_values=xx)
                f = getattr(f, "pooler_output", f)
                feats.append(torch.nn.functional.normalize(f, dim=-1))
            else:                                           # DINOv2: [CLS, mean of patch tokens]
                h = self.model(pixel_values=xx).last_hidden_state
                feats.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], -1))
        return (sum(feats) / 2).float().cpu().numpy()


def variant(image_id, file, k):
    """Variant 0 is the photo itself; k >= 1 is a fixed (seeded) phone-style copy."""
    img = Image.open(DATA / file).convert("RGB")
    if k == 0:
        return img
    return phone_like(img, random.Random(zlib.crc32(f"{image_id}#{k}".encode())))


def embed_all(df, backbone, k=0, batch=16):
    cache_path = DATA / "cache" / f"emb_{backbone.replace('/', '_')}.npz"
    cache = dict(np.load(cache_path)) if cache_path.exists() else {}
    key = (lambda i: i) if k == 0 else (lambda i: f"{i}#{k}")
    todo = [r for r in df.itertuples() if key(r.image_id) not in cache]
    if todo:
        emb = Embedder(backbone)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        for n, i in enumerate(tqdm(range(0, len(todo), batch), desc=f"embedding (variant {k})"), 1):
            chunk = todo[i:i + batch]
            vecs = emb([variant(r.image_id, r.file, k) for r in chunk])
            cache.update({key(r.image_id): v for r, v in zip(chunk, vecs)})
            if n % 40 == 0:   # long runs can be interrupted; keep what is done
                save_cache(cache_path, cache)
        save_cache(cache_path, cache)
    return np.stack([cache[key(i)] for i in df.image_id])


def save_cache(path, cache):
    """Atomic: an interrupted write must not corrupt hours of embeddings."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **cache)
    for attempt in range(40):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:   # another process has the cache open (np.load)
            time.sleep(0.5)
    raise PermissionError(f"could not replace {path}; close other programs using it ({tmp} kept)")


C_GRID = (0.003, 0.03, 0.3, 1.0)


def fit(X, y, C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced")).fit(X, y)


def with_copies(X, y, idx, X_aug):
    """Rows idx plus their phone-style copies (copies always stay with their photo; rows of a
    copy array that are NaN, i.e. not cached, are skipped)."""
    Xs, ys = [X[idx]], [y[idx]]
    for A in X_aug:
        ok = idx[~np.isnan(A[idx]).any(1)]
        Xs.append(A[ok])
        ys.append(y[ok])
    return np.concatenate(Xs), np.concatenate(ys)


def choose_C(X, y, groups, idx, X_aug, folds=3, seed=0):
    """Mean macro-F1 of site-grouped, class-stratified CV inside idx, for each C in C_GRID."""
    scores = {c: [] for c in C_GRID}
    cv = StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
    for a, b in cv.split(idx, y[idx], groups[idx]):
        Xa, ya = with_copies(X, y, idx[a], X_aug)
        for c in C_GRID:
            scores[c].append(f1_score(y[idx[b]], fit(Xa, ya, c).predict(X[idx[b]]), average="macro"))
    best = max(C_GRID, key=lambda c: np.mean(scores[c]))
    return best, {c: round(float(np.mean(v)), 4) for c, v in scores.items()}


def train_head(X, y, split, groups, X_aug, name):
    """Choose C by grouped CV on train+val, refit on all train+val incl. phone-style copies."""
    tv = np.where(np.isin(split, ["train", "val"]))[0]
    C, scores = choose_C(X, y, groups, tv, X_aug)
    print(f"  [{name}] CV macro-F1 by C: {scores} -> C={C}")
    return fit(*with_copies(X, y, tv, X_aug), C), C, scores[C]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="facebook/dinov2-base")
    ap.add_argument("--aug", type=int, default=3, help="phone-style copies per train/val image")
    ap.add_argument("--name", default="probe", help="output name (models/<name>.joblib, results/pred_<name>.csv)")
    args = ap.parse_args()

    df = labeled_dataset(include_reject=True)
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0)
          .reindex(columns=CLASS_NAMES + [REJECT], fill_value=0).to_string())
    X = embed_all(df, args.backbone)
    fitset = df.split.isin(["train", "val"]).values
    X_aug = []
    for k in range(1, args.aug + 1):
        A = np.zeros_like(X)
        A[fitset] = embed_all(df[fitset], args.backbone, k)
        X_aug.append(A)
    split, groups = df.split.values, df.site_id.values

    gate_y = np.where(df.label == REJECT, "reject", "usable")
    gate = gate_c = gate_f1 = None
    if len(set(gate_y[fitset])) == 2:
        gate, gate_c, gate_f1 = train_head(X, gate_y, split, groups, X_aug, "photo check")
    else:
        print("  [photo check] no reject labels: photo check skipped (every photo treated as usable)")
    usable = (df.label != REJECT).values
    y = df.label.values
    if len(set(y[usable & fitset])) < 2:
        raise SystemExit("need labeled train/val images from at least 2 classes")
    typ, typ_c, typ_f1 = train_head(X[usable], y[usable], split[usable], groups[usable],
                                    [A[usable] for A in X_aug], "type")

    MODELS.mkdir(exist_ok=True)
    joblib.dump({"clf": typ, "gate": gate, "backbone": args.backbone, "classes": list(typ.classes_),
                 "cv_macro_f1": {"type": typ_f1, "photo_check": gate_f1}, "C": {"type": typ_c, "photo_check": gate_c},
                 "train_ids": sorted(df.image_id[fitset]), "n_train_usable": int((usable & fitset).sum()),
                 "aug": args.aug, "data_fingerprint": data_fingerprint()}, MODELS / f"{args.name}.joblib")
    print(f"saved {MODELS / (args.name + '.joblib')}")

    te = df[(df.split == "test") & usable].reset_index(drop=True)
    if len(te):
        for name, Xte in ((args.name, X[(split == "test") & usable]), (args.name + "_phonesim", embed_all(te, args.backbone, 1))):
            proba = typ.predict_proba(Xte)
            write_csv(pd.DataFrame({"image_id": te.image_id, "pred": typ.classes_[proba.argmax(1)],
                                    "conf": proba.max(1).round(4),
                                    **{f"p_{c}": proba[:, i].round(4) for i, c in enumerate(typ.classes_)}}),
                      RESULTS / f"pred_{name}.csv")
        print(f"test predictions -> {RESULTS}/pred_{args.name}*.csv; score with: python src/eval.py")


if __name__ == "__main__":
    main()
