"""Train a NEW landslide ensemble from newly reviewed photos; no old artifacts needed.

python tools/train_fresh.py --check-only
python tools/train_fresh.py

Training uses train only, threshold calibration uses val only, and the final evaluation
uses test only. All three public backbones are frozen. Labels must come from actual image
review, never from search keywords. AI-reviewed labels remain explicitly identified.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from common import (CLASS_NAMES, DATA, MODELS, RESULTS, REJECT, data_fingerprint,
                    labeled_dataset, write_csv)

BACKBONES = {"probe_base": "facebook/dinov2-base", "probe_large": "facebook/dinov2-large",
             "probe_siglip": "google/siglip2-base-patch16-224"}


def check(df):
    required = CLASS_NAMES + [REJECT]
    if df.empty:
        raise ValueError("No reviewed images. Collect and review photos first.")
    if df.groupby("site_id").split.nunique().max() > 1:
        raise ValueError("A site appears in multiple splits. Fix site grouping before training.")
    if df.image_id.duplicated().any():
        raise ValueError("Duplicate image IDs in labeled dataset.")
    counts = pd.crosstab(df.split, df.label).reindex(index=["train", "val", "test"],
                                                    columns=required, fill_value=0).fillna(0)
    print(counts.to_string())
    problems = []
    for split, minimum in (("train", 10), ("val", 5), ("test", 5)):
        for label in required:
            rows = df[(df.split == split) & (df.label == label)]
            if len(rows) < minimum:
                problems.append(f"{split}/{label}: {len(rows)} photos, need >= {minimum}")
            if rows.site_id.nunique() < (3 if split == "train" else 2):
                problems.append(f"{split}/{label}: too few independent sites")
    if (df.vlm_shown.astype(str).eq("1") & df.split.eq("test")).any():
        problems.append("Test labels were exposed to model suggestions")
    if problems:
        raise ValueError("Dataset not ready (pilot minima, not an accuracy guarantee):\n" + "\n".join(problems))
    return counts


def confidence_threshold(y, probabilities, classes, target=0.85):
    confidence = probabilities.max(axis=1)
    correct = np.asarray(classes)[probabilities.argmax(axis=1)] == np.asarray(y)
    for threshold in np.arange(0.5, 1.0, 0.025):
        accepted = confidence >= threshold
        if accepted.mean() >= 0.5 and correct[accepted].mean() >= target:
            return float(round(threshold, 3)), True
    # No validated operating point: suppress confident results rather than invent one.
    return 1.0, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--members", nargs="+", choices=list(BACKBONES), default=list(BACKBONES))
    args = parser.parse_args()
    if not (DATA / "metadata.csv").exists() or not (DATA / "splits.csv").exists():
        parser.exit(1, "Collect photos, assign site splits, and review labels before training.\n")
    df = labeled_dataset(include_reject=True)
    try:
        counts = check(df)
    except ValueError as exc:
        parser.exit(1, str(exc) + "\n")
    if args.check_only:
        return
    import joblib
    from sklearn.metrics import accuracy_score, f1_score
    from probe import choose_C, embed_all, fit

    MODELS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    # Keep a fresh run separate until all model files and reports are complete.
    from datetime import datetime, timezone
    run = MODELS / ("fresh_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    run.mkdir()
    train = df.split.eq("train").to_numpy()
    usable = df.label.ne(REJECT).to_numpy()
    val = df.split.eq("val").to_numpy() & usable
    test = df.split.eq("test").to_numpy() & usable
    type_parts, gates = [], []
    for member in dict.fromkeys(args.members):
        backbone = BACKBONES[member]
        X = embed_all(df, backbone, batch=4)
        parts = []
        for mask, labels in ((usable, df.label.to_numpy()),
                             (np.ones(len(df), dtype=bool), np.where(usable, "usable", REJECT))):
            indices = np.flatnonzero(train[mask])
            c, scores = choose_C(X[mask], labels[mask], df.site_id.to_numpy()[mask], indices, [], folds=3)
            head = fit(X[mask][indices], labels[mask][indices], c)
            parts.append((head, c, scores))
        typ, gate = parts[0][0], parts[1][0]
        bundle = {"clf": typ, "gate": gate, "backbone": backbone, "classes": list(typ.classes_),
                  "train_ids": sorted(df.image_id[train]), "data_fingerprint": data_fingerprint(),
                  "aug": 0, "C": {"type": parts[0][1], "photo_check": parts[1][1]},
                  "label_reviewers": sorted(set(df.loc[train, "reviewers"])),
                  "provenance": "newly collected/reviewed dataset; train split only"}
        joblib.dump(bundle, run / f"{member}.joblib")
        probabilities = typ.predict_proba(X)
        type_parts.append(probabilities[:, [list(typ.classes_).index(c) for c in CLASS_NAMES]])
        gates.append(gate.predict_proba(X)[:, list(gate.classes_).index("usable")])
    probs = np.mean(type_parts, axis=0)
    photo_usable = np.mean(gates, axis=0)
    threshold, calibrated = confidence_threshold(df.label[val], probs[val], CLASS_NAMES)
    selection = {"members": list(dict.fromkeys(args.members)), "confident_min": threshold,
                 "backbones": {m: BACKBONES[m] for m in args.members},
                 "selection": "architecture fixed before training; C from grouped train CV; threshold from val",
                 "uncertainty_target_met": calibrated, "data_fingerprint": data_fingerprint()}
    (run / "selected.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    prediction = np.asarray(CLASS_NAMES)[probs.argmax(axis=1)]
    confident = (probs.max(axis=1) >= threshold) & (photo_usable >= 0.5)
    report = {"model_dir": str(run), "selection": selection, "counts": counts.to_dict(),
              "labels": "Scores measure agreement with dataset reviewers; AI labels are not expert ground truth.",
              "test_type_accuracy": float(accuracy_score(df.label[test], prediction[test])),
              "test_type_macro_f1": float(f1_score(df.label[test], prediction[test], labels=CLASS_NAMES, average="macro")),
              "test_usable_confident_coverage": float(confident[test].mean()),
              "test_usable_confident_accuracy": (float(accuracy_score(df.label[test & confident], prediction[test & confident]))
                                                  if (test & confident).any() else None),
              "test_photo_check_accuracy": float(accuracy_score(usable[df.split.eq("test")],
                                                     photo_usable[df.split.eq("test")] >= 0.5)),
              "phone_augmentation": False, "field_validated": False}
    (run / "fresh_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(pd.DataFrame({"image_id": df.image_id[test], "label": df.label[test],
                           "pred": prediction[test], "confidence": probs[test].max(axis=1),
                           "usable": photo_usable[test]}), RESULTS / (run.name + "_test.csv"))
    print(json.dumps(report, indent=2))
    print(f"New artifacts: {run}. Review results before deployment; old models were not overwritten.")


if __name__ == "__main__":
    main()
