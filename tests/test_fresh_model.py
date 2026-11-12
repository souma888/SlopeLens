"""Dataset and calibration guard tests; no torch/downloads required."""
import importlib.util
from pathlib import Path
import unittest

AVAILABLE = all(importlib.util.find_spec(name) for name in ("numpy", "pandas"))
if AVAILABLE:
    import numpy as np
    import pandas as pd
    spec = importlib.util.spec_from_file_location("train_fresh", Path(__file__).resolve().parents[1] / "tools/train_fresh.py")
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)


@unittest.skipUnless(AVAILABLE, "numpy/pandas required for training guard tests")
class FreshModelTests(unittest.TestCase):
    def dataset(self):
        rows = []
        for split, count in (("train", 10), ("val", 5), ("test", 5)):
            for label in fresh.CLASS_NAMES + [fresh.REJECT]:
                for i in range(count):
                    identity = f"{split}-{label}-{i}"
                    rows.append(dict(image_id=identity, site_id=identity, split=split, label=label, vlm_shown="0"))
        return pd.DataFrame(rows)

    def test_complete_pilot_ready(self):
        self.assertEqual(fresh.check(self.dataset()).loc["train", "debris_flow"], 10)

    def test_missing_class_rejected(self):
        df = self.dataset()
        with self.assertRaisesRegex(ValueError, "rotational_slide"):
            fresh.check(df[df.label != "rotational_slide"])

    def test_site_leakage_rejected(self):
        df = self.dataset()
        df.loc[df.split.eq("test"), "site_id"] = df.iloc[0].site_id
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            fresh.check(df)

    def test_suggested_test_labels_rejected(self):
        df = self.dataset()
        df.loc[df.split.eq("test"), "vlm_shown"] = "1"
        with self.assertRaisesRegex(ValueError, "suggestions"):
            fresh.check(df)

    def test_failed_calibration_suppresses_confidence(self):
        threshold, passed = fresh.confidence_threshold(["b"] * 10, np.array([[0.9, 0.1]] * 10), ["a", "b"])
        self.assertEqual(threshold, 1.0)
        self.assertFalse(passed)

    def test_good_calibration(self):
        threshold, passed = fresh.confidence_threshold(["a"] * 10, np.array([[0.9, 0.1]] * 10), ["a", "b"])
        self.assertEqual(threshold, 0.5)
        self.assertTrue(passed)


if __name__ == "__main__":
    unittest.main()
