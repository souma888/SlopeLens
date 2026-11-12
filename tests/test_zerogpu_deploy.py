"""Deployment checks use synthetic artifact bytes, never unpickle or run fake models."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_zerogpu", ROOT / "tools" / "build_zerogpu.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.models = Path(self.tmp.name) / "models"
        self.models.mkdir()
        self.output = Path(self.tmp.name) / "space"

    def trained_fixture(self, **overrides):
        selected = {"members": ["probe_base", "probe_large", "probe_siglip"], "confident_min": 0.6}
        selected.update(overrides)
        (self.models / "selected.json").write_text(json.dumps(selected), encoding="utf-8")
        for member in ("probe_base", "probe_large", "probe_siglip"):
            (self.models / f"{member}.joblib").write_bytes(b"test staging bytes; not a trained model")

    def test_missing_selection_does_not_stage(self):
        with self.assertRaisesRegex(ValueError, "Missing trained selection"):
            builder.stage(self.models, self.output)
        self.assertFalse(self.output.exists())

    def test_selected_head_required(self):
        self.trained_fixture()
        (self.models / "probe_large.joblib").unlink()
        with self.assertRaisesRegex(ValueError, "probe_large"):
            builder.validate_models(self.models)

    def test_rejects_invalid_selection(self):
        for override in ({"members": ["../secrets"]}, {"members": []}, {"confident_min": float("nan")},
                         {"confident_min": 0}, {"confident_min": 2}):
            with self.subTest(override=override):
                self.trained_fixture(**override)
                with self.assertRaises(ValueError):
                    builder.validate_models(self.models)

    def test_stage_includes_calibration_and_integrity_manifest(self):
        import hashlib
        self.trained_fixture()
        (self.models / "dimension_calibration.json").write_text("{}", encoding="utf-8")
        (self.models / "unrelated-secret.txt").write_text("exclude", encoding="utf-8")
        builder.stage(self.models, self.output)
        self.assertTrue((self.output / "models" / "dimension_calibration.json").is_file())
        self.assertTrue((self.output / "space_app.py").is_file())
        self.assertFalse((self.output / "models" / "unrelated-secret.txt").exists())
        manifest = json.loads((self.output / "manifest.json").read_text())
        for name, digest in manifest.items():
            self.assertEqual(hashlib.sha256((self.output / name).read_bytes()).hexdigest(), digest)

    def test_existing_stage_is_not_deleted(self):
        self.trained_fixture()
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "already exists"):
            builder.stage(self.models, self.output)
        self.assertEqual(marker.read_text(), "keep")

    def test_paid_hardware_is_never_reconfigured_or_uploaded(self):
        class API:
            def whoami(self): return {"name": "owner"}
            def create_repo(self, *args, **kwargs):
                self.requested = kwargs["space_hardware"]
            def space_info(self, repo): return SimpleNamespace(card_data={"sdk": "gradio"})
            def get_space_runtime(self, repo):
                return SimpleNamespace(requested_hardware="t4-small", hardware="t4-small")
            def upload_folder(self, **kwargs):
                raise AssertionError("Must not upload to paid hardware")
        api = API()
        with self.assertRaisesRegex(ValueError, "not free ZeroGPU"):
            builder.deploy(api, "SlopeLens", self.output, 1)
        self.assertEqual(api.requested, "zero-a10g")

    def test_running_verified_without_wait(self):
        class API:
            def whoami(self): return {"name": "owner"}
            def create_repo(self, *args, **kwargs): pass
            def space_info(self, repo): return SimpleNamespace(card_data={"sdk": "gradio"})
            def get_space_runtime(self, repo):
                return SimpleNamespace(requested_hardware="zero-a10g", hardware="zero-a10g", stage="RUNNING")
            def upload_folder(self, **kwargs): return SimpleNamespace(commit_url="https://example.test/commit")
        builder.deploy(API(), "SlopeLens", self.output, 1)


if __name__ == "__main__":
    unittest.main()
