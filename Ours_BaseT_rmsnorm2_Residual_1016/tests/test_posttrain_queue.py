"""CPU-only contract checks for queued sampling; no torch or model loads."""

import copy
import fcntl
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SOURCE = Path(__file__).resolve().parents[1] / "scripts/queue_posttrain_heun50.py"
SPEC = importlib.util.spec_from_file_location("posttrain_queue_under_test", SOURCE)
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


class QueueContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "training"
        self.run.mkdir()
        self.checkpoint = self.run / "checkpoint-80.pth"
        self.checkpoint.write_bytes(b"fake checkpoint; never deserialize")
        self.lock = self.run / ".training.lock"
        self.lock.touch()
        self.output = self.root / "sampling"
        self.output.mkdir()
        self.plan = {
            "training_dir": str(self.run),
            "checkpoint": str(self.checkpoint),
            "source_dir": str(self.root / "pinned source"),
            "config": str(self.root / "copied config.yaml"),
            "epoch": 80,
            "checkpoint_stat": {
                "path": str(self.checkpoint), "epoch": 80,
                "size_bytes": self.checkpoint.stat().st_size,
                "mtime_ns": self.checkpoint.stat().st_mtime_ns,
            },
            "model_settings": {
                "dino_time_blend_reverse": True,
                "dino_train_final_norm": True,
            },
        }
        self.job = {"scale": 3.9, "interval": [0.05, 1.0], "output_dir": str(self.output)}

    def finish_training(self):
        (self.run / "exit_code.txt").write_text("0\n")
        (self.run / "finished_at.txt").write_text("2026-10-20T12:00:00+08:00\n")

    def write_results(self):
        payload = b"fake completed npz; only file identity is under test"
        (self.output / "samples.npz").write_bytes(payload)
        self.manifest = {
            "metadata": {
                "epoch": 80, "ema": "ema1", "ema_decay": 0.9999,
                "sampler": "heun", "steps": 50, "cfg": 3.9,
                "cfg_interval": [0.05, 1.0], "seed": 0,
                "gen_bsz": 256, "world_size": 8,
                "checkpoint": copy.deepcopy(self.plan["checkpoint_stat"]),
                "config": copy.deepcopy(self.plan["model_settings"]),
            },
            "shape": [50000, 256, 256, 3], "dtype": "uint8",
            "npz_path": str(self.output / "samples.npz"), "size_bytes": len(payload),
        }
        self.metrics = {
            "num_images": 50000, "source": str(self.output / "samples.npz"),
            "metrics": {"fid": 5.99, "inception_score": 288.76, "fdr6": 20.27},
        }
        self.flush_results()
        (self.output / "exit_code.txt").write_text("0\n")

    def flush_results(self):
        (self.output / "samples.manifest.json").write_text(json.dumps(self.manifest))
        (self.output / "offline_metrics.json").write_text(json.dumps(self.metrics))

    def test_checkpoint_before_training_completion_is_not_ready(self):
        self.assertFalse(queue.training_ready(self.plan)[0])
        (self.run / "exit_code.txt").write_text("0\n")
        self.assertFalse(queue.training_ready(self.plan)[0])

    def test_nonzero_training_exit_is_fatal(self):
        self.finish_training()
        for code in ("1\n", "137\n", "-15\n"):
            with self.subTest(code=code):
                (self.run / "exit_code.txt").write_text(code)
                with self.assertRaisesRegex(RuntimeError, "did not finish successfully"):
                    queue.training_ready(self.plan)

    def test_success_markers_wait_for_training_lock_then_become_ready(self):
        self.finish_training()
        with self.lock.open("a") as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            ready, reason = queue.training_ready(self.plan)
            self.assertFalse(ready)
            self.assertIn("lock", reason)
        self.assertTrue(queue.training_ready(self.plan)[0])
        # A successful probe must itself release the lock.
        with self.lock.open("a") as probe:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_final_checkpoint_must_exist_and_be_nonempty(self):
        self.finish_training()
        self.checkpoint.unlink()
        with self.assertRaisesRegex(RuntimeError, "expected checkpoint is missing"):
            queue.training_ready(self.plan)
        self.checkpoint.touch()
        with self.assertRaisesRegex(RuntimeError, "expected checkpoint is missing"):
            queue.training_ready(self.plan)

    def test_missing_training_lock_is_not_accepted_as_completed(self):
        self.finish_training()
        self.lock.unlink()
        with self.assertRaisesRegex(RuntimeError, "lock file is missing"):
            queue.training_ready(self.plan)

    def test_command_explicitly_overrides_flow_defaults(self):
        for scale in (3.7, 3.9, 4.1, 4.3, 4.5):
            for interval_min in (0.0, 0.05, 0.1):
                with self.subTest(scale=scale, interval_min=interval_min):
                    job = dict(self.job, scale=scale, interval=[interval_min, 1.0])
                    cmd = queue.command(self.plan, job)
                    self.assertEqual(cmd[:5], ["bash", str(Path(self.plan["source_dir"]) / "scripts/sample_portable.sh"),
                                              self.plan["config"], self.plan["checkpoint"], job["output_dir"]])
                    options = dict(arg[2:].split("=", 1) for arg in cmd[5:])
                    self.assertEqual(options, {
                        "sampler": "heun", "steps": "50", "cfg": str(scale),
                        "interval-min": str(interval_min), "interval-max": "1.0",
                        "num-images": "50000", "gen-bsz": "256",
                    })
                    self.assertEqual(queue.command(self.plan, job, preflight=True), cmd + ["--preflight"])

    def test_complete_matching_run_is_reused_without_launch(self):
        self.write_results()
        self.assertEqual(queue.completed(self.plan, self.job), self.metrics["metrics"])
        with mock.patch.object(queue.subprocess, "run", side_effect=AssertionError("must not launch")):
            self.assertEqual(queue.run_job(self.plan, self.job), self.metrics["metrics"])

    def test_incomplete_or_failed_sampling_is_not_reported_complete(self):
        self.write_results()
        (self.output / "exit_code.txt").unlink()
        self.assertIsNone(queue.completed(self.plan, self.job))
        (self.output / "exit_code.txt").write_text("1\n")
        self.assertIsNone(queue.completed(self.plan, self.job))

    def test_reuse_rejects_mismatched_sampler_cfg_interval_and_model(self):
        self.write_results()
        original = copy.deepcopy(self.manifest)
        changes = {
            "cfg": 4.1, "cfg_interval": [0.1, 1.0], "sampler": "flow_dpm_solver",
            "steps": 100, "ema": "ema2", "epoch": 160, "world_size": 4, "gen_bsz": 128,
        }
        for field, value in changes.items():
            with self.subTest(field=field):
                self.manifest = copy.deepcopy(original)
                self.manifest["metadata"][field] = value
                self.flush_results()
                with self.assertRaisesRegex(RuntimeError, "manifest setting mismatch"):
                    queue.completed(self.plan, self.job)
        self.manifest = copy.deepcopy(original)
        self.manifest["metadata"]["config"]["dino_train_final_norm"] = False
        self.flush_results()
        with self.assertRaisesRegex(RuntimeError, "model setting mismatch: dino_train_final_norm"):
            queue.completed(self.plan, self.job)

    def test_reuse_rejects_checkpoint_identity_changes(self):
        self.write_results()
        original = copy.deepcopy(self.manifest)
        for field, value in (("path", "/wrong/checkpoint-80.pth"), ("epoch", 160),
                             ("size_bytes", 999), ("mtime_ns", 0)):
            with self.subTest(field=field):
                self.manifest = copy.deepcopy(original)
                self.manifest["metadata"]["checkpoint"][field] = value
                self.flush_results()
                with self.assertRaisesRegex(RuntimeError, "checkpoint identity mismatch"):
                    queue.completed(self.plan, self.job)

    def test_old_baseline_can_omit_false_final_norm_flag_but_trainable_run_cannot(self):
        self.write_results()
        del self.manifest["metadata"]["config"]["dino_train_final_norm"]
        self.flush_results()
        with self.assertRaisesRegex(RuntimeError, "dino_train_final_norm"):
            queue.sample_metadata(self.plan, self.job)
        self.plan["model_settings"]["dino_train_final_norm"] = False
        queue.sample_metadata(self.plan, self.job)

    def test_reuse_requires_expected_image_shape_and_export_size(self):
        self.write_results()
        original = copy.deepcopy(self.manifest)
        changes = {
            "shape": [1000, 256, 256, 3], "dtype": "float32",
            "npz_path": str(self.root / "wrong.npz"), "size_bytes": 1,
        }
        for field, value in changes.items():
            with self.subTest(field=field):
                self.manifest = copy.deepcopy(original)
                self.manifest[field] = value
                self.flush_results()
                with self.assertRaises(RuntimeError):
                    queue.completed(self.plan, self.job)

    def test_reuse_requires_metrics_for_exact_sample_archive(self):
        self.write_results()
        original = copy.deepcopy(self.metrics)
        for field, value in (("num_images", 1000), ("source", str(self.root / "other.npz"))):
            with self.subTest(field=field):
                self.metrics = copy.deepcopy(original)
                self.metrics[field] = value
                self.flush_results()
                with self.assertRaisesRegex(RuntimeError, "metrics source mismatch"):
                    queue.completed(self.plan, self.job)

    def test_nonfinite_metrics_are_never_marked_complete(self):
        self.write_results()
        for metric in ("fid", "inception_score", "fdr6"):
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(metric=metric, value=value):
                    self.metrics["metrics"] = {"fid": 5.99, "inception_score": 288.76, "fdr6": 20.27}
                    self.metrics["metrics"][metric] = value
                    self.flush_results()
                    with self.assertRaisesRegex(RuntimeError, "Nonfinite offline metrics"):
                        queue.completed(self.plan, self.job)


if __name__ == "__main__":
    unittest.main()
