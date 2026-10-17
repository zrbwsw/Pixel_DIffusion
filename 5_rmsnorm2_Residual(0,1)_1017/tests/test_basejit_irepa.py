"""Focused contract tests; the CUDA integration probe is launched explicitly."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
VERIFY_SCRIPT = ROOT / "scripts" / "verify_baseline_parity.py"
spec = importlib.util.spec_from_file_location("verify_baseline_parity", VERIFY_SCRIPT)
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


def test_comparison_rejects_rng_and_gradient_changes():
    """Equal zero outputs alone cannot satisfy this verifier."""
    reference = {
        "rng": {"after_forward": {"cuda": {"sha256": "original"}}},
        "states": {"nonzero_probe/gradients": {"qkv.weight": {"sha256": "original"}}},
    }
    candidate = {
        "rng": {"after_forward": {"cuda": {"sha256": "changed"}}},
        "states": {"nonzero_probe/gradients": {"qkv.weight": {"sha256": "changed"}}},
    }
    paths = {entry["path"] for entry in verify.compare_reports(reference, candidate)}
    assert "rng/after_forward/cuda/sha256" in paths
    assert "states/nonzero_probe/gradients/qkv.weight/sha256" in paths


def test_native_baseline_cpu_in_separate_processes(tmp_path):
    """Independent imports, factory/config, two optimizer steps, EMA and CFG."""
    reference = ROOT.parent / "iREPA"
    if not (reference / "jit" / "denoiser.py").is_file():
        import pytest
        pytest.skip("Sibling original iREPA checkout is needed only for parity verification")
    subprocess.run([
        sys.executable, str(VERIFY_SCRIPT), "--mode", "cpu", "--output", str(tmp_path),
    ], check=True, timeout=240)
    report = json.loads((tmp_path / "comparison.json").read_text())
    assert report["passed"]
    assert not report["differences"]
    assert all(item["exact"] for item in report["tensor_comparison"].values())
    candidate = json.loads((tmp_path / "candidate.json").read_text())
    assert candidate["structure"]["hooks_before_probe"] == 0
    assert not candidate["structure"]["generation_modules_imported"]
    assert candidate["structure"]["class"] == "Denoiser"
    assert "nonzero_probe/gradients" in candidate["states"]
    assert candidate["cfg_forward_batch_sizes"] == [1, 1, 2, 2, 1, 2, 1, 2]
