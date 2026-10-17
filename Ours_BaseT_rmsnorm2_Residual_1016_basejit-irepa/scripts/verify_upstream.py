"""Verify the pinned numerical implementation without importing torch."""
import hashlib
import json
from pathlib import Path


def verify(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "upstream_manifest.json").read_text())
    mismatches = []
    for name, expected in manifest["sha256"].items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            mismatches.append(name)
    if mismatches:
        raise RuntimeError("Pinned upstream files changed: " + ", ".join(mismatches))
    return {"verified_files": len(manifest["sha256"]), "irepa_commit": manifest["irepa_commit"]}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
