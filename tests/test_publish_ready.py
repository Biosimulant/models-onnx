from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run_script(name: str) -> None:
    subprocess.run([PY, str(ROOT / "scripts" / name)], cwd=ROOT, check=True)


def test_manifests_are_publish_ready() -> None:
    run_script("validate_manifests.py")


def test_entrypoints_instantiate() -> None:
    run_script("check_entrypoints.py")


def test_publish_ready_static_audit() -> None:
    run_script("audit_publish_ready.py")


def test_onnx_runtime_and_visual_smoke() -> None:
    run_script("validate_onnx_runtime.py")
