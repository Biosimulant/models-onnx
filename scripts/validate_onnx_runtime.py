from __future__ import annotations

import importlib
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
LABS = ROOT / "labs"
BSIM_SRC = Path("/Volumes/dem-ssd/imp/projects/Nitoons/Biosimulant/bsim-active/biosim/src")


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def split_entrypoint(entrypoint: str) -> tuple[str, str]:
    return tuple(entrypoint.split(":", 1)) if ":" in entrypoint else tuple(entrypoint.rsplit(".", 1))  # type: ignore[return-value]


def clear_src() -> None:
    for key in [name for name in sys.modules if name == "src" or name.startswith("src.")]:
        sys.modules.pop(key, None)


@contextmanager
def import_root(path: Path):
    sys.path.insert(0, str(BSIM_SRC))
    sys.path.insert(0, str(path))
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)
        sys.path = [item for item in sys.path if item not in {str(path), str(BSIM_SRC)}]
        clear_src()


def finite_count(value: Any, limit: int = 100000) -> int:
    if hasattr(value, "value"):
        value = value.value
    if isinstance(value, dict) and set(value.keys()) == {"payload"}:
        value = value["payload"]
    arr = np.asarray(value)
    if arr.dtype == object:
        flat: list[float] = []

        def walk(item: Any) -> None:
            if len(flat) >= limit:
                return
            if hasattr(item, "tolist"):
                walk(item.tolist())
            elif isinstance(item, dict):
                for child in item.values():
                    walk(child)
            elif isinstance(item, (list, tuple)):
                for child in item:
                    walk(child)
            elif isinstance(item, (int, float)) and math.isfinite(float(item)):
                flat.append(float(item))

        walk(value)
        return len(flat)
    return int(np.isfinite(arr).sum())


def sample_signal(name: str, spec, source: str):
    from biosim.signals import ArraySignal, RecordSignal, ScalarSignal

    if spec.signal_type == "array":
        dtype = np.int64 if str(spec.dtype).startswith("int") else np.float32
        shape = tuple(int(dim) for dim in spec.shape)
        value = np.ones(shape, dtype=dtype)
        if np.issubdtype(value.dtype, np.integer) and "mask" in name:
            value = np.ones(shape, dtype=dtype)
        elif np.issubdtype(value.dtype, np.integer):
            # ESM exports emit NaNs for padding token id 1 when every position is
            # masked as active. Use a simple non-padding token id for synthetic
            # runtime smoke; this is still not scientific sequence evidence.
            value = np.full(shape, 5, dtype=dtype)
        return ArraySignal(source=source, name=name, value=value, emitted_at=0.0, spec=spec)
    if spec.signal_type == "record":
        return RecordSignal(source=source, name=name, value={key: 1 for key in (spec.schema or {"payload": "json"})}, emitted_at=0.0, spec=spec)
    return ScalarSignal(source=source, name=name, value=1.0, emitted_at=0.0, spec=spec)


def visual_data_ok(lab_name: str, visual: dict[str, Any]) -> None:
    render = visual.get("render")
    data = visual.get("data")
    if not isinstance(data, dict):
        raise AssertionError(f"{lab_name}: {render} visual has no data")
    if render == "bar":
        if not data.get("items"):
            raise AssertionError(f"{lab_name}: empty bar")
        if "categories" in data or "values" in data:
            raise AssertionError(f"{lab_name}: old bar schema")
    elif render == "timeseries":
        if not data.get("series") or any(not item.get("points") for item in data["series"]):
            raise AssertionError(f"{lab_name}: empty timeseries")
    elif render == "scatter":
        if not data.get("points"):
            raise AssertionError(f"{lab_name}: empty scatter")
    elif render == "heatmap":
        values = data.get("values")
        if not values or not isinstance(values, list) or any(not isinstance(row, list) or not row for row in values):
            raise AssertionError(f"{lab_name}: empty heatmap")
    elif render == "table":
        rows = data.get("rows")
        if not rows:
            raise AssertionError(f"{lab_name}: empty table")
        prompts = {row[0] for row in rows if isinstance(row, list) and row}
        if "Scientific question" in prompts and "Observed answer" not in prompts:
            raise AssertionError(f"{lab_name}: Q/A table lacks Observed answer")
    else:
        raise AssertionError(f"{lab_name}: unsupported render {render!r}")


def run_lab(lab_dir: Path) -> None:
    lab = load_yaml(lab_dir / "lab.yaml")
    aliases = {item["alias"]: item["path"] for item in lab["models"]}
    core_alias = next(alias for alias, path in aliases.items() if path == "models/core")
    core_manifest = load_yaml(lab_dir / "models/core/model.yaml")
    visual_manifest = load_yaml(lab_dir / "models/visualisation/model.yaml")

    with import_root(lab_dir / "models/core"):
        module_name, attr = split_entrypoint(core_manifest["biosim"]["entrypoint"])
        core_cls = getattr(importlib.import_module(module_name), attr)
        core = core_cls(**core_manifest["biosim"].get("init_kwargs", {}))
        inputs = {name: sample_signal(name, spec, core_alias) for name, spec in core.inputs().items()}
        core.set_inputs(inputs)
        core.advance_window(0.0, 0.01)
        outputs = core.get_outputs()
        expected = {
            item.get("name") if isinstance(item, dict) else str(item)
            for item in core_manifest["io"]["outputs"]
        }
        if not expected <= set(outputs):
            raise AssertionError(f"{lab_dir.name}: missing outputs {expected - set(outputs)}")
        for name, signal in outputs.items():
            if finite_count(signal) == 0:
                raise AssertionError(f"{lab_dir.name}: output {name} has no finite numeric data")

    with import_root(lab_dir / "models/visualisation"):
        module_name, attr = split_entrypoint(visual_manifest["biosim"]["entrypoint"])
        visual_cls = getattr(importlib.import_module(module_name), attr)
        visual = visual_cls(**visual_manifest["biosim"].get("init_kwargs", {}))
        visual.setup({})
        visual.set_inputs(outputs)
        visual.advance_window(0.0, 0.01)
        visuals = visual.visualize()
        if not visuals:
            raise AssertionError(f"{lab_dir.name}: visualisation emitted no visuals")
        for item in visuals:
            visual_data_ok(lab_dir.name, item)


def main() -> None:
    import onnxruntime  # noqa: F401

    failures: list[str] = []
    for lab_dir in sorted(p for p in LABS.iterdir() if p.is_dir()):
        try:
            run_lab(lab_dir)
            print(f"ok {lab_dir.name}")
        except Exception as exc:
            failures.append(f"{lab_dir.name}: {exc}")
            print(f"FAIL {lab_dir.name}: {exc}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Validated ONNX runtime and visual smoke for {len(list(LABS.iterdir()))} lab(s).")


if __name__ == "__main__":
    main()
