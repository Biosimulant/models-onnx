# SPDX-FileCopyrightText: 2025-present Demi <bjaiye1@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Improve ONNX lab visual evidence without changing source ONNX artifacts."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LABS = ROOT / "labs"


CORE_HELPERS = '''MAX_SERIALIZED_TENSOR_VALUES = 4096
TENSOR_EVIDENCE_SAMPLE_VALUES = 32
TENSOR_PREVIEW_SIZE = 32
TENSOR_PREVIEW_CHANNELS = 4


def _downsample_1d(values: Any, size: int = TENSOR_PREVIEW_SIZE) -> list[float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return []
    if arr.size <= size:
        return [float(x) for x in arr.tolist()]
    idx = np.linspace(0, arr.size - 1, num=size).astype(int)
    return [float(arr[i]) for i in idx]


def _downsample_2d(values: Any, size: int = TENSOR_PREVIEW_SIZE) -> list[list[float]]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return [[float(arr)]]
    if arr.ndim == 1:
        return [_downsample_1d(arr, size)]
    arr = np.squeeze(arr)
    while arr.ndim > 2:
        arr = arr[arr.shape[0] // 2]
    if arr.ndim == 1:
        return [_downsample_1d(arr, size)]
    rows = np.linspace(0, arr.shape[0] - 1, num=min(size, arr.shape[0])).astype(int)
    cols = np.linspace(0, arr.shape[1] - 1, num=min(size, arr.shape[1])).astype(int)
    sampled = arr[np.ix_(rows, cols)]
    return [[float(x) for x in row] for row in sampled.tolist()]


def _channel_axis_preview(arr: Any) -> tuple[np.ndarray, bool]:
    data = np.asarray(arr)
    while data.ndim > 2 and data.shape[0] == 1:
        data = data[0]
    has_channels = data.ndim >= 3 and 1 < data.shape[0] <= 16
    return data, has_channels


def _preview_matrices(arr: Any) -> list[dict[str, Any]]:
    data = np.asarray(arr)
    if not np.issubdtype(data.dtype, np.number):
        return []
    data = np.nan_to_num(data.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    viewed, has_channels = _channel_axis_preview(data)
    previews: list[dict[str, Any]] = []
    if has_channels:
        for channel in range(min(TENSOR_PREVIEW_CHANNELS, viewed.shape[0])):
            channel_data = viewed[channel]
            previews.append({
                "label": f"channel_{channel}",
                "values": _downsample_2d(channel_data),
            })
        return previews
    previews.append({"label": "tensor_preview", "values": _downsample_2d(viewed)})
    return previews


def _channel_stats(arr: Any) -> list[dict[str, Any]]:
    data = np.asarray(arr)
    if not np.issubdtype(data.dtype, np.number):
        return []
    data = data.astype(float)
    viewed, has_channels = _channel_axis_preview(data)
    if not has_channels:
        return []
    rows: list[dict[str, Any]] = []
    for channel in range(min(TENSOR_PREVIEW_CHANNELS, viewed.shape[0])):
        values = viewed[channel].reshape(-1)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        rows.append({
            "label": f"channel_{channel}",
            "mean": float(np.mean(finite)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "positive_fraction": float(np.mean(finite > 0.0)),
            "finite_count": int(finite.size),
        })
    return rows


def _compact_tensor_evidence(value: Any) -> Any:
    if not hasattr(value, "shape") and not hasattr(value, "__array__"):
        return value
    arr = np.asarray(value)
    if arr.dtype == object or arr.ndim == 0:
        return value
    if not np.issubdtype(arr.dtype, np.number):
        return {
            "payload_kind": "compact_tensor_evidence",
            "shape": [int(dim) for dim in arr.shape],
            "dtype": str(arr.dtype),
            "element_count": int(arr.size),
            "sample": [],
            "previews": [],
            "channel_stats": [],
        }
    finite = arr[np.isfinite(arr)]
    if arr.size <= MAX_SERIALIZED_TENSOR_VALUES:
        return value
    sample = finite[:TENSOR_EVIDENCE_SAMPLE_VALUES].astype(float).tolist()
    evidence = {
        "payload_kind": "compact_tensor_evidence",
        "shape": [int(dim) for dim in arr.shape],
        "dtype": str(arr.dtype),
        "element_count": int(arr.size),
        "finite_count": int(finite.size),
        "sample": sample,
        "previews": _preview_matrices(arr),
        "channel_stats": _channel_stats(arr),
    }
    if finite.size:
        evidence.update({
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "mean": float(np.mean(finite)),
        })
    return evidence
'''


VISUAL_SOURCE = '''# SPDX-FileCopyrightText: 2025-present Demi <bjaiye1@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Publication-facing visualisation module for source-derived ONNX labs."""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

from biosim import BioModule
from biosim.signals import BioSignal, SignalSpec


def _unwrap(value: Any) -> Any:
    if hasattr(value, "value"):
        value = value.value
    if isinstance(value, Mapping) and set(value.keys()) == {"payload"}:
        return value["payload"]
    return value


def _flatten_numbers(value: Any, limit: int = 20000) -> list[float]:
    value = _unwrap(value)
    out: list[float] = []

    def walk(item: Any) -> None:
        if len(out) >= limit:
            return
        if hasattr(item, "tolist"):
            walk(item.tolist())
        elif isinstance(item, Mapping):
            if item.get("payload_kind") == "compact_tensor_evidence":
                for key in ("sample",):
                    walk(item.get(key))
                return
            for child in item.values():
                walk(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                walk(child)
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            number = float(item)
            if math.isfinite(number):
                out.append(number)

    walk(value)
    return out


def _shape(value: Any) -> list[int]:
    value = _unwrap(value)
    if isinstance(value, Mapping) and isinstance(value.get("shape"), list):
        return [int(dim) for dim in value["shape"]]
    if hasattr(value, "shape"):
        return [int(dim) for dim in value.shape]
    shape: list[int] = []
    while isinstance(value, list):
        shape.append(len(value))
        value = value[0] if value else None
    return shape


def _is_compact_evidence(value: Any) -> bool:
    value = _unwrap(value)
    return isinstance(value, Mapping) and value.get("payload_kind") == "compact_tensor_evidence"


def _evidence_summary(value: Mapping[str, Any]) -> str:
    shape = "x".join(str(dim) for dim in value.get("shape", [])) or "-"
    pieces = [
        f"shape={shape}",
        f"dtype={value.get('dtype', 'unknown')}",
        f"n={value.get('element_count', '-')}",
    ]
    for key in ("min", "max", "mean"):
        item = value.get(key)
        if isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item)):
            pieces.append(f"{key}={float(item):.4g}")
    return ", ".join(pieces)


def _summary_text(value: Any) -> str:
    value = _unwrap(value)
    if isinstance(value, Mapping) and value.get("payload_kind") == "compact_tensor_evidence":
        return _evidence_summary(value)
    if not isinstance(value, Mapping):
        return "Runtime output was emitted as a tensor payload."
    parts = []
    for key, item in value.items():
        if isinstance(item, float):
            parts.append(f"{key}={item:.4g}")
        else:
            parts.append(f"{key}={item}")
    return ", ".join(parts) if parts else "Runtime summary was emitted."


def _matrix_from_numbers(numbers: list[float], width: int = 16) -> list[list[float]]:
    if not numbers:
        return []
    rows: list[list[float]] = []
    for start in range(0, min(len(numbers), width * width), width):
        rows.append(numbers[start : start + width])
    return rows


class OnnxPublicationVisualisation(BioModule):
    def __init__(
        self,
        *,
        lab_title: str,
        source_type: str,
        task: str,
        question: str,
        answer_style: str,
        caveat: str,
        dominant_module: str,
        public_input_note: str,
        input_contracts: list[dict[str, Any]],
        output_contracts: list[dict[str, Any]],
        visual_inputs: dict[str, dict[str, str]],
    ) -> None:
        self.lab_title = lab_title
        self.source_type = source_type
        self.task = task
        self.question = question
        self.answer_style = answer_style
        self.caveat = caveat
        self.dominant_module = dominant_module
        self.public_input_note = public_input_note
        self.input_contracts = list(input_contracts)
        self.output_contracts = list(output_contracts)
        self.visual_inputs = dict(visual_inputs)
        self._inputs: dict[str, BioSignal] = {}

    def inputs(self) -> dict[str, SignalSpec]:
        return {
            name: SignalSpec.record(schema=dict(schema), description=f"Visualisation input for `{name}` from the ONNX core wrapper.")
            for name, schema in self.visual_inputs.items()
        }

    def outputs(self) -> dict[str, SignalSpec]:
        return {}

    def setup(self, config: Optional[dict[str, Any]] = None) -> None:
        self._inputs = {}

    def set_inputs(self, inputs: dict[str, BioSignal]) -> None:
        self._inputs = dict(inputs or {})

    def advance_window(self, start: float, end: float) -> None:
        return None

    def get_outputs(self) -> dict[str, BioSignal]:
        return {}

    def _schema_rows(self) -> list[list[str]]:
        rows: list[list[str]] = []
        for item in self.input_contracts:
            rows.append(["input", str(item.get("name")), str(item.get("dtype")), str(item.get("shape"))])
        for item in self.output_contracts:
            rows.append(["output", str(item.get("name")), str(item.get("dtype")), str(item.get("shape"))])
        return rows or [["contract", "unavailable", "unknown", "unknown"]]

    def _summary_rows(self) -> list[list[str]]:
        rows: list[list[str]] = []
        for port, signal in sorted(self._inputs.items()):
            value = _unwrap(signal)
            shape = _shape(value)
            if _is_compact_evidence(value):
                rows.append([port, "tensor evidence", "x".join(str(dim) for dim in shape), _summary_text(value)])
            elif isinstance(value, Mapping) and set(value.keys()) != {"payload"}:
                rows.append([port, "record", "-", _summary_text(value)])
            else:
                numbers = _flatten_numbers(value)
                if numbers:
                    rows.append([
                        port,
                        "tensor",
                        "x".join(str(dim) for dim in shape) if shape else "scalar/list",
                        f"n={len(numbers)}, min={min(numbers):.4g}, max={max(numbers):.4g}, mean={sum(numbers) / len(numbers):.4g}",
                    ])
        return rows or [["runtime", "none", "-", "No runtime signal has been received yet."]]

    def _heatmap_visuals(self) -> list[dict[str, Any]]:
        visuals: list[dict[str, Any]] = []
        for port, signal in sorted(self._inputs.items()):
            value = _unwrap(signal)
            if isinstance(value, Mapping) and value.get("payload_kind") == "compact_tensor_evidence":
                for preview in list(value.get("previews") or [])[:2]:
                    values = preview.get("values") if isinstance(preview, Mapping) else None
                    if isinstance(values, list) and values:
                        visuals.append({
                            "render": "heatmap",
                            "description": "Downsampled preview from the source ONNX output tensor.",
                            "data": {
                                "title": f"{port} {preview.get('label', 'tensor preview')}",
                                "values": values,
                            },
                        })
                continue
            numbers = _flatten_numbers(value, limit=256)
            if len(numbers) >= 4:
                visuals.append({
                    "render": "heatmap",
                    "description": "Compact value heatmap from the emitted ONNX tensor.",
                    "data": {
                        "title": f"{port} value heatmap",
                        "values": _matrix_from_numbers(numbers),
                    },
                })
        return visuals[:3]

    def _bar_items(self) -> list[dict[str, float | str]]:
        items: list[dict[str, float | str]] = []
        for port, signal in sorted(self._inputs.items()):
            value = _unwrap(signal)
            if isinstance(value, Mapping) and value.get("payload_kind") == "compact_tensor_evidence":
                for row in list(value.get("channel_stats") or [])[:8]:
                    if not isinstance(row, Mapping):
                        continue
                    mean = row.get("mean")
                    if isinstance(mean, (int, float)) and not isinstance(mean, bool) and math.isfinite(float(mean)):
                        items.append({"label": f"{port}.{row.get('label', 'channel')} mean", "value": float(mean)})
                if items:
                    continue
                for key in ("mean", "max", "min"):
                    item = value.get(key)
                    if isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item)):
                        items.append({"label": f"{port} {key}", "value": float(item)})
                continue
            if isinstance(value, Mapping) and set(value.keys()) != {"payload"}:
                for key, item in value.items():
                    if isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item)):
                        items.append({"label": f"{port}.{key}", "value": float(item)})
            else:
                numbers = _flatten_numbers(value, limit=32)
                if 0 < len(numbers) <= 32:
                    for index, item in enumerate(numbers[:16]):
                        items.append({"label": f"{port}[{index}]", "value": float(item)})
                elif numbers:
                    items.append({"label": f"{port} mean", "value": float(sum(numbers) / len(numbers))})
                    items.append({"label": f"{port} max", "value": float(max(numbers))})
            if len(items) >= 12:
                break
        return items[:12]

    def visualize(self) -> Optional[list[dict[str, Any]]]:
        summary_rows = self._summary_rows()
        answer = summary_rows[0][3] if summary_rows else self.answer_style
        visuals: list[dict[str, Any]] = [
            {
                "render": "table",
                "description": "Direct interpretation of this source-derived ONNX run.",
                "data": {
                    "title": f"{self.lab_title} - source-faithful ONNX run",
                    "columns": ["Prompt", "Answer"],
                    "rows": [
                        ["Scientific question", self.question],
                        ["Observed answer", answer],
                        ["Evidence", self.answer_style],
                        ["Dominant module", self.dominant_module],
                        ["Caveat", self.caveat],
                    ],
                },
            },
        ]
        visuals.extend(self._heatmap_visuals())
        bar_items = self._bar_items()
        if bar_items:
            visuals.append(
                {
                    "render": "bar",
                    "description": "Compact numeric summary of emitted ONNX outputs.",
                    "data": {
                        "title": "Runtime numeric evidence",
                        "items": bar_items,
                    },
                }
            )
        visuals.extend([
            {
                "render": "table",
                "description": "Verified ONNX input and output contract.",
                "data": {
                    "title": "ONNX graph contract",
                    "columns": ["Direction", "Graph name", "Dtype", "Shape"],
                    "rows": self._schema_rows(),
                },
            },
            {
                "render": "table",
                "description": "Compact runtime output summary from the core ONNX wrapper.",
                "data": {
                    "title": "Runtime tensor evidence",
                    "columns": ["Port", "Kind", "Shape", "Evidence"],
                    "rows": summary_rows,
                },
            },
            {
                "render": "table",
                "description": "Public input traceability.",
                "data": {
                    "title": "Input contract caveat",
                    "columns": ["Field", "Value"],
                    "rows": [
                        ["Source type", self.source_type],
                        ["Task", self.task],
                        ["Public input mapping", self.public_input_note],
                    ],
                },
            },
        ])
        return visuals
'''


def replace_core_helpers(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"MAX_SERIALIZED_TENSOR_VALUES = 4096\nTENSOR_EVIDENCE_SAMPLE_VALUES = 64\n\n\n"
        r"def _compact_tensor_evidence\(value: Any\) -> Any:\n.*?\n\n\n(?=def _(flatten|normalize_ids)\()",
        re.S,
    )
    new_text, count = pattern.subn(CORE_HELPERS + "\n\n", text)
    if count != 1:
        return False
    path.write_text(new_text, encoding="utf-8")
    return True


def update_cleanup_script() -> bool:
    path = ROOT / "scripts" / "cleanup_publish_ready.py"
    text = path.read_text(encoding="utf-8")
    new_text, count = re.subn(
        r"VISUAL_SOURCE = '''.*?'''\n\n\nTEST_SOURCE =",
        "VISUAL_SOURCE = '''" + VISUAL_SOURCE + "'''\n\n\nTEST_SOURCE =",
        text,
        count=1,
        flags=re.S,
    )
    if count != 1:
        return False
    path.write_text(new_text, encoding="utf-8")
    return True


def main() -> None:
    core_updated = 0
    for path in sorted(LABS.glob("*/models/core/src/imported_*_model.py")):
        if replace_core_helpers(path):
            core_updated += 1
        else:
            raise RuntimeError(f"Could not update core helper block in {path}")

    visual_updated = 0
    for path in sorted(LABS.glob("*/models/visualisation/src/onnx_visualisation.py")):
        path.write_text(VISUAL_SOURCE, encoding="utf-8")
        visual_updated += 1

    if not update_cleanup_script():
        raise RuntimeError("Could not update cleanup_publish_ready.py VISUAL_SOURCE")

    print(f"Updated {core_updated} core wrappers")
    print(f"Updated {visual_updated} visualisation wrappers")
    print("Updated cleanup_publish_ready.py visual template")


if __name__ == "__main__":
    main()
