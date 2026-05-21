# SPDX-FileCopyrightText: 2025-present Demi <bjaiye1@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Imported MONAI ONNX wrapper for Swin UNETR BTCV Segmentation."""

from __future__ import annotations

import importlib

import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from biosim import BioModule
from biosim.signals import (AcceptedSignalProfile, ArraySignal, BioSignal, EventSignal, RecordSignal, ScalarSignal, SignalSpec)

MAX_SERIALIZED_TENSOR_VALUES = 4096
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


def _flatten(value: Any) -> List[float]:
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
        return _flatten(value.tolist())
    if isinstance(value, list):
        out: List[float] = []
        for item in value:
            out.extend(_flatten(item))
        return out
    if isinstance(value, tuple):
        return _flatten(list(value))
    if isinstance(value, (int, float)):
        return [float(value)]
    return []


def _reshape(flat: Sequence[float], shape: Sequence[int]) -> Any:
    if len(shape) == 1:
        return list(flat[: shape[0]])
    stride = 1
    for dim in shape[1:]:
        stride *= dim
    return [_reshape(flat[idx * stride : (idx + 1) * stride], shape[1:]) for idx in range(shape[0])]


def _schema_type(value):
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    return "json"


def _signal_value(signal):
    value = signal.value
    if isinstance(value, dict) and set(value.keys()) == {"payload"}:
        return value["payload"]
    return value


def _generic_input_spec(description=None):
    return SignalSpec.record(
        schema={"payload": "json"},
        accepted_profiles=(
            AcceptedSignalProfile(signal_type="record", schema={"payload": "json"}),
            AcceptedSignalProfile(signal_type="scalar"),
        ),
        description=description,
    )


def _make_signal(*, source, name, value, emitted_at, spec=None):
    value = _compact_tensor_evidence(value)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if spec is None:
        if isinstance(value, dict):
            spec = SignalSpec.record(schema={str(key): _schema_type(item) for key, item in value.items()})
        elif isinstance(value, (list, tuple)):
            spec = SignalSpec.record(schema={"payload": "json"})
        else:
            spec = SignalSpec.scalar(dtype=_schema_type(value))

    if spec.signal_type == "scalar":
        return ScalarSignal(source=source, name=name, value=value, emitted_at=emitted_at, spec=spec)
    if spec.signal_type == "array":
        return ArraySignal(source=source, name=name, value=value, emitted_at=emitted_at, spec=spec)
    if spec.signal_type == "event":
        event_value = value
        if spec.schema is not None and not (isinstance(value, dict) and set(value.keys()) == set(spec.schema.keys())):
            event_value = {"payload": value}
        return EventSignal(source=source, name=name, value=event_value, emitted_at=emitted_at, spec=spec)

    record_value = value
    if not isinstance(value, dict) or set(value.keys()) != set((spec.schema or {}).keys()):
        record_value = {"payload": value}
    return RecordSignal(source=source, name=name, value=record_value, emitted_at=emitted_at, spec=spec)

class MonaiSwinUnetrBtcvSegmentation(BioModule):
    """Run the imported Swin UNETR segmentation ONNX graph as a BioModule."""

    def __init__(
        self,
        model_path: str = "artifacts/model.onnx",
        input_port: str = "volume_tensor",
        output_port: str = "class_probabilities",
        summary_port: str = "segmentation_summary",
        model_input_name: Optional[str] = None,
        model_output_name: Optional[str] = None,
        input_shape: Sequence[int] = (1, 1, 96, 96, 96),
        output_shape: Sequence[int] = (1, 14, 96, 96, 96),
        base_dir: Optional[str] = None,
        integration_step: float = 0.01,
        session_factory: Optional[Callable[[str], Any]] = None,
        providers: Optional[Sequence[str]] = None,
    ) -> None:
        self.integration_step = float(integration_step)
        self.model_path = model_path
        self.input_port = input_port
        self.output_port = output_port
        self.summary_port = summary_port
        self.model_input_name = model_input_name
        self.model_output_name = model_output_name
        self.input_shape = tuple(int(x) for x in input_shape)
        self.output_shape = tuple(int(x) for x in output_shape)
        self.base_dir = Path(base_dir).resolve() if base_dir else Path(__file__).resolve().parents[1]
        self._session_factory = session_factory
        self.providers = list(providers or ["CPUExecutionProvider"])
        self._session: Any = None
        self._latest_input = _reshape([0.0] * self._element_count(self.input_shape), self.input_shape)
        self._outputs: Dict[str, BioSignal] = {}

    def _element_count(self, shape: Sequence[int]) -> int:
        total = 1
        for dim in shape:
            total *= dim
        return total

    def _resolved_model_path(self) -> str:
        path = Path(self.model_path)
        if path.is_absolute():
            return str(path)
        return str((self.base_dir / path).resolve())

    def _default_session_factory(self) -> Callable[[str], Any]:
        ort = importlib.import_module("onnxruntime")
        return lambda model_path: ort.InferenceSession(model_path, providers=self.providers)

    def _ensure_session(self) -> Any:
        if self._session is None:
            factory = self._session_factory or self._default_session_factory()
            self._session = factory(self._resolved_model_path())
            inputs = getattr(self._session, "get_inputs", lambda: [])()
            outputs = getattr(self._session, "get_outputs", lambda: [])()
            if inputs and self.model_input_name is None:
                self.model_input_name = str(inputs[0].name)
            if outputs and self.model_output_name is None:
                self.model_output_name = str(outputs[0].name)
        return self._session

    def inputs(self) -> dict[str, SignalSpec]:
        return {
            self.input_port: SignalSpec.array(
                dtype="float32",
                shape=self.input_shape,
                description="Input tensor matching the source ONNX graph input shape.",
            ),
        }

    def outputs(self) -> dict[str, SignalSpec]:
        return {
            self.output_port: SignalSpec.record(
                schema={"payload": "json"},
                description="Compact tensor evidence derived from the source ONNX graph output.",
            ),
            self.summary_port: SignalSpec.record(
                schema={"mean_probability": "float", "voxel_count": "int"},
                description="Summary statistics derived from the prediction tensor.",
            ),
        }

    def set_inputs(self, signals: Dict[str, BioSignal]) -> None:
        signal = signals.get(self.input_port)
        if signal is None:
            return
        flat = _flatten(_signal_value(signal))
        needed = self._element_count(self.input_shape)
        flat = flat[:needed] + [0.0] * max(0, needed - len(flat))
        self._latest_input = _reshape(flat, self.input_shape)

    def advance_window(self, start: float, end: float) -> None:
        t = float(end)
        session = self._ensure_session()
        output_name = self.model_output_name or self.output_port
        input_name = self.model_input_name or self.input_port
        result = session.run([output_name], {input_name: np.asarray(self._latest_input, dtype=np.float32)})
        prediction = result[0] if result else _reshape([0.0] * self._element_count(self.output_shape), self.output_shape)
        flat = _flatten(prediction)
        source = getattr(self, "_world_name", self.__class__.__name__)
        self._outputs = {
            self.output_port: _make_signal(source=source, name=self.output_port, value=prediction, emitted_at=t, spec=self.outputs().get(self.output_port) if 'self' in locals() else None),
            self.summary_port: _make_signal(source=source, name=self.summary_port, value={"mean_probability": (sum(flat) / len(flat)) if flat else 0.0, "voxel_count": len(flat)}, emitted_at=t, spec=self.outputs().get(self.summary_port) if 'self' in locals() else None),
        }

    def get_outputs(self) -> Dict[str, BioSignal]:
        return dict(self._outputs)
