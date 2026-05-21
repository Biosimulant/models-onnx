# SPDX-FileCopyrightText: 2025-present Demi <bjaiye1@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Imported transformer ONNX wrapper for biobert-base-cased-v1.1."""

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


def _normalize_ids(raw: Any, length: int) -> List[int]:
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if isinstance(raw, list):
        values = [int(x) for x in raw if isinstance(x, (int, float))]
    elif isinstance(raw, tuple):
        values = [int(x) for x in raw if isinstance(x, (int, float))]
    elif isinstance(raw, (int, float)):
        values = [int(raw)]
    else:
        values = []
    values = values[:length]
    while len(values) < length:
        values.append(0)
    return values


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

class HfDmisLabBiobertBaseCasedV11(BioModule):
    """Run the imported BioBERT ONNX export behind the BioModule interface."""

    def __init__(
        self,
        model_path: str = "artifacts/model.onnx",
        token_port: str = "token_ids",
        mask_port: str = "attention_mask",
        sequence_port: str = "sequence_embeddings",
        pooled_port: str = "pooled_embedding",
        token_input_name: Optional[str] = None,
        mask_input_name: Optional[str] = None,
        segment_input_name: Optional[str] = None,
        sequence_output_name: Optional[str] = None,
        pooled_output_name: Optional[str] = None,
        seq_length: int = 16,
        base_dir: Optional[str] = None,
        integration_step: float = 0.01,
        session_factory: Optional[Callable[[str], Any]] = None,
        providers: Optional[Sequence[str]] = None,
    ) -> None:
        self.integration_step = float(integration_step)
        self.model_path = model_path
        self.token_port = token_port
        self.mask_port = mask_port
        self.sequence_port = sequence_port
        self.pooled_port = pooled_port
        self.token_input_name = token_input_name
        self.mask_input_name = mask_input_name
        self.segment_input_name = segment_input_name
        self.sequence_output_name = sequence_output_name
        self.pooled_output_name = pooled_output_name
        self.seq_length = seq_length
        self.base_dir = Path(base_dir).resolve() if base_dir else Path(__file__).resolve().parents[1]
        self._session_factory = session_factory
        self.providers = list(providers or ["CPUExecutionProvider"])
        self._session: Any = None
        self._token_ids = [0] * seq_length
        self._attention_mask = [1] * seq_length
        self._outputs: Dict[str, BioSignal] = {}

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
            if inputs and self.token_input_name is None:
                self.token_input_name = str(inputs[0].name)
            if len(inputs) > 1 and self.mask_input_name is None:
                self.mask_input_name = str(inputs[1].name)
            if len(inputs) > 2 and self.segment_input_name is None:
                self.segment_input_name = str(inputs[2].name)
            if outputs and self.sequence_output_name is None:
                self.sequence_output_name = str(outputs[0].name)
            if len(outputs) > 1 and self.pooled_output_name is None:
                self.pooled_output_name = str(outputs[1].name)
        return self._session

    def inputs(self) -> dict[str, SignalSpec]:
        return {
            self.token_port: SignalSpec.array(
                dtype="int64",
                shape=(self.seq_length,),
                description="Token IDs for the source ONNX transformer input.",
            ),
            self.mask_port: SignalSpec.array(
                dtype="int64",
                shape=(self.seq_length,),
                description="Attention mask for the source ONNX transformer input.",
            ),
        }

    def outputs(self) -> dict[str, SignalSpec]:
        return {
            self.sequence_port: SignalSpec.record(
                schema={"payload": "json"},
                description="Token-level embedding tensor emitted by the ONNX graph.",
            ),
            self.pooled_port: SignalSpec.record(
                schema={"payload": "json"},
                description="Pooled embedding tensor emitted by the ONNX graph or derived from the first token output when the graph has no pooled output.",
            ),
        }

    def set_inputs(self, signals: Dict[str, BioSignal]) -> None:
        token_signal = signals.get(self.token_port)
        mask_signal = signals.get(self.mask_port)
        if token_signal is not None:
            self._token_ids = _normalize_ids(_signal_value(token_signal), self.seq_length)
        if mask_signal is not None:
            self._attention_mask = _normalize_ids(_signal_value(mask_signal), self.seq_length)

    def advance_window(self, start: float, end: float) -> None:
        t = float(end)
        session = self._ensure_session()
        sequence_name = self.sequence_output_name or self.sequence_port
        pooled_name = self.pooled_output_name or self.pooled_port
        token_name = self.token_input_name or self.token_port
        mask_name = self.mask_input_name or self.mask_port
        feed_dict = {
            token_name: np.asarray([self._token_ids], dtype=np.int64),
            mask_name: np.asarray([self._attention_mask], dtype=np.int64),
        }
        if self.segment_input_name is not None:
            feed_dict[self.segment_input_name] = np.asarray([[0] * self.seq_length], dtype=np.int64)
        requested_outputs = [sequence_name]
        if self.pooled_output_name is not None:
            requested_outputs.append(pooled_name)
        result = session.run(requested_outputs, feed_dict)
        sequence_embeddings = result[0] if result else [[[0.0] * 768 for _ in range(self.seq_length)]]
        if len(result) > 1:
            pooled_embedding = result[1]
        else:
            first_row = sequence_embeddings[0] if sequence_embeddings else []
            first_token = first_row[0] if first_row else []
            pooled_embedding = [first_token] if first_token else [[0.0] * 768]
        source = getattr(self, "_world_name", self.__class__.__name__)
        self._outputs = {
            self.sequence_port: _make_signal(source=source, name=self.sequence_port, value=sequence_embeddings, emitted_at=t, spec=self.outputs().get(self.sequence_port) if 'self' in locals() else None),
            self.pooled_port: _make_signal(source=source, name=self.pooled_port, value=pooled_embedding, emitted_at=t, spec=self.outputs().get(self.pooled_port) if 'self' in locals() else None),
        }

    def get_outputs(self) -> Dict[str, BioSignal]:
        return dict(self._outputs)
