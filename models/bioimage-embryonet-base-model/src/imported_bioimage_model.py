# SPDX-FileCopyrightText: 2025-present Demi <bjaiye1@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Imported BioImage ONNX wrapper for EmbryoNet base model."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from biosim import BioModule
from biosim.signals import BioSignal, SignalMetadata


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


class BioimageEmbryonetBaseModel(BioModule):
    """Run the imported EmbryoNet ONNX graph as a BioModule."""

    def __init__(
        self,
        model_path: str = "artifacts/model.onnx",
        image_port: str = "image",
        conv_time_port: str = "conv_time_map",
        fc_time_port: str = "fc_time_scalar",
        output_port: str = "development_scores",
        summary_port: str = "prediction_summary",
        image_input_name: Optional[str] = None,
        conv_time_input_name: Optional[str] = None,
        fc_time_input_name: Optional[str] = None,
        output_name: Optional[str] = None,
        image_shape: Sequence[int] = (1, 3, 224, 224),
        conv_time_shape: Sequence[int] = (1, 1, 224, 224),
        fc_time_shape: Sequence[int] = (1, 1),
        output_dim: int = 14,
        base_dir: Optional[str] = None,
        min_dt: float = 0.01,
        session_factory: Optional[Callable[[str], Any]] = None,
        providers: Optional[Sequence[str]] = None,
    ) -> None:
        self.min_dt = min_dt
        self.model_path = model_path
        self.image_port = image_port
        self.conv_time_port = conv_time_port
        self.fc_time_port = fc_time_port
        self.output_port = output_port
        self.summary_port = summary_port
        self.image_input_name = image_input_name
        self.conv_time_input_name = conv_time_input_name
        self.fc_time_input_name = fc_time_input_name
        self.output_name = output_name
        self.image_shape = tuple(int(x) for x in image_shape)
        self.conv_time_shape = tuple(int(x) for x in conv_time_shape)
        self.fc_time_shape = tuple(int(x) for x in fc_time_shape)
        self.output_dim = int(output_dim)
        self.base_dir = Path(base_dir).resolve() if base_dir else Path(__file__).resolve().parents[1]
        self._session_factory = session_factory
        self.providers = list(providers or ["CPUExecutionProvider"])
        self._session: Any = None
        self._image = _reshape([0.0] * self._element_count(self.image_shape), self.image_shape)
        self._conv_time = _reshape([0.0] * self._element_count(self.conv_time_shape), self.conv_time_shape)
        self._fc_time = _reshape([0.0] * self._element_count(self.fc_time_shape), self.fc_time_shape)
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
            if inputs and self.image_input_name is None:
                self.image_input_name = str(inputs[0].name)
            if len(inputs) > 1 and self.conv_time_input_name is None:
                self.conv_time_input_name = str(inputs[1].name)
            if len(inputs) > 2 and self.fc_time_input_name is None:
                self.fc_time_input_name = str(inputs[2].name)
            if outputs and self.output_name is None:
                self.output_name = str(outputs[0].name)
        return self._session

    def inputs(self) -> Set[str]:
        return {self.image_port, self.conv_time_port, self.fc_time_port}

    def outputs(self) -> Set[str]:
        return {self.output_port, self.summary_port}

    def _update_tensor(self, signal: BioSignal, shape: Sequence[int]) -> Any:
        flat = _flatten(signal.value)
        needed = self._element_count(shape)
        flat = flat[:needed] + [0.0] * max(0, needed - len(flat))
        return _reshape(flat, shape)

    def set_inputs(self, signals: Dict[str, BioSignal]) -> None:
        image_signal = signals.get(self.image_port)
        conv_time_signal = signals.get(self.conv_time_port)
        fc_time_signal = signals.get(self.fc_time_port)
        if image_signal is not None:
            self._image = self._update_tensor(image_signal, self.image_shape)
        if conv_time_signal is not None:
            self._conv_time = self._update_tensor(conv_time_signal, self.conv_time_shape)
        if fc_time_signal is not None:
            self._fc_time = self._update_tensor(fc_time_signal, self.fc_time_shape)

    def advance_to(self, t: float) -> None:
        session = self._ensure_session()
        output_name = self.output_name or self.output_port
        feed_dict = {
            self.image_input_name or self.image_port: self._image,
            self.conv_time_input_name or self.conv_time_port: self._conv_time,
            self.fc_time_input_name or self.fc_time_port: self._fc_time,
        }
        result = session.run([output_name], feed_dict)
        scores = result[0] if result else [0.0] * self.output_dim
        flat_scores = _flatten(scores)
        top_idx = max(range(len(flat_scores)), key=flat_scores.__getitem__) if flat_scores else -1
        top_score = flat_scores[top_idx] if top_idx >= 0 else 0.0
        source = getattr(self, "_world_name", self.__class__.__name__)
        self._outputs = {
            self.output_port: BioSignal(
                source=source,
                name=self.output_port,
                value=scores,
                time=t,
                metadata=SignalMetadata(description="EmbryoNet development-stage scores", dtype="float32", shape=(self.output_dim,), kind="state"),
            ),
            self.summary_port: BioSignal(
                source=source,
                name=self.summary_port,
                value={"top_stage_index": top_idx, "top_score": top_score},
                time=t,
                metadata=SignalMetadata(description="Top EmbryoNet stage prediction summary", kind="metric"),
            ),
        }

    def get_outputs(self) -> Dict[str, BioSignal]:
        return dict(self._outputs)
