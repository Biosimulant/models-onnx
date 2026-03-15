# SPDX-FileCopyrightText: 2025-present Demi <bjaiye1@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Imported BioImage ONNX wrapper for HPA Bestfitting Densenet."""

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


class BioimageHpaBestfittingDensenet(BioModule):
    """Run the imported HPA Densenet ONNX graph as a BioModule."""

    def __init__(
        self,
        model_path: str = "artifacts/model.onnx",
        input_port: str = "image",
        class_port: str = "class_probabilities",
        feature_port: str = "feature_embeddings",
        summary_port: str = "prediction_summary",
        model_input_name: Optional[str] = None,
        class_output_name: Optional[str] = None,
        feature_output_name: Optional[str] = None,
        input_shape: Sequence[int] = (1, 4, 1024, 1024),
        class_dim: int = 28,
        feature_dim: int = 1024,
        input_scale: float = 0.003921568627,
        base_dir: Optional[str] = None,
        min_dt: float = 0.01,
        session_factory: Optional[Callable[[str], Any]] = None,
        providers: Optional[Sequence[str]] = None,
    ) -> None:
        self.min_dt = min_dt
        self.model_path = model_path
        self.input_port = input_port
        self.class_port = class_port
        self.feature_port = feature_port
        self.summary_port = summary_port
        self.model_input_name = model_input_name
        self.class_output_name = class_output_name
        self.feature_output_name = feature_output_name
        self.input_shape = tuple(int(x) for x in input_shape)
        self.class_dim = int(class_dim)
        self.feature_dim = int(feature_dim)
        self.input_scale = float(input_scale)
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
            if outputs and self.class_output_name is None:
                self.class_output_name = str(outputs[0].name)
            if len(outputs) > 1 and self.feature_output_name is None:
                self.feature_output_name = str(outputs[1].name)
        return self._session

    def inputs(self) -> Set[str]:
        return {self.input_port}

    def outputs(self) -> Set[str]:
        return {self.class_port, self.feature_port, self.summary_port}

    def set_inputs(self, signals: Dict[str, BioSignal]) -> None:
        signal = signals.get(self.input_port)
        if signal is None:
            return
        flat = _flatten(signal.value)
        needed = self._element_count(self.input_shape)
        flat = [(x * self.input_scale) for x in (flat[:needed] + [0.0] * max(0, needed - len(flat)))]
        self._latest_input = _reshape(flat, self.input_shape)

    def advance_to(self, t: float) -> None:
        session = self._ensure_session()
        input_name = self.model_input_name or self.input_port
        outputs = [self.class_output_name or self.class_port]
        if self.feature_output_name is not None:
            outputs.append(self.feature_output_name)
        result = session.run(outputs, {input_name: self._latest_input})
        classes = result[0] if result else [[0.0] * self.class_dim]
        features = result[1] if len(result) > 1 else [[0.0] * self.feature_dim]
        flat_classes = _flatten(classes)
        top_idx = max(range(len(flat_classes)), key=flat_classes.__getitem__) if flat_classes else -1
        top_score = flat_classes[top_idx] if top_idx >= 0 else 0.0
        source = getattr(self, "_world_name", self.__class__.__name__)
        self._outputs = {
            self.class_port: BioSignal(
                source=source,
                name=self.class_port,
                value=classes,
                time=t,
                metadata=SignalMetadata(description="HPA class probabilities", dtype="float32", shape=(1, self.class_dim), kind="state"),
            ),
            self.feature_port: BioSignal(
                source=source,
                name=self.feature_port,
                value=features,
                time=t,
                metadata=SignalMetadata(description="HPA feature embedding vector", dtype="float32", shape=(1, self.feature_dim), kind="state"),
            ),
            self.summary_port: BioSignal(
                source=source,
                name=self.summary_port,
                value={"top_class_index": top_idx, "top_score": top_score},
                time=t,
                metadata=SignalMetadata(description="Top HPA class prediction summary", kind="metric"),
            ),
        }

    def get_outputs(self) -> Dict[str, BioSignal]:
        return dict(self._outputs)
