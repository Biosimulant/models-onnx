# SPDX-FileCopyrightText: 2025-present Demi <bjaiye1@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Imported BioImage ONNX wrapper for HyLFM-Net-stat."""

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


class BioimageHylfmNetStat(BioModule):
    """Run the imported HyLFM-Net ONNX graph as a BioModule."""

    def __init__(
        self,
        model_path: str = "artifacts/model.onnx",
        input_port: str = "lf_image",
        output_port: str = "reconstruction_volume",
        summary_port: str = "prediction_summary",
        model_input_name: Optional[str] = None,
        model_output_name: Optional[str] = None,
        input_shape: Sequence[int] = (1, 1, 1235, 1425),
        output_shape: Sequence[int] = (1, 1, 49, 244, 284),
        base_dir: Optional[str] = None,
        min_dt: float = 0.01,
        session_factory: Optional[Callable[[str], Any]] = None,
        providers: Optional[Sequence[str]] = None,
    ) -> None:
        self.min_dt = min_dt
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

    def inputs(self) -> Set[str]:
        return {self.input_port}

    def outputs(self) -> Set[str]:
        return {self.output_port, self.summary_port}

    def set_inputs(self, signals: Dict[str, BioSignal]) -> None:
        signal = signals.get(self.input_port)
        if signal is None:
            return
        flat = _flatten(signal.value)
        needed = self._element_count(self.input_shape)
        flat = flat[:needed] + [0.0] * max(0, needed - len(flat))
        self._latest_input = _reshape(flat, self.input_shape)

    def advance_to(self, t: float) -> None:
        session = self._ensure_session()
        output_name = self.model_output_name or self.output_port
        input_name = self.model_input_name or self.input_port
        result = session.run([output_name], {input_name: self._latest_input})
        volume = result[0] if result else _reshape([0.0] * self._element_count(self.output_shape), self.output_shape)
        flat = _flatten(volume)
        source = getattr(self, "_world_name", self.__class__.__name__)
        self._outputs = {
            self.output_port: BioSignal(
                source=source,
                name=self.output_port,
                value=volume,
                time=t,
                metadata=SignalMetadata(description="HyLFM reconstruction volume", dtype="float32", shape=self.output_shape, kind="state"),
            ),
            self.summary_port: BioSignal(
                source=source,
                name=self.summary_port,
                value={
                    "mean_value": (sum(flat) / len(flat)) if flat else 0.0,
                    "voxel_count": len(flat),
                },
                time=t,
                metadata=SignalMetadata(description="Summary statistics over the HyLFM reconstruction volume", kind="metric"),
            ),
        }

    def get_outputs(self) -> Dict[str, BioSignal]:
        return dict(self._outputs)
