# SPDX-FileCopyrightText: 2025-present Demi <bjaiye1@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Imported transformer ONNX wrapper for Bio_ClinicalBERT."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from biosim import BioModule
from biosim.signals import BioSignal, SignalMetadata


def _normalize_ids(raw: Any, length: int) -> List[int]:
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


class HfEmilyalsentzerBioClinicalbert(BioModule):
    """Run the imported Bio_ClinicalBERT ONNX export behind the BioModule interface."""

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
        min_dt: float = 0.01,
        session_factory: Optional[Callable[[str], Any]] = None,
        providers: Optional[Sequence[str]] = None,
    ) -> None:
        self.min_dt = min_dt
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

    def inputs(self) -> Set[str]:
        return {self.token_port, self.mask_port}

    def outputs(self) -> Set[str]:
        return {self.sequence_port, self.pooled_port}

    def set_inputs(self, signals: Dict[str, BioSignal]) -> None:
        token_signal = signals.get(self.token_port)
        mask_signal = signals.get(self.mask_port)
        if token_signal is not None:
            self._token_ids = _normalize_ids(token_signal.value, self.seq_length)
        if mask_signal is not None:
            self._attention_mask = _normalize_ids(mask_signal.value, self.seq_length)

    def advance_to(self, t: float) -> None:
        session = self._ensure_session()
        sequence_name = self.sequence_output_name or self.sequence_port
        pooled_name = self.pooled_output_name or self.pooled_port
        token_name = self.token_input_name or self.token_port
        mask_name = self.mask_input_name or self.mask_port
        feed_dict = {
            token_name: [self._token_ids],
            mask_name: [self._attention_mask],
        }
        if self.segment_input_name is not None:
            feed_dict[self.segment_input_name] = [[0] * self.seq_length]
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
            self.sequence_port: BioSignal(
                source=source,
                name=self.sequence_port,
                value=sequence_embeddings,
                time=t,
                metadata=SignalMetadata(
                    description="Bio_ClinicalBERT token-level embedding tensor",
                    dtype="float32",
                    shape=(1, self.seq_length, 768),
                    kind="state",
                ),
            ),
            self.pooled_port: BioSignal(
                source=source,
                name=self.pooled_port,
                value=pooled_embedding,
                time=t,
                metadata=SignalMetadata(
                    description="Bio_ClinicalBERT pooled embedding vector",
                    dtype="float32",
                    shape=(1, 768),
                    kind="state",
                ),
            ),
        }

    def get_outputs(self) -> Dict[str, BioSignal]:
        return dict(self._outputs)
