"""Scaffold the first lighter transformer export batch into models-onnx."""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
from pathlib import Path


BATCH = [
    {
        "rank": "81",
        "slug": "hf-emilyalsentzer-bio-clinicalbert",
        "title": "Bio_ClinicalBERT",
        "source_url": "https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT",
        "domain": "clinical-nlp",
        "task": "embedding-generation",
    },
    {
        "rank": "83",
        "slug": "hf-facebook-esm2-t6-8m-ur50d",
        "title": "esm2_t6_8M_UR50D",
        "source_url": "https://huggingface.co/facebook/esm2_t6_8M_UR50D",
        "domain": "protein-language-model",
        "task": "embedding-generation",
    },
    {
        "rank": "84",
        "slug": "hf-dmis-lab-biobert-base-cased-v1-1",
        "title": "biobert-base-cased-v1.1",
        "source_url": "https://huggingface.co/dmis-lab/biobert-base-cased-v1.1",
        "domain": "biomedical-nlp",
        "task": "embedding-generation",
    },
    {
        "rank": "86",
        "slug": "hf-facebook-esm2-t12-35m-ur50d",
        "title": "esm2_t12_35M_UR50D",
        "source_url": "https://huggingface.co/facebook/esm2_t12_35M_UR50D",
        "domain": "protein-language-model",
        "task": "embedding-generation",
    },
    {
        "rank": "87",
        "slug": "hf-allenai-scibert-scivocab-uncased",
        "title": "scibert_scivocab_uncased",
        "source_url": "https://huggingface.co/allenai/scibert_scivocab_uncased",
        "domain": "scientific-nlp",
        "task": "embedding-generation",
    },
]


WRAPPER_TEMPLATE = """# SPDX-FileCopyrightText: 2025-present Demi <bjaiye1@gmail.com>
#
# SPDX-License-Identifier: MIT
\"\"\"Scaffolded transformer ONNX wrapper for {title}.\"\"\"

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


class {class_name}(BioModule):
    \"\"\"Scaffolded transformer wrapper pending ONNX export and tokenizer contract.\"\"\"

    def __init__(
        self,
        model_path: str = "artifacts/model.onnx",
        token_port: str = "token_ids",
        mask_port: str = "attention_mask",
        sequence_port: str = "sequence_embeddings",
        pooled_port: str = "pooled_embedding",
        token_input_name: Optional[str] = None,
        mask_input_name: Optional[str] = None,
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
        self.sequence_output_name = sequence_output_name
        self.pooled_output_name = pooled_output_name
        self.seq_length = seq_length
        self.base_dir = Path(base_dir).resolve() if base_dir else Path(__file__).resolve().parents[1]
        self._session_factory = session_factory
        self.providers = list(providers or ["CPUExecutionProvider"])
        self._session: Any = None
        self._token_ids = [0] * seq_length
        self._attention_mask = [1] * seq_length
        self._outputs: Dict[str, BioSignal] = {{}}

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
            if outputs and self.sequence_output_name is None:
                self.sequence_output_name = str(outputs[0].name)
            if len(outputs) > 1 and self.pooled_output_name is None:
                self.pooled_output_name = str(outputs[1].name)
        return self._session

    def inputs(self) -> Set[str]:
        return {{self.token_port, self.mask_port}}

    def outputs(self) -> Set[str]:
        return {{self.sequence_port, self.pooled_port}}

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
        result = session.run(
            [sequence_name, pooled_name],
            {{
                token_name: [self._token_ids],
                mask_name: [self._attention_mask],
            }},
        )
        sequence_embeddings = result[0] if result else [[[0.0] * 4 for _ in range(self.seq_length)]]
        pooled_embedding = result[1] if len(result) > 1 else [[0.0, 0.0, 0.0, 0.0]]
        source = getattr(self, "_world_name", self.__class__.__name__)
        self._outputs = {{
            self.sequence_port: BioSignal(
                source=source,
                name=self.sequence_port,
                value=sequence_embeddings,
                time=t,
                metadata=SignalMetadata(
                    description="{title} token-level embedding tensor",
                    dtype="float32",
                    shape=(1, self.seq_length, 4),
                    kind="state",
                ),
            ),
            self.pooled_port: BioSignal(
                source=source,
                name=self.pooled_port,
                value=pooled_embedding,
                time=t,
                metadata=SignalMetadata(
                    description="{title} pooled embedding vector",
                    dtype="float32",
                    shape=(1, 4),
                    kind="state",
                ),
            ),
        }}

    def get_outputs(self) -> Dict[str, BioSignal]:
        return dict(self._outputs)
"""


TEST_TEMPLATE = """from __future__ import annotations

from src.imported_transformer_model import {class_name}


class _FakeSession:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def get_inputs(self):
        return [type("In", (), {{"name": "input_ids"}})(), type("In", (), {{"name": "attention_mask"}})()]

    def get_outputs(self):
        return [type("Out", (), {{"name": "sequence_embeddings"}})(), type("Out", (), {{"name": "pooled_embedding"}})()]

    def run(self, outputs, feed_dict):
        batch = len(next(iter(feed_dict.values())))
        seq_length = len(feed_dict["input_ids"][0])
        sequence = [[[float(i), 0.0, 0.0, 1.0] for i in range(seq_length)] for _ in range(batch)]
        pooled = [[sum(feed_dict["attention_mask"][0]), 0.0, 0.0, 1.0]]
        return [sequence, pooled]


def test_imported_transformer_model_emits_sequence_and_pooled_outputs() -> None:
    tokens = type("Sig", (), {{"value": [1, 2, 3, 4]}})()
    mask = type("Sig", (), {{"value": [1, 1, 1, 1]}})()
    model = {class_name}(session_factory=_FakeSession)
    model.set_inputs({{"token_ids": tokens, "attention_mask": mask}})
    model.advance_to(0.1)
    outputs = model.get_outputs()
    assert "sequence_embeddings" in outputs
    assert "pooled_embedding" in outputs
"""


def _class_name(slug: str) -> str:
    return "".join(part.capitalize() for part in re.split(r"[^a-zA-Z0-9]+", slug) if part)


def scaffold_model(root: Path, item: dict[str, str]) -> None:
    model_dir = root / "models" / item["slug"]
    if model_dir.exists():
        shutil.rmtree(model_dir)
    src_dir = model_dir / "src"
    tests_dir = model_dir / "tests"
    artifacts_dir = model_dir / "artifacts"
    src_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    class_name = _class_name(item["slug"])
    model_yaml = f"""schema_version: "2.0"
title: "Transformer: {class_name}"
description: "Scaffolded Hugging Face transformer import for {item['title']} pending tokenizer contract validation and ONNX export."
standard: onnx
tags: [transformer, onnx, {item['domain']}, scaffold]
authors: ["Biosimulant Team"]
biosim:
  entrypoint: "src.imported_transformer_model:{class_name}"
  init_kwargs:
    model_path: "artifacts/model.onnx"
io:
  inputs:
    - token_ids
    - attention_mask
  outputs:
    - sequence_embeddings
    - pooled_embedding
runtime:
  dependencies:
    packages:
      - onnxruntime==1.22.1
onnx:
  task: {item['task']}
  model_file: "artifacts/model.onnx"
  inputs:
    - name: input_ids
      dtype: int64
      shape: [1, 16]
    - name: attention_mask
      dtype: int64
      shape: [1, 16]
  outputs:
    - name: sequence_embeddings
      dtype: float32
      shape: [1, 16, 4]
    - name: pooled_embedding
      dtype: float32
      shape: [1, 4]
source:
  model: "{item['source_url']}"
"""
    (model_dir / "model.yaml").write_text(model_yaml, encoding="utf-8")
    (src_dir / "imported_transformer_model.py").write_text(
        WRAPPER_TEMPLATE.format(title=item["title"].replace('"', "'"), class_name=class_name),
        encoding="utf-8",
    )
    (tests_dir / "test_imported_transformer_model.py").write_text(
        TEST_TEMPLATE.format(class_name=class_name),
        encoding="utf-8",
    )
    (model_dir / "import-metadata.json").write_text(json.dumps(item, indent=2), encoding="utf-8")
    (artifacts_dir / "README.md").write_text(
        "Scaffolded transformer import. Validate the tokenizer/preprocessing contract and export a stable ONNX artifact before indexing this model.\n",
        encoding="utf-8",
    )
    (artifacts_dir / "source-model-url.txt").write_text(item["source_url"] + "\n", encoding="utf-8")
    print(f"Scaffolded {item['slug']}")


def update_import_plan(root: Path) -> None:
    metadata_root = Path(
        os.environ.get("MODELS_ONNX_METADATA_DIR", root.parent / "models-onnx-metadata")
    )
    plan_path = metadata_root / "inventory" / "import-plan.csv"
    with plan_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = rows[0].keys()
    batch_ranks = {item["rank"] for item in BATCH}
    for row in rows:
        if row["rank"] in batch_ranks:
            row["import_status"] = "scaffolded"
            row["blocking_reason"] = "export-hf-transformer-to-onnx"
    with plan_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for item in BATCH:
        scaffold_model(root, item)
    update_import_plan(root)


if __name__ == "__main__":
    main()
