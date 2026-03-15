"""Scaffold the first MONAI bundle batch into models-onnx."""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
from pathlib import Path


BATCH = [
    {
        "rank": "45",
        "slug": "monai-spleen-ct-segmentation",
        "title": "Spleen CT Segmentation",
        "source_url": "https://api.ngc.nvidia.com/v2/models/nvidia/monaihosting/spleen_ct_segmentation/versions/0.1.0/files/spleen_ct_segmentation_v0.1.0.zip",
        "modality": "ct",
        "task": "segmentation",
    },
    {
        "rank": "46",
        "slug": "monai-pancreas-ct-dints-segmentation",
        "title": "Pancreas CT DINTS Segmentation",
        "source_url": "https://api.ngc.nvidia.com/v2/models/nvidia/monaihosting/pancreas_ct_dints_segmentation/versions/0.1.0/files/pancreas_ct_dints_segmentation_v0.1.0.zip",
        "modality": "ct",
        "task": "segmentation",
    },
    {
        "rank": "47",
        "slug": "monai-brats-mri-segmentation",
        "title": "BRATS MRI Segmentation",
        "source_url": "https://api.ngc.nvidia.com/v2/models/nvidia/monaihosting/brats_mri_segmentation/versions/0.1.0/files/brats_mri_segmentation_v0.1.0.zip",
        "modality": "mri",
        "task": "segmentation",
    },
    {
        "rank": "48",
        "slug": "monai-spleen-deepedit-annotation",
        "title": "Spleen DeepEdit Annotation",
        "source_url": "https://api.ngc.nvidia.com/v2/models/nvidia/monaihosting/spleen_deepedit_annotation/versions/0.1.0/files/spleen_deepedit_annotation_v0.1.0.zip",
        "modality": "medical-imaging",
        "task": "annotation",
    },
    {
        "rank": "49",
        "slug": "monai-swin-unetr-btcv-segmentation",
        "title": "Swin UNETR BTCV Segmentation",
        "source_url": "https://api.ngc.nvidia.com/v2/models/nvidia/monaihosting/swin_unetr_btcv_segmentation/versions/0.1.0/files/swin_unetr_btcv_segmentation_v0.1.0.zip",
        "modality": "medical-imaging",
        "task": "segmentation",
    },
]


WRAPPER_TEMPLATE = """# SPDX-FileCopyrightText: 2025-present Demi <bjaiye1@gmail.com>
#
# SPDX-License-Identifier: MIT
\"\"\"Imported MONAI ONNX wrapper scaffold for {title}.\"\"\"

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from biosim import BioModule
from biosim.signals import BioSignal, SignalMetadata


def _flatten(value: Any) -> List[float]:
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


class {class_name}(BioModule):
    \"\"\"Scaffolded MONAI wrapper pending bundle inspection and ONNX export.\"\"\"

    def __init__(
        self,
        model_path: str = "artifacts/model.onnx",
        input_port: str = "volume_tensor",
        output_port: str = "class_probabilities",
        summary_port: str = "segmentation_summary",
        model_input_name: Optional[str] = None,
        model_output_name: Optional[str] = None,
        input_shape: Sequence[int] = (1, 1, 4, 4, 4),
        output_shape: Sequence[int] = (1, 2, 4, 4, 4),
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
        self._outputs: Dict[str, BioSignal] = {{}}

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
        return {{self.input_port}}

    def outputs(self) -> Set[str]:
        return {{self.output_port, self.summary_port}}

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
        result = session.run([output_name], {{input_name: self._latest_input}})
        prediction = result[0] if result else _reshape([0.0] * self._element_count(self.output_shape), self.output_shape)
        flat = _flatten(prediction)
        source = getattr(self, "_world_name", self.__class__.__name__)
        self._outputs = {{
            self.output_port: BioSignal(
                source=source,
                name=self.output_port,
                value=prediction,
                time=t,
                metadata=SignalMetadata(description="{title} output tensor", dtype="float32", shape=self.output_shape, kind="state"),
            ),
            self.summary_port: BioSignal(
                source=source,
                name=self.summary_port,
                value={{"mean_probability": (sum(flat) / len(flat)) if flat else 0.0, "voxel_count": len(flat)}},
                time=t,
                metadata=SignalMetadata(description="Summary statistics over the latest MONAI output tensor", kind="metric"),
            ),
        }}

    def get_outputs(self) -> Dict[str, BioSignal]:
        return dict(self._outputs)
"""


TEST_TEMPLATE = """from __future__ import annotations

from src.imported_monai_model import {class_name}


class _FakeSession:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def get_inputs(self):
        return [type("In", (), {{"name": "x"}})()]

    def get_outputs(self):
        return [type("Out", (), {{"name": "y"}})()]

    def run(self, outputs, feed_dict):
        return [next(iter(feed_dict.values()))]


def test_imported_monai_model_emits_prediction_and_summary() -> None:
    signal = type("Sig", (), {{"value": [[[[[0.0] * 4 for _ in range(4)] for _ in range(4)]]],}})()
    model = {class_name}(session_factory=_FakeSession)
    model.set_inputs({{"volume_tensor": signal}})
    model.advance_to(0.1)
    outputs = model.get_outputs()
    assert "class_probabilities" in outputs
    assert "segmentation_summary" in outputs
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
title: "MONAI: {class_name}"
description: "Scaffolded MONAI import for {item['title']} pending bundle inspection and ONNX export."
standard: onnx
tags: [monai, onnx, {item['modality']}, {item['task']}, scaffold]
authors: ["Biosimulant Team"]
biosim:
  entrypoint: "src.imported_monai_model:{class_name}"
  init_kwargs:
    model_path: "artifacts/model.onnx"
io:
  inputs:
    - volume_tensor
  outputs:
    - class_probabilities
    - segmentation_summary
runtime:
  dependencies:
    packages:
      - onnxruntime==1.22.1
source:
  bundle: "{item['source_url']}"
"""
    (model_dir / "model.yaml").write_text(model_yaml, encoding="utf-8")
    (src_dir / "imported_monai_model.py").write_text(
        WRAPPER_TEMPLATE.format(title=item["title"].replace('"', "'"), class_name=class_name),
        encoding="utf-8",
    )
    (tests_dir / "test_imported_monai_model.py").write_text(
        TEST_TEMPLATE.format(class_name=class_name),
        encoding="utf-8",
    )
    (model_dir / "import-metadata.json").write_text(json.dumps(item, indent=2), encoding="utf-8")
    (artifacts_dir / "README.md").write_text(
        "Scaffolded MONAI import. Inspect the source bundle and export a stable ONNX artifact before indexing this model.\n",
        encoding="utf-8",
    )
    (artifacts_dir / "source-bundle-url.txt").write_text(item["source_url"] + "\n", encoding="utf-8")
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
            row["blocking_reason"] = "export-monai-bundle-to-onnx"
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
