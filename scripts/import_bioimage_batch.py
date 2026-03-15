"""Import the first BioImage ONNX batch into models-onnx."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from urllib.request import urlopen


BATCH = [
    {
        "rank": "1",
        "slug": "bioimage-nucleisegmentationboundarymodel",
        "rdf_source": "https://bioimage-io.github.io/collection-bioimage-io/rdfs/10.5281/zenodo.5764892/6647674/rdf.yaml",
    },
    {
        "rank": "26",
        "slug": "bioimage-hpa-bestfitting-inceptionv3",
        "rdf_source": "https://bioimage-io.github.io/collection-bioimage-io/rdfs/10.5281/zenodo.5910854/6539073/rdf.yaml",
    },
    {
        "rank": "30",
        "slug": "bioimage-hpa-bestfitting-densenet",
        "rdf_source": "https://bioimage-io.github.io/collection-bioimage-io/rdfs/10.5281/zenodo.5910163/5942853/rdf.yaml",
    },
    {
        "rank": "34",
        "slug": "bioimage-embryonet-base-model",
        "rdf_source": "https://bioimage-io.github.io/collection-bioimage-io/rdfs/10.5281/zenodo.7315440/7315441/rdf.yaml",
    },
    {
        "rank": "36",
        "slug": "bioimage-hylfm-net-stat",
        "rdf_source": "https://bioimage-io.github.io/collection-bioimage-io/rdfs/10.5281/zenodo.7614645/7642674/rdf.yaml",
    },
]


WRAPPER_TEMPLATE = """# SPDX-FileCopyrightText: 2025-present Demi <bjaiye1@gmail.com>
#
# SPDX-License-Identifier: MIT
\"\"\"Imported BioImage ONNX wrapper for {model_name}.\"\"\"

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
    \"\"\"Thin imported BioImage ONNX wrapper.\"\"\"

    def __init__(
        self,
        model_path: str = "artifacts/model.onnx",
        input_port: str = "{input_name}",
        output_port: str = "{output_name}",
        summary_port: str = "prediction_summary",
        model_input_name: Optional[str] = None,
        model_output_name: Optional[str] = None,
        input_shape: Sequence[int] = {input_shape},
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
        self.base_dir = Path(base_dir).resolve() if base_dir else Path(__file__).resolve().parents[1]
        self._session_factory = session_factory
        self.providers = list(providers or ["CPUExecutionProvider"])
        self._session: Any = None
        self._latest_input = _reshape([0.0] * self._element_count(), self.input_shape)
        self._outputs: Dict[str, BioSignal] = {{}}

    def _element_count(self) -> int:
        total = 1
        for dim in self.input_shape:
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
        needed = self._element_count()
        flat = flat[:needed] + [0.0] * max(0, needed - len(flat))
        self._latest_input = _reshape(flat, self.input_shape)

    def advance_to(self, t: float) -> None:
        session = self._ensure_session()
        output_name = self.model_output_name or self.output_port
        input_name = self.model_input_name or self.input_port
        result = session.run([output_name], {{input_name: self._latest_input}})
        prediction = result[0] if result else self._latest_input
        flat = _flatten(prediction)
        source = getattr(self, "_world_name", self.__class__.__name__)
        self._outputs = {{
            self.output_port: BioSignal(
                source=source,
                name=self.output_port,
                value=prediction,
                time=t,
                metadata=SignalMetadata(
                    description="{description}",
                    dtype="float32",
                    kind="state",
                ),
            ),
            self.summary_port: BioSignal(
                source=source,
                name=self.summary_port,
                value={{"mean": (sum(flat) / len(flat)) if flat else 0.0, "nonzero_fraction": sum(1 for x in flat if x > 0.5) / max(1, len(flat))}},
                time=t,
                metadata=SignalMetadata(description="Summary statistics for the imported BioImage ONNX output", kind="metric"),
            ),
        }}

    def get_outputs(self) -> Dict[str, BioSignal]:
        return dict(self._outputs)
"""


TEST_TEMPLATE = """from __future__ import annotations

from src.imported_bioimage_model import {class_name}


class _FakeSession:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def get_inputs(self):
        return [type("In", (), {{"name": "x"}})()]

    def get_outputs(self):
        return [type("Out", (), {{"name": "y"}})()]

    def run(self, outputs, feed_dict):
        return [next(iter(feed_dict.values()))]


def test_imported_bioimage_model_emits_prediction_and_summary() -> None:
    signal = type("Sig", (), {{"value": {sample_value}}})()
    model = {class_name}(session_factory=_FakeSession)
    model.set_inputs({{{input_port!r}: signal}})
    model.advance_to(0.1)
    outputs = model.get_outputs()
    assert {output_port!r} in outputs
    assert "prediction_summary" in outputs
"""


def _extract_scalar(text: str, key: str) -> str:
    match = re.search(rf"(?m)^{re.escape(key)}:\s*(.+)$", text)
    if not match:
        raise ValueError(f"Missing scalar: {key}")
    return match.group(1).strip().strip("'\"")


def _extract_list(text: str, key: str) -> list[str]:
    lines = _extract_top_level_block(text, key)
    values: list[str] = []
    for line in lines:
        if not line.startswith("- "):
            break
        values.append(line[2:].strip())
    return values


def _extract_top_level_block(text: str, key: str) -> list[str]:
    lines = text.splitlines()
    collecting = False
    block: list[str] = []
    for line in lines:
        if not collecting:
            if line == f"{key}:":
                collecting = True
            continue
        if line and not line.startswith(" ") and line.endswith(":"):
            break
        block.append(line)
    return block


def _extract_first_io_name(text: str, section: str) -> str:
    for line in _extract_top_level_block(text, section):
        stripped = line.strip()
        if stripped.startswith("name:"):
            return stripped.split(":", 1)[1].strip()
    raise ValueError(f"Missing {section} name")


def _extract_input_shape(text: str) -> list[int]:
    block = _extract_top_level_block(text, "inputs")
    values: list[int] = []
    in_min = False
    for line in block:
        if line.strip() == "min:":
            in_min = True
            continue
        if in_min:
            stripped = line.strip()
            if stripped.startswith("- "):
                values.append(int(float(stripped.split("- ", 1)[1])))
                continue
            break
    return values or [1, 1, 64, 64]


def _extract_onnx_url(text: str) -> str:
    block = _extract_top_level_block(text, "weights")
    in_onnx = False
    pending_source = False
    for line in block:
        if line.startswith("  onnx:"):
            in_onnx = True
            continue
        if in_onnx and line.startswith("  ") and not line.startswith("    "):
            break
        if in_onnx and pending_source and line.strip():
            return line.strip()
        if in_onnx and line.strip().startswith("source:"):
            value = line.strip().split(":", 1)[1].strip()
            if value:
                return value
            pending_source = True
    raise ValueError("Missing ONNX source URL")


def _class_name_from_slug(slug: str) -> str:
    parts = re.split(r"[^a-zA-Z0-9]+", slug)
    return "".join(part.capitalize() for part in parts if part)


def _sample_value(shape: list[int]) -> str:
    if len(shape) == 4:
        return "[[[[0.0] * {w} for _ in range({h})] for _ in range({c})] for _ in range({b})]".format(
            b=shape[0], c=shape[1], h=shape[2], w=shape[3]
        )
    return "[0.0]"


def import_model(root: Path, item: dict[str, str], *, download_artifacts: bool) -> None:
    rdf_text = urlopen(item["rdf_source"]).read().decode("utf-8", "ignore").replace("\r\n", "\n")
    name = _extract_scalar(rdf_text, "name")
    description = _extract_scalar(rdf_text, "description")
    license_name = _extract_scalar(rdf_text, "license")
    tags = _extract_list(rdf_text, "tags")
    input_name = _extract_first_io_name(rdf_text, "inputs")
    output_name = _extract_first_io_name(rdf_text, "outputs")
    input_shape = _extract_input_shape(rdf_text)
    onnx_url = _extract_onnx_url(rdf_text)

    model_dir = root / "models" / item["slug"]
    if model_dir.exists():
        shutil.rmtree(model_dir)
    src_dir = model_dir / "src"
    tests_dir = model_dir / "tests"
    artifacts_dir = model_dir / "artifacts"
    src_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    class_name = _class_name_from_slug(item["slug"])
    tags_yaml = ", ".join(["bioimage", "onnx"] + tags[:5])
    model_yaml = f"""schema_version: "2.0"
title: "BioImage: {class_name}"
description: "{description}"
standard: onnx
tags: [{tags_yaml}]
authors: ["Biosimulant Team"]
biosim:
  entrypoint: "src.imported_bioimage_model:{class_name}"
  init_kwargs:
    model_path: "artifacts/model.onnx"
io:
  inputs:
    - {input_name}
  outputs:
    - {output_name}
    - prediction_summary
runtime:
  dependencies:
    packages:
      - onnxruntime==1.22.1
onnx:
  task: segmentation
  model_file: "artifacts/model.onnx"
  inputs:
    - name: {input_name}
      dtype: float32
      shape: {json.dumps(input_shape)}
  outputs:
    - name: {output_name}
      dtype: float32
license: {license_name}
source:
  rdf: "{item['rdf_source']}"
  onnx: "{onnx_url}"
"""
    (model_dir / "model.yaml").write_text(model_yaml, encoding="utf-8")
    (src_dir / "imported_bioimage_model.py").write_text(
        WRAPPER_TEMPLATE.format(
            model_name=name,
            class_name=class_name,
            input_name=input_name,
            output_name=output_name,
            input_shape=tuple(input_shape),
            description=description.replace('"', "'"),
        ),
        encoding="utf-8",
    )
    (tests_dir / "test_imported_bioimage_model.py").write_text(
        TEST_TEMPLATE.format(
            class_name=class_name,
            input_port=input_name,
            output_port=output_name,
            sample_value=_sample_value(input_shape),
        ),
        encoding="utf-8",
    )
    metadata = {
        "rank": item["rank"],
        "slug": item["slug"],
        "name": name,
        "license": license_name,
        "tags": tags,
        "input_name": input_name,
        "output_name": output_name,
        "input_shape": input_shape,
        "rdf_source": item["rdf_source"],
        "onnx_source": onnx_url,
    }
    (model_dir / "import-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (artifacts_dir / "README.md").write_text(
        (
            "Imported ONNX artifact downloaded from the BioImage RDF recorded in import-metadata.json.\n"
            if download_artifacts
            else "Scaffolded import. Fetch the BioImage ONNX artifact from import-metadata.json before indexing this model.\n"
        ),
        encoding="utf-8",
    )
    if download_artifacts:
        subprocess.run(
            [
                "curl",
                "-L",
                "--fail",
                "--retry",
                "3",
                "--continue-at",
                "-",
                "-o",
                str(artifacts_dir / "model.onnx"),
                onnx_url,
            ],
            check=True,
        )
        print(f"Imported {item['slug']}")
    else:
        (artifacts_dir / "source-url.txt").write_text(onnx_url + "\n", encoding="utf-8")
        print(f"Scaffolded {item['slug']}")


def update_index(root: Path) -> None:
    index_path = root / "biosim-index.yaml"
    lines = index_path.read_text(encoding="utf-8").splitlines()
    model_lines = [f"  - models/{item['slug']}/model.yaml" for item in BATCH]
    existing = set(lines)
    insert_at = 1
    for line in reversed(model_lines):
        if line not in existing:
            lines.insert(insert_at, line)
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_import_plan(root: Path, *, status: str, blocker: str) -> None:
    metadata_root = Path(
        os.environ.get("MODELS_ONNX_METADATA_DIR", root.parent / "models-onnx-metadata")
    )
    plan_path = metadata_root / "inventory" / "import-plan.csv"
    with plan_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = handle.readline()
    batch_ranks = {item["rank"] for item in BATCH}
    for row in rows:
        if row["rank"] in batch_ranks:
            row["import_status"] = status
            row["blocking_reason"] = blocker
    with plan_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download-artifacts",
        action="store_true",
        help="Download the published ONNX artifacts instead of creating metadata-only scaffolds.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    for item in BATCH:
        import_model(root, item, download_artifacts=args.download_artifacts)
    if args.download_artifacts:
        update_index(root)
        update_import_plan(root, status="imported", blocker="")
    else:
        update_import_plan(root, status="scaffolded", blocker="fetch-published-onnx-artifact")


if __name__ == "__main__":
    main()
