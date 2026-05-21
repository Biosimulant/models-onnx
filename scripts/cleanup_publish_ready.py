from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
LABS = ROOT / "labs"
TMP = ROOT / "tmp"


CONFIG: dict[str, dict[str, Any]] = {
    "bioimage-embryonet-base-model": {
        "slug": "bioimage-embryonet-embryo-stage-classification",
        "source_type": "BioImage ONNX",
        "context": "EmbryoNet embryo-stage classification from an imported BioImage.io ONNX graph.",
        "question": "What does the source ONNX graph emit for this preprocessed embryo image tensor?",
        "answer_style": "Report the top score index and tensor evidence without interpreting synthetic input as embryo biology.",
        "caveat": "Runtime smoke uses synthetic tensors unless source sample data is added; scores are graph outputs, not validated embryo-development evidence.",
        "visual_scope": "provenance, graph schema, development-score summary, top stage index",
        "init_ports": {
            "image_port": "embryo_image",
            "conv_time_port": "convolution_time_map",
            "fc_time_port": "stage_time_scalar",
            "output_port": "embryo_development_scores",
            "summary_port": "prediction_summary",
        },
        "inputs": ["embryo_image", "convolution_time_map", "stage_time_scalar"],
        "outputs": ["embryo_development_scores", "prediction_summary"],
        "dominant_module": "EmbryoNet classification head",
        "public_input_note": "Inputs are already-preprocessed ONNX tensors: `data`, `t_for_conv`, and `t_for_fc`.",
    },
    "bioimage-hpa-bestfitting-densenet": {
        "source_type": "BioImage ONNX",
        "context": "Human Protein Atlas image classification and feature embedding from a BioImage.io DenseNet ONNX graph.",
        "question": "What class-score and embedding tensors does this source HPA DenseNet graph emit for the supplied preprocessed cell image tensor?",
        "answer_style": "Report score and embedding statistics without inventing cell-label semantics.",
        "caveat": "Class labels are not asserted unless source metadata provides them; synthetic input validates runtime only.",
        "visual_scope": "provenance, graph schema, class-score stats, feature stats",
        "init_ports": {
            "input_port": "cell_image_tensor",
            "class_port": "classification_scores",
            "feature_port": "feature_embedding",
            "summary_port": "prediction_summary",
        },
        "inputs": ["cell_image_tensor"],
        "outputs": ["classification_scores", "feature_embedding", "prediction_summary"],
        "dominant_module": "HPA DenseNet classification and feature heads",
        "public_input_note": "Input maps to ONNX graph input `image` and is an already-preprocessed cell image tensor.",
    },
    "bioimage-hpa-bestfitting-inceptionv3": {
        "source_type": "BioImage ONNX",
        "context": "Human Protein Atlas image classification and feature embedding from a BioImage.io InceptionV3 ONNX graph.",
        "question": "What class-score and embedding tensors does this source HPA InceptionV3 graph emit for the supplied preprocessed cell image tensor?",
        "answer_style": "Report score and embedding statistics without inventing cell-label semantics.",
        "caveat": "Class labels are not asserted unless source metadata provides them; synthetic input validates runtime only.",
        "visual_scope": "provenance, graph schema, class-score stats, feature stats",
        "init_ports": {
            "input_port": "cell_image_tensor",
            "class_port": "classification_scores",
            "feature_port": "feature_embedding",
            "summary_port": "prediction_summary",
        },
        "inputs": ["cell_image_tensor"],
        "outputs": ["classification_scores", "feature_embedding", "prediction_summary"],
        "dominant_module": "HPA InceptionV3 classification and feature heads",
        "public_input_note": "Input maps to ONNX graph input `image` and is an already-preprocessed cell image tensor.",
    },
    "bioimage-hylfm-net-stat": {
        "slug": "bioimage-hylfm-light-field-reconstruction",
        "source_type": "BioImage ONNX",
        "context": "HyLFM light-field microscopy reconstruction from a BioImage.io ONNX graph.",
        "question": "What reconstruction tensor does the source HyLFM ONNX graph emit for this preprocessed light-field image tensor?",
        "answer_style": "Report reconstruction shape and value statistics as source graph output evidence.",
        "caveat": "Synthetic input validates wiring only; image quality requires source or experimental light-field data.",
        "visual_scope": "provenance, graph schema, reconstruction tensor stats and shape evidence",
        "init_ports": {
            "input_port": "light_field_image",
            "output_port": "reconstruction_volume",
            "summary_port": "prediction_summary",
        },
        "inputs": ["light_field_image"],
        "outputs": ["reconstruction_volume", "prediction_summary"],
        "dominant_module": "HyLFM reconstruction graph",
        "public_input_note": "Input maps to ONNX graph input `0` and is an already-preprocessed light-field image tensor.",
    },
    "bioimage-nucleisegmentationboundarymodel": {
        "slug": "bioimage-nuclei-segmentation-boundary",
        "source_type": "BioImage ONNX",
        "context": "Nuclei boundary segmentation from a BioImage.io ONNX graph.",
        "question": "What segmentation-logit tensor does this source nuclei-boundary ONNX graph emit for the supplied preprocessed nuclei image?",
        "answer_style": "Report logit tensor statistics and pixel-count evidence without claiming segmentation quality from synthetic input.",
        "caveat": "Synthetic input validates ONNX execution only; segmentation accuracy needs source validation images.",
        "visual_scope": "provenance, graph schema, segmentation tensor stats and mask/logit summary",
        "init_ports": {
            "input_port": "nuclei_image",
            "output_port": "segmentation_logits",
            "summary_port": "prediction_summary",
        },
        "inputs": ["nuclei_image"],
        "outputs": ["segmentation_logits", "prediction_summary"],
        "dominant_module": "Nuclei boundary segmentation graph",
        "public_input_note": "Input maps to ONNX graph input `input.1` and is an already-preprocessed nuclei image tensor.",
    },
    "monai-brats-mri-segmentation": {
        "source_type": "MONAI ONNX export",
        "context": "BRATS multimodal MRI segmentation exported from a MONAI bundle.",
        "question": "What segmentation prediction tensor does the source MONAI BRATS MRI ONNX export emit for this multimodal MRI volume tensor?",
        "answer_style": "Report output tensor shape, channel count, and value statistics.",
        "caveat": "Synthetic input validates ONNX execution only; clinical segmentation quality requires source validation data.",
        "visual_scope": "MRI channel schema, segmentation tensor stats",
        "init_ports": {"input_port": "multimodal_mri_volume", "output_port": "segmentation_prediction_tensor", "summary_port": "segmentation_summary"},
        "inputs": ["multimodal_mri_volume"],
        "outputs": ["segmentation_prediction_tensor", "segmentation_summary"],
        "dominant_module": "MONAI BRATS MRI segmentation graph",
        "public_input_note": "Input maps to ONNX graph input `image` and is a four-channel preprocessed MRI volume tensor.",
    },
    "monai-pancreas-ct-dints-segmentation": {
        "source_type": "MONAI ONNX export",
        "context": "Pancreas CT DiNTS segmentation exported from a MONAI bundle.",
        "question": "What segmentation prediction tensor does the source MONAI pancreas CT ONNX export emit for this CT volume tensor?",
        "answer_style": "Report output tensor shape, channel count, and value statistics.",
        "caveat": "Synthetic input validates ONNX execution only; clinical segmentation quality requires source validation data.",
        "visual_scope": "medical-imaging provenance, shape/schema, class/channel stats",
        "init_ports": {"input_port": "ct_volume", "output_port": "segmentation_prediction_tensor", "summary_port": "segmentation_summary"},
        "inputs": ["ct_volume"],
        "outputs": ["segmentation_prediction_tensor", "segmentation_summary"],
        "dominant_module": "MONAI pancreas CT segmentation graph",
        "public_input_note": "Input maps to ONNX graph input `image` and is a preprocessed CT volume tensor.",
    },
    "monai-spleen-ct-segmentation": {
        "source_type": "MONAI ONNX export",
        "context": "Spleen CT segmentation exported from a MONAI bundle.",
        "question": "What segmentation prediction tensor does the source MONAI spleen CT ONNX export emit for this CT volume tensor?",
        "answer_style": "Report output tensor shape, channel count, and value statistics.",
        "caveat": "Synthetic input validates ONNX execution only; clinical segmentation quality requires source validation data.",
        "visual_scope": "medical-imaging provenance, shape/schema, class/channel stats",
        "init_ports": {"input_port": "ct_volume", "output_port": "segmentation_prediction_tensor", "summary_port": "segmentation_summary"},
        "inputs": ["ct_volume"],
        "outputs": ["segmentation_prediction_tensor", "segmentation_summary"],
        "dominant_module": "MONAI spleen CT segmentation graph",
        "public_input_note": "Input maps to ONNX graph input `image` and is a preprocessed CT volume tensor.",
    },
    "monai-spleen-deepedit-annotation": {
        "source_type": "MONAI ONNX export",
        "context": "Spleen DeepEdit annotation exported from a MONAI bundle.",
        "question": "What annotation prediction tensor does the source MONAI DeepEdit ONNX export emit for this CT volume plus guidance-channel tensor?",
        "answer_style": "Report output tensor shape, channel count, and value statistics.",
        "caveat": "The three input channels include guidance channels from the source DeepEdit contract; synthetic input validates wiring only.",
        "visual_scope": "annotation input contract and output tensor stats",
        "init_ports": {"input_port": "ct_volume_with_guidance_channels", "output_port": "segmentation_prediction_tensor", "summary_port": "segmentation_summary"},
        "inputs": ["ct_volume_with_guidance_channels"],
        "outputs": ["segmentation_prediction_tensor", "segmentation_summary"],
        "dominant_module": "MONAI DeepEdit annotation graph",
        "public_input_note": "Input maps to ONNX graph input `image` and is a preprocessed CT volume with guidance channels.",
    },
    "monai-swin-unetr-btcv-segmentation": {
        "source_type": "MONAI ONNX export",
        "context": "BTCV multi-organ CT segmentation exported from a MONAI Swin UNETR bundle.",
        "question": "What segmentation prediction tensor does the source MONAI BTCV ONNX export emit for this CT volume tensor?",
        "answer_style": "Report output tensor shape, channel count, and value statistics.",
        "caveat": "Synthetic input validates ONNX execution only; organ-label interpretation requires source label metadata.",
        "visual_scope": "medical-imaging provenance, shape/schema, class/channel stats",
        "init_ports": {"input_port": "ct_volume", "output_port": "segmentation_prediction_tensor", "summary_port": "segmentation_summary"},
        "inputs": ["ct_volume"],
        "outputs": ["segmentation_prediction_tensor", "segmentation_summary"],
        "dominant_module": "MONAI BTCV Swin UNETR segmentation graph",
        "public_input_note": "Input maps to ONNX graph input `image` and is a preprocessed CT volume tensor.",
    },
    "hf-allenai-scibert-scivocab-uncased": {
        "source_type": "Hugging Face ONNX export",
        "context": "SciBERT scientific-text embedding exported to ONNX.",
        "question": "What embedding tensors does the SciBERT ONNX export emit for these already-tokenized scientific text inputs?",
        "answer_style": "Report sequence and pooled embedding statistics.",
        "caveat": "No tokenizer is bundled, so public inputs are token tensors rather than raw text.",
        "visual_scope": "tokenizer caveat, embedding norms/stats, schema",
        "init_ports": {"token_port": "scientific_text_token_ids", "mask_port": "scientific_text_attention_mask"},
        "inputs": ["scientific_text_token_ids", "scientific_text_attention_mask"],
        "outputs": ["sequence_embeddings", "pooled_embedding"],
        "dominant_module": "SciBERT embedding graph",
        "public_input_note": "Inputs map to ONNX `input_ids` and `attention_mask`; tokenizer preprocessing is not bundled.",
    },
    "hf-dmis-lab-biobert-base-cased-v1-1": {
        "source_type": "Hugging Face ONNX export",
        "context": "BioBERT biomedical-text embedding exported to ONNX.",
        "question": "What embedding tensors does the BioBERT ONNX export emit for these already-tokenized biomedical text inputs?",
        "answer_style": "Report sequence and pooled embedding statistics.",
        "caveat": "No tokenizer is bundled, so public inputs are token tensors rather than raw text.",
        "visual_scope": "tokenizer caveat, embedding norms/stats, schema",
        "init_ports": {"token_port": "biomedical_text_token_ids", "mask_port": "biomedical_text_attention_mask"},
        "inputs": ["biomedical_text_token_ids", "biomedical_text_attention_mask"],
        "outputs": ["sequence_embeddings", "pooled_embedding"],
        "dominant_module": "BioBERT embedding graph",
        "public_input_note": "Inputs map to ONNX `input_ids` and `attention_mask`; tokenizer preprocessing is not bundled.",
    },
    "hf-emilyalsentzer-bio-clinicalbert": {
        "source_type": "Hugging Face ONNX export",
        "context": "ClinicalBERT clinical-text embedding exported to ONNX.",
        "question": "What embedding tensors does the ClinicalBERT ONNX export emit for these already-tokenized clinical text inputs?",
        "answer_style": "Report sequence and pooled embedding statistics.",
        "caveat": "No tokenizer is bundled, so public inputs are token tensors rather than raw text.",
        "visual_scope": "tokenizer caveat, embedding norms/stats, schema",
        "init_ports": {"token_port": "clinical_text_token_ids", "mask_port": "clinical_text_attention_mask"},
        "inputs": ["clinical_text_token_ids", "clinical_text_attention_mask"],
        "outputs": ["sequence_embeddings", "pooled_embedding"],
        "dominant_module": "ClinicalBERT embedding graph",
        "public_input_note": "Inputs map to ONNX `input_ids` and `attention_mask`; tokenizer preprocessing is not bundled.",
    },
    "hf-facebook-esm2-t12-35m-ur50d": {
        "source_type": "Hugging Face ONNX export",
        "context": "ESM-2 t12 protein-language embedding exported to ONNX.",
        "question": "What protein embedding tensors does the ESM-2 t12 ONNX export emit for these already-tokenized protein sequence inputs?",
        "answer_style": "Report sequence and pooled embedding statistics.",
        "caveat": "No tokenizer is bundled, so public inputs are token tensors rather than amino-acid sequence strings.",
        "visual_scope": "protein-token caveat, embedding norms/stats, schema",
        "init_ports": {"token_port": "protein_sequence_token_ids", "mask_port": "protein_sequence_attention_mask"},
        "inputs": ["protein_sequence_token_ids", "protein_sequence_attention_mask"],
        "outputs": ["sequence_embeddings", "pooled_embedding"],
        "dominant_module": "ESM-2 protein embedding graph",
        "public_input_note": "Inputs map to ONNX `input_ids` and `attention_mask`; sequence tokenization is not bundled.",
    },
    "hf-facebook-esm2-t6-8m-ur50d": {
        "source_type": "Hugging Face ONNX export",
        "context": "ESM-2 t6 protein-language embedding exported to ONNX.",
        "question": "What protein embedding tensors does the ESM-2 t6 ONNX export emit for these already-tokenized protein sequence inputs?",
        "answer_style": "Report sequence and pooled embedding statistics.",
        "caveat": "No tokenizer is bundled, so public inputs are token tensors rather than amino-acid sequence strings.",
        "visual_scope": "protein-token caveat, embedding norms/stats, schema",
        "init_ports": {"token_port": "protein_sequence_token_ids", "mask_port": "protein_sequence_attention_mask"},
        "inputs": ["protein_sequence_token_ids", "protein_sequence_attention_mask"],
        "outputs": ["sequence_embeddings", "pooled_embedding"],
        "dominant_module": "ESM-2 protein embedding graph",
        "public_input_note": "Inputs map to ONNX `input_ids` and `attention_mask`; sequence tokenization is not bundled.",
    },
}


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
        f"dtype={value.get(\'dtype\', \'unknown\')}",
        f"n={value.get(\'element_count\', \'-\')}",
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
                                "title": f"{port} {preview.get(\'label\', \'tensor preview\')}",
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
                        items.append({"label": f"{port}.{row.get(\'label\', \'channel\')} mean", "value": float(mean)})
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


TEST_SOURCE = '''from __future__ import annotations

import importlib
import sys
from pathlib import Path

import yaml


CORE = Path(__file__).resolve().parents[1]
BSIM_SRC = Path("/Volumes/dem-ssd/imp/projects/Nitoons/Biosimulant/bsim-active/biosim/src")
if str(BSIM_SRC) not in sys.path:
    sys.path.insert(0, str(BSIM_SRC))
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))


def _load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


class _FakeValue:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSession:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        manifest = _load_yaml(CORE / "model.yaml")
        onnx = manifest.get("onnx", {})
        self._inputs = [_FakeValue(str(item["name"])) for item in onnx.get("inputs", [])]
        self._outputs = [_FakeValue(str(item["name"])) for item in onnx.get("outputs", [])]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, outputs, feed_dict):
        result = []
        for index, _name in enumerate(outputs):
            if index == 0:
                result.append([[0.1, 0.2, 0.3]])
            else:
                result.append([[0.4, 0.5, 0.6]])
        return result


class _Signal:
    def __init__(self, value):
        self.value = value


def _entrypoint():
    manifest = _load_yaml(CORE / "model.yaml")
    module_name, attr = manifest["biosim"]["entrypoint"].split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr), manifest


def test_core_wrapper_emits_declared_outputs_with_fake_session() -> None:
    cls, manifest = _entrypoint()
    kwargs = dict(manifest["biosim"].get("init_kwargs", {}))
    kwargs["session_factory"] = _FakeSession
    for key in ("input_shape", "image_shape", "conv_time_shape", "fc_time_shape", "output_shape"):
        if key in kwargs:
            kwargs[key] = [1] if key != "fc_time_shape" else [1, 1]
    module = cls(**kwargs)
    signals = {}
    for name in module.inputs():
        signals[name] = _Signal([1])
    module.set_inputs(signals)
    module.advance_window(0.0, 0.1)
    outputs = module.get_outputs()
    assert set(manifest["io"]["outputs"]) <= set(outputs)
    assert all(signal.value is not None for signal in outputs.values())
'''


def slug_to_alias(slug: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", slug).strip("_").lower()


def load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_yaml(path: Path, data: Any) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=False), encoding="utf-8")


def schema_for_output(port: str, model_yaml: dict[str, Any]) -> dict[str, str]:
    for item in model_yaml.get("io", {}).get("outputs", []):
        if isinstance(item, dict) and item.get("name") == port:
            schema = item.get("schema")
            if isinstance(schema, dict):
                return {str(k): str(v) for k, v in schema.items()}
    if port == "prediction_summary":
        output_names = {str(item.get("name")) for item in model_yaml.get("onnx", {}).get("outputs", [])}
        task = str(model_yaml.get("onnx", {}).get("task", ""))
        if task == "classification" and "output" in output_names:
            return {"top_stage_index": "int", "top_score": "float"}
        if task == "classification":
            return {"top_class_index": "int", "top_score": "float"}
        if task == "segmentation":
            return {"mean_logit": "float", "pixel_count": "int"}
        if task == "image-reconstruction":
            return {"mean_value": "float", "voxel_count": "int"}
    if port == "segmentation_summary":
        return {"mean_probability": "float", "voxel_count": "int"}
    return {"payload": "json"}


def normalize_io(model_yaml: dict[str, Any], cfg: dict[str, Any]) -> None:
    model_yaml.setdefault("biosim", {}).setdefault("init_kwargs", {}).update(cfg["init_ports"])
    model_yaml["io"] = {
        "inputs": [
            {
                "name": name,
                "maps_to": graph_name,
                "description": cfg["public_input_note"],
            }
            for name, graph_name in zip(cfg["inputs"], [str(item.get("name")) for item in model_yaml.get("onnx", {}).get("inputs", [])])
        ],
        "outputs": [
            {
                "name": name,
                "maps_to": str((model_yaml.get("onnx", {}).get("outputs", [{}]) + [{}])[min(index, len(model_yaml.get("onnx", {}).get("outputs", [])) - 1)].get("name", name))
                if model_yaml.get("onnx", {}).get("outputs")
                else name,
                "description": output_description(name, cfg),
            }
            for index, name in enumerate(cfg["outputs"])
        ],
    }


def output_description(name: str, cfg: dict[str, Any]) -> str:
    if name == "prediction_summary":
        return "Compact summary derived from the ONNX graph output tensor."
    if name == "segmentation_summary":
        return "Shape and value summary derived from the ONNX segmentation output tensor."
    if "embedding" in name:
        return "Compact embedding evidence derived from the source ONNX graph output."
    return "Compact tensor evidence derived from the source-derived ONNX graph output."


def visual_manifest(model_yaml: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    visual_inputs = {}
    for port in cfg["outputs"]:
        visual_inputs[port] = schema_for_output(port, model_yaml)
    return {
        "schema_version": "2.0",
        "title": f"{model_yaml.get('title', 'ONNX model')} Visualisation",
        "description": "Publication-facing visualisation for source-faithful ONNX runtime evidence.",
        "standard": "other",
        "tags": ["onnx", "visualisation", "publish-ready"],
        "authors": ["Biosimulant Team"],
        "biosim": {
            "communication_step": 0.01,
            "entrypoint": "src.onnx_visualisation:OnnxPublicationVisualisation",
            "init_kwargs": {
                "lab_title": model_yaml.get("title", "ONNX lab"),
                "source_type": cfg["source_type"],
                "task": str(model_yaml.get("onnx", {}).get("task", "onnx-inference")),
                "question": cfg["question"],
                "answer_style": cfg["answer_style"],
                "caveat": cfg["caveat"],
                "dominant_module": cfg["dominant_module"],
                "public_input_note": cfg["public_input_note"],
                "input_contracts": model_yaml.get("onnx", {}).get("inputs", []),
                "output_contracts": model_yaml.get("onnx", {}).get("outputs", []),
                "visual_inputs": visual_inputs,
            },
        },
        "io": {"inputs": [{"name": name, "description": f"Runtime evidence from core output `{name}`."} for name in cfg["outputs"]], "outputs": []},
    }


def lab_manifest(slug: str, model_yaml: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    alias = slug_to_alias(slug)
    return {
        "schema_version": "2.0",
        "title": f"{model_yaml.get('title', slug)} Lab",
        "description": f"{cfg['context']} This lab exposes source-faithful ONNX runtime evidence without claiming scientific validity beyond the source artifact.",
        "models": [
            {"alias": alias, "path": "models/core"},
            {"alias": "visualisation", "path": "models/visualisation"},
        ],
        "wiring": [{"from": f"{alias}.{port}", "to": [f"visualisation.{port}"]} for port in cfg["outputs"]],
        "runtime": {"duration": 0.02, "communication_step": 0.01, "initial_inputs": {}},
    }


def readme_text(slug: str, model_yaml: dict[str, Any], cfg: dict[str, Any]) -> str:
    source = model_yaml.get("source", {})
    source_lines = "\n".join(f"- `{key}`: {value}" for key, value in source.items()) or "- Source metadata unavailable."
    inputs = "\n".join(f"- `{name}`: {cfg['public_input_note']}" for name in cfg["inputs"])
    outputs = "\n".join(f"- `{name}`: {output_description(name, cfg)}" for name in cfg["outputs"])
    return f"""# {model_yaml.get('title', slug)}

This Biosimulant lab wraps a source-derived ONNX artifact and reports runtime evidence from the bundled graph.

## Scientific Context

{cfg['context']}

## Source And Artifact

{source_lines}

- ONNX artifact: `models/core/artifacts/model.onnx`
- Runtime: ONNX Runtime CPU execution provider
- Validation scope: graph load, input/output contract, finite synthetic inference, Biosimulant wiring, and visual payloads

## Inputs

{inputs}

The lab does not add raw image, raw text, raw protein sequence, tokenizer, or medical-image preprocessing unless that
preprocessing is bundled and validated. Public inputs are already-preprocessed tensors matching the ONNX graph contract.

## Outputs

{outputs}

## Visualisations

The visualisation answers:

> {cfg['question']}

It shows provenance, graph schema, runtime tensor evidence, and a conservative caveat. Synthetic inference is a runtime
smoke test, not biological, clinical, or microscopy performance evidence.

<!-- BIOSIMULANT_VISUALS_START -->
<!-- BIOSIMULANT_VISUALS_END -->

## Caveat

{cfg['caveat']}
"""


def planning_rows(final_labs: list[tuple[str, str, dict[str, Any], dict[str, Any]]]) -> list[dict[str, str]]:
    rows = []
    for old_slug, new_slug, model_yaml, cfg in final_labs:
        source = model_yaml.get("source", {})
        source_artifact = "models/core/artifacts/model.onnx"
        input_map = "; ".join(
            f"{public}->{graph.get('name')} ({graph.get('dtype')}, {graph.get('shape')})"
            for public, graph in zip(cfg["inputs"], model_yaml.get("onnx", {}).get("inputs", []))
        )
        output_map = "; ".join(
            f"{public}->{graph.get('name')} ({graph.get('dtype')}, {graph.get('shape')})"
            for public, graph in zip(cfg["outputs"], model_yaml.get("onnx", {}).get("outputs", []))
        )
        rows.append(
            {
                "current_lab_folder": old_slug,
                "proposed_lab_slug": new_slug,
                "source_type": cfg["source_type"],
                "upstream_source": json.dumps(source, sort_keys=True),
                "scientific_context": cfg["context"],
                "keep_fix_orphan": "keep",
                "reason": "Bundled ONNX artifact and provenance metadata are present; runtime validation required before publication.",
                "source_artifact": source_artifact,
                "runtime_wrapper": str(model_yaml.get("biosim", {}).get("entrypoint", "")),
                "proposed_public_inputs": input_map,
                "proposed_public_outputs": output_map,
                "visualisation_scope": cfg["visual_scope"],
                "scientific_question": cfg["question"],
                "answer_style": cfg["answer_style"],
                "caveats": cfg["caveat"],
                "validation_required": "onnxruntime graph load; synthetic inference; finite outputs; visual smoke; README asset audit",
            }
        )
    return rows


def main() -> None:
    TMP.mkdir(exist_ok=True)
    labs_to_process: list[tuple[str, str, Path, dict[str, Any], dict[str, Any]]] = []
    for old_slug, cfg in sorted(CONFIG.items()):
        current = LABS / old_slug
        new_slug = cfg.get("slug", old_slug)
        target = LABS / new_slug
        lab_dir = target if target.exists() else current
        if not lab_dir.exists():
            raise FileNotFoundError(f"Missing lab directory for {old_slug}: {lab_dir}")
        model_dir = lab_dir / "model"
        core_dir = lab_dir / "models" / "core"
        if not model_dir.exists() and core_dir.exists():
            model_dir = core_dir
        model_yaml = load_yaml(model_dir / "model.yaml")
        labs_to_process.append((old_slug, new_slug, lab_dir, model_yaml, cfg))

    rows = planning_rows([(old, new, model, cfg) for old, new, _lab, model, cfg in labs_to_process])
    plan_path = TMP / "publish_cleanup_plan.csv"
    with plan_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for old_slug, new_slug, lab_dir, model_yaml, cfg in labs_to_process:
        target = LABS / new_slug
        if lab_dir != target:
            if target.exists():
                raise FileExistsError(f"Cannot rename {lab_dir} to existing {target}")
            shutil.move(str(lab_dir), str(target))
            lab_dir = target

        old_model = lab_dir / "model"
        core = lab_dir / "models" / "core"
        visual = lab_dir / "models" / "visualisation"
        assets = lab_dir / "assets"
        if old_model.exists():
            core.parent.mkdir(exist_ok=True)
            shutil.move(str(old_model), str(core))
        core.mkdir(parents=True, exist_ok=True)
        visual.mkdir(parents=True, exist_ok=True)
        assets.mkdir(exist_ok=True)
        (visual / "src").mkdir(exist_ok=True)
        (visual / "tests").mkdir(exist_ok=True)

        model_yaml = load_yaml(core / "model.yaml")
        normalize_io(model_yaml, cfg)
        write_yaml(core / "model.yaml", model_yaml)
        (visual / "src" / "onnx_visualisation.py").write_text(VISUAL_SOURCE, encoding="utf-8")
        write_yaml(visual / "model.yaml", visual_manifest(model_yaml, cfg))
        write_yaml(lab_dir / "lab.yaml", lab_manifest(new_slug, model_yaml, cfg))
        (lab_dir / "README.md").write_text(readme_text(new_slug, model_yaml, cfg), encoding="utf-8")
        (core / "tests" / "test_core_wrapper.py").write_text(TEST_SOURCE, encoding="utf-8")

    print(f"Wrote {plan_path}")
    print(f"Processed {len(labs_to_process)} ONNX labs")


if __name__ == "__main__":
    main()
