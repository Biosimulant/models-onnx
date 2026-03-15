"""Export an ESM checkpoint to ONNX and validate it with ONNX Runtime."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoTokenizer, EsmModel


class _EsmExportWrapper(torch.nn.Module):
    def __init__(self, model: EsmModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):  # type: ignore[override]
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0, :]
        return outputs.last_hidden_state, pooled


def export(model_id: str, output_path: Path, sample_sequence: str) -> dict[str, object]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = EsmModel.from_pretrained(model_id)
    model.eval()
    wrapped = _EsmExportWrapper(model)

    encoded = tokenizer(sample_sequence, return_tensors="pt")
    inputs = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            (inputs["input_ids"], inputs["attention_mask"]),
            str(output_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state", "pooled_output"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "last_hidden_state": {0: "batch", 1: "sequence"},
                "pooled_output": {0: "batch"},
            },
            opset_version=17,
        )

    ref_outputs = wrapped(inputs["input_ids"], inputs["attention_mask"])
    ref_last_hidden = ref_outputs[0].detach().cpu().numpy()
    ref_pooled = ref_outputs[1].detach().cpu().numpy()

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    ort_outputs = session.run(
        ["last_hidden_state", "pooled_output"],
        {
            "input_ids": inputs["input_ids"].cpu().numpy(),
            "attention_mask": inputs["attention_mask"].cpu().numpy(),
        },
    )
    np.testing.assert_allclose(ort_outputs[0], ref_last_hidden, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(ort_outputs[1], ref_pooled, atol=1e-4, rtol=1e-4)

    return {
        "source_model_id": model_id,
        "artifact": str(output_path),
        "artifact_size_bytes": output_path.stat().st_size,
        "inputs": [
            {"name": x.name, "shape": x.shape, "dtype": x.type}
            for x in session.get_inputs()
        ],
        "outputs": [
            {"name": x.name, "shape": x.shape, "dtype": x.type}
            for x in session.get_outputs()
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output", required=True)
    parser.add_argument(
        "--sample-sequence",
        default="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQLR",
        help="Protein sequence used to build dummy inputs for export validation.",
    )
    args = parser.parse_args()

    info = export(args.model_id, Path(args.output), args.sample_sequence)
    metadata_path = Path(args.metadata_output)
    metadata_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
