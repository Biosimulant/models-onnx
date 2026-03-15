"""Generate the actionable import plan for the active 15-model shortlist."""
from __future__ import annotations

import csv
import os
from pathlib import Path


def map_row(row: dict[str, str]) -> dict[str, str]:
    bucket = row["onnx_readiness_bucket"]
    if bucket == "standardized-near-ready":
        family = "bioimage-onnx-image-model"
        batch = "B1-bioimage"
        status = "planned"
        blocker = "verify-weight-format"
    elif bucket == "convertible-needs-export":
        family = "monai-onnx-image-model"
        batch = "B2-monai"
        status = "planned"
        blocker = "inspect-bundle-and-export"
    elif bucket == "likely-convertible-transformer":
        family = "transformer-onnx-sequence-encoder"
        batch = "B3-transformers"
        status = "planned"
        blocker = "export-and-preprocessing-contract"
    else:
        raise ValueError(f"Unsupported shortlist bucket: {bucket}")

    return {
        "rank": row["rank"],
        "proposed_slug": row["proposed_slug"],
        "source_collection": row["source_collection"],
        "source_name": row["source_name"],
        "onnx_readiness_bucket": bucket,
        "implementation_family": family,
        "target_model_path": f"models/{row['proposed_slug']}/model.yaml",
        "import_batch": batch,
        "import_status": status,
        "blocking_reason": blocker,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    metadata_root = Path(
        os.environ.get("MODELS_ONNX_METADATA_DIR", root.parent / "models-onnx-metadata")
    )
    triage_path = metadata_root / "inventory" / "published-candidates-triage.csv"
    out_path = metadata_root / "inventory" / "import-plan.csv"

    with triage_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    mapped = [map_row(row) for row in rows]
    fieldnames = [
        "rank",
        "proposed_slug",
        "source_collection",
        "source_name",
        "onnx_readiness_bucket",
        "implementation_family",
        "target_model_path",
        "import_batch",
        "import_status",
        "blocking_reason",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mapped)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
