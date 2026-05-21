from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
LABS = ROOT / "labs"

GENERIC_INPUTS = {"input", "image", "volume_tensor", "input_tensor", "token_ids", "attention_mask", "lf_image"}
FORBIDDEN_LABEL_CLAIMS = {"disease", "diagnosis", "benign", "malignant", "tumor grade", "cell type label"}


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def main() -> None:
    failures: list[str] = []
    labs = sorted(p for p in LABS.iterdir() if p.is_dir())
    if len(labs) != 15:
        failures.append(f"expected 15 labs, found {len(labs)}")
    if list(LABS.glob("*/model")):
        failures.append("legacy top-level model directories remain under labs")
    for lab_dir in labs:
        if not (lab_dir / "README.md").exists():
            failures.append(f"{lab_dir.name}: missing README.md")
        if not (lab_dir / "assets").is_dir():
            failures.append(f"{lab_dir.name}: missing assets directory")
        core = load_yaml(lab_dir / "models/core/model.yaml")
        visual = load_yaml(lab_dir / "models/visualisation/model.yaml")
        graph_inputs = {str(item.get("name")) for item in core.get("onnx", {}).get("inputs", [])}
        graph_outputs = {str(item.get("name")) for item in core.get("onnx", {}).get("outputs", [])}
        for item in core.get("io", {}).get("inputs", []):
            name = item.get("name") if isinstance(item, dict) else item
            if name in GENERIC_INPUTS:
                failures.append(f"{lab_dir.name}: generic public input {name}")
            maps_to = str(item.get("maps_to", "")) if isinstance(item, dict) else ""
            if maps_to not in graph_inputs:
                failures.append(f"{lab_dir.name}: input {name} maps to missing graph input {maps_to}")
        for item in core.get("io", {}).get("outputs", []):
            maps_to = str(item.get("maps_to", "")) if isinstance(item, dict) else ""
            if maps_to and maps_to not in graph_outputs:
                failures.append(f"{lab_dir.name}: output {item.get('name')} maps to missing graph output {maps_to}")
            text = f"{item.get('name', '')} {item.get('description', '')}".lower() if isinstance(item, dict) else str(item).lower()
            if any(claim in text for claim in FORBIDDEN_LABEL_CLAIMS):
                failures.append(f"{lab_dir.name}: output may contain unsupported biological/clinical label claim")
        rows = visual.get("biosim", {}).get("init_kwargs", {})
        if "synthetic" not in str(rows.get("caveat", "")).lower() and "tokenizer" not in str(rows.get("caveat", "")).lower():
            failures.append(f"{lab_dir.name}: visual caveat does not state synthetic/tokenizer limitation")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Publish-ready audit passed for {len(labs)} ONNX lab(s).")


if __name__ == "__main__":
    main()
