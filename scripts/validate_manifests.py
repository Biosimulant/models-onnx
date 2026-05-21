from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
LABS = ROOT / "labs"


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def main() -> None:
    model_count = 0
    lab_count = 0
    failures: list[str] = []
    for lab_dir in sorted(p for p in LABS.iterdir() if p.is_dir()):
        lab_count += 1
        lab_yaml_path = lab_dir / "lab.yaml"
        if not lab_yaml_path.exists():
            failures.append(f"{lab_dir.name}: missing lab.yaml")
            continue
        lab = load_yaml(lab_yaml_path)
        paths = {item.get("path") for item in lab.get("models", [])}
        if "models/core" not in paths:
            failures.append(f"{lab_dir.name}: missing models/core in lab.yaml")
        if "models/visualisation" not in paths:
            failures.append(f"{lab_dir.name}: missing models/visualisation in lab.yaml")
        for item in lab.get("models", []):
            model_dir = lab_dir / str(item.get("path", ""))
            manifest_path = model_dir / "model.yaml"
            if not manifest_path.exists():
                failures.append(f"{lab_dir.name}: missing model manifest at {item.get('path')}")
                continue
            model_count += 1
            manifest = load_yaml(manifest_path)
            if "biosim" not in manifest or "entrypoint" not in manifest["biosim"]:
                failures.append(f"{manifest_path}: missing biosim.entrypoint")
            if item.get("path") == "models/core":
                if manifest.get("standard") != "onnx":
                    failures.append(f"{manifest_path}: core standard is not onnx")
                artifact = model_dir / str(manifest.get("onnx", {}).get("model_file", "artifacts/model.onnx"))
                if not artifact.exists():
                    failures.append(f"{manifest_path}: missing ONNX artifact {artifact}")
                if not manifest.get("source"):
                    failures.append(f"{manifest_path}: missing source provenance")
                public_inputs = [entry.get("name") if isinstance(entry, dict) else entry for entry in manifest.get("io", {}).get("inputs", [])]
                if not public_inputs:
                    failures.append(f"{manifest_path}: no public inputs declared")
                for entry in manifest.get("io", {}).get("inputs", []):
                    if not isinstance(entry, dict) or not entry.get("maps_to") or not entry.get("description"):
                        failures.append(f"{manifest_path}: input lacks traceable maps_to/description: {entry!r}")
        wired = lab.get("wiring", [])
        if not wired:
            failures.append(f"{lab_dir.name}: missing core-to-visualisation wiring")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Validated {model_count} model manifest(s) and {lab_count} lab manifest(s).")


if __name__ == "__main__":
    main()
