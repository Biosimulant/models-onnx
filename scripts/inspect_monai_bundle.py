"""Inspect a MONAI NGC bundle and record whether it already contains ONNX."""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path


def inspect_bundle(source_url: str) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as td:
        archive_path = Path(td) / "bundle.zip"
        subprocess.run(
            ["curl", "-L", "--fail", "--retry", "3", "-o", str(archive_path), source_url],
            check=True,
        )
        with zipfile.ZipFile(archive_path) as zf:
            names = sorted(zf.namelist())
            return {
                "source_url": source_url,
                "archive_size_bytes": archive_path.stat().st_size,
                "archive_files": names,
                "contains_onnx": any(name.endswith(".onnx") for name in names),
                "contains_torchscript": any(name.endswith(".ts") for name in names),
                "contains_pytorch": any(name.endswith(".pt") or name.endswith(".pth") for name in names),
            }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_url", help="NGC bundle URL")
    parser.add_argument("output_json", help="Path to write inspection JSON")
    args = parser.parse_args()

    result = inspect_bundle(args.source_url)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
