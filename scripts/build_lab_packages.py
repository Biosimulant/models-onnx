#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build portable Biosimulant lab packages for the ONNX lab repository.

This script is intentionally small and deterministic so CI can stage large ONNX
labs without relying on a locally installed Biosimulant desktop binary.  It
preserves each lab's source artifacts, rewrites embedded model paths to the
portable `owned/models/...` layout, and emits `.bsilab` archives accepted by the
Hub package registry.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


EXCLUDED_NAMES = {
    ".DS_Store",
    ".biosimulant-project.json",
}
EXCLUDED_PARTS = {
    "__pycache__",
    ".pytest_cache",
}


@dataclass(frozen=True)
class BuiltPackage:
    lab_slug: str
    package_name: str
    version: str
    output_path: Path
    logical_sha256: str
    size_bytes: int


def slugify_for_package_name(title: str) -> str:
    value = title.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    if not value:
        raise ValueError("Cannot derive package name from empty title")
    return value


def sanitize_owned_name(alias: str) -> str:
    value = alias.strip().replace("\\", "/").split("/")[-1]
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    if not value:
        raise ValueError(f"Cannot derive owned model name from alias {alias!r}")
    return value


def should_skip(path: Path) -> bool:
    if path.name in EXCLUDED_NAMES:
        return True
    return any(part in EXCLUDED_PARTS for part in path.parts)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def dump_yaml(data: dict[str, Any]) -> bytes:
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=False)
    return text.encode("utf-8")


def copy_tree_filtered(source: Path, target: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Expected directory: {source}")
    for item in sorted(source.rglob("*")):
        rel = item.relative_to(source)
        if should_skip(rel) or should_skip(item):
            continue
        destination = target / rel
        if item.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        else:
            if item.suffix == ".onnx" and is_git_lfs_pointer(item):
                raise ValueError(
                    f"{item} is a Git LFS pointer, not an ONNX artifact. "
                    "Run `git lfs pull` before packaging."
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, destination)


def is_git_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            prefix = handle.read(128)
    except OSError:
        return False
    return prefix.startswith(b"version https://git-lfs.github.com/spec/v1")


def stage_lab(lab_dir: Path, staging_dir: Path) -> dict[str, Any]:
    lab_manifest = read_yaml(lab_dir / "lab.yaml")
    payload_dir = staging_dir / "payload"
    payload_dir.mkdir(parents=True, exist_ok=True)

    for item in sorted(lab_dir.iterdir()):
        if should_skip(item):
            continue
        if item.name in {"lab.yaml", "models"}:
            continue
        destination = payload_dir / item.name
        if item.is_dir():
            copy_tree_filtered(item, destination)
        elif item.is_file():
            shutil.copy2(item, destination)

    rewritten_models: list[dict[str, Any]] = []
    referenced_names: set[str] = set()
    for model in lab_manifest.get("models", []):
        if not isinstance(model, dict):
            raise ValueError(f"{lab_dir}/lab.yaml contains a non-mapping model entry")
        alias = str(model.get("alias") or "")
        model_path = str(model.get("path") or "")
        if not alias or not model_path:
            raise ValueError(f"{lab_dir}/lab.yaml model entries require alias and path")
        source_model_dir = lab_dir / model_path
        owned_name = sanitize_owned_name(alias)
        if owned_name in referenced_names:
            raise ValueError(f"{lab_dir}/lab.yaml has duplicate owned model name {owned_name}")
        referenced_names.add(owned_name)
        target_rel = Path("owned") / "models" / owned_name
        copy_tree_filtered(source_model_dir, payload_dir / target_rel)

        rewritten = dict(model)
        rewritten.pop("export", None)
        provenance = rewritten.get("provenance")
        if not isinstance(provenance, dict):
            provenance = {}
        provenance["owned_path"] = target_rel.as_posix()
        rewritten["provenance"] = provenance
        rewritten["path"] = target_rel.as_posix()
        rewritten_models.append(rewritten)

    lab_manifest["models"] = rewritten_models
    (payload_dir / "lab.yaml").write_bytes(dump_yaml(lab_manifest))
    return lab_manifest


def package_entries(staging_dir: Path) -> dict[str, bytes]:
    entries: dict[str, bytes] = {}
    for file_path in sorted(staging_dir.rglob("*")):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(staging_dir).as_posix()
        if should_skip(Path(rel)):
            continue
        entries[rel] = file_path.read_bytes()
    return entries


def logical_sha256(entries: dict[str, bytes]) -> str:
    hasher = hashlib.sha256()
    for name in sorted(entries):
        if name in {"package.yaml", "integrity/sha256sums.txt"}:
            continue
        if name.rsplit("/", 1)[-1] == ".biosimulant-project.json":
            continue
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(entries[name])
        hasher.update(b"\0")
    return hasher.hexdigest()


def checksums_text(entries: dict[str, bytes]) -> bytes:
    lines: list[str] = []
    for name in sorted(entries):
        if name == "integrity/sha256sums.txt":
            continue
        digest = hashlib.sha256(entries[name]).hexdigest()
        lines.append(f"{digest}  {name}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def finalize_entries(
    entries: dict[str, bytes],
    *,
    title: str,
    description: str | None,
    package_name: str,
    version: str,
) -> tuple[dict[str, bytes], str]:
    logical_sha = logical_sha256(entries)
    package_yaml = {
        "schema_version": "1.0",
        "package_type": "lab",
        "title": title,
        "package": package_name,
        "version": version,
        "description": description,
        "visibility": "private",
        "entry_manifest": "payload/lab.yaml",
        "sha256": logical_sha,
        "tags": [],
        "authors": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    entries = dict(entries)
    entries["package.yaml"] = dump_yaml(package_yaml)
    entries["integrity/sha256sums.txt"] = checksums_text(entries)
    return entries, logical_sha


def write_zip(entries: dict[str, bytes], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(entries):
            archive.writestr(name, entries[name])
    return output_path.stat().st_size


def build_lab_package(lab_dir: Path, output_dir: Path, version: str) -> BuiltPackage:
    if not (lab_dir / "lab.yaml").is_file():
        raise FileNotFoundError(f"Missing lab.yaml in {lab_dir}")
    staging_dir = output_dir / ".staging" / lab_dir.name
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    try:
        lab_manifest = stage_lab(lab_dir, staging_dir)
        title = str(lab_manifest.get("title") or lab_dir.name)
        description = lab_manifest.get("description")
        if description is not None:
            description = str(description)
        package_name = slugify_for_package_name(title)
        entries = package_entries(staging_dir)
        finalized, logical_sha = finalize_entries(
            entries,
            title=title,
            description=description,
            package_name=package_name,
            version=version,
        )
        output_path = output_dir / f"{package_name}.bsilab"
        size = write_zip(finalized, output_path)
        return BuiltPackage(
            lab_slug=lab_dir.name,
            package_name=package_name,
            version=version,
            output_path=output_path,
            logical_sha256=logical_sha,
            size_bytes=size,
        )
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labs-dir",
        type=Path,
        default=Path("labs"),
        help="Directory containing lab folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/release-packages"),
        help="Directory where .bsilab packages will be written.",
    )
    parser.add_argument(
        "--version",
        default=f"0.1.0+{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        help="Package version to stamp into package.yaml.",
    )
    parser.add_argument(
        "labs",
        nargs="*",
        help="Optional lab slugs to package. Defaults to every lab in --labs-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labs_dir = args.labs_dir.resolve()
    if args.labs:
        lab_dirs = [labs_dir / slug for slug in args.labs]
    else:
        lab_dirs = sorted(path for path in labs_dir.iterdir() if (path / "lab.yaml").is_file())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    built = [build_lab_package(lab_dir, args.output_dir.resolve(), args.version) for lab_dir in lab_dirs]
    for package in built:
        print(
            "\t".join(
                [
                    package.lab_slug,
                    package.package_name,
                    package.version,
                    str(package.output_path),
                    package.logical_sha256,
                    str(package.size_bytes),
                ]
            )
        )


if __name__ == "__main__":
    main()
