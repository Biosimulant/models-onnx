from __future__ import annotations

import importlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
LABS = ROOT / "labs"
BSIM_SRC = Path("/Volumes/dem-ssd/imp/projects/Nitoons/Biosimulant/bsim-active/biosim/src")


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def split_entrypoint(entrypoint: str) -> tuple[str, str]:
    return tuple(entrypoint.split(":", 1)) if ":" in entrypoint else tuple(entrypoint.rsplit(".", 1))  # type: ignore[return-value]


def clear_src() -> None:
    for key in [name for name in sys.modules if name == "src" or name.startswith("src.")]:
        sys.modules.pop(key, None)


@contextmanager
def import_root(path: Path):
    sys.path.insert(0, str(BSIM_SRC))
    sys.path.insert(0, str(path))
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)
        sys.path = [item for item in sys.path if item not in {str(path), str(BSIM_SRC)}]
        clear_src()


def main() -> None:
    from biosim import BioModule

    count = 0
    failures: list[str] = []
    for manifest_path in sorted(LABS.glob("*/models/*/model.yaml")):
        manifest = load_yaml(manifest_path)
        module_name, attr = split_entrypoint(manifest["biosim"]["entrypoint"])
        try:
            with import_root(manifest_path.parent):
                cls = getattr(importlib.import_module(module_name), attr)
                module = cls(**manifest["biosim"].get("init_kwargs", {}))
                if not isinstance(module, BioModule):
                    failures.append(f"{manifest_path}: entrypoint did not instantiate BioModule")
        except Exception as exc:
            failures.append(f"{manifest_path}: {exc}")
        count += 1
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Validated entrypoints for {count} model manifest(s).")


if __name__ == "__main__":
    main()
