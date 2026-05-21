from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LABS = ROOT / "labs"


def main() -> None:
    failures: list[str] = []
    for lab_dir in sorted(p for p in LABS.iterdir() if p.is_dir()):
        readme = lab_dir / "README.md"
        assets = lab_dir / "assets"
        if not readme.exists():
            failures.append(f"{lab_dir.name}: missing README.md")
            continue
        text = readme.read_text(encoding="utf-8")
        if "BIOSIMULANT_VISUALS_START" not in text or "BIOSIMULANT_VISUALS_END" not in text:
            failures.append(f"{lab_dir.name}: missing README visual markers")
        for png in sorted(assets.glob("*.png")):
            rel = f"assets/{png.name}"
            if rel not in text:
                failures.append(f"{lab_dir.name}: generated asset {rel} is not referenced")
    if failures:
        raise SystemExit("\n".join(failures))
    print("README asset audit passed.")


if __name__ == "__main__":
    main()
