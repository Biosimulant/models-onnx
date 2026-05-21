from __future__ import annotations

import importlib
import os
import re
import sys
import textwrap
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFont

import validate_onnx_runtime as runtime


ROOT = Path(__file__).resolve().parents[1]
LABS = ROOT / "labs"
BSIM_SRC = Path("/Volumes/dem-ssd/imp/projects/Nitoons/Biosimulant/bsim-active/biosim/src")
README_START = "<!-- BIOSIMULANT_VISUALS_START -->"
README_END = "<!-- BIOSIMULANT_VISUALS_END -->"


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


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


def slugify(value: str, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:80]
    return slug or fallback


def collect_visuals(lab_dir: Path) -> list[dict[str, Any]]:
    core_manifest = load_yaml(lab_dir / "models/core/model.yaml")
    visual_manifest = load_yaml(lab_dir / "models/visualisation/model.yaml")
    lab = load_yaml(lab_dir / "lab.yaml")
    core_alias = next(item["alias"] for item in lab["models"] if item["path"] == "models/core")

    with import_root(lab_dir / "models/core"):
        module_name, attr = runtime.split_entrypoint(core_manifest["biosim"]["entrypoint"])
        core_cls = getattr(importlib.import_module(module_name), attr)
        core = core_cls(**core_manifest["biosim"].get("init_kwargs", {}))
        inputs = {name: runtime.sample_signal(name, spec, core_alias) for name, spec in core.inputs().items()}
        core.set_inputs(inputs)
        core.advance_window(0.0, 0.01)
        outputs = core.get_outputs()

    with import_root(lab_dir / "models/visualisation"):
        module_name, attr = runtime.split_entrypoint(visual_manifest["biosim"]["entrypoint"])
        visual_cls = getattr(importlib.import_module(module_name), attr)
        visual = visual_cls(**visual_manifest["biosim"].get("init_kwargs", {}))
        visual.setup({})
        visual.set_inputs(outputs)
        visual.advance_window(0.0, 0.01)
        return list(visual.visualize() or [])


def font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, *, width: int, fill: str, fnt, line_gap: int = 8) -> int:
    x, y = xy
    approx = max(20, width // max(8, fnt.size // 2))
    for paragraph in str(text).splitlines() or [""]:
        for line in textwrap.wrap(paragraph, width=approx) or [""]:
            draw.text((x, y), line, fill=fill, font=fnt)
            y += fnt.size + line_gap
        y += line_gap
    return y


def render_visual(path: Path, visual: dict[str, Any]) -> None:
    data = visual.get("data") or {}
    title = str(data.get("title") or visual.get("description") or "ONNX visual")
    rows = data.get("rows") or []
    columns = data.get("columns") or []
    items = data.get("items") or []
    values = data.get("values") or []
    image = Image.new("RGB", (1600, 1000), "#101418")
    draw = ImageDraw.Draw(image)
    title_font = font(44, bold=True)
    body_font = font(27)
    small_font = font(23)
    y = 54
    y = draw_wrapped(draw, (64, y), title, width=1450, fill="#f4f7fb", fnt=title_font, line_gap=10)
    desc = str(visual.get("description") or "")
    if desc:
        y = draw_wrapped(draw, (64, y + 8), desc, width=1450, fill="#a8b3c5", fnt=body_font)
    y += 24
    if rows:
        if columns:
            draw.text((64, y), " | ".join(str(col) for col in columns), fill="#77c7ff", font=body_font)
            y += 44
        for row in rows[:12]:
            text = "  -  ".join(str(cell) for cell in row)
            y = draw_wrapped(draw, (64, y), text, width=1450, fill="#edf2f7", fnt=small_font, line_gap=6)
            y += 6
    elif items:
        max_value = max(abs(float(item.get("value", 0.0))) for item in items) or 1.0
        for item in items[:10]:
            label = str(item.get("label", "value"))
            value = float(item.get("value", 0.0))
            draw.text((64, y), f"{label}: {value:.4g}", fill="#edf2f7", font=small_font)
            bar_w = int(900 * abs(value) / max_value)
            draw.rectangle((520, y + 6, 520 + bar_w, y + 28), fill="#4cc9f0")
            y += 54
    elif values:
        matrix = [[float(cell) for cell in row] for row in values if isinstance(row, list)]
        flat = [cell for row in matrix for cell in row]
        if matrix and flat:
            lo = min(flat)
            hi = max(flat)
            span = hi - lo if hi > lo else 1.0
            rows_n = len(matrix)
            cols_n = max(len(row) for row in matrix)
            left, top = 96, y
            width, height = 1400, min(760, max(260, rows_n * 24))
            cell_w = max(4, width / max(cols_n, 1))
            cell_h = max(4, height / max(rows_n, 1))
            for r, row in enumerate(matrix):
                for c, value in enumerate(row):
                    ratio = max(0.0, min(1.0, (value - lo) / span))
                    red = int(45 + ratio * 210)
                    green = int(80 + (1.0 - abs(ratio - 0.5) * 2.0) * 130)
                    blue = int(210 - ratio * 160)
                    draw.rectangle(
                        (
                            left + c * cell_w,
                            top + r * cell_h,
                            left + (c + 1) * cell_w,
                            top + (r + 1) * cell_h,
                        ),
                        fill=(red, green, blue),
                    )
            draw.text((96, top + height + 18), f"min={lo:.4g}  max={hi:.4g}", fill="#a8b3c5", font=small_font)
        else:
            draw.text((64, y), "No renderable heatmap values.", fill="#edf2f7", font=body_font)
    else:
        draw.text((64, y), "No renderable data.", fill="#edf2f7", font=body_font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def update_readme(lab_dir: Path, captures: list[tuple[str, str]]) -> None:
    readme = lab_dir / "README.md"
    text = readme.read_text(encoding="utf-8")
    block = [README_START]
    for title, rel in captures:
        block.append("")
        block.append(f"![{title}]({rel})")
    block.append("")
    block.append(README_END)
    new_block = "\n".join(block)
    if README_START in text and README_END in text:
        before, rest = text.split(README_START, 1)
        _old, after = rest.split(README_END, 1)
        text = before.rstrip() + "\n\n" + new_block + after
    else:
        text = text.rstrip() + "\n\n" + new_block + "\n"
    readme.write_text(text, encoding="utf-8")


def main() -> None:
    args = list(sys.argv[1:])
    force = "--force" in args
    args = [arg for arg in args if arg != "--force"]
    labs = [Path(arg).resolve() for arg in args] if args else sorted(p for p in LABS.iterdir() if p.is_dir())
    for lab_dir in labs:
        assets = lab_dir / "assets"
        if force:
            for png in assets.glob("*.png"):
                png.unlink()
        if list(assets.glob("*.png")):
            continue
        visuals = collect_visuals(lab_dir)
        captures: list[tuple[str, str]] = []
        for index, visual in enumerate(visuals, start=1):
            title = str((visual.get("data") or {}).get("title") or visual.get("description") or f"visual {index}")
            filename = f"{index:02d}-{slugify(title, f'visual-{index}')}.png"
            out = assets / filename
            render_visual(out, visual)
            captures.append((title, f"assets/{filename}"))
        update_readme(lab_dir, captures)
        print(f"Rendered {len(captures)} fallback asset(s) for {lab_dir.name}")


if __name__ == "__main__":
    main()
