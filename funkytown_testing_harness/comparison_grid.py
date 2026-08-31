"""Build a labeled side-by-side comparison grid image from a logged
run's output images (run_test.py/lora_test.py's runs/*.csv) - one column
per queued row that already has an output image, each labeled with
whatever varied for that row (model name, or LoRA/weight combo), matching
the style of an informal hand-made A/B comparison strip: a white header
band above each column with its label, images butted directly together
below with no gaps.
"""

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

LABEL_HEIGHT_RATIO = 0.08  # header band height as a fraction of panel width
MIN_LABEL_HEIGHT = 50
PANEL_MAX_WIDTH = 400
MAX_COLUMNS = 10  # more panels than this spill into additional output files, not additional rows


def _load_font(size):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _clean_label(raw, is_lora_run):
    """"detail_slider.safetensors=1.0; other.safetensors=0.5" (lora_test.py's
    "LoRAs" column) -> "detail_slider:1.0, other:0.5"; a bare model filename
    (run_test.py's "Model" column) -> just its stem, extension dropped."""
    if not is_lora_run:
        return Path(raw).stem
    parts = []
    for pair in raw.split(";"):
        pair = pair.strip()
        if "=" in pair:
            name, weight = pair.split("=", 1)
            parts.append(f"{Path(name.strip()).stem}:{weight.strip()}")
        elif pair:
            parts.append(pair)
    return ", ".join(parts) or raw


def read_run_rows(log_path):
    """[(label, filename_prefix), ...] in queue order for a runs/*.csv log,
    skipping rows with no Filename Prefix (skipped/errored rows never
    produced an image)."""
    with open(log_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    is_lora_run = bool(rows) and "LoRAs" in rows[0]
    label_col = "LoRAs" if is_lora_run else "Model"
    return [
        (_clean_label(row[label_col], is_lora_run), row["Filename Prefix"])
        for row in rows
        if row.get("Filename Prefix") and label_col in row
    ]


def resolve_row_image(comfyui_output_dir, prefix):
    """First image file matching a Filename Prefix under ComfyUI's output
    folder (ComfyUI appends its own numeric counter/extension), or None if
    nothing's there yet."""
    prefix_path = Path(comfyui_output_dir) / prefix
    if not prefix_path.parent.is_dir():
        return None
    matches = sorted(prefix_path.parent.glob(prefix_path.name + "_*"))
    return matches[0] if matches else None


def _chunked(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


def _render_single_row_grid(panels, output_path):
    """Render one MAX_COLUMNS-or-fewer-wide single-row grid image to
    output_path - the images-butted-together-with-a-label-band-above-each
    layout, no wrapping (that's handled by build_comparison_grid chunking
    the panels before calling this)."""
    images = [Image.open(path).convert("RGB") for _label, path in panels]
    panel_width = min(PANEL_MAX_WIDTH, min(im.width for im in images))
    resized = [
        im.resize((panel_width, round(im.height * panel_width / im.width)), Image.LANCZOS)
        for im in images
    ]
    panel_height = max(im.height for im in resized)

    label_height = max(MIN_LABEL_HEIGHT, round(panel_width * LABEL_HEIGHT_RATIO))
    font = _load_font(round(label_height * 0.5))

    grid = Image.new("RGB", (panel_width * len(resized), label_height + panel_height), "white")
    draw = ImageDraw.Draw(grid)

    for i, ((label, _path), im) in enumerate(zip(panels, resized)):
        x = i * panel_width
        grid.paste(im, (x, label_height))
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((x + (panel_width - text_w) / 2, (label_height - text_h) / 2 - bbox[1]), label, fill="black", font=font)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)


def build_comparison_grid(log_path, comfyui_output_dir, output_path, selected_images=None):
    """Build one or more single-row comparison grid PNGs, up to
    MAX_COLUMNS panels each - a run with more panels than that produces
    additional numbered files (output_path's stem gets a "_2", "_3", ...
    suffix) rather than a taller single image, so each file stays a plain
    single row. Returns the list of Path objects written, in order
    (output_path itself is always first).

    selected_images, if given, restricts the grid to just those image paths
    (e.g. a caller's own checked/selected subset) - a row whose resolved
    image isn't in that set is skipped, though its label is still looked up
    from the log the normal way. None (the default) means every queued row
    that currently has an image.

    Raises ValueError if fewer than 2 rows end up with an image to include -
    nothing meaningful to compare with 0 or 1."""
    entries = read_run_rows(Path(log_path))
    selected_set = {Path(p) for p in selected_images} if selected_images is not None else None

    panels = []
    for label, prefix in entries:
        image_path = resolve_row_image(comfyui_output_dir, prefix)
        if image_path and (selected_set is None or image_path in selected_set):
            panels.append((label, image_path))

    if len(panels) < 2:
        raise ValueError(
            f"Need at least 2 output images to build a comparison grid - found {len(panels)}. "
            "The run may still be in progress, nothing was queued successfully, or too few images are selected."
        )

    output_path = Path(output_path)
    output_paths = []
    for i, chunk in enumerate(_chunked(panels, MAX_COLUMNS)):
        chunk_path = output_path if i == 0 else output_path.with_stem(f"{output_path.stem}_{i + 1}")
        _render_single_row_grid(chunk, chunk_path)
        output_paths.append(chunk_path)

    return output_paths
