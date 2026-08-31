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
MAX_COLUMNS = 4  # more panels than this wrap onto additional rows below


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


def build_comparison_grid(log_path, comfyui_output_dir, output_path, selected_images=None):
    """Build a comparison grid PNG at output_path - a single row for up to
    MAX_COLUMNS panels, wrapping onto additional rows below for more than
    that (each wrapped row gets its own header, mirrored underneath that
    row's images too, so a row's labels are never far from its images).

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

    images = [Image.open(path).convert("RGB") for _label, path in panels]
    panel_width = min(PANEL_MAX_WIDTH, min(im.width for im in images))
    resized = [
        im.resize((panel_width, round(im.height * panel_width / im.width)), Image.LANCZOS)
        for im in images
    ]
    panel_height = max(im.height for im in resized)

    label_height = max(MIN_LABEL_HEIGHT, round(panel_width * LABEL_HEIGHT_RATIO))
    font = _load_font(round(label_height * 0.5))

    labels = [label for label, _path in panels]
    columns = min(MAX_COLUMNS, len(resized))
    num_rows = -(-len(resized) // columns)  # ceil division
    mirror_header = num_rows > 1
    row_height = label_height * (2 if mirror_header else 1) + panel_height

    grid = Image.new("RGB", (panel_width * columns, row_height * num_rows), "white")
    draw = ImageDraw.Draw(grid)

    def draw_label(x, y, label):
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((x + (panel_width - text_w) / 2, y + (label_height - text_h) / 2 - bbox[1]), label, fill="black", font=font)

    for i, (im, label) in enumerate(zip(resized, labels)):
        row, col = divmod(i, columns)
        x = col * panel_width
        row_top = row * row_height
        draw_label(x, row_top, label)
        grid.paste(im, (x, row_top + label_height))
        if mirror_header:
            draw_label(x, row_top + label_height + panel_height, label)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    return output_path
