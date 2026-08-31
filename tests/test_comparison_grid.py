import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from funkytown_testing_harness.comparison_grid import (
    _clean_label,
    build_comparison_grid,
    read_run_rows,
    resolve_row_image,
)


def write_run_test_log(log_path, rows):
    """rows: [(model, filename_prefix, status), ...]"""
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "KSampler Overrides", "Prompt ID", "Status", "Filename Prefix", "Detail"])
        for model, prefix, status in rows:
            writer.writerow([model, "{}", "p1" if status == "queued" else "", status, prefix, ""])


def write_lora_test_log(log_path, rows):
    """rows: [(loras_label, filename_prefix, status), ...]"""
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "LoRAs", "Prompt ID", "Status", "Filename Prefix", "Detail"])
        for loras, prefix, status in rows:
            writer.writerow(["modelA.safetensors", loras, "p1" if status == "queued" else "", status, prefix, ""])


def make_png(path, size=(100, 150), color="red"):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


class CleanLabelTests(unittest.TestCase):
    def test_model_run_strips_extension(self):
        self.assertEqual(_clean_label("krea2SATDirtyrealism_krea2SAT.safetensors", is_lora_run=False), "krea2SATDirtyrealism_krea2SAT")

    def test_lora_run_single_lora(self):
        self.assertEqual(_clean_label("detail_slider.safetensors=1.0", is_lora_run=True), "detail_slider:1.0")

    def test_lora_run_combined_loras(self):
        self.assertEqual(
            _clean_label("detail_slider.safetensors=1.0; skindetails.safetensors=0.5", is_lora_run=True),
            "detail_slider:1.0, skindetails:0.5",
        )


class ReadRunRowsTests(unittest.TestCase):
    def test_run_test_style_log_uses_model_column(self):
        tmpdir = Path(tempfile.mkdtemp())
        log_path = tmpdir / "log.csv"
        write_run_test_log(log_path, [
            ("modelA.safetensors", "tests/x/0001_modelA", "queued"),
            ("modelB.safetensors", "tests/x/0002_modelB", "queued"),
        ])
        rows = read_run_rows(log_path)
        self.assertEqual(rows, [("modelA", "tests/x/0001_modelA"), ("modelB", "tests/x/0002_modelB")])

    def test_lora_test_style_log_uses_loras_column(self):
        tmpdir = Path(tempfile.mkdtemp())
        log_path = tmpdir / "log.csv"
        write_lora_test_log(log_path, [
            ("detail_slider.safetensors=1.0", "tests/x/0001_modelA__detail_slider_w1_0", "queued"),
        ])
        rows = read_run_rows(log_path)
        self.assertEqual(rows, [("detail_slider:1.0", "tests/x/0001_modelA__detail_slider_w1_0")])

    def test_skipped_and_error_rows_excluded(self):
        tmpdir = Path(tempfile.mkdtemp())
        log_path = tmpdir / "log.csv"
        write_run_test_log(log_path, [
            ("modelA.safetensors", "tests/x/0001_modelA", "queued"),
            ("modelB.safetensors", "", "error"),
        ])
        rows = read_run_rows(log_path)
        self.assertEqual(rows, [("modelA", "tests/x/0001_modelA")])


class ResolveRowImageTests(unittest.TestCase):
    def test_finds_matching_file(self):
        tmpdir = Path(tempfile.mkdtemp())
        output_dir = tmpdir / "output"
        make_png(output_dir / "tests" / "x" / "0001_modelA_00001_.png")
        result = resolve_row_image(output_dir, "tests/x/0001_modelA")
        self.assertEqual(result.name, "0001_modelA_00001_.png")

    def test_returns_none_when_missing(self):
        tmpdir = Path(tempfile.mkdtemp())
        output_dir = tmpdir / "output"
        result = resolve_row_image(output_dir, "tests/x/does_not_exist")
        self.assertIsNone(result)


class BuildComparisonGridTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.output_dir = self.tmpdir / "output"
        self.log_path = self.tmpdir / "log.csv"

    def test_builds_grid_with_correct_dimensions_and_column_count(self):
        write_run_test_log(self.log_path, [
            ("modelA.safetensors", "tests/x/0001_modelA", "queued"),
            ("modelB.safetensors", "tests/x/0002_modelB", "queued"),
            ("modelC.safetensors", "tests/x/0003_modelC", "queued"),
        ])
        make_png(self.output_dir / "tests" / "x" / "0001_modelA_00001_.png", size=(200, 300))
        make_png(self.output_dir / "tests" / "x" / "0002_modelB_00001_.png", size=(200, 300))
        make_png(self.output_dir / "tests" / "x" / "0003_modelC_00001_.png", size=(200, 300))

        out_path = self.tmpdir / "grid.png"
        result = build_comparison_grid(self.log_path, self.output_dir, out_path)

        self.assertEqual(result, out_path)
        self.assertTrue(out_path.is_file())
        with Image.open(out_path) as grid:
            self.assertEqual(grid.width, 200 * 3)  # 3 columns, no downscale needed (200 < PANEL_MAX_WIDTH)
            self.assertGreater(grid.height, 300)  # panel height plus label band

    def test_downscales_panels_wider_than_max(self):
        write_run_test_log(self.log_path, [
            ("modelA.safetensors", "tests/x/0001_modelA", "queued"),
            ("modelB.safetensors", "tests/x/0002_modelB", "queued"),
        ])
        make_png(self.output_dir / "tests" / "x" / "0001_modelA_00001_.png", size=(1000, 1500))
        make_png(self.output_dir / "tests" / "x" / "0002_modelB_00001_.png", size=(1000, 1500))

        out_path = self.tmpdir / "grid.png"
        build_comparison_grid(self.log_path, self.output_dir, out_path)
        with Image.open(out_path) as grid:
            self.assertEqual(grid.width, 400 * 2)  # PANEL_MAX_WIDTH = 400

    def test_still_in_progress_run_uses_only_images_that_exist(self):
        write_run_test_log(self.log_path, [
            ("modelA.safetensors", "tests/x/0001_modelA", "queued"),
            ("modelB.safetensors", "tests/x/0002_modelB", "queued"),
            ("modelC.safetensors", "tests/x/0003_modelC", "queued"),
        ])
        # Only 2 of the 3 have actually finished rendering.
        make_png(self.output_dir / "tests" / "x" / "0001_modelA_00001_.png", size=(200, 300))
        make_png(self.output_dir / "tests" / "x" / "0002_modelB_00001_.png", size=(200, 300))

        out_path = self.tmpdir / "grid.png"
        build_comparison_grid(self.log_path, self.output_dir, out_path)
        with Image.open(out_path) as grid:
            self.assertEqual(grid.width, 200 * 2)  # only the 2 that exist

    def test_fewer_than_two_images_raises(self):
        write_run_test_log(self.log_path, [
            ("modelA.safetensors", "tests/x/0001_modelA", "queued"),
        ])
        make_png(self.output_dir / "tests" / "x" / "0001_modelA_00001_.png")

        with self.assertRaises(ValueError):
            build_comparison_grid(self.log_path, self.output_dir, self.tmpdir / "grid.png")

    def test_zero_images_raises(self):
        write_run_test_log(self.log_path, [
            ("modelA.safetensors", "tests/x/0001_modelA", "queued"),
        ])
        with self.assertRaises(ValueError):
            build_comparison_grid(self.log_path, self.output_dir, self.tmpdir / "grid.png")

    def test_four_or_fewer_panels_stay_a_single_row(self):
        rows = [(f"model{i}.safetensors", f"tests/x/000{i}_model{i}", "queued") for i in range(4)]
        write_run_test_log(self.log_path, rows)
        for i in range(4):
            make_png(self.output_dir / "tests" / "x" / f"000{i}_model{i}_00001_.png", size=(100, 150))

        out_path = self.tmpdir / "grid.png"
        build_comparison_grid(self.log_path, self.output_dir, out_path)
        with Image.open(out_path) as grid:
            self.assertEqual(grid.width, 100 * 4)
            self.assertEqual(grid.height, 150 + 50)  # one label band, no mirrored bottom header

    def test_more_than_four_panels_wrap_to_a_second_row_with_mirrored_header(self):
        rows = [(f"model{i}.safetensors", f"tests/x/000{i}_model{i}", "queued") for i in range(6)]
        write_run_test_log(self.log_path, rows)
        for i in range(6):
            make_png(self.output_dir / "tests" / "x" / f"000{i}_model{i}_00001_.png", size=(100, 150))

        out_path = self.tmpdir / "grid.png"
        build_comparison_grid(self.log_path, self.output_dir, out_path)
        with Image.open(out_path) as grid:
            self.assertEqual(grid.width, 100 * 4)  # capped at 4 columns
            # 2 rows, each: label band + panel + mirrored label band underneath
            self.assertEqual(grid.height, 2 * (150 + 50 + 50))

    def test_wrapped_row_places_images_by_position_within_its_own_row(self):
        # 5 panels -> row 0 has columns 0-3, row 1 has just column 0 (index 4).
        # Confirm the 5th image lands at the start of row 1, not appended
        # to a nonexistent 5th column of row 0.
        rows = [(f"model{i}.safetensors", f"tests/x/000{i}_model{i}", "queued") for i in range(5)]
        write_run_test_log(self.log_path, rows)
        colors = ["red", "green", "blue", "yellow", "purple"]
        for i, color in enumerate(colors):
            make_png(self.output_dir / "tests" / "x" / f"000{i}_model{i}_00001_.png", size=(20, 20), color=color)

        out_path = self.tmpdir / "grid.png"
        build_comparison_grid(self.log_path, self.output_dir, out_path)
        with Image.open(out_path) as grid:
            label_height = 50
            row_height = label_height * 2 + 20  # mirrored header, since 2 rows
            # Sample a pixel from the middle of each panel's image area.
            row1_y = label_height + 10
            row2_y = row_height + label_height + 10
            self.assertEqual(grid.getpixel((10, row1_y)), (255, 0, 0))    # red, row0 col0
            self.assertEqual(grid.getpixel((30, row1_y)), (0, 128, 0))    # green, row0 col1
            self.assertEqual(grid.getpixel((10, row2_y)), (128, 0, 128))  # purple, row1 col0 (5th image)

    def test_selected_images_restricts_which_rows_are_included(self):
        write_run_test_log(self.log_path, [
            ("modelA.safetensors", "tests/x/0001_modelA", "queued"),
            ("modelB.safetensors", "tests/x/0002_modelB", "queued"),
            ("modelC.safetensors", "tests/x/0003_modelC", "queued"),
        ])
        img_a = self.output_dir / "tests" / "x" / "0001_modelA_00001_.png"
        img_b = self.output_dir / "tests" / "x" / "0002_modelB_00001_.png"
        img_c = self.output_dir / "tests" / "x" / "0003_modelC_00001_.png"
        make_png(img_a, size=(200, 300))
        make_png(img_b, size=(200, 300))
        make_png(img_c, size=(200, 300))

        out_path = self.tmpdir / "grid.png"
        build_comparison_grid(self.log_path, self.output_dir, out_path, selected_images=[img_a, img_c])
        with Image.open(out_path) as grid:
            self.assertEqual(grid.width, 200 * 2)  # only the 2 selected, not all 3 queued

    def test_selected_images_accepts_string_paths_too(self):
        write_run_test_log(self.log_path, [
            ("modelA.safetensors", "tests/x/0001_modelA", "queued"),
            ("modelB.safetensors", "tests/x/0002_modelB", "queued"),
        ])
        img_a = self.output_dir / "tests" / "x" / "0001_modelA_00001_.png"
        img_b = self.output_dir / "tests" / "x" / "0002_modelB_00001_.png"
        make_png(img_a, size=(200, 300))
        make_png(img_b, size=(200, 300))

        out_path = self.tmpdir / "grid.png"
        # Deliberately pass plain strings, not Path objects - a GUI caller
        # working with Qt item data would naturally have strings.
        build_comparison_grid(self.log_path, self.output_dir, out_path, selected_images=[str(img_a), str(img_b)])
        with Image.open(out_path) as grid:
            self.assertEqual(grid.width, 200 * 2)

    def test_fewer_than_two_selected_images_raises(self):
        write_run_test_log(self.log_path, [
            ("modelA.safetensors", "tests/x/0001_modelA", "queued"),
            ("modelB.safetensors", "tests/x/0002_modelB", "queued"),
            ("modelC.safetensors", "tests/x/0003_modelC", "queued"),
        ])
        img_a = self.output_dir / "tests" / "x" / "0001_modelA_00001_.png"
        make_png(img_a, size=(200, 300))
        make_png(self.output_dir / "tests" / "x" / "0002_modelB_00001_.png", size=(200, 300))
        make_png(self.output_dir / "tests" / "x" / "0003_modelC_00001_.png", size=(200, 300))

        with self.assertRaises(ValueError):
            build_comparison_grid(self.log_path, self.output_dir, self.tmpdir / "grid.png", selected_images=[img_a])


if __name__ == "__main__":
    unittest.main()
