import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from funkytown_testing_harness.lora_test import build_template, check_model_present, load_config, run


def make_template():
    """A minimal but structurally valid API-format workflow: one model
    loader, one Power Lora Loader (with a couple of pre-existing slots), one
    KSampler, one SaveImage."""
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "placeholder.safetensors", "weight_dtype": "default"}},
        "2": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 1, "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
                "model": ["22", 0], "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["5", 0],
            },
        },
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "a test prompt", "clip": ["1", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["1", 0]}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 3}},
        "6": {"class_type": "SaveImage", "inputs": {"filename_prefix": "orig", "images": ["2", 0]}},
        "22": {
            "class_type": "Power Lora Loader (rgthree)",
            "inputs": {
                "model": ["1", 0],
                "clip": ["1", 0],
                "lora_1": {"on": False, "lora": "detail_slider.safetensors", "strength": 2},
                "lora_2": {"on": False, "lora": "other_lora.safetensors", "strength": 1},
            },
        },
    }


class BuildTemplateTests(unittest.TestCase):
    @patch("funkytown_testing_harness.lora_test.load_live_template")
    def test_fetches_using_configured_source_workflow(self, mock_load):
        mock_load.return_value = make_template()
        config = {"source_workflow": "my_live_workflow.json"}
        build_template(config, "http://fake-server")
        mock_load.assert_called_once_with("http://fake-server", "my_live_workflow.json")

    @patch("funkytown_testing_harness.lora_test.load_live_template")
    def test_applies_positive_prompt_override(self, mock_load):
        mock_load.return_value = make_template()
        config = {"source_workflow": "wf.json", "positive_prompt": "a new prompt"}
        template = build_template(config, "http://fake-server")
        self.assertEqual(template["3"]["inputs"]["text"], "a new prompt")


class CheckModelPresentTests(unittest.TestCase):
    @patch("funkytown_testing_harness.lora_test.fetch_available_models")
    def test_passes_when_model_present(self, mock_fetch):
        mock_fetch.return_value = {"modelA.safetensors"}
        check_model_present("modelA.safetensors", make_template(), "http://fake")  # should not raise

    @patch("funkytown_testing_harness.lora_test.fetch_available_models")
    def test_exits_when_model_missing(self, mock_fetch):
        mock_fetch.return_value = {"modelA.safetensors"}
        with self.assertRaises(SystemExit):
            check_model_present("does_not_exist.safetensors", make_template(), "http://fake")


class RunEndToEndTests(unittest.TestCase):
    """Exercises the full run() flow: always-fresh fetch (mocked) + single-model
    fixed + one-LoRA-at-a-time weight sweep, with ComfyUI network calls mocked out."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.tmp_path = Path(self.tmpdir.name)

        self.config = {
            "name": "unit_test_lora_run",
            "source_workflow": "krea2_basic_t2i.json",
            "model": "modelA.safetensors",
            "server": "http://fake",
            "loras": [
                {"lora": "detail_slider.safetensors", "weights": [0.5, 1.0, 1.5]},
            ],
        }
        self.config_path = self.tmp_path / "config.json"
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")

        self.runs_dir = self.tmp_path / "runs"
        self.queued = []  # list of (server, workflow_copy, client_id)

        def fake_queue_prompt(server, workflow, client_id):
            self.queued.append((server, copy.deepcopy(workflow), client_id))
            return {"prompt_id": f"fake-{len(self.queued)}", "node_errors": {}}

        patcher_queue = patch("funkytown_testing_harness.lora_test.queue_prompt", side_effect=fake_queue_prompt)
        patcher_load_template = patch(
            "funkytown_testing_harness.lora_test.load_live_template",
            side_effect=lambda server, source_workflow: make_template(),
        )
        patcher_fetch = patch(
            "funkytown_testing_harness.lora_test.fetch_available_models",
            return_value={"modelA.safetensors"},
        )
        patcher_runs_dir = patch("funkytown_testing_harness.lora_test.RUNS_DIR", self.runs_dir)
        self.mock_queue_prompt = patcher_queue.start()
        self.mock_load_template = patcher_load_template.start()
        patcher_fetch.start()
        patcher_runs_dir.start()
        self.addCleanup(patcher_queue.stop)
        self.addCleanup(patcher_load_template.stop)
        self.addCleanup(patcher_fetch.stop)
        self.addCleanup(patcher_runs_dir.stop)

    def test_queues_once_per_weight(self):
        run(self.config_path)
        self.assertEqual(len(self.queued), 3)  # 0.5, 1.0, 1.5

    def test_model_is_set_on_every_queued_variant(self):
        run(self.config_path)
        for _server, wf, _client_id in self.queued:
            self.assertEqual(wf["1"]["inputs"]["unet_name"], "modelA.safetensors")

    def test_each_run_isolates_the_target_lora_at_its_weight(self):
        run(self.config_path)
        strengths = sorted(wf["22"]["inputs"]["lora_1"]["strength"] for _s, wf, _c in self.queued)
        self.assertEqual(strengths, [0.5, 1.0, 1.5])
        for _server, wf, _client_id in self.queued:
            self.assertTrue(wf["22"]["inputs"]["lora_1"]["on"])
            self.assertFalse(wf["22"]["inputs"]["lora_2"]["on"])  # other slot always off

    def test_batch_size_from_workflow_is_never_overridden(self):
        run(self.config_path)
        for _server, wf, _client_id in self.queued:
            self.assertEqual(wf["5"]["inputs"]["batch_size"], 3)

    def test_filename_prefix_encodes_lora_and_weight(self):
        run(self.config_path)
        prefixes = {wf["6"]["inputs"]["filename_prefix"] for _s, wf, _c in self.queued}
        self.assertIn("tests/unit_test_lora_run/detail_slider_w0_5", prefixes)
        self.assertIn("tests/unit_test_lora_run/detail_slider_w1_0", prefixes)
        self.assertIn("tests/unit_test_lora_run/detail_slider_w1_5", prefixes)

    def test_log_csv_has_one_row_per_weight(self):
        run(self.config_path)
        log_files = list(self.runs_dir.glob("unit_test_lora_run_*.csv"))
        self.assertEqual(len(log_files), 1)
        with open(log_files[0], newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        self.assertEqual(len(rows) - 1, 3)  # header + 3 data rows

    def test_missing_model_aborts_before_queuing(self):
        self.config["model"] = "does_not_exist.safetensors"
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        with self.assertRaises(SystemExit):
            run(self.config_path)
        self.assertEqual(len(self.queued), 0)

    def test_missing_lora_slot_is_skipped_not_fatal(self):
        self.config["loras"] = [{"lora": "does_not_exist_lora.safetensors", "weights": [1.0]}]
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        run(self.config_path)  # should not raise
        self.assertEqual(len(self.queued), 0)
        log_files = list(self.runs_dir.glob("unit_test_lora_run_*.csv"))
        with open(log_files[0], newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        self.assertEqual(rows[1][3], "skipped")


if __name__ == "__main__":
    unittest.main()
