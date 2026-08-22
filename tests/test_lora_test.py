import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from funkytown_testing_harness.lora_test import build_template, config_models, load_config, resolve_present_models, run


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


class ConfigModelsTests(unittest.TestCase):
    def test_single_model_key_normalized_to_list(self):
        self.assertEqual(config_models({"model": "modelA.safetensors"}), ["modelA.safetensors"])

    def test_models_key_used_as_is(self):
        self.assertEqual(
            config_models({"models": ["modelA.safetensors", "modelB.safetensors"]}),
            ["modelA.safetensors", "modelB.safetensors"],
        )

    def test_models_key_takes_precedence_over_model(self):
        config = {"model": "modelA.safetensors", "models": ["modelB.safetensors"]}
        self.assertEqual(config_models(config), ["modelB.safetensors"])

    def test_neither_key_exits(self):
        with self.assertRaises(SystemExit):
            config_models({})


class ResolvePresentModelsTests(unittest.TestCase):
    @patch("funkytown_testing_harness.lora_test.fetch_available_models")
    def test_single_model_present(self, mock_fetch):
        mock_fetch.return_value = {"modelA.safetensors"}
        present = resolve_present_models(["modelA.safetensors"], make_template(), "http://fake")
        self.assertEqual(present, ["modelA.safetensors"])

    @patch("funkytown_testing_harness.lora_test.fetch_available_models")
    def test_exits_when_model_missing(self, mock_fetch):
        mock_fetch.return_value = {"modelA.safetensors"}
        with self.assertRaises(SystemExit):
            resolve_present_models(["does_not_exist.safetensors"], make_template(), "http://fake")

    @patch("funkytown_testing_harness.lora_test.fetch_available_models")
    def test_multiple_models_all_present(self, mock_fetch):
        mock_fetch.return_value = {"modelA.safetensors", "modelB.safetensors"}
        present = resolve_present_models(["modelA.safetensors", "modelB.safetensors"], make_template(), "http://fake")
        self.assertEqual(present, ["modelA.safetensors", "modelB.safetensors"])

    @patch("funkytown_testing_harness.lora_test.fetch_available_models")
    def test_one_missing_is_skipped_but_others_still_run(self, mock_fetch):
        mock_fetch.return_value = {"modelA.safetensors"}
        present = resolve_present_models(["modelA.safetensors", "does_not_exist.safetensors"], make_template(), "http://fake")
        self.assertEqual(present, ["modelA.safetensors"])

    @patch("funkytown_testing_harness.lora_test.fetch_available_models")
    def test_all_missing_exits(self, mock_fetch):
        mock_fetch.return_value = set()
        with self.assertRaises(SystemExit):
            resolve_present_models(["modelA.safetensors", "modelB.safetensors"], make_template(), "http://fake")


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

    def test_filename_prefix_encodes_model_lora_and_weight(self):
        run(self.config_path)
        prefixes = {wf["6"]["inputs"]["filename_prefix"] for _s, wf, _c in self.queued}
        self.assertIn("tests/unit_test_lora_run/modelA__detail_slider_w0_5", prefixes)
        self.assertIn("tests/unit_test_lora_run/modelA__detail_slider_w1_0", prefixes)
        self.assertIn("tests/unit_test_lora_run/modelA__detail_slider_w1_5", prefixes)

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
        self.assertEqual(rows[1][3], "skipped")  # header: Model, LoRAs, Prompt ID, Status, Filename Prefix, Detail


class RunMultiModelTests(unittest.TestCase):
    """"models": [...] - every present model run against every LoRA combination."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.tmp_path = Path(self.tmpdir.name)

        self.config = {
            "name": "unit_test_multi_model_run",
            "source_workflow": "krea2_basic_t2i.json",
            "models": ["modelA.safetensors", "modelB.safetensors"],
            "server": "http://fake",
            "loras": [
                {"lora": "detail_slider.safetensors", "weights": [0.5, 1.0]},
            ],
        }
        self.config_path = self.tmp_path / "config.json"
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")

        self.runs_dir = self.tmp_path / "runs"
        self.queued = []

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
            return_value={"modelA.safetensors", "modelB.safetensors"},
        )
        patcher_runs_dir = patch("funkytown_testing_harness.lora_test.RUNS_DIR", self.runs_dir)
        patcher_queue.start()
        patcher_load_template.start()
        patcher_fetch.start()
        patcher_runs_dir.start()
        self.addCleanup(patcher_queue.stop)
        self.addCleanup(patcher_load_template.stop)
        self.addCleanup(patcher_fetch.stop)
        self.addCleanup(patcher_runs_dir.stop)

    def test_queues_every_model_times_every_combination(self):
        run(self.config_path)
        self.assertEqual(len(self.queued), 4)  # 2 models x 2 weights

    def test_every_model_lora_pairing_is_represented(self):
        run(self.config_path)
        pairings = {
            (wf["1"]["inputs"]["unet_name"], wf["22"]["inputs"]["lora_1"]["strength"])
            for _s, wf, _c in self.queued
        }
        self.assertEqual(
            pairings,
            {
                ("modelA.safetensors", 0.5), ("modelA.safetensors", 1.0),
                ("modelB.safetensors", 0.5), ("modelB.safetensors", 1.0),
            },
        )

    def test_filename_prefix_encodes_model(self):
        run(self.config_path)
        prefixes = {wf["6"]["inputs"]["filename_prefix"] for _s, wf, _c in self.queued}
        self.assertIn("tests/unit_test_multi_model_run/modelA__detail_slider_w0_5", prefixes)
        self.assertIn("tests/unit_test_multi_model_run/modelB__detail_slider_w0_5", prefixes)

    def test_one_missing_model_is_skipped_but_others_still_run(self):
        self.config["models"] = ["modelA.safetensors", "does_not_exist.safetensors"]
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        run(self.config_path)
        self.assertEqual(len(self.queued), 2)  # only modelA x 2 weights

    def test_all_models_missing_aborts_before_queuing(self):
        self.config["models"] = ["does_not_exist_a.safetensors", "does_not_exist_b.safetensors"]
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        with self.assertRaises(SystemExit):
            run(self.config_path)
        self.assertEqual(len(self.queued), 0)


class RunCombinedModeTests(unittest.TestCase):
    """combine_loras: true - every listed LoRA active together, one run per
    combination across the cartesian product of their weight lists."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.tmp_path = Path(self.tmpdir.name)

        self.config = {
            "name": "unit_test_combined_run",
            "source_workflow": "krea2_basic_t2i.json",
            "model": "modelA.safetensors",
            "server": "http://fake",
            "combine_loras": True,
            "loras": [
                {"lora": "detail_slider.safetensors", "weights": [0.5, 1.0]},
                {"lora": "other_lora.safetensors", "weights": [1.0, 2.0]},
            ],
        }
        self.config_path = self.tmp_path / "config.json"
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")

        self.runs_dir = self.tmp_path / "runs"
        self.queued = []

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
        patcher_load_template.start()
        patcher_fetch.start()
        patcher_runs_dir.start()
        self.addCleanup(patcher_queue.stop)
        self.addCleanup(patcher_load_template.stop)
        self.addCleanup(patcher_fetch.stop)
        self.addCleanup(patcher_runs_dir.stop)

    def test_queues_the_full_cartesian_product(self):
        run(self.config_path)
        self.assertEqual(len(self.queued), 4)  # 2 weights x 2 weights

    def test_both_loras_active_together_in_every_run(self):
        run(self.config_path)
        for _server, wf, _client_id in self.queued:
            self.assertTrue(wf["22"]["inputs"]["lora_1"]["on"])  # detail_slider.safetensors
            self.assertTrue(wf["22"]["inputs"]["lora_2"]["on"])  # other_lora.safetensors

    def test_every_weight_pairing_is_represented(self):
        run(self.config_path)
        pairings = {
            (wf["22"]["inputs"]["lora_1"]["strength"], wf["22"]["inputs"]["lora_2"]["strength"])
            for _s, wf, _c in self.queued
        }
        self.assertEqual(pairings, {(0.5, 1.0), (0.5, 2.0), (1.0, 1.0), (1.0, 2.0)})

    def test_filename_prefix_encodes_both_loras(self):
        run(self.config_path)
        prefixes = {wf["6"]["inputs"]["filename_prefix"] for _s, wf, _c in self.queued}
        self.assertIn("tests/unit_test_combined_run/modelA__detail_slider_w0_5__other_lora_w1_0", prefixes)
        self.assertIn("tests/unit_test_combined_run/modelA__detail_slider_w1_0__other_lora_w2_0", prefixes)

    def test_whole_combination_skipped_if_any_lora_missing(self):
        self.config["loras"] = [
            {"lora": "detail_slider.safetensors", "weights": [0.5]},
            {"lora": "does_not_exist.safetensors", "weights": [1.0]},
        ]
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        run(self.config_path)
        self.assertEqual(len(self.queued), 0)
        log_files = list(self.runs_dir.glob("unit_test_combined_run_*.csv"))
        with open(log_files[0], newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        self.assertEqual(rows[1][3], "skipped")


class BuildCombinationsTests(unittest.TestCase):
    def test_isolated_mode_one_pair_per_combination(self):
        from funkytown_testing_harness.lora_test import build_combinations

        loras = [
            {"lora": "a.safetensors", "weights": [1, 2]},
            {"lora": "b.safetensors", "weights": [3]},
        ]
        combos = build_combinations(loras, combine=False)
        self.assertEqual(combos, [[("a.safetensors", 1)], [("a.safetensors", 2)], [("b.safetensors", 3)]])

    def test_combined_mode_cartesian_product(self):
        from funkytown_testing_harness.lora_test import build_combinations

        loras = [
            {"lora": "a.safetensors", "weights": [1, 2]},
            {"lora": "b.safetensors", "weights": [3, 4]},
        ]
        combos = build_combinations(loras, combine=True)
        self.assertEqual(len(combos), 4)
        as_sets = [set(c) for c in combos]
        self.assertIn({("a.safetensors", 1), ("b.safetensors", 3)}, as_sets)
        self.assertIn({("a.safetensors", 2), ("b.safetensors", 4)}, as_sets)

    def test_combined_mode_single_lora_behaves_like_isolated(self):
        from funkytown_testing_harness.lora_test import build_combinations

        loras = [{"lora": "a.safetensors", "weights": [1, 2, 3]}]
        self.assertEqual(build_combinations(loras, combine=True), build_combinations(loras, combine=False))


if __name__ == "__main__":
    unittest.main()
