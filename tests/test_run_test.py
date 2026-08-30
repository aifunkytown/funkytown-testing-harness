import copy
import csv
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from funkytown_testing_harness.run_test import (
    apply_ksampler_overrides,
    build_template,
    find_ksampler_node_id,
    load_config,
    resolve_present_models,
    run,
)


def strip_run_id(prefix):
    """Prefixes now embed a per-run random hex id (run_test.run()'s run_id)
    and a per-variant zero-padded queue_index between "tests/<name>/" and
    the rest - strip both out so assertions on the rest of the format
    don't need to know their values, while still confirming they're
    actually there and correctly shaped."""
    return re.sub(r"^(tests/[^/]+)/[0-9a-f]{8}/\d{4}_", r"\1/", prefix)


def make_template_with_lora():
    """Same as make_template() but with a Power Lora Loader node added, for
    exercising the keyword LoRA-routing wiring."""
    wf = make_template()
    wf["7"] = {
        "class_type": "Power Lora Loader (rgthree)",
        "inputs": {
            "model": ["1", 0],
            "clip": ["1", 0],
            "lora_1": {"on": False, "lora": "some_lora.safetensors", "strength": 0.8},
        },
    }
    return wf


def make_template():
    """A minimal but structurally valid API-format workflow: one model loader,
    one KSampler, one SaveImage. batch_size is set to a distinctive value (3)
    so tests can confirm it's never touched."""
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "placeholder.safetensors", "weight_dtype": "default"}},
        "2": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 1, "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
                "model": ["1", 0], "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["5", 0],
            },
        },
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "a test prompt", "clip": ["1", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["1", 0]}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 3}},
        "6": {"class_type": "SaveImage", "inputs": {"filename_prefix": "orig", "images": ["2", 0]}},
    }


class ApplyKsamplerOverridesTests(unittest.TestCase):
    def test_applies_known_keys(self):
        wf = make_template()
        apply_ksampler_overrides(wf, "2", {"sampler_name": "er_sde", "steps": 10, "cfg": 2.0, "scheduler": "simple"})
        inputs = wf["2"]["inputs"]
        self.assertEqual(inputs["sampler_name"], "er_sde")
        self.assertEqual(inputs["steps"], 10)
        self.assertEqual(inputs["cfg"], 2.0)
        self.assertEqual(inputs["scheduler"], "simple")

    def test_leaves_unspecified_keys_untouched(self):
        wf = make_template()
        apply_ksampler_overrides(wf, "2", {"steps": 20})
        self.assertEqual(wf["2"]["inputs"]["sampler_name"], "euler")  # untouched
        self.assertEqual(wf["2"]["inputs"]["steps"], 20)

    def test_empty_overrides_changes_nothing(self):
        wf = make_template()
        before = dict(wf["2"]["inputs"])
        apply_ksampler_overrides(wf, "2", {})
        self.assertEqual(wf["2"]["inputs"], before)

    def test_ignores_unrecognized_keys_without_applying_them(self):
        wf = make_template()
        apply_ksampler_overrides(wf, "2", {"model": ["99", 0], "bogus_key": "x"})
        self.assertEqual(wf["2"]["inputs"]["model"], ["1", 0])  # connection untouched
        self.assertNotIn("bogus_key", wf["2"]["inputs"])

    def test_never_touches_batch_size(self):
        wf = make_template()
        apply_ksampler_overrides(wf, "2", {"steps": 99, "cfg": 5, "sampler_name": "dpmpp_2m", "scheduler": "karras"})
        self.assertEqual(wf["5"]["inputs"]["batch_size"], 3)


class FindKsamplerNodeIdTests(unittest.TestCase):
    def test_finds_ksampler(self):
        wf = make_template()
        self.assertEqual(find_ksampler_node_id(wf), "2")

    def test_returns_none_when_absent(self):
        wf = {"1": {"class_type": "SaveImage", "inputs": {}}}
        self.assertIsNone(find_ksampler_node_id(wf))


class ResolvePresentModelsTests(unittest.TestCase):
    def setUp(self):
        self.template = make_template()

    @patch("funkytown_testing_harness.run_test.fetch_available_models")
    def test_all_present_returns_all(self, mock_fetch):
        mock_fetch.return_value = {"modelA.safetensors", "modelB.safetensors"}
        models = [{"model": "modelA.safetensors"}, {"model": "modelB.safetensors"}]
        present = resolve_present_models(models, self.template, "http://fake")
        self.assertEqual([m["model"] for m in present], ["modelA.safetensors", "modelB.safetensors"])

    @patch("funkytown_testing_harness.run_test.fetch_available_models")
    def test_missing_model_is_skipped_but_run_continues(self, mock_fetch):
        mock_fetch.return_value = {"modelA.safetensors", "modelB.safetensors"}
        models = [
            {"model": "modelA.safetensors"},
            {"model": "modelB.safetensors"},
            {"model": "does_not_exist.safetensors"},
        ]
        present = resolve_present_models(models, self.template, "http://fake")
        self.assertEqual([m["model"] for m in present], ["modelA.safetensors", "modelB.safetensors"])

    @patch("funkytown_testing_harness.run_test.fetch_available_models")
    def test_fewer_than_two_present_raises(self, mock_fetch):
        mock_fetch.return_value = {"modelA.safetensors"}
        models = [{"model": "modelA.safetensors"}, {"model": "does_not_exist.safetensors"}]
        with self.assertRaises(SystemExit):
            resolve_present_models(models, self.template, "http://fake")

    @patch("funkytown_testing_harness.run_test.fetch_available_models")
    def test_single_configured_model_raises(self, mock_fetch):
        mock_fetch.return_value = {"modelA.safetensors"}
        models = [{"model": "modelA.safetensors"}]
        with self.assertRaises(SystemExit):
            resolve_present_models(models, self.template, "http://fake")

    @patch("funkytown_testing_harness.run_test.fetch_available_models")
    def test_zero_present_raises(self, mock_fetch):
        mock_fetch.return_value = set()
        models = [{"model": "modelA.safetensors"}, {"model": "modelB.safetensors"}]
        with self.assertRaises(SystemExit):
            resolve_present_models(models, self.template, "http://fake")


class BuildTemplateTests(unittest.TestCase):
    """build_template() always fetches live (mocked here) and reapplies
    strip_loras/positive_prompt from config on top, every call."""

    @patch("funkytown_testing_harness.run_test.load_live_template")
    def test_fetches_using_configured_source_workflow(self, mock_load):
        mock_load.return_value = make_template()
        config = {"source_workflow": "my_live_workflow.json"}
        build_template(config, "http://fake-server")
        mock_load.assert_called_once_with("http://fake-server", "my_live_workflow.json")

    @patch("funkytown_testing_harness.run_test.load_live_template")
    def test_does_not_strip_loras_unless_requested(self, mock_load):
        # No Power Lora Loader node in this fixture - if strip_loras were
        # mistakenly invoked it would just warn, not fail, so this mainly
        # documents that omitting the key is the default (covered for real
        # in test_live_workflow.py's StripLorasTests).
        mock_load.return_value = make_template()
        config = {"source_workflow": "wf.json"}
        template = build_template(config, "http://fake-server")
        self.assertEqual(template, make_template())

    @patch("funkytown_testing_harness.run_test.load_live_template")
    def test_applies_positive_prompt_override(self, mock_load):
        mock_load.return_value = make_template()
        config = {"source_workflow": "wf.json", "positive_prompt": "a brand new prompt"}
        template = build_template(config, "http://fake-server")
        self.assertEqual(template["3"]["inputs"]["text"], "a brand new prompt")


class RunEndToEndTests(unittest.TestCase):
    """Exercises the full run() flow: always-fresh fetch (mocked) + model swap +
    the single-config-vs-multiple-configs behavior, with ComfyUI network calls
    mocked out."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.tmp_path = Path(self.tmpdir.name)

        self.config = {
            "name": "unit_test_run",
            "source_workflow": "krea2_basic_t2i.json",
            "server": "http://fake",
            "models": [
                {
                    "model": "modelA.safetensors",
                    "configs": [
                        {"sampler_name": "euler", "steps": 4},
                        {"sampler_name": "er_sde", "steps": 10, "cfg": 2.0, "scheduler": "simple"},
                    ],
                },
                {
                    "model": "modelB.safetensors",
                    # no "configs" key at all - should run once with workflow defaults
                },
            ],
        }
        self.config_path = self.tmp_path / "config.json"
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")

        self.runs_dir = self.tmp_path / "runs"

        self.queued = []  # list of (server, workflow_copy, client_id)

        def fake_queue_prompt(server, workflow, client_id):
            self.queued.append((server, copy.deepcopy(workflow), client_id))
            return {"prompt_id": f"fake-{len(self.queued)}", "node_errors": {}}

        patcher_queue = patch("funkytown_testing_harness.run_test.queue_prompt", side_effect=fake_queue_prompt)
        patcher_load_template = patch(
            "funkytown_testing_harness.run_test.load_live_template",
            side_effect=lambda server, source_workflow: make_template(),
        )
        patcher_fetch = patch(
            "funkytown_testing_harness.run_test.fetch_available_models",
            return_value={"modelA.safetensors", "modelB.safetensors"},
        )
        patcher_runs_dir = patch("funkytown_testing_harness.run_test.RUNS_DIR", self.runs_dir)
        self.mock_queue_prompt = patcher_queue.start()
        self.mock_load_template = patcher_load_template.start()
        patcher_fetch.start()
        patcher_runs_dir.start()
        self.addCleanup(patcher_queue.stop)
        self.addCleanup(patcher_load_template.stop)
        self.addCleanup(patcher_fetch.stop)
        self.addCleanup(patcher_runs_dir.stop)

    def test_always_fetches_fresh_using_configured_source_workflow(self):
        run(self.config_path)
        self.mock_load_template.assert_called_once_with("http://fake", "krea2_basic_t2i.json")

    def test_model_with_multiple_configs_runs_once_per_config(self):
        run(self.config_path)
        model_a_runs = [q for q in self.queued if q[1]["1"]["inputs"]["unet_name"] == "modelA.safetensors"]
        self.assertEqual(len(model_a_runs), 2)

    def test_model_with_no_configs_runs_once_with_workflow_defaults(self):
        run(self.config_path)
        model_b_runs = [q for q in self.queued if q[1]["1"]["inputs"]["unet_name"] == "modelB.safetensors"]
        self.assertEqual(len(model_b_runs), 1)
        inputs = model_b_runs[0][1]["2"]["inputs"]
        self.assertEqual(inputs["sampler_name"], "euler")  # workflow default, untouched
        self.assertEqual(inputs["steps"], 4)               # workflow default, untouched

    def test_each_config_applies_its_own_ksampler_overrides(self):
        run(self.config_path)
        model_a_runs = [q for q in self.queued if q[1]["1"]["inputs"]["unet_name"] == "modelA.safetensors"]
        samplers = sorted(r[1]["2"]["inputs"]["sampler_name"] for r in model_a_runs)
        self.assertEqual(samplers, ["er_sde", "euler"])
        steps = sorted(r[1]["2"]["inputs"]["steps"] for r in model_a_runs)
        self.assertEqual(steps, [4, 10])

    def test_batch_size_from_workflow_is_never_overridden(self):
        run(self.config_path)
        for _server, wf, _client_id in self.queued:
            self.assertEqual(wf["5"]["inputs"]["batch_size"], 3)

    def test_filename_prefix_gets_cfg_suffix_only_when_multiple_configs(self):
        run(self.config_path)
        prefixes = {strip_run_id(wf["6"]["inputs"]["filename_prefix"]) for _s, wf, _c in self.queued}
        self.assertIn("tests/unit_test_run/modelA_cfg0", prefixes)
        self.assertIn("tests/unit_test_run/modelA_cfg1", prefixes)
        self.assertIn("tests/unit_test_run/modelB", prefixes)  # no suffix - only one config

    def test_filename_prefix_shares_one_run_id_per_run(self):
        run(self.config_path)
        prefixes = [wf["6"]["inputs"]["filename_prefix"] for _s, wf, _c in self.queued]
        run_ids = {re.match(r"tests/[^/]+/([0-9a-f]{8})/", p).group(1) for p in prefixes}
        self.assertEqual(len(run_ids), 1)  # every row in one run() call shares the same run_id

    def test_filename_prefix_queue_index_sorts_in_actual_queue_order(self):
        # modelA sorts alphabetically before modelB, but modelB is queued
        # LAST here (2 configs) - queue_index should still reflect the
        # actual order things were queued in, not model name order.
        run(self.config_path)
        by_queue_order = [wf["6"]["inputs"]["filename_prefix"] for _s, wf, _c in self.queued]
        by_filename_sort = sorted(by_queue_order)
        self.assertEqual(by_queue_order, by_filename_sort)

    def test_two_separate_runs_get_different_run_ids(self):
        run(self.config_path)
        first_run_ids = {re.match(r"tests/[^/]+/([0-9a-f]{8})/", wf["6"]["inputs"]["filename_prefix"]).group(1) for _s, wf, _c in self.queued}
        self.queued.clear()
        run(self.config_path)
        second_run_ids = {re.match(r"tests/[^/]+/([0-9a-f]{8})/", wf["6"]["inputs"]["filename_prefix"]).group(1) for _s, wf, _c in self.queued}
        self.assertTrue(first_run_ids.isdisjoint(second_run_ids))

    def test_total_queue_calls_match_model_and_config_counts(self):
        run(self.config_path)
        self.assertEqual(len(self.queued), 3)  # modelA x2 configs + modelB x1

    def test_log_csv_has_one_row_per_queued_variant(self):
        run(self.config_path)
        log_files = list(self.runs_dir.glob("unit_test_run_*.csv"))
        self.assertEqual(len(log_files), 1)
        with open(log_files[0], newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        self.assertEqual(len(rows) - 1, 3)  # header + 3 data rows

    def test_missing_model_is_skipped_and_run_still_succeeds(self):
        self.config["models"].append({"model": "does_not_exist.safetensors"})
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        run(self.config_path)
        self.assertEqual(len(self.queued), 3)  # unaffected - the missing one contributes nothing

    def test_fewer_than_two_present_models_aborts_before_queuing(self):
        self.config["models"] = [{"model": "modelA.safetensors"}]
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        with self.assertRaises(SystemExit):
            run(self.config_path)
        self.assertEqual(len(self.queued), 0)

    def test_strip_loras_and_positive_prompt_are_reapplied_every_run_when_configured(self):
        self.config["strip_loras"] = True
        self.config["positive_prompt"] = "a totally different sfw prompt"
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        run(self.config_path)
        for _server, wf, _client_id in self.queued:
            self.assertEqual(wf["3"]["inputs"]["text"], "a totally different sfw prompt")


class RunLoraRuleRoutingTests(unittest.TestCase):
    """Confirms run() wires comfy_prompt_tools' keyword LoRA routing into
    every queued variant, on top of whatever strip_loras/positive_prompt
    already did - see live_workflow.apply_lora_rules for the unit-level
    behavior."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.tmp_path = Path(self.tmpdir.name)

        self.config = {
            "name": "lora_rule_test",
            "source_workflow": "krea2_basic_t2i.json",
            "server": "http://fake",
            "positive_prompt": "a scene with a furry character",
            "models": [{"model": "modelA.safetensors"}, {"model": "modelB.safetensors"}],
        }
        self.config_path = self.tmp_path / "config.json"
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")

        self.runs_dir = self.tmp_path / "runs"
        self.queued = []

        def fake_queue_prompt(server, workflow, client_id):
            self.queued.append((server, copy.deepcopy(workflow), client_id))
            return {"prompt_id": f"fake-{len(self.queued)}", "node_errors": {}}

        patcher_queue = patch("funkytown_testing_harness.run_test.queue_prompt", side_effect=fake_queue_prompt)
        patcher_load_template = patch(
            "funkytown_testing_harness.run_test.load_live_template",
            side_effect=lambda server, source_workflow: make_template_with_lora(),
        )
        patcher_fetch = patch(
            "funkytown_testing_harness.run_test.fetch_available_models",
            return_value={"modelA.safetensors", "modelB.safetensors"},
        )
        patcher_runs_dir = patch("funkytown_testing_harness.run_test.RUNS_DIR", self.runs_dir)
        patcher_select = patch(
            "funkytown_testing_harness.live_workflow.select_loras",
            return_value=[("some_lora.safetensors", 0.75)],
        )
        self.mock_queue_prompt = patcher_queue.start()
        patcher_load_template.start()
        patcher_fetch.start()
        patcher_runs_dir.start()
        self.mock_select_loras = patcher_select.start()
        self.addCleanup(patcher_queue.stop)
        self.addCleanup(patcher_load_template.stop)
        self.addCleanup(patcher_fetch.stop)
        self.addCleanup(patcher_runs_dir.stop)
        self.addCleanup(patcher_select.stop)

    def test_keyword_matched_lora_turned_on(self):
        run(self.config_path)
        self.assertEqual(len(self.queued), 2)
        for _server, wf, _client_id in self.queued:
            self.assertTrue(wf["7"]["inputs"]["lora_1"]["on"])
            self.assertEqual(wf["7"]["inputs"]["lora_1"]["strength"], 0.75)

    def test_routes_against_the_effective_prompt_text(self):
        run(self.config_path)
        self.mock_select_loras.assert_called_with("a scene with a furry character")

    def test_strip_loras_removes_the_slot_so_routing_has_nothing_to_act_on(self):
        # strip_loras deletes every lora_N slot outright (not just turns it
        # off), so a keyword match afterward has no slot left to turn on -
        # see live_workflow.apply_lora_rules's docstring.
        self.config["strip_loras"] = True
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        run(self.config_path)
        for _server, wf, _client_id in self.queued:
            self.assertNotIn("lora_1", wf["7"]["inputs"])


class RunPromptSweepTests(unittest.TestCase):
    """"positive_prompts": [...] - every model x KSampler-config combination
    run once per prompt in the list."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.tmp_path = Path(self.tmpdir.name)

        self.config = {
            "name": "unit_test_prompt_sweep",
            "source_workflow": "krea2_basic_t2i.json",
            "server": "http://fake",
            "positive_prompts": ["a red car", "a blue car", "a green car"],
            "models": [
                {"model": "modelA.safetensors"},
                {"model": "modelB.safetensors"},
            ],
        }
        self.config_path = self.tmp_path / "config.json"
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")

        self.runs_dir = self.tmp_path / "runs"
        self.queued = []

        def fake_queue_prompt(server, workflow, client_id):
            self.queued.append((server, copy.deepcopy(workflow), client_id))
            return {"prompt_id": f"fake-{len(self.queued)}", "node_errors": {}}

        patcher_queue = patch("funkytown_testing_harness.run_test.queue_prompt", side_effect=fake_queue_prompt)
        patcher_load_template = patch(
            "funkytown_testing_harness.run_test.load_live_template",
            side_effect=lambda server, source_workflow: make_template(),
        )
        patcher_fetch = patch(
            "funkytown_testing_harness.run_test.fetch_available_models",
            return_value={"modelA.safetensors", "modelB.safetensors"},
        )
        patcher_runs_dir = patch("funkytown_testing_harness.run_test.RUNS_DIR", self.runs_dir)
        patcher_queue.start()
        patcher_load_template.start()
        patcher_fetch.start()
        patcher_runs_dir.start()
        self.addCleanup(patcher_queue.stop)
        self.addCleanup(patcher_load_template.stop)
        self.addCleanup(patcher_fetch.stop)
        self.addCleanup(patcher_runs_dir.stop)

    def test_queues_every_model_times_every_prompt(self):
        run(self.config_path)
        self.assertEqual(len(self.queued), 6)  # 2 models x 3 prompts

    def test_every_prompt_is_applied_to_the_workflow(self):
        run(self.config_path)
        prompts_used = {wf["3"]["inputs"]["text"] for _s, wf, _c in self.queued}
        self.assertEqual(prompts_used, {"a red car", "a blue car", "a green car"})

    def test_filename_prefix_encodes_prompt_index(self):
        run(self.config_path)
        prefixes = {strip_run_id(wf["6"]["inputs"]["filename_prefix"]) for _s, wf, _c in self.queued}
        self.assertIn("tests/unit_test_prompt_sweep/prompt0_modelA", prefixes)
        self.assertIn("tests/unit_test_prompt_sweep/prompt2_modelB", prefixes)

    def test_log_csv_gains_prompt_columns(self):
        run(self.config_path)
        log_files = list(self.runs_dir.glob("unit_test_prompt_sweep_*.csv"))
        self.assertEqual(len(log_files), 1)
        with open(log_files[0], newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        self.assertEqual(rows[0][:2], ["Prompt Index", "Prompt"])
        self.assertEqual(len(rows) - 1, 6)  # header + 6 data rows

    def test_both_prompt_keys_given_aborts_before_queuing(self):
        self.config["positive_prompt"] = "a single override"
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        with self.assertRaises(SystemExit):
            run(self.config_path)
        self.assertEqual(len(self.queued), 0)


if __name__ == "__main__":
    unittest.main()
