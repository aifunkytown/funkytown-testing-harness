import json
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

from funkytown_testing_harness.live_workflow import (
    fetch_live_workflow,
    is_api_format,
    load_live_template,
    set_positive_prompt,
    strip_loras,
)


def make_template_with_lora():
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "placeholder.safetensors"}},
        "2": {"class_type": "KSampler", "inputs": {"model": ["7", 0], "positive": ["3", 0], "negative": ["4", 0]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "original prompt", "clip": ["1", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["1", 0]}},
        "7": {
            "class_type": "Power Lora Loader (rgthree)",
            "inputs": {
                "model": ["1", 0],
                "clip": ["1", 0],
                "lora_1": {"on": False, "lora": "some_lora.safetensors", "strength": 0.8},
                "lora_2": {"on": True, "lora": "other_lora.safetensors", "strength": 1.0},
            },
        },
    }


class IsApiFormatTests(unittest.TestCase):
    def test_valid_api_format(self):
        self.assertTrue(is_api_format({"1": {"class_type": "KSampler"}}))

    def test_empty_dict_is_not_api_format(self):
        self.assertFalse(is_api_format({}))

    def test_ui_format_is_not_api_format(self):
        self.assertFalse(is_api_format({"nodes": [], "links": []}))

    def test_non_dict_is_not_api_format(self):
        self.assertFalse(is_api_format(["not", "a", "dict"]))


class StripLorasTests(unittest.TestCase):
    def test_removes_all_lora_slots(self):
        wf = make_template_with_lora()
        strip_loras(wf)
        inputs = wf["7"]["inputs"]
        self.assertNotIn("lora_1", inputs)
        self.assertNotIn("lora_2", inputs)

    def test_keeps_model_and_clip_passthrough(self):
        wf = make_template_with_lora()
        strip_loras(wf)
        self.assertEqual(wf["7"]["inputs"]["model"], ["1", 0])
        self.assertEqual(wf["7"]["inputs"]["clip"], ["1", 0])

    def test_noop_when_no_lora_loader_present(self):
        wf = {"1": {"class_type": "KSampler", "inputs": {}}}
        strip_loras(wf)  # should not raise
        self.assertEqual(wf, {"1": {"class_type": "KSampler", "inputs": {}}})


class SetPositivePromptTests(unittest.TestCase):
    def test_overwrites_positive_prompt_text(self):
        wf = make_template_with_lora()
        set_positive_prompt(wf, "a brand new sfw prompt")
        self.assertEqual(wf["3"]["inputs"]["text"], "a brand new sfw prompt")

    def test_leaves_negative_prompt_untouched(self):
        wf = make_template_with_lora()
        set_positive_prompt(wf, "a brand new sfw prompt")
        self.assertEqual(wf["4"]["inputs"]["text"], "")

    def test_noop_when_no_positive_node_found(self):
        wf = {"1": {"class_type": "KSampler", "inputs": {}}}
        set_positive_prompt(wf, "anything")  # should not raise


class FetchLiveWorkflowTests(unittest.TestCase):
    @patch("funkytown_testing_harness.live_workflow.urllib.request.urlopen")
    def test_requests_correct_url_and_parses_json(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"nodes": [], "links": []}).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        result = fetch_live_workflow("http://fake-server", "my_workflow.json")

        self.assertEqual(result, {"nodes": [], "links": []})
        called_url = mock_urlopen.call_args[0][0]
        self.assertIn("http://fake-server/api/userdata/", called_url)
        self.assertIn("my_workflow.json", called_url)

    @patch("funkytown_testing_harness.live_workflow.urllib.request.urlopen")
    def test_url_error_exits(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("connection refused")
        with self.assertRaises(SystemExit):
            fetch_live_workflow("http://fake-server", "my_workflow.json")


class LoadLiveTemplateTests(unittest.TestCase):
    @patch("funkytown_testing_harness.live_workflow.fetch_live_workflow")
    def test_already_api_format_passthrough(self, mock_fetch):
        api_wf = {"1": {"class_type": "KSampler", "inputs": {}}}
        mock_fetch.return_value = api_wf
        result = load_live_template("http://fake", "wf.json")
        self.assertEqual(result, api_wf)

    @patch("funkytown_testing_harness.live_workflow.fetch_live_workflow")
    def test_unwraps_prompt_wrapper(self, mock_fetch):
        api_wf = {"1": {"class_type": "KSampler", "inputs": {}}}
        mock_fetch.return_value = {"prompt": api_wf, "extra_data": {}}
        result = load_live_template("http://fake", "wf.json")
        self.assertEqual(result, api_wf)

    @patch("funkytown_testing_harness.live_workflow.convert_ui_workflow_via_browser")
    @patch("funkytown_testing_harness.live_workflow.fetch_live_workflow")
    def test_ui_format_triggers_browser_conversion(self, mock_fetch, mock_convert):
        ui_wf = {"nodes": [{"id": 1}], "links": []}
        api_wf = {"1": {"class_type": "KSampler", "inputs": {}}}
        mock_fetch.return_value = ui_wf
        mock_convert.return_value = api_wf

        result = load_live_template("http://fake-server", "wf.json")

        mock_convert.assert_called_once_with("http://fake-server", ui_wf)
        self.assertEqual(result, api_wf)

    @patch("funkytown_testing_harness.live_workflow.fetch_live_workflow")
    def test_unrecognized_format_exits(self, mock_fetch):
        mock_fetch.return_value = {"totally": "unrecognized"}
        with self.assertRaises(SystemExit):
            load_live_template("http://fake", "wf.json")


if __name__ == "__main__":
    unittest.main()
