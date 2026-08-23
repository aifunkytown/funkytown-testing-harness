import unittest

from funkytown_testing_harness.lora_swap import find_lora_slot_keys, set_multiple_loras, set_single_lora


def make_lora_node():
    return {
        "22": {
            "class_type": "Power Lora Loader (rgthree)",
            "inputs": {
                "model": ["36", 0],
                "clip": ["59", 0],
                "lora_1": {"on": False, "lora": "styleA.safetensors", "strength": 0.9},
                "lora_2": {"on": True, "lora": "detail_slider.safetensors", "strength": 2},
                "lora_3": {"on": False, "lora": "skindetails.safetensors", "strength": 1},
            },
        }
    }


class FindLoraSlotKeysTests(unittest.TestCase):
    def test_finds_all_slot_keys(self):
        wf = make_lora_node()
        self.assertEqual(sorted(find_lora_slot_keys(wf, "22")), ["lora_1", "lora_2", "lora_3"])

    def test_excludes_non_slot_inputs(self):
        wf = make_lora_node()
        keys = find_lora_slot_keys(wf, "22")
        self.assertNotIn("model", keys)
        self.assertNotIn("clip", keys)

    def test_empty_when_no_slots(self):
        wf = {"22": {"class_type": "Power Lora Loader (rgthree)", "inputs": {"model": ["1", 0], "clip": ["1", 0]}}}
        self.assertEqual(find_lora_slot_keys(wf, "22"), [])


class SetSingleLoraTests(unittest.TestCase):
    def test_turns_on_target_and_sets_strength(self):
        wf = make_lora_node()
        found = set_single_lora(wf, "22", "skindetails.safetensors", 1.75)
        self.assertTrue(found)
        self.assertEqual(wf["22"]["inputs"]["lora_3"]["on"], True)
        self.assertEqual(wf["22"]["inputs"]["lora_3"]["strength"], 1.75)

    def test_turns_off_every_other_slot(self):
        wf = make_lora_node()
        set_single_lora(wf, "22", "skindetails.safetensors", 1.0)
        self.assertFalse(wf["22"]["inputs"]["lora_1"]["on"])
        self.assertFalse(wf["22"]["inputs"]["lora_2"]["on"])

    def test_isolates_even_a_previously_on_slot(self):
        # lora_2 starts "on": True in the fixture - confirm it gets turned off
        # when a different lora is the target.
        wf = make_lora_node()
        set_single_lora(wf, "22", "styleA.safetensors", 0.5)
        self.assertFalse(wf["22"]["inputs"]["lora_2"]["on"])
        self.assertTrue(wf["22"]["inputs"]["lora_1"]["on"])

    def test_does_not_touch_model_clip_passthrough(self):
        wf = make_lora_node()
        set_single_lora(wf, "22", "styleA.safetensors", 0.5)
        self.assertEqual(wf["22"]["inputs"]["model"], ["36", 0])
        self.assertEqual(wf["22"]["inputs"]["clip"], ["59", 0])

    def test_returns_false_and_changes_nothing_when_lora_not_found(self):
        wf = make_lora_node()
        before = {k: dict(v) for k, v in wf["22"]["inputs"].items() if isinstance(v, dict)}
        found = set_single_lora(wf, "22", "does_not_exist.safetensors", 1.0)
        self.assertFalse(found)
        after = {k: dict(v) for k, v in wf["22"]["inputs"].items() if isinstance(v, dict)}
        self.assertEqual(before, after)

    def test_weight_sweep_each_call_independent(self):
        wf = make_lora_node()
        for weight in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
            set_single_lora(wf, "22", "styleA.safetensors", weight)
            self.assertEqual(wf["22"]["inputs"]["lora_1"]["strength"], weight)
            self.assertTrue(wf["22"]["inputs"]["lora_1"]["on"])
            self.assertFalse(wf["22"]["inputs"]["lora_2"]["on"])
            self.assertFalse(wf["22"]["inputs"]["lora_3"]["on"])


class SetMultipleLorasTests(unittest.TestCase):
    def test_turns_on_all_targets_at_their_own_strength(self):
        wf = make_lora_node()
        missing = set_multiple_loras(wf, "22", [("styleA.safetensors", 0.3), ("skindetails.safetensors", 1.75)])
        self.assertEqual(missing, [])
        self.assertTrue(wf["22"]["inputs"]["lora_1"]["on"])
        self.assertEqual(wf["22"]["inputs"]["lora_1"]["strength"], 0.3)
        self.assertTrue(wf["22"]["inputs"]["lora_3"]["on"])
        self.assertEqual(wf["22"]["inputs"]["lora_3"]["strength"], 1.75)

    def test_turns_off_every_slot_not_targeted(self):
        wf = make_lora_node()
        # lora_2 starts "on": True in the fixture - confirm it's turned off
        # since it's not one of the targets this time.
        set_multiple_loras(wf, "22", [("styleA.safetensors", 0.3), ("skindetails.safetensors", 1.75)])
        self.assertFalse(wf["22"]["inputs"]["lora_2"]["on"])

    def test_all_or_nothing_when_one_target_missing(self):
        wf = make_lora_node()
        before = {k: dict(v) for k, v in wf["22"]["inputs"].items() if isinstance(v, dict)}
        missing = set_multiple_loras(wf, "22", [("styleA.safetensors", 0.3), ("does_not_exist.safetensors", 1.0)])
        self.assertEqual(missing, ["does_not_exist.safetensors"])
        after = {k: dict(v) for k, v in wf["22"]["inputs"].items() if isinstance(v, dict)}
        self.assertEqual(before, after)  # nothing touched, including the target that WAS found

    def test_does_not_touch_model_clip_passthrough(self):
        wf = make_lora_node()
        set_multiple_loras(wf, "22", [("styleA.safetensors", 0.3)])
        self.assertEqual(wf["22"]["inputs"]["model"], ["36", 0])
        self.assertEqual(wf["22"]["inputs"]["clip"], ["59", 0])

    def test_single_pair_behaves_like_set_single_lora(self):
        wf_a = make_lora_node()
        wf_b = make_lora_node()
        set_single_lora(wf_a, "22", "skindetails.safetensors", 1.75)
        set_multiple_loras(wf_b, "22", [("skindetails.safetensors", 1.75)])
        self.assertEqual(wf_a, wf_b)


if __name__ == "__main__":
    unittest.main()
