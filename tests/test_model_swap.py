import unittest

from funkytown_testing_harness.model_swap import find_model_loader_nodes, set_model


def make_workflow(loader_class_type, loader_field):
    return {
        "1": {
            "class_type": loader_class_type,
            "inputs": {loader_field: "original.safetensors", "weight_dtype": "default"},
        },
        "2": {
            "class_type": "KSampler",
            "inputs": {"model": ["1", 0]},
        },
    }


class FindModelLoaderNodesTests(unittest.TestCase):
    def test_finds_unet_loader(self):
        wf = make_workflow("UNETLoader", "unet_name")
        self.assertEqual(find_model_loader_nodes(wf), [("1", "unet_name", "UNETLoader")])

    def test_finds_checkpoint_loader_simple(self):
        wf = make_workflow("CheckpointLoaderSimple", "ckpt_name")
        self.assertEqual(find_model_loader_nodes(wf), [("1", "ckpt_name", "CheckpointLoaderSimple")])

    def test_finds_checkpoint_loader(self):
        wf = make_workflow("CheckpointLoader", "ckpt_name")
        self.assertEqual(find_model_loader_nodes(wf), [("1", "ckpt_name", "CheckpointLoader")])

    def test_ignores_unrelated_node_types(self):
        wf = {"1": {"class_type": "KSampler", "inputs": {}}, "2": {"class_type": "SaveImage", "inputs": {}}}
        self.assertEqual(find_model_loader_nodes(wf), [])

    def test_finds_multiple_loader_nodes(self):
        wf = make_workflow("UNETLoader", "unet_name")
        wf["3"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "other.safetensors"}}
        found = sorted(find_model_loader_nodes(wf))
        expected = sorted([("1", "unet_name", "UNETLoader"), ("3", "ckpt_name", "CheckpointLoaderSimple")])
        self.assertEqual(found, expected)


class SetModelTests(unittest.TestCase):
    def test_sets_unet_name(self):
        wf = make_workflow("UNETLoader", "unet_name")
        changed = set_model(wf, "new_model.safetensors")
        self.assertEqual(wf["1"]["inputs"]["unet_name"], "new_model.safetensors")
        self.assertEqual(changed, [("1", "unet_name", "UNETLoader")])

    def test_sets_ckpt_name(self):
        wf = make_workflow("CheckpointLoaderSimple", "ckpt_name")
        set_model(wf, "new_checkpoint.safetensors")
        self.assertEqual(wf["1"]["inputs"]["ckpt_name"], "new_checkpoint.safetensors")

    def test_leaves_other_inputs_untouched(self):
        wf = make_workflow("UNETLoader", "unet_name")
        set_model(wf, "new_model.safetensors")
        self.assertEqual(wf["1"]["inputs"]["weight_dtype"], "default")

    def test_leaves_unrelated_nodes_untouched(self):
        wf = make_workflow("UNETLoader", "unet_name")
        before = dict(wf["2"]["inputs"])
        set_model(wf, "new_model.safetensors")
        self.assertEqual(wf["2"]["inputs"], before)

    def test_returns_empty_list_when_no_loader_present(self):
        wf = {"1": {"class_type": "KSampler", "inputs": {}}}
        changed = set_model(wf, "anything.safetensors")
        self.assertEqual(changed, [])
        self.assertNotIn("anything.safetensors", str(wf))

    def test_updates_all_recognized_loader_nodes_to_same_model(self):
        wf = make_workflow("UNETLoader", "unet_name")
        wf["3"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "old.safetensors"}}
        set_model(wf, "shared_model.safetensors")
        self.assertEqual(wf["1"]["inputs"]["unet_name"], "shared_model.safetensors")
        self.assertEqual(wf["3"]["inputs"]["ckpt_name"], "shared_model.safetensors")


if __name__ == "__main__":
    unittest.main()
