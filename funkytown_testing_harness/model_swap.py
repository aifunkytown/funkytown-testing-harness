"""Swap the model a workflow loads, without touching anything else in the graph.

Different ComfyUI node types load a model under different input names -
UNETLoader for diffusion-only weights (e.g. Krea2), CheckpointLoaderSimple for
a combined checkpoint file. MODEL_LOADER_FIELDS maps recognized loader
class_types to the input key that holds the model filename.
"""

MODEL_LOADER_FIELDS = {
    "UNETLoader": "unet_name",
    "CheckpointLoaderSimple": "ckpt_name",
    "CheckpointLoader": "ckpt_name",
}


def find_model_loader_nodes(workflow):
    """Return [(node_id, field_name, class_type), ...] for every recognized
    model-loader node."""
    found = []
    for node_id, node in workflow.items():
        class_type = node.get("class_type", "")
        field = MODEL_LOADER_FIELDS.get(class_type)
        if field:
            found.append((node_id, field, class_type))
    return found


def set_model(workflow, model_filename):
    """Point every recognized model-loader node in workflow at model_filename.
    Returns the list of (node_id, field_name, class_type) that were changed."""
    changed = find_model_loader_nodes(workflow)
    for node_id, field, _class_type in changed:
        workflow[node_id]["inputs"][field] = model_filename
    return changed
