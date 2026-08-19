"""Toggle a single LoRA on (at a given strength) in a workflow's Power Lora
Loader (rgthree) node, isolating it - every other configured LoRA slot in
that node gets turned off. For sweeping one LoRA's weight at a time rather
than combining several LoRAs together (see comfy_prompt_tools.rerun_prompts_comfyui's
apply_loras/LORA_RULES for that combined-keyword-matching use case instead).

A slot can only be toggled, not created - the Power Lora Loader node must
already have an entry for the target LoRA filename (added via ComfyUI's
"+ Add Lora" widget beforehand), same constraint as LORA_RULES.

Finding the Power Lora Loader node itself is done via
comfy_prompt_tools.rerun_prompts_comfyui.find_power_lora_loader_id - this
module only operates on a node_id once you have it.
"""


def find_lora_slot_keys(workflow, lora_node_id):
    """Return the list of input keys on lora_node_id that look like LoRA
    slots (structural match, same heuristic as find_power_lora_loader_id:
    a dict value with 'on'/'lora'/'strength' keys)."""
    inputs = workflow[lora_node_id].get("inputs", {})
    return [
        key for key, value in inputs.items()
        if isinstance(value, dict) and {"on", "lora", "strength"} <= set(value.keys())
    ]


def set_single_lora(workflow, lora_node_id, lora_filename, strength):
    """Turn on exactly the slot matching lora_filename at the given strength,
    and turn every other LoRA slot on lora_node_id off. Returns True if the
    target slot was found (and the change applied), False otherwise - in
    which case nothing is turned off either, so a typo'd/missing LoRA name
    doesn't silently disable everything else."""
    inputs = workflow[lora_node_id]["inputs"]
    slot_keys = find_lora_slot_keys(workflow, lora_node_id)

    target_key = next((k for k in slot_keys if inputs[k].get("lora") == lora_filename), None)
    if target_key is None:
        return False

    for key in slot_keys:
        inputs[key]["on"] = key == target_key
    inputs[target_key]["strength"] = strength
    return True
