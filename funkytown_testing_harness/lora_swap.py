"""Toggle LoRA slot(s) on (at given strength(s)) in a workflow's Power Lora
Loader (rgthree) node - either a single LoRA isolated (every other slot
forced off) or several combined together at once, still with every
unlisted slot forced off.

A slot can only be toggled, not created - the Power Lora Loader node must
already have an entry for each target LoRA filename (added via ComfyUI's
"+ Add Lora" widget beforehand). See also
comfy_prompt_tools.rerun_prompts_comfyui's apply_loras/LORA_RULES for a
different combined-LoRA use case (keyword-matched against prompt text rather
than an explicit weight sweep).

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
    return not set_multiple_loras(workflow, lora_node_id, [(lora_filename, strength)])


def set_multiple_loras(workflow, lora_node_id, lora_weight_pairs):
    """Turn on exactly the slots matching each (lora_filename, strength) pair
    given, all at once - for combining several LoRAs together in one run,
    rather than isolating a single one. Every other LoRA slot on lora_node_id
    is turned off. Returns the list of lora_filename values from
    lora_weight_pairs that had no matching slot (empty list = all found and
    applied) - all-or-nothing: if any target is missing, nothing is changed
    at all, matching set_single_lora's guarantee that a typo'd/missing LoRA
    name doesn't silently touch the rest of the node."""
    inputs = workflow[lora_node_id]["inputs"]
    slot_keys = find_lora_slot_keys(workflow, lora_node_id)
    targets = dict(lora_weight_pairs)  # lora_filename -> strength

    slot_for_lora = {}
    for key in slot_keys:
        slot_lora = inputs[key].get("lora")
        if slot_lora in targets:
            slot_for_lora[slot_lora] = key

    missing = [name for name in targets if name not in slot_for_lora]
    if missing:
        return missing

    for key in slot_keys:
        inputs[key]["on"] = key in slot_for_lora.values()
    for lora_filename, key in slot_for_lora.items():
        inputs[key]["strength"] = targets[lora_filename]

    return []
