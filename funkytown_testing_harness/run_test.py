"""
Run a model-comparison test-case config against a running ComfyUI server: the
same workflow/prompt, swapped across a list of models (each with its own
optional KSampler settings), one queued generation per (model, KSampler
config) combination.

The source workflow is always fetched fresh from ComfyUI and converted on
every run (see live_workflow.py) - there's no static snapshot on disk to go
stale, and no separate sync step to remember before running a test.

There is no pass/fail here. This submits each variant to ComfyUI and logs what
was queued (model, KSampler settings, prompt_id, output filename prefix) to a
CSV under runs/, so you can compare the resulting images yourself.

Config file format (JSON):
    {
        "name": "model_testing",
        "source_workflow": "krea2_basic_t2i.json",
        "strip_loras": true,
        "positive_prompt": "A high-resolution realistic photo of ...",
        "server": "http://127.0.0.1:8000",
        "models": [
            {
                "model": "krea2SATDirtyrealism_krea2SAT.safetensors",
                "configs": [
                    {"sampler_name": "euler", "steps": 8, "cfg": 1, "scheduler": "beta"}
                ]
            },
            {
                "model": "bf95Krea2DarkRealism_v325.safetensors",
                "configs": [
                    {"sampler_name": "er_sde", "steps": 10, "cfg": 1.0, "scheduler": "simple"}
                ]
            }
        ]
    }

- "source_workflow" - filename of a workflow saved in ComfyUI's own
  user/default/workflows folder. Pulled fresh from ComfyUI and converted to
  API format every time this runs (requires playwright - see
  live_workflow.py). Its batch_size and anything else not explicitly
  overridden below is used exactly as it currently is in ComfyUI.
- "strip_loras" / "positive_prompt" - optional, reapplied to the freshly
  fetched workflow every run: clears the Power Lora Loader node and/or
  overwrites the positive prompt text. After these, comfy_prompt_tools'
  keyword -> LoRA routing (lora_rules.json / lora_rules.local.json) is
  applied against whatever the effective prompt text ends up being (the
  override above, or the workflow's own default if none given) - but only
  a LoRA slot that still structurally exists can be turned on this way, so
  "strip_loras": true (which removes every slot outright, not just turns
  them off) leaves nothing for it to act on. Leave strip_loras unset/false
  if you want keyword-matched LoRAs to actually take effect.
- "positive_prompts" - optional list of prompt strings, mutually exclusive
  with "positive_prompt" (config is rejected if both are given). Sweeps
  every model/config combination once per prompt in the list - e.g. 2
  models and 3 prompts queues 6 (or more, with multiple configs) runs. The
  CSV log gains "Prompt Index"/"Prompt" columns and each output filename
  prefix gets a "promptN_" segment, only when this is used.
- "group_by_model" - optional, default false. Only matters with
  "positive_prompts" (2+ prompts) - by default every model is queued once
  per prompt (prompt-major order), which cycles back through every model
  on each prompt boundary and so forces ComfyUI to reload the checkpoint
  model on nearly every queued item. Setting this queues every
  prompt/config for one model before moving to the next (model-major
  order) instead - the model loads once and stays loaded for everything
  it needs, cutting down how often ComfyUI has to swap models. See
  iter_variants(). No effect with a single prompt. Since this order no
  longer naturally clusters a prompt's images together by queue_index (see
  below), each filename prefix also gains a short hash of its prompt text
  ahead of queue_index in this mode - see prompt_short_hash().
- "models" - list of model objects, each with:
  - "model" - a model filename. The workflow's model-loader node
    (UNETLoader for diffusion-only weights like Krea2, or
    CheckpointLoaderSimple for a combined checkpoint) is repointed at it.
    Checked against ComfyUI's own live model list (/object_info) before
    running - one not currently installed is skipped with a warning. At
    least 1 configured model must be present or the run aborts with an
    error; a single present model is fine - useful for just testing
    prompts against one model rather than comparing several.
  - "configs" - optional list of KSampler overrides (seed, steps, cfg,
    sampler_name, scheduler, denoise). Omit for the workflow's own KSampler
    settings, EXCEPT seed - every prompt gets its own fresh random seed
    (see random_seed() in live_workflow.py), shared by every model/config
    tested against that specific prompt so the model/config is the only
    thing that changes between them, regardless of whatever numeric seed
    the live-fetched workflow happens to have. A config can still
    explicitly set "seed" to override that for just its own entries; give
    multiple entries to run that model once per entry.
- "server" - optional, defaults to http://127.0.0.1:8000.

Output filename prefix (and so the folder images land in under ComfyUI's
output directory) is
"tests/<name>/<run_id>/<queue_index>_<model stem>[_cfgN]" - run_id is a
short (8 hex char) random id generated fresh each run() call, so two runs
sharing the same "name" land in separate folders instead of comingling
their images together. queue_index is a zero-padded 4-digit counter over
every variant queued this run (starting at 0001, in queue order), so
sorting the output folder by filename always matches the order they were
actually queued in, regardless of how model names alphabetize - except
with "group_by_model" and 2+ prompts, where an 8-char prompt hash leads
queue_index instead (see "group_by_model" above and prompt_short_hash() in
live_workflow.py), trading "sorted by name matches queue order" for
"sorted by name groups each prompt's images together across every model".

Usage:
    python -m funkytown_testing_harness.run_test configs/model-testing-config.json
"""

import argparse
import copy
import csv
import datetime
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

# comfy-prompt-tools is a sibling checkout, not an installed package - see
# README. If something (e.g. the GUI, using a custom path from settings)
# already made it importable, that takes precedence over the sibling guess.
try:
    from comfy_prompt_tools.rerun_prompts_comfyui import find_save_image_node_ids, queue_prompt
except ImportError:
    _COMFY_PROMPT_TOOLS = Path(__file__).resolve().parent.parent.parent / "comfy-prompt-tools"
    sys.path.insert(0, str(_COMFY_PROMPT_TOOLS))
    try:
        from comfy_prompt_tools.rerun_prompts_comfyui import find_save_image_node_ids, queue_prompt
    except ImportError:
        sys.exit(
            f"Error: could not import comfy_prompt_tools from {_COMFY_PROMPT_TOOLS}.\n"
            "Expected comfy-prompt-tools checked out as a sibling directory next to "
            "funkytown-testing-harness (or already importable via sys.path)."
        )

from funkytown_testing_harness.live_workflow import apply_lora_rules, config_prompts, find_ksampler_node_id, load_live_template, prompt_short_hash, random_seed, set_positive_prompt, set_seed, strip_loras
from funkytown_testing_harness.model_swap import find_model_loader_nodes, set_model

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"

KSAMPLER_OVERRIDE_KEYS = {"seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"}


def load_config(config_path):
    return json.loads(config_path.read_text(encoding="utf-8"))


def build_template(config, server):
    source_workflow = config["source_workflow"]
    print(f"Fetching '{source_workflow}' fresh from ComfyUI at {server} ...")
    template = load_live_template(server, source_workflow)
    print(f"Converted: {len(template)} active node(s).")

    if config.get("strip_loras"):
        strip_loras(template)
    if config.get("positive_prompt"):
        set_positive_prompt(template, config["positive_prompt"])

    return template


def fetch_available_models(server, class_type, field):
    """Query ComfyUI's own /object_info for the live list of model filenames
    it currently recognizes for a given loader node type - this accounts for
    extra_model_paths.yaml and whatever's actually installed, rather than
    guessing at filesystem layout."""
    url = f"{server}/object_info/{class_type}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            info = json.loads(resp.read().decode("utf-8"))
        return set(info[class_type]["input"]["required"][field][0])
    except (urllib.error.URLError, KeyError, IndexError, TypeError) as e:
        sys.exit(f"Error: could not fetch available models from {url}: {e}")


def resolve_present_models(models_config, template, server):
    """Check each configured model against ComfyUI's live model list. Returns
    the subset that are actually present (skipping - with a warning - any
    that aren't), or exits with an error if none are present. A single
    present model is fine - this tool doesn't require a comparison."""
    loader_nodes = find_model_loader_nodes(template)
    if not loader_nodes:
        sys.exit("Error: no recognized model-loader node (UNETLoader/CheckpointLoader) found in workflow.")

    # Cache available-model sets per (class_type, field) - a workflow only ever
    # has one loader node in practice, but this stays correct if it had more.
    available_cache = {}
    present = []
    for entry in models_config:
        model_name = entry["model"]
        found = False
        for _node_id, field, class_type in loader_nodes:
            key = (class_type, field)
            if key not in available_cache:
                available_cache[key] = fetch_available_models(server, class_type, field)
            if model_name in available_cache[key]:
                found = True
                break
        if found:
            present.append(entry)
        else:
            print(f"[{model_name}] Skipping: not found on this ComfyUI server", file=sys.stderr)

    if not present:
        sys.exit(
            f"Error: none of the {len(models_config)} configured model(s) are present on this ComfyUI server."
        )
    return present


def iter_variants(prompts, present_models, group_by_model=False):
    """Yields (p_idx, prompt_text, entry, config_index, overrides) for every
    variant to queue, in the exact order they'll actually be queued (and so
    numbered in output filenames/queue_index).

    Default order is prompt-major: every model (and its configs) once per
    prompt, prompt by prompt. With more than one prompt, this cycles back
    through every model on each prompt boundary, forcing ComfyUI to swap
    the loaded checkpoint model on nearly every queued item.

    group_by_model=True reorders to model-major instead: every prompt (and
    each model's own configs) for one model, before moving to the next
    model - the model loads once and stays loaded for everything that
    model needs, instead of being swapped out and back in on every prompt
    boundary. Only actually changes anything when there's more than one
    prompt; with a single prompt the two orders are identical."""
    if group_by_model:
        for entry in present_models:
            configs = entry.get("configs") or [{}]
            for i, overrides in enumerate(configs):
                for p_idx, prompt_text in enumerate(prompts):
                    yield p_idx, prompt_text, entry, i, overrides
    else:
        for p_idx, prompt_text in enumerate(prompts):
            for entry in present_models:
                configs = entry.get("configs") or [{}]
                for i, overrides in enumerate(configs):
                    yield p_idx, prompt_text, entry, i, overrides


def apply_ksampler_overrides(workflow, ksampler_id, overrides):
    unknown = set(overrides) - KSAMPLER_OVERRIDE_KEYS
    if unknown:
        print(f"  warning: ignoring unrecognized KSampler override key(s): {sorted(unknown)}", file=sys.stderr)
    for key, value in overrides.items():
        if key in KSAMPLER_OVERRIDE_KEYS:
            workflow[ksampler_id]["inputs"][key] = value


def run(config_path):
    config = load_config(config_path)
    server = config.get("server", "http://127.0.0.1:8000")
    name = config.get("name", config_path.stem)

    print(f"Test case: {name}")
    template = build_template(config, server)

    save_ids = find_save_image_node_ids(template)
    ksampler_id = find_ksampler_node_id(template)
    client_id = str(uuid.uuid4())
    # Leads with a shortened timestamp (not just a random id) so the output
    # folder itself says when the run happened and sorts chronologically by
    # name - the short random suffix still guarantees two runs starting in
    # the same second never land in the same folder.
    run_id = f"{datetime.datetime.now():%y%m%d_%H%M%S}_{uuid.uuid4().hex[:4]}"

    present_models = resolve_present_models(config["models"], template, server)
    prompts = config_prompts(config)
    multi_prompt = len(prompts) > 1
    group_by_model = bool(config.get("group_by_model"))
    # One random seed per prompt (not per queued variant, and not shared
    # across prompts either) - see random_seed()'s docstring.
    prompt_seeds = [random_seed() for _ in prompts]

    RUNS_DIR.mkdir(exist_ok=True)
    log_path = RUNS_DIR / f"{name}_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"

    print(f"Models present ({len(present_models)}/{len(config['models'])}): "
          f"{', '.join(m['model'] for m in present_models)}")
    if multi_prompt:
        print(f"Prompts: {len(prompts)}")
        if group_by_model:
            print("Queuing grouped by model (all prompts/configs per model before moving to the next).")
    print()

    header = ["Model", "KSampler Overrides", "Prompt ID", "Status", "Filename Prefix", "Detail"]
    if multi_prompt:
        header = ["Prompt Index", "Prompt"] + header

    with open(log_path, "w", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(header)

        variants = iter_variants(prompts, present_models, group_by_model)
        for queue_index, (p_idx, prompt_text, entry, i, overrides) in enumerate(variants, start=1):
            model = entry["model"]
            configs = entry.get("configs") or [{}]

            wf = copy.deepcopy(template)
            set_model(wf, model)
            if prompt_text:
                set_positive_prompt(wf, prompt_text)
            apply_lora_rules(wf)
            set_seed(wf, prompt_seeds[p_idx])  # before overrides, so an explicit "seed" in a config below still wins

            if overrides:
                if not ksampler_id:
                    print(f"[{model}] Warning: KSampler overrides given but no KSampler node found", file=sys.stderr)
                else:
                    apply_ksampler_overrides(wf, ksampler_id, overrides)

            suffix = f"_cfg{i}" if len(configs) > 1 else ""
            prompt_part = f"prompt{p_idx}_" if multi_prompt else ""
            # group_by_model reorders queuing to model-major, so the
            # queue_index that otherwise leads each prefix would group
            # filenames by model when sorted by name instead of by prompt -
            # leading with a short hash of the prompt text here restores
            # "sort by name to group one prompt's images together" (see
            # prompt_short_hash()).
            hash_part = f"{prompt_short_hash(prompt_text)}_" if (multi_prompt and group_by_model) else ""
            prefix = f"tests/{name}/{run_id}/{hash_part}{queue_index:04d}_{prompt_part}{Path(model).stem}{suffix}"
            for save_id in save_ids:
                wf[save_id]["inputs"]["filename_prefix"] = prefix

            overrides_summary = json.dumps(overrides) if overrides else "(workflow defaults)"
            row_prefix = [p_idx, prompt_text] if multi_prompt else []

            try:
                result = queue_prompt(server, wf, client_id)
            except urllib.error.URLError as e:
                print(f"[{model}] Failed to queue: {e}", file=sys.stderr)
                writer.writerow(row_prefix + [model, overrides_summary, "", "error", prefix, f"Failed to queue: {e}"])
                continue

            node_errors = result.get("node_errors")
            prompt_id = result.get("prompt_id")
            if node_errors:
                print(f"[{model}] node errors: {node_errors}")
                writer.writerow(row_prefix + [model, overrides_summary, prompt_id or "", "error", prefix, json.dumps(node_errors)])
                continue

            print(f"[{model}] {overrides_summary} -> queued as prompt_id={prompt_id}, output prefix '{prefix}'")
            writer.writerow(row_prefix + [model, overrides_summary, prompt_id, "queued", prefix, ""])
            log_file.flush()
            time.sleep(0.2)

    print(f"\nAll variants queued. Log written to: {log_path}")
    print("ComfyUI processes its queue in the background - check its window or output folder for results.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="Path to a test-case config JSON file")
    args = parser.parse_args()
    run(Path(args.config))


if __name__ == "__main__":
    main()
