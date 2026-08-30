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
- "models" - list of model objects, each with:
  - "model" - a model filename. The workflow's model-loader node
    (UNETLoader for diffusion-only weights like Krea2, or
    CheckpointLoaderSimple for a combined checkpoint) is repointed at it.
    Checked against ComfyUI's own live model list (/object_info) before
    running - one not currently installed is skipped with a warning. If
    fewer than 2 configured models are present, the run aborts with an
    error - comparing a single model isn't what this tool is for.
  - "configs" - optional list of KSampler overrides (seed, steps, cfg,
    sampler_name, scheduler, denoise). Omit for the workflow's own KSampler
    settings; give multiple entries to run that model once per entry.
- "server" - optional, defaults to http://127.0.0.1:8000.

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

from funkytown_testing_harness.live_workflow import apply_lora_rules, config_prompts, load_live_template, set_positive_prompt, strip_loras
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


def find_ksampler_node_id(workflow):
    for node_id, node in workflow.items():
        if "KSampler" in node.get("class_type", ""):
            return node_id
    return None


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
    that aren't), or exits with an error if fewer than 2 are present."""
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

    if len(present) < 2:
        sys.exit(
            f"Error: only {len(present)} of {len(models_config)} configured model(s) are present on this "
            "ComfyUI server. This tool compares models against each other, so at least 2 must be present."
        )
    return present


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

    present_models = resolve_present_models(config["models"], template, server)
    prompts = config_prompts(config)
    multi_prompt = len(prompts) > 1

    RUNS_DIR.mkdir(exist_ok=True)
    log_path = RUNS_DIR / f"{name}_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"

    print(f"Models present ({len(present_models)}/{len(config['models'])}): "
          f"{', '.join(m['model'] for m in present_models)}")
    if multi_prompt:
        print(f"Prompts: {len(prompts)}")
    print()

    header = ["Model", "KSampler Overrides", "Prompt ID", "Status", "Filename Prefix", "Detail"]
    if multi_prompt:
        header = ["Prompt Index", "Prompt"] + header

    with open(log_path, "w", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(header)

        for p_idx, prompt_text in enumerate(prompts):
            for entry in present_models:
                model = entry["model"]
                configs = entry.get("configs") or [{}]

                for i, overrides in enumerate(configs):
                    wf = copy.deepcopy(template)
                    set_model(wf, model)
                    if prompt_text:
                        set_positive_prompt(wf, prompt_text)
                    apply_lora_rules(wf)

                    if overrides:
                        if not ksampler_id:
                            print(f"[{model}] Warning: KSampler overrides given but no KSampler node found", file=sys.stderr)
                        else:
                            apply_ksampler_overrides(wf, ksampler_id, overrides)

                    suffix = f"_cfg{i}" if len(configs) > 1 else ""
                    prompt_part = f"prompt{p_idx}_" if multi_prompt else ""
                    prefix = f"tests/{name}/{prompt_part}{Path(model).stem}{suffix}"
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
