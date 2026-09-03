"""
Run a LoRA-weight sweep against a running ComfyUI server: one or more models
and a fixed workflow, LoRA(s) turned on at each weight given for them, one
queued generation per (model, LoRA combination) pair.

Two LoRA modes, controlled by "combine_loras" in the config:
  - Isolated (default, combine_loras absent/false): one LoRA turned on at a
    time (every other LoRA slot forced off), at each of its weights - one
    combination per (LoRA, weight) pair, LoRAs never mixed together.
  - Combined (combine_loras: true): every listed LoRA turned on together,
    one combination per pairing across the cartesian product of all their
    weight lists - e.g. 2 LoRAs with 3 weights each makes 9 combinations,
    each with both LoRAs active simultaneously at one weight pairing.

Every present model is run against every LoRA combination - e.g. 2 models
and 4 LoRA combinations queues all 8 pairings.

The source workflow is always fetched fresh from ComfyUI and converted on
every run (see live_workflow.py), same as run_test.py.

There is no pass/fail here. This submits each variant to ComfyUI and logs what
was queued (model, LoRA(s) and weight(s), prompt_id, output filename prefix)
to a CSV under runs/, so you can compare the resulting images yourself.

Config file format (JSON):
    {
        "name": "lora_testing",
        "source_workflow": "krea2_basic_t2i.json",
        "models": ["krea2SATDirtyrealism_krea2SAT.safetensors", "bf95Krea2DarkRealism_v325.safetensors"],
        "positive_prompt": "A high-resolution realistic photo of ...",
        "combine_loras": true,
        "server": "http://127.0.0.1:8000",
        "loras": [
            {"lora": "detail_slider_krea2_loraholic.safetensors", "weights": [1, 2, 3]},
            {"lora": "skindetails_krea2_loraholic.safetensors", "weights": [0.5, 1.0]}
        ]
    }

- "source_workflow" - filename of a workflow saved in ComfyUI's own
  user/default/workflows folder. Pulled fresh from ComfyUI and converted to
  API format every time this runs (requires playwright - see
  live_workflow.py).
- "models" - list of model filenames (or "model" - a single filename - for
  the older single-model form; equivalent to a one-item "models" list).
  Each is checked against ComfyUI's own live model list (/object_info)
  before running - one not currently installed is skipped with a warning.
  The run aborts with an error if none of them are present.
- "positive_prompt" - optional, reapplied every run: overwrites the positive
  CLIPTextEncode node's text.
- "positive_prompts" - optional list of prompt strings, mutually exclusive
  with "positive_prompt" (config is rejected if both are given). Sweeps
  every model/LoRA combination once per prompt in the list - e.g. 2 models,
  4 LoRA combinations, and 3 prompts queues 24 runs. The CSV log gains
  "Prompt Index"/"Prompt" columns and each output filename prefix gets a
  "promptN_" segment, only when this is used.
- "combine_loras" - optional, default false. See the two LoRA modes above.
  After each combination's own LoRA slot(s) are set, comfy_prompt_tools'
  keyword -> LoRA routing (lora_rules.json / lora_rules.local.json) is
  applied against the effective prompt text, same as run_test.py - except
  it never touches whichever LoRA(s) this specific combination is already
  testing, so a keyword rule's fixed preset strength can't silently
  overwrite the exact weight being swept.
- "loras" - list of LoRA objects, each with:
  - "lora" - filename of a LoRA slot that must already exist in the
    workflow's Power Lora Loader (rgthree) node (added there via ComfyUI's
    "+ Add Lora" widget beforehand - a slot can be toggled but not created).
    One not found there is skipped with a warning (the whole combination is
    skipped in combined mode if any of its LoRAs aren't found).
  - "weights" - list of strength values for that LoRA.
- "server" - optional, defaults to http://127.0.0.1:8000.

Output filename prefix (and so the folder images land in under ComfyUI's
output directory) is
"tests/<name>/<run_id>/<queue_index>_<model stem>__<lora>_w<weight>...",
same run_id and queue_index scheme as run_test.py - run_id a short (8 hex
char) random id generated fresh each run() call, so two runs sharing the
same "name" land in separate folders; queue_index a zero-padded 4-digit
counter over every combination queued this run (starting at 0001, in queue
order), so sorting the output folder by filename always matches actual
queue order regardless of how model/LoRA names alphabetize.

Usage:
    python -m funkytown_testing_harness.lora_test configs/lora-testing-config.json
"""

import argparse
import copy
import csv
import datetime
import itertools
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

# comfy-prompt-tools is a sibling checkout, not an installed package - see
# README. If something already made it importable, that takes precedence
# over the sibling guess.
try:
    from comfy_prompt_tools.rerun_prompts_comfyui import find_power_lora_loader_id, find_save_image_node_ids, queue_prompt
except ImportError:
    _COMFY_PROMPT_TOOLS = Path(__file__).resolve().parent.parent.parent / "comfy-prompt-tools"
    sys.path.insert(0, str(_COMFY_PROMPT_TOOLS))
    try:
        from comfy_prompt_tools.rerun_prompts_comfyui import find_power_lora_loader_id, find_save_image_node_ids, queue_prompt
    except ImportError:
        sys.exit(
            f"Error: could not import comfy_prompt_tools from {_COMFY_PROMPT_TOOLS}.\n"
            "Expected comfy-prompt-tools checked out as a sibling directory next to "
            "funkytown-testing-harness (or already importable via sys.path)."
        )

from funkytown_testing_harness.live_workflow import apply_lora_rules, config_prompts, load_live_template, set_positive_prompt
from funkytown_testing_harness.lora_swap import set_multiple_loras
from funkytown_testing_harness.model_swap import find_model_loader_nodes, set_model

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def load_config(config_path):
    return json.loads(config_path.read_text(encoding="utf-8"))


def build_template(config, server):
    source_workflow = config["source_workflow"]
    print(f"Fetching '{source_workflow}' fresh from ComfyUI at {server} ...")
    template = load_live_template(server, source_workflow)
    print(f"Converted: {len(template)} active node(s).")

    if config.get("positive_prompt"):
        set_positive_prompt(template, config["positive_prompt"])

    return template


def config_models(config):
    """Normalize "models" (list) / "model" (single, older form) into a list."""
    if "models" in config:
        return list(config["models"])
    if "model" in config:
        return [config["model"]]
    sys.exit("Error: config must specify either 'models' (a list) or 'model' (a single filename).")


def fetch_available_models(server, class_type, field):
    """Same approach as run_test.py: ask ComfyUI's own /object_info rather
    than guessing at filesystem layout."""
    url = f"{server}/object_info/{class_type}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            info = json.loads(resp.read().decode("utf-8"))
        return set(info[class_type]["input"]["required"][field][0])
    except (urllib.error.URLError, KeyError, IndexError, TypeError) as e:
        sys.exit(f"Error: could not fetch available models from {url}: {e}")


def resolve_present_models(models, template, server):
    """Check each configured model against ComfyUI's live model list. Returns
    the subset that are actually present (skipping - with a warning - any
    that aren't), or exits with an error if none are present."""
    loader_nodes = find_model_loader_nodes(template)
    if not loader_nodes:
        sys.exit("Error: no recognized model-loader node (UNETLoader/CheckpointLoader) found in workflow.")

    available_cache = {}
    present = []
    for model_name in models:
        found = False
        for _node_id, field, class_type in loader_nodes:
            key = (class_type, field)
            if key not in available_cache:
                available_cache[key] = fetch_available_models(server, class_type, field)
            if model_name in available_cache[key]:
                found = True
                break
        if found:
            present.append(model_name)
        else:
            print(f"[{model_name}] Skipping: not found on this ComfyUI server", file=sys.stderr)

    if not present:
        sys.exit(f"Error: none of the {len(models)} configured model(s) are present on this ComfyUI server.")
    return present


def build_combinations(loras_config, combine):
    """Return a list of combinations to run, each a list of (lora, weight)
    pairs. combine=False: each combination has exactly one pair (isolated
    one-at-a-time sweep). combine=True: the cartesian product across every
    LoRA's weight list, each combination containing one pair per configured
    LoRA (all active together in that run)."""
    if not combine:
        return [[(entry["lora"], weight)] for entry in loras_config for weight in entry["weights"]]

    per_lora_pairs = [[(entry["lora"], w) for w in entry["weights"]] for entry in loras_config]
    return [list(combo) for combo in itertools.product(*per_lora_pairs)]


def weight_label(weight):
    return f"{weight}".replace(".", "_").replace("-", "neg")


def combo_description(combo):
    return "; ".join(f"{lora}={weight}" for lora, weight in combo)


def combo_prefix(name, run_id, queue_index, model, combo, prompt_idx=None):
    lora_part = "__".join(f"{Path(lora).stem}_w{weight_label(weight)}" for lora, weight in combo)
    prompt_part = f"prompt{prompt_idx}_" if prompt_idx is not None else ""
    return f"tests/{name}/{run_id}/{queue_index:04d}_{prompt_part}{Path(model).stem}__{lora_part}"


def run(config_path):
    config = load_config(config_path)
    server = config.get("server", "http://127.0.0.1:8000")
    name = config.get("name", config_path.stem)
    combine = bool(config.get("combine_loras"))

    print(f"Test case: {name}")
    template = build_template(config, server)

    present_models = resolve_present_models(config_models(config), template, server)
    prompts = config_prompts(config)
    multi_prompt = len(prompts) > 1

    lora_node_id = find_power_lora_loader_id(template)
    if not lora_node_id:
        sys.exit("Error: no Power Lora Loader (rgthree) node found in workflow.")

    save_ids = find_save_image_node_ids(template)
    client_id = str(uuid.uuid4())
    # Leads with a shortened timestamp (not just a random id) so the output
    # folder itself says when the run happened and sorts chronologically by
    # name - the short random suffix still guarantees two runs starting in
    # the same second never land in the same folder.
    run_id = f"{datetime.datetime.now():%y%m%d_%H%M%S}_{uuid.uuid4().hex[:4]}"

    combinations = build_combinations(config["loras"], combine)

    RUNS_DIR.mkdir(exist_ok=True)
    log_path = RUNS_DIR / f"{name}_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"

    mode = "combined" if combine else "isolated"
    total_runs = len(present_models) * len(combinations) * len(prompts)
    print(f"Models ({len(present_models)}/{len(config_models(config))}): {', '.join(present_models)}")
    print(f"LoRAs ({len(config['loras'])}): {', '.join(l['lora'] for l in config['loras'])}")
    prompt_summary = f" x {len(prompts)} prompt(s)" if multi_prompt else ""
    print(f"Mode: {mode} - {len(present_models)} model(s) x {len(combinations)} combination(s)"
          f"{prompt_summary} = {total_runs} run(s)\n")

    header = ["Model", "LoRAs", "Prompt ID", "Status", "Filename Prefix", "Detail"]
    if multi_prompt:
        header = ["Prompt Index", "Prompt"] + header

    with open(log_path, "w", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(header)

        queue_index = 0
        for p_idx, prompt_text in enumerate(prompts):
            for model in present_models:
                for combo in combinations:
                    queue_index += 1
                    label = combo_description(combo)
                    wf = copy.deepcopy(template)
                    set_model(wf, model)
                    if prompt_text:
                        set_positive_prompt(wf, prompt_text)
                    missing = set_multiple_loras(wf, lora_node_id, combo)
                    row_prefix = [p_idx, prompt_text] if multi_prompt else []

                    if missing:
                        print(f"[{model}] [{label}] Skipping: no matching LoRA slot for: {', '.join(missing)}", file=sys.stderr)
                        writer.writerow(row_prefix + [model, label, "", "skipped", "", f"No matching LoRA slot for: {', '.join(missing)}"])
                        continue

                    apply_lora_rules(wf, exclude={lora_filename for lora_filename, _weight in combo})

                    prefix = combo_prefix(name, run_id, queue_index, model, combo, p_idx if multi_prompt else None)
                    for save_id in save_ids:
                        wf[save_id]["inputs"]["filename_prefix"] = prefix

                    try:
                        result = queue_prompt(server, wf, client_id)
                    except urllib.error.URLError as e:
                        print(f"[{model}] [{label}] Failed to queue: {e}", file=sys.stderr)
                        writer.writerow(row_prefix + [model, label, "", "error", prefix, f"Failed to queue: {e}"])
                        continue

                    node_errors = result.get("node_errors")
                    prompt_id = result.get("prompt_id")
                    if node_errors:
                        print(f"[{model}] [{label}] node errors: {node_errors}")
                        writer.writerow(row_prefix + [model, label, prompt_id or "", "error", prefix, json.dumps(node_errors)])
                        continue

                    print(f"[{model}] [{label}] -> queued as prompt_id={prompt_id}, output prefix '{prefix}'")
                    writer.writerow(row_prefix + [model, label, prompt_id, "queued", prefix, ""])
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
