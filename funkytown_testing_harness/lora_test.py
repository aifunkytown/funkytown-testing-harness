"""
Run a LoRA-weight sweep against a running ComfyUI server: one fixed model and
workflow, LoRA(s) turned on at each weight given for them, one queued
generation per combination.

Two modes, controlled by "combine_loras" in the config:
  - Isolated (default, combine_loras absent/false): one LoRA turned on at a
    time (every other LoRA slot forced off), at each of its weights - one
    queued run per (LoRA, weight) pair, LoRAs never mixed together.
  - Combined (combine_loras: true): every listed LoRA turned on together,
    one queued run per combination across the cartesian product of all their
    weight lists - e.g. 2 LoRAs with 3 weights each queues 9 runs, each with
    both LoRAs active simultaneously at one weight pairing.

The source workflow is always fetched fresh from ComfyUI and converted on
every run (see live_workflow.py), same as run_test.py.

There is no pass/fail here. This submits each variant to ComfyUI and logs what
was queued (LoRA(s) and weight(s), prompt_id, output filename prefix) to a
CSV under runs/, so you can compare the resulting images yourself.

This only supports a single fixed model for now, not multiple models the way
run_test.py compares models - see model_swap.py/run_test.py for the
equivalent pattern this could follow if that's ever needed.

Config file format (JSON):
    {
        "name": "lora_testing",
        "source_workflow": "krea2_basic_t2i.json",
        "model": "krea2SATDirtyrealism_krea2SAT.safetensors",
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
- "model" - a single model filename. Checked against ComfyUI's own live model
  list (/object_info) before running; the run aborts with an error if it
  isn't present.
- "positive_prompt" - optional, reapplied every run: overwrites the positive
  CLIPTextEncode node's text.
- "combine_loras" - optional, default false. See the two modes above.
- "loras" - list of LoRA objects, each with:
  - "lora" - filename of a LoRA slot that must already exist in the
    workflow's Power Lora Loader (rgthree) node (added there via ComfyUI's
    "+ Add Lora" widget beforehand - a slot can be toggled but not created).
    One not found there is skipped with a warning (the whole combination is
    skipped in combined mode if any of its LoRAs aren't found).
  - "weights" - list of strength values for that LoRA.
- "server" - optional, defaults to http://127.0.0.1:8000.

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

from funkytown_testing_harness.live_workflow import load_live_template, set_positive_prompt
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


def check_model_present(model_name, template, server):
    loader_nodes = find_model_loader_nodes(template)
    if not loader_nodes:
        sys.exit("Error: no recognized model-loader node (UNETLoader/CheckpointLoader) found in workflow.")

    for _node_id, field, class_type in loader_nodes:
        if model_name in fetch_available_models(server, class_type, field):
            return
    sys.exit(f"Error: model '{model_name}' not found on this ComfyUI server.")


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


def combo_prefix(name, combo):
    return f"tests/{name}/" + "__".join(f"{Path(lora).stem}_w{weight_label(weight)}" for lora, weight in combo)


def run(config_path):
    config = load_config(config_path)
    server = config.get("server", "http://127.0.0.1:8000")
    name = config.get("name", config_path.stem)
    model = config["model"]
    combine = bool(config.get("combine_loras"))

    print(f"Test case: {name}")
    template = build_template(config, server)

    check_model_present(model, template, server)
    set_model(template, model)

    lora_node_id = find_power_lora_loader_id(template)
    if not lora_node_id:
        sys.exit("Error: no Power Lora Loader (rgthree) node found in workflow.")

    save_ids = find_save_image_node_ids(template)
    client_id = str(uuid.uuid4())

    combinations = build_combinations(config["loras"], combine)

    RUNS_DIR.mkdir(exist_ok=True)
    log_path = RUNS_DIR / f"{name}_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"

    mode = "combined" if combine else "isolated"
    print(f"Model: {model}")
    print(f"LoRAs ({len(config['loras'])}): {', '.join(l['lora'] for l in config['loras'])}")
    print(f"Mode: {mode} - {len(combinations)} combination(s) to run\n")

    with open(log_path, "w", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["LoRAs", "Prompt ID", "Status", "Filename Prefix", "Detail"])

        for combo in combinations:
            label = combo_description(combo)
            wf = copy.deepcopy(template)
            missing = set_multiple_loras(wf, lora_node_id, combo)
            if missing:
                print(f"[{label}] Skipping: no matching LoRA slot for: {', '.join(missing)}", file=sys.stderr)
                writer.writerow([label, "", "skipped", "", f"No matching LoRA slot for: {', '.join(missing)}"])
                continue

            prefix = combo_prefix(name, combo)
            for save_id in save_ids:
                wf[save_id]["inputs"]["filename_prefix"] = prefix

            try:
                result = queue_prompt(server, wf, client_id)
            except urllib.error.URLError as e:
                print(f"[{label}] Failed to queue: {e}", file=sys.stderr)
                writer.writerow([label, "", "error", prefix, f"Failed to queue: {e}"])
                continue

            node_errors = result.get("node_errors")
            prompt_id = result.get("prompt_id")
            if node_errors:
                print(f"[{label}] node errors: {node_errors}")
                writer.writerow([label, prompt_id or "", "error", prefix, json.dumps(node_errors)])
                continue

            print(f"[{label}] -> queued as prompt_id={prompt_id}, output prefix '{prefix}'")
            writer.writerow([label, prompt_id, "queued", prefix, ""])
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
