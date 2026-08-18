"""Fetch a ComfyUI workflow live and convert it to API format, every time -
no static snapshot on disk, no manual re-sync step to remember.

Converting a saved (UI-format) workflow into the flat API format ComfyUI's
/prompt endpoint expects means resolving bypassed nodes, reroutes, and
subgraphs correctly - logic that only exists in ComfyUI's own frontend JS
(window.app.graphToPrompt), with no server-side equivalent. So this drives a
headless browser to do exactly what ComfyUI's own Workflow menu -> Export
(API) does.

Requires (only actually exercised when source_workflow is still in UI format,
which is the common case):
    pip install playwright
    playwright install chromium
"""

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

_COMFY_PROMPT_TOOLS = Path(__file__).resolve().parent.parent.parent / "comfy-prompt-tools"
if str(_COMFY_PROMPT_TOOLS) not in sys.path:
    sys.path.insert(0, str(_COMFY_PROMPT_TOOLS))

try:
    from comfy_prompt_tools.rerun_prompts_comfyui import find_power_lora_loader_id, find_prompt_node_ids
except ImportError:
    sys.exit(
        f"Error: could not import comfy_prompt_tools from {_COMFY_PROMPT_TOOLS}.\n"
        "Expected comfy-prompt-tools checked out as a sibling directory next to "
        "funkytown-testing-harness."
    )


def fetch_live_workflow(server, source_workflow):
    """Pull a workflow ComfyUI has saved under its own user/default/workflows
    folder, in whatever format (UI or API) it's currently saved in."""
    encoded = urllib.parse.quote(f"workflows/{source_workflow}", safe="")
    url = f"{server}/api/userdata/{encoded}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        sys.exit(f"Error: could not fetch '{source_workflow}' from ComfyUI at {url}: {e}")


def convert_ui_workflow_via_browser(server, ui_workflow):
    """Convert a saved (UI-format) ComfyUI workflow into API-format by driving
    ComfyUI's own frontend conversion logic (app.loadGraphData / app.graphToPrompt)
    through a headless browser. This correctly handles bypassed nodes, primitive
    nodes, and subgraphs, since it's the exact same code ComfyUI's UI uses."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.goto(server, wait_until="domcontentloaded")
            page.wait_for_function(
                "window.app && typeof window.app.graphToPrompt === 'function'", timeout=30000
            )
            return page.evaluate(
                """async (wf) => {
                    await window.app.loadGraphData(wf);
                    const result = await window.app.graphToPrompt();
                    return result.output;
                }""",
                ui_workflow,
            )
        finally:
            browser.close()


def is_api_format(data):
    """True if data looks like a flat API-format workflow (node-id -> {class_type, inputs})."""
    return isinstance(data, dict) and bool(data) and all(
        isinstance(v, dict) and "class_type" in v for v in data.values()
    )


def load_live_template(server, source_workflow):
    """Fetch source_workflow fresh from ComfyUI and return it as an
    API-format dict - converting via the browser first if it's still saved
    in UI format (the common case)."""
    raw = fetch_live_workflow(server, source_workflow)

    if "prompt" in raw and is_api_format(raw.get("prompt")):
        return raw["prompt"]
    if is_api_format(raw):
        return raw
    if "nodes" in raw and "links" in raw:
        template = convert_ui_workflow_via_browser(server, raw)
        if not is_api_format(template):
            sys.exit(f"Error: conversion of '{source_workflow}' did not produce a valid API-format workflow.")
        return template

    sys.exit(f"Error: '{source_workflow}' doesn't look like a recognizable ComfyUI workflow export.")


def strip_loras(template):
    """Clear every lora_N slot from the Power Lora Loader (rgthree) node,
    leaving just its model/clip passthrough."""
    lora_node_id = find_power_lora_loader_id(template)
    if not lora_node_id:
        print("  warning: strip_loras requested but no Power Lora Loader node found", file=sys.stderr)
        return
    inputs = template[lora_node_id]["inputs"]
    keep = {k: v for k, v in inputs.items() if k in ("model", "clip")}
    inputs.clear()
    inputs.update(keep)


def set_positive_prompt(template, text):
    """Overwrite the positive CLIPTextEncode node's text."""
    positive_id, _negative_id = find_prompt_node_ids(template)
    if not positive_id:
        print("  warning: positive_prompt given but no positive prompt node found", file=sys.stderr)
        return
    template[positive_id]["inputs"]["text"] = text
