# funkytown-testing-harness

A config-driven sweep runner built on top of
[comfy-prompt-tools](https://github.com/aifunkytown/comfy-prompt-tools). It
takes a workflow/prompt straight from a running ComfyUI server and pushes it
once per entry in a list - e.g. the same prompt against several different
models - so you can compare the resulting images side by side.

**There is no pass/fail here.** Image quality/content can't be asserted
automatically. Each run just queues generations and logs what was submitted;
you judge the output yourself.

## Setup

This project imports directly from `comfy_prompt_tools` rather than installing
it as a package, so it expects `comfy-prompt-tools` checked out as a **sibling
directory** next to this one. The parent folder can be named/located anything
you like - only the sibling relationship matters:

```
your-workspace/
├── comfy-prompt-tools/
└── funkytown-testing-harness/
```

The source workflow is **always fetched fresh from ComfyUI and converted on
every run** - there's no static snapshot on disk that can go stale, and
nothing to manually re-sync before running a test. That conversion (resolving
bypassed nodes/subgraphs correctly) only exists in ComfyUI's own frontend JS,
so it's driven through a headless browser:

```bash
pip install playwright
playwright install chromium
```

Building a comparison grid (see below) needs Pillow:

```bash
pip install Pillow
```

(The unit tests need nothing beyond the Python standard library plus
Pillow - ComfyUI and the browser are mocked out there.)

## Desktop GUI

A PySide6 desktop front end for building and running a config without
hand-editing JSON lives in a separate project:
[funkytown-testing-harness-gui](https://github.com/aifunkytown/funkytown-testing-harness-gui)
(expects this repo checked out as a sibling directory, same as this repo
expects `comfy-prompt-tools`).

## Running a test case

```bash
python -m funkytown_testing_harness.run_test configs/model-testing-config.json
```

`model-testing-config.json` is a generic, reusable config - edit it in place
for whatever comparison you want to run next, rather than creating a new named
config file per experiment. See [`examples/`](examples/) for a sample of
exactly what gets fetched and run.

### Config file format

```json
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
```

- **`source_workflow`** - filename of a workflow saved in ComfyUI's own
  `user/default/workflows` folder. Pulled fresh from ComfyUI and converted to
  API format on every run - whatever is currently live in ComfyUI (model,
  batch_size, sampler defaults, prompt, anything) is what gets used, with no
  manual step to keep it current.
- **`strip_loras`** - optional. Reapplied to the freshly fetched workflow
  every run: clears every `lora_N` slot from the Power Lora Loader (rgthree)
  node, leaving just its model/clip passthrough.
- **`positive_prompt`** - optional. Reapplied every run: overwrites the
  positive CLIPTextEncode node's text, regardless of whatever prompt happens
  to be live in ComfyUI at the time.
- **Keyword LoRA routing** - after the above, `comfy-prompt-tools`'
  `rerun_prompts_comfyui.py` keyword -> LoRA rules (`lora_rules.json` /
  `lora_rules.local.json`) are checked against the effective prompt text
  (the override above, or whatever's live in ComfyUI if none given) and any
  matching LoRA is turned on - same rules and matching logic that script
  uses for a rerun. Not configurable here; it always runs. Only a LoRA slot
  that still structurally exists can be turned on this way, so `strip_loras:
  true` (which removes every slot outright) leaves nothing for it to act on
  - leave `strip_loras` unset/false for keyword-matched LoRAs to take effect.
- **`positive_prompts`** - optional list of prompt strings, mutually
  exclusive with `positive_prompt` (the config is rejected if both are
  given). Sweeps every model/config combination once per prompt - e.g. 2
  models and 3 prompts queues 6 (or more, with multiple KSampler configs)
  runs. The log CSV gains `Prompt Index`/`Prompt` columns and each output
  filename prefix gets a `promptN_` segment, only when this is used.
- **`models`** - list of model objects, each with:
  - **`model`** - a model filename. The workflow's model-loader node
    (`UNETLoader` for diffusion-only weights like Krea2, or
    `CheckpointLoaderSimple` for a combined checkpoint) is repointed at it.
    Checked against ComfyUI's own live model list (via `/object_info`) before
    running - one that isn't currently installed is **skipped with a
    warning** rather than failing the whole run. If fewer than 2 configured
    models turn out to be present, the whole run **aborts with an error**,
    since this tool exists to compare models against each other.
  - **`configs`** - optional list of KSampler overrides for that model:
    `sampler_name`, `steps`, `cfg`, `scheduler`, `seed`, `denoise`. Only the
    keys you give are changed; everything else in the workflow's KSampler
    node stays as-is. Omit `configs` (or leave it empty) to run that model
    once with the workflow's own KSampler settings. Give multiple entries to
    run that model once per entry - e.g. the same model at several step
    counts.
- **`server`** - optional, defaults to `http://127.0.0.1:8000`.

`EmptyLatentImage.batch_size` is never touched by any of the above - whatever
the live workflow currently has is what gets used.

Each variant's output goes to ComfyUI's output folder under
`tests/<name>/<run_id>/<queue_index>_<model filename stem>[_cfg<N>]/...`
(the `_cfg<N>` suffix only appears when a model has more than one entry
under `configs`). `run_id` is a short random id generated fresh each run,
so two runs sharing the same `name` land in separate folders instead of
comingling images. `queue_index` is a zero-padded 4-digit counter over
every variant queued this run (`0001`, `0002`, ...), so sorting the output
folder by filename always matches the order things were actually queued
in, regardless of how model names happen to alphabetize.

### Example: fetch, strip LoRAs, swap to an SFW prompt, compare two models

[`examples/krea2_basic_t2i_sfw_example.json`](examples/krea2_basic_t2i_sfw_example.json)
is a saved snapshot of exactly what `configs/model-testing-config.json`
produces: `krea2_basic_t2i.json` pulled fresh from ComfyUI, LoRAs stripped,
and the positive prompt swapped to a specific SFW one - a slender woman
walking through a farmers market. It's committed as a concrete reference for
what the fetch-and-edit pipeline outputs; it isn't read by `run_test.py`
itself (which always re-fetches live rather than using a saved copy).

Run that exact case with:

```bash
python -m funkytown_testing_harness.run_test configs/model-testing-config.json
```

This queues the SFW prompt above against both
`krea2SATDirtyrealism_krea2SAT.safetensors` (euler, 8 steps) and
`bf95Krea2DarkRealism_v325.safetensors` (er_sde, 10 steps), so you can compare
how the two models render the same scene.

## Running a LoRA weight sweep

```bash
python -m funkytown_testing_harness.lora_test configs/lora-testing-config.json
```

Same idea as model testing, but for LoRA weights: one or more models and a
fixed workflow, one queued run per (model, LoRA combination) pair - e.g. 2
models and 4 LoRA combinations queues all 8 pairings. Two LoRA modes,
controlled by `combine_loras`:

- **Isolated** (default): one LoRA turned on at a time (every other LoRA
  slot forced off), at each of its weights - LoRAs are never mixed together.
- **Combined** (`"combine_loras": true`): every listed LoRA turned on
  *together*, one combination per pairing across the cartesian product of
  all their weight lists - e.g. 2 LoRAs with 2 weights each makes 4
  pairings, each with both LoRAs active simultaneously.

### Config file format

```json
{
    "name": "lora_testing",
    "source_workflow": "krea2_basic_t2i.json",
    "models": ["krea2SATDirtyrealism_krea2SAT.safetensors", "bf95Krea2DarkRealism_v325.safetensors"],
    "positive_prompt": "A high-resolution realistic photo of ...",
    "combine_loras": true,
    "server": "http://127.0.0.1:8000",
    "loras": [
        {
            "lora": "detail_slider_krea2_loraholic.safetensors",
            "weights": [1.0, 3.0]
        },
        {
            "lora": "Cinematic_Krea2_2_c1n3m4t1c_st6000.safetensors",
            "weights": [0.8, 1.5]
        }
    ]
}
```

- **`models`** - list of model filenames (or **`model`** - a single
  filename - for the older single-model form; equivalent to a one-item
  `models` list). Each is checked against ComfyUI's live model list the same
  way as `run_test.py` - one not present is **skipped with a warning**
  rather than failing the whole run; the run **aborts with an error** only
  if *none* of them are present.
- **`combine_loras`** - optional, default `false`. See the two LoRA modes above.
- **`loras`** - list of LoRA objects, each with:
  - **`lora`** - filename of a LoRA slot that must already exist in the
    workflow's Power Lora Loader (rgthree) node (added there via ComfyUI's
    "+ Add Lora" widget beforehand - a slot can be toggled on/off and given
    a strength, but not created through the API). One not found there is
    **skipped with a warning**, logged as `skipped` in the output CSV. In
    combined mode, if any LoRA in a combination is missing, that whole
    combination is skipped rather than partially applying the rest.
  - **`weights`** - list of strength values for that LoRA. In isolated mode,
    one run per value; in combined mode, one axis of the cartesian product.

Same keyword LoRA routing as `run_test.py` (see above) is applied to each
combination after its own LoRA slot(s) are set - except it never touches
whichever LoRA(s) that specific combination is already sweeping, so a
keyword rule's fixed preset strength can't silently overwrite the exact
weight being tested. Since a combination's *other* slots are only ever
turned off (not removed - unlike `strip_loras`), a keyword match on an
unrelated LoRA can still turn it on here.
- `source_workflow`/`positive_prompt`/`positive_prompts`/`server` work
  exactly as in `run_test.py` - a prompt sweep is crossed with the model x
  LoRA-combination space, so 2 models, 4 LoRA combinations, and 3 prompts
  queues 24 runs.

Output goes to `tests/<name>/<run_id>/<queue_index>_<model stem>__<lora filename stem>_w<weight>/...`
in isolated mode, or `tests/<name>/<run_id>/<queue_index>_<model stem>__<lora1 stem>_w<weight1>__<lora2 stem>_w<weight2>/...`
(joined with `__`, one segment per LoRA) in combined mode - `run_id` and
`queue_index` work exactly as in `run_test.py` (see above): a short random
id per run, plus a zero-padded 4-digit queue-order counter so sorting by
filename matches actual queue order. The run log at
`runs/<name>_<timestamp>.csv` has one row per (model, combination) pairing,
with `Model` and `LoRAs` columns describing what ran.

### Workflow-related files are gitignored

`workflows/` is excluded from git, and so are `configs/model-testing-config.json`
and `configs/lora-testing-config.json` specifically - all three can carry
real (sometimes NSFW) prompt text via `positive_prompt` as you edit them for
your own experiments. `examples/` is the deliberate exception: files there
are vetted SFW references meant to be committed and visible, not your live
working config.

## Output

Each run writes a log CSV to `runs/<name>_<timestamp>.csv` with one row per
(model, KSampler config) combination: the model name, the overrides used,
ComfyUI's `prompt_id`, status (`queued`/`error`), the output filename prefix
used, and any error detail. `runs/` is also gitignored.

## Comparison grids

`funkytown_testing_harness.comparison_grid.build_comparison_grid(log_path,
comfyui_output_dir, output_path, selected_images=None)` builds a single
labeled image from a run's log: one column per queued row that already has
an output image on disk (a still-in-progress run just uses whichever ones
exist so far), labeled with that row's model name (`run_test.py` logs) or
LoRA/weight combo (`lora_test.py` logs) in a white header band above it,
images butted directly together with no gaps. Up to 4 columns per row; more
than that wraps onto additional rows below, each with its own header
mirrored underneath its images too (so a row's labels are never far from it
once there's more than one row). `selected_images`, if given, restricts the
grid to just that subset of the run's image paths (a row's label is still
looked up normally, just skipped if its image isn't in the set) - this is
how the GUI's Results tab "Create Grid" button only grids whatever's
checked in its thumbnail gallery, rather than every image in the run.
Raises `ValueError` if fewer than 2 rows end up included - nothing
meaningful to compare with 0 or 1. There's no CLI entry point for this yet.

## Running the unit tests

```bash
python -m unittest discover -s tests -v
```

Covers `model_swap.py` (finding/repointing model-loader nodes), `lora_swap.py`
(isolating a single LoRA slot on/off at a strength), `live_workflow.py`
(fetching/converting from ComfyUI, stripping LoRAs, overriding the prompt),
and the config logic in `run_test.py`/`lora_test.py` - KSampler overrides,
checking models against ComfyUI's live model list, skipping missing
models/LoRAs, erroring below 2 present models (or a missing single model for
LoRA testing), the multi-value expansion (one queued run per config/weight,
`batch_size` never touched), and the `"positive_prompts"` sweep (crossed
with whatever other axes are configured, mutually exclusive with
`"positive_prompt"`). ComfyUI and the browser are fully mocked out, so these
run without a server up. Also covers `comparison_grid.py` (label formatting,
row-wrapping past 4 columns with mirrored headers, and image resolution/
downscaling) using small synthetic PNGs, no real ComfyUI output needed.

## Adding a new axis to sweep (e.g. seed)

Model and KSampler settings, multiple models for LoRA testing
(`lora_test.py`'s `"models"` list), single-LoRA and combined-LoRA weight
sweeps, and a prompt sweep (`"positive_prompts"`, both scripts) are
covered. To extend further - sweeping seed, for instance - follow the same
pattern: a small helper alongside `model_swap.py`/`lora_swap.py`, called
from `run()` for each combination wanted. `comfy_prompt_tools.rerun_prompts_comfyui`
already has a reusable piece for seed sweeps - `find_seed_inputs`.
