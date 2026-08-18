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
directory** next to this one:

```
Claude Projects/
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

(The unit tests need nothing beyond the Python standard library - ComfyUI and
the browser are mocked out there.)

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
`tests/<name>/<model filename stem>[_cfg<N>]/...` (the `_cfg<N>` suffix only
appears when a model has more than one entry under `configs`) so runs are easy
to tell apart.

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

### Workflow-related files are gitignored

`workflows/` is excluded from git, and so is `configs/model-testing-config.json`
specifically - both can carry real (sometimes NSFW) prompt text via
`positive_prompt` as you edit them for your own experiments. `examples/` is
the deliberate exception: files there are vetted SFW references meant to be
committed and visible, not your live working config.

## Output

Each run writes a log CSV to `runs/<name>_<timestamp>.csv` with one row per
(model, KSampler config) combination: the model name, the overrides used,
ComfyUI's `prompt_id`, status (`queued`/`error`), the output filename prefix
used, and any error detail. `runs/` is also gitignored.

## Running the unit tests

```bash
python -m unittest discover -s tests -v
```

Covers `model_swap.py` (finding/repointing model-loader nodes), `live_workflow.py`
(fetching/converting from ComfyUI, stripping LoRAs, overriding the prompt),
and the config logic in `run_test.py` - KSampler overrides, checking models
against ComfyUI's live model list, skipping missing ones, erroring below 2
present models, and the single-config-vs-multiple-configs expansion (one
queued run per config, `batch_size` never touched). ComfyUI and the browser
are fully mocked out, so these run without a server up.

## Adding a new axis to sweep (e.g. LoRA, seed)

Model and KSampler settings (sampler/steps/cfg/scheduler/seed/denoise) are
covered. To sweep something else - a LoRA on/off, for instance - add a small
helper function alongside the ones in `model_swap.py`, then call it from
`run()` in `run_test.py` for each combination you want.
`comfy_prompt_tools.rerun_prompts_comfyui` already has reusable pieces for
some of this - e.g. `find_power_lora_loader_id` for LoRA toggling.
