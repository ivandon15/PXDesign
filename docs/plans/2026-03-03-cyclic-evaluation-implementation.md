# Cyclic Evaluation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Thread the `cyclic` flag from the input YAML/JSON into both the AF2 and Protenix evaluation filters so cyclic-aware scoring activates automatically when `cyclic: true` is set.

**Architecture:** Two changes: (1) `pipeline.py:main()` reads `cyclic` from the input JSON and sets two eval config flags before the pipeline runs; (2) `pxdbench/tasks/base.py` (installed package) passes `is_cyclic` from cfg into `ProtenixFilter.predict()`. No new files needed.

**Tech Stack:** Python, pxdbench (installed at `/data/kelsey/miniforge3/envs/pxdesign/lib/python3.11/site-packages/pxdbench/`), PXDesign pipeline

---

## Background: What already exists

- `pxdbench/tools/af2/main_af2_complex.py:complex_prediction()` — accepts `is_cyclic=False`, calls `add_cyclic_offset(prediction_model)` when True ✅
- `AF2ComplexPredictor.predict()` — passes `is_cyclic=self.cfg.get("is_cyclic", False)` to `complex_prediction()` ✅
- `ProtenixFilter.predict()` — accepts `is_cyclic=False`, calls `make_is_cyclic_mask_feat()` when True ✅
- `pxd_configs/eval.py` — has `"is_cyclic": False` at binder level and `"af2": {"is_cyclic": False}` ✅
- **Missing:** `pipeline.py:main()` never reads `cyclic` from input JSON to set eval configs
- **Missing:** `base.py:protenix_predict()` never passes `is_cyclic` to `ProtenixFilter.predict()`

## Key file paths

- `pxdesign/runner/pipeline.py` — main pipeline, where eval configs are set
- `/data/kelsey/miniforge3/envs/pxdesign/lib/python3.11/site-packages/pxdbench/tasks/base.py` — `protenix_predict()` method
- `pxdesign/utils/inputs.py` — already sets `generation[0]["cyclic"]` from YAML ✅

---

### Task 1: Wire `cyclic` from input JSON into eval configs in `pipeline.py`

**Files:**
- Modify: `pxdesign/runner/pipeline.py` (around line 348, inside `main()`)

**Context:**
In `main()`, after `orig_inputs` is loaded from the JSON file (line ~349), we need to read `cyclic` and set two eval config fields. The relevant block is:

```python
# existing code (lines ~349-354):
with open(configs.input_json_path, "r") as f:
    orig_inputs = json.load(f)
for x in orig_inputs:
    convert_to_bioassembly_dict(x, configs.dump_dir)
configs.input_json_path = os.path.join(configs.dump_dir, "pipeline_input.json")
with open(configs.input_json_path, "w") as f:
    json.dump(orig_inputs, f, indent=4)
```

**Step 1: Add cyclic propagation after loading orig_inputs**

In `pxdesign/runner/pipeline.py`, find the block inside `if DIST_WRAPPER.rank == 0:` that loads `orig_inputs`. Add these lines immediately after `orig_inputs = json.load(f)`:

```python
        # Propagate cyclic flag from input JSON into eval configs
        cyclic = orig_inputs[0].get("generation", [{}])[0].get("cyclic", False)
        if cyclic:
            configs.eval.binder.is_cyclic = True
            configs.eval.binder.tools.af2.is_cyclic = True
```

The full block after the edit should look like:

```python
    if DIST_WRAPPER.rank == 0:
        save_config(configs, os.path.join(configs.dump_dir, "config.yaml"))
        with open(configs.input_json_path, "r") as f:
            orig_inputs = json.load(f)
        # Propagate cyclic flag from input JSON into eval configs
        cyclic = orig_inputs[0].get("generation", [{}])[0].get("cyclic", False)
        if cyclic:
            configs.eval.binder.is_cyclic = True
            configs.eval.binder.tools.af2.is_cyclic = True
        for x in orig_inputs:
            convert_to_bioassembly_dict(x, configs.dump_dir)
        configs.input_json_path = os.path.join(configs.dump_dir, "pipeline_input.json")
        with open(configs.input_json_path, "w") as f:
            json.dump(orig_inputs, f, indent=4)
```

**Step 2: Verify the edit looks correct**

Run:
```bash
grep -n "is_cyclic\|cyclic" /data/kelsey/code/PXDesign/pxdesign/runner/pipeline.py
```
Expected output: lines showing the new `cyclic` and `is_cyclic` assignments.

**Step 3: Commit**

```bash
cd /data/kelsey/code/PXDesign
git add pxdesign/runner/pipeline.py
git commit -m "feat: propagate cyclic flag from input JSON into eval configs"
```

---

### Task 2: Pass `is_cyclic` from cfg into `ProtenixFilter.predict()` in `base.py`

**Files:**
- Modify: `/data/kelsey/miniforge3/envs/pxdesign/lib/python3.11/site-packages/pxdbench/tasks/base.py` (around line 301, `protenix_predict()` method)

**Context:**
`protenix_predict()` currently calls `ptx_filter.predict(...)` without passing `is_cyclic`. The `ProtenixFilter.predict()` signature already has `is_cyclic=False`. We just need to read it from `self.cfg` and pass it through.

Current code (lines ~301-340):
```python
def protenix_predict(self, data_list, orig_seqs=None, is_large=False):
    ptx_cfg = self.cfg.tools.ptx if is_large else self.cfg.tools.ptx_mini
    ptx_filter = self.get_ptx(is_large)
    dump_dir = os.path.join(
        self.out_dir, "ptx_pred" if is_large else "ptx_mini_pred"
    )
    binder_chain_idx = 0 if self.binder_chains[0] == "A" else None
    json_path = ptx_filter.prepare_json(...)
    pred_pdb_paths = ptx_filter.predict(
        input_json_path=json_path,
        design_pdb_dir=self.pdb_dir,
        data_list=data_list,
        dump_dir=dump_dir,
        seed=self.seed,
        N_sample=ptx_cfg.N_sample,
        N_step=ptx_cfg.N_step,
        step_scale_eta=ptx_cfg.step_scale_eta,
        gamma0=ptx_cfg.gamma0,
        N_cycle=ptx_cfg.N_cycle,
        binder_chain_idx=binder_chain_idx,
        use_msa=ptx_cfg.get("use_msa", True),
        suffix="_mini" if not is_large else "",
    )
    return pred_pdb_paths
```

**Step 1: Add `is_cyclic` to the `ptx_filter.predict()` call**

Find the `pred_pdb_paths = ptx_filter.predict(` call in `protenix_predict()` and add `is_cyclic=self.cfg.get("is_cyclic", False),` to the argument list:

```python
    pred_pdb_paths = ptx_filter.predict(
        input_json_path=json_path,
        design_pdb_dir=self.pdb_dir,
        data_list=data_list,
        dump_dir=dump_dir,
        seed=self.seed,
        N_sample=ptx_cfg.N_sample,
        N_step=ptx_cfg.N_step,
        step_scale_eta=ptx_cfg.step_scale_eta,
        gamma0=ptx_cfg.gamma0,
        N_cycle=ptx_cfg.N_cycle,
        binder_chain_idx=binder_chain_idx,
        use_msa=ptx_cfg.get("use_msa", True),
        suffix="_mini" if not is_large else "",
        is_cyclic=self.cfg.get("is_cyclic", False),
    )
```

**Step 2: Verify the edit**

Run:
```bash
grep -n "is_cyclic" /data/kelsey/miniforge3/envs/pxdesign/lib/python3.11/site-packages/pxdbench/tasks/base.py
```
Expected: one line showing `is_cyclic=self.cfg.get("is_cyclic", False),`

**Step 3: Clear Python bytecode cache for pxdbench**

```bash
find /data/kelsey/miniforge3/envs/pxdesign/lib/python3.11/site-packages/pxdbench -name "*.pyc" -delete
```

**Step 4: Commit the patched file into the repo for tracking**

Copy the patched file into the repo so it's tracked:
```bash
cp /data/kelsey/miniforge3/envs/pxdesign/lib/python3.11/site-packages/pxdbench/tasks/base.py \
   /data/kelsey/code/PXDesign/Protenix-0.5.0-pxd/pxdbench_patches/tasks_base.py
```

Wait — actually, since pxdbench is a separate installed package (not part of this repo), just commit a note. Skip the copy step. Instead:

```bash
cd /data/kelsey/code/PXDesign
git add pxdesign/runner/pipeline.py  # already committed in Task 1
git commit -m "fix: pass is_cyclic from cfg into ProtenixFilter.predict in pxdbench base.py" --allow-empty
```

Actually, since `base.py` is in the installed package (not in this repo), just note the change was made to the installed package. No git commit needed for Task 2 — the change is in the installed package.

---

### Task 3: Smoke test end-to-end with a cyclic YAML

**Context:**
Use the existing `examples/PDL1_quick_cyclic.yaml` (already open in the IDE) to verify the cyclic flag flows through to eval.

**Step 1: Add a debug print to confirm `is_cyclic` is set**

Temporarily add to `pxdesign/runner/pipeline.py` after the cyclic block:
```python
        if cyclic:
            configs.eval.binder.is_cyclic = True
            configs.eval.binder.tools.af2.is_cyclic = True
            print(f"[DEBUG cyclic eval] is_cyclic set: binder={configs.eval.binder.is_cyclic}, af2={configs.eval.binder.tools.af2.is_cyclic}")
```

**Step 2: Run a quick dry-run**

Check that the JSON produced from the YAML has `cyclic: true`:
```bash
cd /data/kelsey/code/PXDesign
conda run -n pxdesign python -c "
from pxdesign.utils.inputs import parse_yaml_to_json
import json
result = parse_yaml_to_json('examples/PDL1_quick_cyclic.yaml')
print(json.dumps(result[0]['generation'], indent=2))
"
```
Expected output:
```json
[
  {
    "type": "protein",
    "length": <N>,
    "count": 1,
    "cyclic": true
  }
]
```

**Step 3: Remove the debug print**

Remove the `print(f"[DEBUG cyclic eval]...")` line added in Step 1.

**Step 4: Commit**

```bash
cd /data/kelsey/code/PXDesign
git add pxdesign/runner/pipeline.py
git commit -m "feat: cyclic eval wiring complete - pipeline propagates cyclic to AF2 and Protenix filters"
```

---

## Summary of changes

| File | Change |
|------|--------|
| `pxdesign/runner/pipeline.py` | Read `cyclic` from input JSON, set `configs.eval.binder.is_cyclic` and `configs.eval.binder.tools.af2.is_cyclic` |
| `pxdbench/tasks/base.py` (installed) | Add `is_cyclic=self.cfg.get("is_cyclic", False)` to `ptx_filter.predict()` call |

No new files. No new config keys needed (they already exist in `pxd_configs/eval.py`).
