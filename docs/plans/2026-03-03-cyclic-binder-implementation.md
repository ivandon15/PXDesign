# Cyclic Binder Generation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `cyclic: true` YAML flag so the PXDesign diffusion model generates head-to-tail cyclized binders by injecting a ring-graph shortest-path offset into `RelativePositionEncoding`.

**Architecture:** The `cyclic` flag flows YAML → JSON `generation` dict → `proteinChain` sequence entry → `binder_cyclic_offset` feature tensor → `RelativePositionEncoding.forward()`. Only the binder-binder token subblock of `rel_pos_index` is overridden; target tokens are untouched.

**Tech Stack:** Python, PyTorch, NumPy, biotite, PXDesign codebase

---

### Task 1: Propagate `cyclic` flag through YAML → JSON

**Files:**
- Modify: `pxdesign/utils/inputs.py:48-140`

**Step 1: Read `cyclic` from YAML and add to `generation` dict**

In `parse_yaml_to_json()`, after reading `binder_length`, read the optional `cyclic` flag and include it in the `generation` list entry:

```python
# after: binder_length = int(cfg["binder_length"])
cyclic = bool(cfg.get("cyclic", False))

# in json_task construction, change generation to:
"generation": [
    {
        "type": "protein",
        "length": binder_length,
        "count": 1,
        "cyclic": cyclic,
    }
],
```

**Step 2: Verify manually**

```bash
cd /data/kelsey/code/PXDesign
python -c "
from pxdesign.utils.inputs import parse_yaml_to_json
import json, tempfile, os

yaml_content = '''
target:
  file: examples/5o45.cif
  chains:
    A:
      crop: [\"1-116\"]
binder_length: 10
cyclic: true
'''
with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False) as f:
    f.write(yaml_content)
    tmp = f.name

result = parse_yaml_to_json(tmp)
print(json.dumps(result[0]['generation'], indent=2))
os.unlink(tmp)
"
```

Expected output:
```json
[
  {
    "type": "protein",
    "length": 10,
    "count": 1,
    "cyclic": true
  }
]
```

**Step 3: Commit**

```bash
git add pxdesign/utils/inputs.py
git commit -m "feat: propagate cyclic flag from YAML into generation dict"
```

---

### Task 2: Propagate `cyclic` through `make_gen_sequences()`

**Files:**
- Modify: `pxdesign/data/infer_data_pipeline.py:332-353`

**Step 1: Pass `cyclic` into the `proteinChain` dict**

In `make_gen_sequences()`, read `cyclic` from each `gen_seq_dict` and include it:

```python
def make_gen_sequences(self, json_dict):
    if "sequences" not in json_dict:
        json_dict["sequences"] = []

    for gen_seq_dict in json_dict.get("generation", {}):
        assert "sequence" not in gen_seq_dict
        length = gen_seq_dict["length"]
        count = gen_seq_dict["count"]
        cyclic = gen_seq_dict.get("cyclic", False)   # NEW
        one_dict = {
            "proteinChain": {
                "sequence": "j" * length,
                "count": count,
                "sequence_type": "design",
                "use_msa": False,
                "cyclic": cyclic,                    # NEW
            }
        }
        json_dict["sequences"].append(one_dict)
    return json_dict
```

**Step 2: Verify manually**

```bash
python -c "
from pxdesign.data.infer_data_pipeline import InferenceDataset
json_dict = {
    'generation': [{'type': 'protein', 'length': 5, 'count': 1, 'cyclic': True}]
}
ds = object.__new__(InferenceDataset)
result = ds.make_gen_sequences({'sequences': [], **json_dict})
print(result['sequences'][0]['proteinChain'])
"
```

Expected: `{'sequence': 'jjjjj', 'count': 1, 'sequence_type': 'design', 'use_msa': False, 'cyclic': True}`

**Step 3: Commit**

```bash
git add pxdesign/data/infer_data_pipeline.py
git commit -m "feat: propagate cyclic flag into proteinChain sequence entry"
```

---

### Task 3: Compute `binder_cyclic_offset` feature tensor

**Files:**
- Modify: `pxdesign/data/json_to_feature.py:317-366`

**Step 1: Add `compute_cyclic_offset` helper and inject into `add_design_features()`**

Add the helper function at module level (after imports):

```python
def _compute_cyclic_offset(L: int) -> np.ndarray:
    """Shortest-path signed distance on a ring graph of length L.
    offset[i,j] = signed shortest path from j to i on a ring.
    For a linear chain: offset[0, L-1] = -(L-1).
    For a cyclic chain: offset[0, L-1] = -1.
    """
    i = np.arange(L)
    ij = np.stack([i, i + L], -1)
    offset = i[:, None] - i[None, :]
    c_offset = np.abs(ij[:, None, :, None] - ij[None, :, None, :]).min((2, 3))
    a = c_offset < np.abs(offset)
    c_offset[a] = -c_offset[a]
    return (c_offset * np.sign(offset)).astype(np.int32)
```

Then in `add_design_features()`, after building `design_token_mask`, add:

```python
# Cyclic binder offset
# self.single_sample_dict["sequences"] contains the processed sequence list.
# The binder (design) chain is the last proteinChain with cyclic=True.
cyclic = False
for seq in self.single_sample_dict.get("sequences", []):
    pc = seq.get("proteinChain", {})
    if pc.get("sequence_type") == "design" and pc.get("cyclic", False):
        cyclic = True
        break

if cyclic:
    N_total = int(condi_token_mask.shape[0])
    binder_indices = (~condi_token_mask).nonzero(as_tuple=True)[0]
    L = int(binder_indices.shape[0])
    cyc_block = torch.from_numpy(_compute_cyclic_offset(L))  # [L, L]
    binder_cyclic_offset = torch.zeros(N_total, N_total, dtype=torch.int32)
    idx = binder_indices
    binder_cyclic_offset[idx[:, None], idx[None, :]] = cyc_block
    feature_dict["binder_cyclic_offset"] = binder_cyclic_offset
```

**Step 2: Verify the offset values manually**

```bash
python -c "
import numpy as np
from pxdesign.data.json_to_feature import _compute_cyclic_offset

L = 6
off = _compute_cyclic_offset(L)
print('Shape:', off.shape)
print('offset[0, L-1] =', off[0, L-1], '  (should be -1 for cyclic, not', -(L-1), ')')
print('offset[0, 1]   =', off[0, 1],   '  (should be -1)')
print('offset[1, 0]   =', off[1, 0],   '  (should be +1)')
print(off)
"
```

Expected: `offset[0, L-1] = -1`, `offset[0, 1] = -1`, `offset[1, 0] = 1`

**Step 3: Commit**

```bash
git add pxdesign/data/json_to_feature.py
git commit -m "feat: compute binder_cyclic_offset feature for cyclic binders"
```

---

### Task 4: Override `rel_pos_index` in `RelativePositionEncoding`

**Files:**
- Modify: `pxdesign/model/embedders.py:262-273`

**Step 1: Inject cyclic override after `rel_pos_index` is computed**

In `RelativePositionEncoding.forward()`, the current code is:

```python
rel_pos_index = (
    input_feature_dict["residue_index"][..., :, None]
    - input_feature_dict["residue_index"][..., None, :]
)

d_residue = torch.clip(
    input=rel_pos_index + self.r_max,
    ...
```

Add the override between these two blocks:

```python
rel_pos_index = (
    input_feature_dict["residue_index"][..., :, None]
    - input_feature_dict["residue_index"][..., None, :]
)

# Override binder-binder subblock with cyclic shortest-path offset
if "binder_cyclic_offset" in input_feature_dict:
    cyc = input_feature_dict["binder_cyclic_offset"].to(rel_pos_index.device)
    mask = cyc != 0
    rel_pos_index = torch.where(mask, cyc.to(rel_pos_index.dtype), rel_pos_index)

d_residue = torch.clip(
    input=rel_pos_index + self.r_max,
    ...
```

**Step 2: Verify the override is applied correctly**

```bash
python -c "
import torch
from pxdesign.model.embedders import RelativePositionEncoding

rpe = RelativePositionEncoding(r_max=32, s_max=2, c_z=128)
rpe.eval()

N = 5   # 3 target + 2 binder
feat = {
    'residue_index': torch.tensor([1, 2, 3, 1, 2]),
    'asym_id':       torch.tensor([1, 1, 1, 2, 2]),
    'entity_id':     torch.tensor([1, 1, 1, 2, 2]),
    'sym_id':        torch.tensor([1, 1, 1, 2, 2]),
    'token_index':   torch.arange(N),
}

# Without cyclic: rel_pos[3,4] = 1-2 = -1 (already neighbors, trivial case)
# Use L=4 binder to make it interesting
N2 = 7  # 3 target + 4 binder
feat2 = {
    'residue_index': torch.tensor([1, 2, 3, 1, 2, 3, 4]),
    'asym_id':       torch.tensor([1, 1, 1, 2, 2, 2, 2]),
    'entity_id':     torch.tensor([1, 1, 1, 2, 2, 2, 2]),
    'sym_id':        torch.tensor([1, 1, 1, 2, 2, 2, 2]),
    'token_index':   torch.arange(N2),
}

# Cyclic offset for L=4 binder: offset[0,3] should be -1
import numpy as np
from pxdesign.data.json_to_feature import _compute_cyclic_offset
cyc_block = torch.from_numpy(_compute_cyclic_offset(4))
bco = torch.zeros(N2, N2, dtype=torch.int32)
bco[3:, 3:] = cyc_block
feat2['binder_cyclic_offset'] = bco

out = rpe(feat2)
print('Output shape:', out.shape)  # [7, 7, 128]
print('No error — cyclic override applied successfully')
"
```

Expected: `Output shape: torch.Size([7, 7, 128])` with no errors.

**Step 3: Commit**

```bash
git add pxdesign/model/embedders.py
git commit -m "feat: override rel_pos_index with cyclic offset in RelativePositionEncoding"
```

---

### Task 5: End-to-end smoke test with a real YAML

**Files:**
- Read: `examples/PDL1_quick_start.yaml`

**Step 1: Create a cyclic test YAML**

```bash
cat > /tmp/test_cyclic.yaml << 'EOF'
target:
  file: "./examples/5o45.cif"
  chains:
    A:
      crop: ["1-116"]
      hotspots: [40, 99, 107]
      msa: "./examples/msa/PDL1/0"

binder_length: 20
cyclic: true
EOF
```

**Step 2: Validate the YAML parses correctly**

```bash
cd /data/kelsey/code/PXDesign
python -m pxdesign check-input --yaml /tmp/test_cyclic.yaml
```

Expected: `✅ YAML file is valid.`

**Step 3: Run a minimal inference (1 sample, 10 steps)**

```bash
pxdesign infer \
  -i /tmp/test_cyclic.yaml \
  -o /tmp/test_cyclic_out \
  --N_sample 1 \
  --N_step 10 \
  --dtype bf16
```

Expected: completes without error, output CIF written to `/tmp/test_cyclic_out/`.

**Step 4: Verify non-cyclic still works (regression)**

```bash
pxdesign infer \
  -i examples/PDL1_quick_start.yaml \
  -o /tmp/test_linear_out \
  --N_sample 1 \
  --N_step 10 \
  --dtype bf16
```

Expected: completes without error.

**Step 5: Commit**

```bash
git add docs/plans/
git commit -m "docs: add cyclic binder implementation plan"
```

---

## Future work (not in this plan)

- Modify AF2 evaluation filter (BindCraft-style `add_cyclic_offset`)
- Modify Protenix-0.5.0-pxd `RelativePositionEncoding` the same way
- Add `covalent_bonds` N-C entry to output JSON for downstream validation
