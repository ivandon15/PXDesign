# Cyclic Binder Design — Feature Design Doc

Date: 2026-03-03

## Goal

Add a `cyclic: true` flag to the PXDesign YAML config so that de novo binders
are generated with head-to-tail (N-to-C backbone) cyclization awareness during
diffusion inference.

## Scope

**In scope:** Generation (PXDesign diffusion model).
**Out of scope (future work):** Evaluation filters (AF2, Protenix). These will
need separate modifications to score cyclic peptides correctly.

---

## Background

PXDesign uses a `RelativePositionEncoding` module that computes pairwise
`rel_pos_index = residue_index[i] - residue_index[j]` for all token pairs.
For a linear chain of length L, `rel_pos(1, L) = -(L-1)`, telling the model
the termini are far apart. For a cyclic peptide they are neighbors, so the
correct value is `-1`.

BindCraft solves this with `add_cyclic_offset` (AF2-based), and AF3 uses
`CycFeatures` with a graph shortest-path approach. We adopt the same ring-graph
shortest-path formula for PXDesign.

---

## Design

### 1. YAML interface

```yaml
binder_length: 30
cyclic: true        # new optional flag, default false

target:
  file: "./examples/5o45.cif"
  chains:
    A:
      hotspots: [40, 99, 107]
```

### 2. Data flow

```
YAML
 └─ parse_yaml_to_json()          [pxdesign/utils/inputs.py]
     └─ generation[{..., "cyclic": true}]
         └─ make_gen_sequences()  [pxdesign/data/infer_data_pipeline.py]
             └─ proteinChain[{..., "cyclic": true}]
                 └─ add_design_features()  [pxdesign/data/json_to_feature.py]
                     └─ feature_dict["binder_cyclic_offset"]  [N_token, N_token]
                         └─ RelativePositionEncoding.forward()  [pxdesign/model/embedders.py]
                             └─ overrides rel_pos_index on binder-binder subblock
```

### 3. Cyclic offset formula

```python
def compute_cyclic_offset(L: int) -> np.ndarray:
    """Shortest-path signed distance on a ring graph of length L."""
    i = np.arange(L)
    ij = np.stack([i, i + L], -1)
    offset = i[:, None] - i[None, :]
    c_offset = np.abs(ij[:, None, :, None] - ij[None, :, None, :]).min((2, 3))
    a = c_offset < np.abs(offset)
    c_offset[a] = -c_offset[a]
    return c_offset * np.sign(offset)  # [L, L]
```

A full `[N_token, N_token]` tensor `binder_cyclic_offset` is initialized to
zeros. The binder-binder subblock is identified via `design_token_mask`
(`~condi_token_mask`) and filled with the cyclic offset. This is added to
`feature_dict` only when `cyclic=true`.

### 4. Model change

In `RelativePositionEncoding.forward()`, after computing `rel_pos_index`,
override the binder-binder block when `binder_cyclic_offset` is present:

```python
if "binder_cyclic_offset" in input_feature_dict:
    cyc = input_feature_dict["binder_cyclic_offset"]
    mask = cyc != 0
    rel_pos_index = torch.where(mask, cyc, rel_pos_index)
```

This only affects the binder-binder subblock; target-target and
target-binder pairs are unchanged.

---

## Files to modify

| File | Change |
|------|--------|
| `pxdesign/utils/inputs.py` | Read `cyclic` from YAML, pass into `generation` dict |
| `pxdesign/data/infer_data_pipeline.py` | Propagate `cyclic` from `generation` to `proteinChain` in `make_gen_sequences()` |
| `pxdesign/data/json_to_feature.py` | Compute `binder_cyclic_offset` in `add_design_features()` |
| `pxdesign/model/embedders.py` | Override `rel_pos_index` in `RelativePositionEncoding.forward()` |

---

## Future work

- Modify AF2 evaluation filter to apply cyclic offset (BindCraft-style)
- Modify Protenix evaluation filter (`Protenix-0.5.0-pxd`) to apply cyclic
  offset in its own `RelativePositionEncoding`
- Consider adding a `covalent_bonds` entry in the output JSON to represent
  the N-C bond for downstream structure validation
