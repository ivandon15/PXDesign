# Cyclic Evaluation Design

**Goal:** Thread the `cyclic` flag from the input YAML/JSON into the AF2 and Protenix evaluation filters so cyclic-aware scoring is applied when `cyclic: true` is set.

**Architecture:** Read `cyclic` from the first task's generation list in `pipeline.py:main()` after loading the input JSON. Propagate it to two eval config fields that `pxdbench` already checks.

**Tech Stack:** Python, pxdbench (AF2ComplexPredictor, ProtenixFilter), PXDesign pipeline

---

## Data Flow

```
YAML: cyclic: true
  ↓ parse_yaml_to_json()
generation[0]["cyclic"] = True  (already done)
  ↓ pipeline.py:main()
configs.eval.binder.is_cyclic = True
configs.eval.binder.tools.af2.is_cyclic = True
  ↓ BinderTask.af2_complex_predict()
AF2ComplexPredictor → complex_prediction(is_cyclic=True) → add_cyclic_offset()
  ↓ BinderTask.protenix_predict()
ProtenixFilter.predict(is_cyclic=True) → make_is_cyclic_mask_feat()
```

## Change

Single location: `pxdesign/runner/pipeline.py:main()`

After loading `orig_inputs` and before constructing `DesignPipeline`, read `cyclic` from the first task:

```python
cyclic = orig_inputs[0].get("generation", [{}])[0].get("cyclic", False)
if cyclic:
    configs.eval.binder.is_cyclic = True
    configs.eval.binder.tools.af2.is_cyclic = True
```

## What pxdbench already does

- `AF2ComplexPredictor.predict()` passes `is_cyclic=self.cfg.get("is_cyclic", False)` to `complex_prediction()`
- `complex_prediction()` calls `add_cyclic_offset(prediction_model)` when `is_cyclic=True`
- `ProtenixFilter.predict()` accepts `is_cyclic=False`; when True calls `make_is_cyclic_mask_feat()`
- `BinderTask` reads `self.cfg.get("is_cyclic", False)` — but currently never passes it to `protenix_predict()`; needs to be wired

## Additional fix needed in pxdbench

`BinderTask.protenix_predict()` in `base.py` does not currently pass `is_cyclic` to `ProtenixFilter.predict()`. Since pxdbench is an installed package, we need to check if `BinderTask` reads `is_cyclic` from cfg and passes it through. If not, we patch `base.py` in the installed package.
