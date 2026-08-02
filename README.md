# xgboost-bridge

**Status: pre-1.0, under construction.** Most of this repository does not
exist yet. There is no install-and-use path today; see below for what's
built so far.

## What this is

`xgboost-bridge` exports trained XGBoost models as portable JSON artifacts.
`xgboost-predictor` runs zero-dependency JavaScript inference on those
artifacts in browsers, edge runtimes, and Node.

| Package | Registry | Role |
|---|---|---|
| `xgboost-bridge` | PyPI | Python export, artifact inspection, reference predictor |
| `xgboost-predictor` | npm | Zero-dependency browser/edge/Node inference |

## Why this exists

The standard path for taking an XGBoost model out of Python — converting it
to ONNX and running it elsewhere — fails silently. The conversion succeeds,
inference runs, and the predictions are wrong, with no exception and no
warning. This project exists because a crash is an acceptable outcome and a
plausible wrong number is not. Every design choice in this repository
follows from that failure mode.

## 1.0 scope

Binary classification, regression, and Cox survival. Multi-class objectives
are explicitly out of scope for this release; see `DECISIONS.md`.

## Repository layout

```
packages/python/src/xgboost_bridge/   Python package (export, inspection, reference predictor)
packages/js/src/                      TypeScript package (browser/edge/Node inference)
fixtures/                             fixture corpus + generators
schema/                               JSON Schema for the artifact format (not yet present)
probes/                               recorded empirical findings (not yet present)
```

## More detail

- `COMPAT.md` — compatibility and support policy (feature-key strictness,
  XGBoost version boundary, Python version floors).
- `DECISIONS.md` — durable engineering decisions and their rationale.
