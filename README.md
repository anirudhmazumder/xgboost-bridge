# xgboost-bridge

`xgboost-bridge` exports trained XGBoost models as portable JSON artifacts.
`xgboost-predictor` runs zero-dependency JavaScript inference on those
artifacts in browsers, edge runtimes, and Node.

| Package | Registry | Role |
|---|---|---|
| [`xgboost-bridge`](https://pypi.org/project/xgboost-bridge/) | PyPI | Python export, artifact inspection, reference predictor |
| [`xgboost-predictor`](https://www.npmjs.com/package/xgboost-predictor) | npm | Zero-dependency browser/edge/Node inference |

## Why this exists

The standard way to take an XGBoost model out of Python — converting it to
ONNX and running it elsewhere — **fails silently**. The conversion succeeds,
inference runs, and the predictions are wrong, with no exception and no
warning raised at any step. Nothing tells you that anything went wrong; you
find out later, when the numbers a deployed model produces don't match the
numbers the same model produced in Python.

This project exists because a crash is an acceptable outcome and a plausible
wrong number is not. Every design choice in this repository follows from
that failure mode: strict validation instead of best-effort defaults, a
hand-rolled float32 arithmetic path instead of a platform math function that
might round differently on a different machine, and a version ceiling that
refuses an XGBoost build nobody has actually verified against — see
`docs/DECISIONS.md` for the specific measurements behind each of those choices.

## 1.0 scope

Binary classification, regression, and Cox survival objectives —
`binary:logistic`, `reg:squarederror`, `survival:cox`. **`gbtree` only**:
`dart` and `gblinear` raise a loud, specific error at export rather than
being silently approximated (see below). Multi-class objectives
(`multi:softmax`, `multi:softprob`) are out of scope for this release and
raise on export; no structural space is reserved for them, so adding
multi-class later is a new format version rather than filling in a
pre-reserved field.

## Install

```bash
# Read exported artifacts and run the reference predictor. Requires only numpy.
pip install xgboost-bridge

# Export models too. Requires Python >=3.12, because XGBoost 3.3.0 does.
pip install "xgboost-bridge[export]"
```

```bash
npm install xgboost-predictor
```

See `COMPAT.md` for the two `requires-python` floors and why they differ,
and the exact XGBoost and Node versions this has been verified against.

## Quick start

### Export a model (Python)

Every code example on this page was run against a real fitted model; see
`docs/DECISIONS.md` for the project's take on trusting an example that wasn't.

```python
import numpy as np
import xgboost as xgb

from xgboost_bridge import export_model, to_json

feature_names = ["feature_0", "feature_1", "feature_2"]

rng = np.random.default_rng(0)
X = rng.uniform(-2.0, 2.0, size=(200, 3))
y = (X[:, 0] + 0.5 * X[:, 1] - X[:, 2] > 0.0).astype(np.float64)

dtrain = xgb.DMatrix(X, label=y, feature_names=feature_names, nthread=1)
booster = xgb.train(
    {"objective": "binary:logistic", "max_depth": 3, "eta": 0.3, "nthread": 1, "seed": 0},
    dtrain,
    num_boost_round=10,
)

# feature_names is required whenever the model was fit from a bare array —
# a model with no feature names cannot be exported (D021).
artifact = export_model(booster, feature_names=feature_names)
document = to_json(artifact)  # a deterministic JSON string; write it wherever you like
```

`document` starts:

```json
{"feature_names":["feature_0","feature_1","feature_2"],"format_version":1,"intercept":0.18048842251300812,"objective":"binary:logistic","output_transform":"sigmoid","provenance":{"base_score":"[5.45E-1]","exporter_version":"1.0.0rc1","xgboost_version":"3.3.0"},"trees":[{"default_left":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"left_children":[1,3,5,7,9,11,13,-1,-1,-1,-1,-1,-1,-1,-1],"node_values":[0.10408...
```

The Python package also ships a reference predictor, useful for checking an
artifact before it ever reaches JavaScript:

```python
import json
from xgboost_bridge import Predictor

predictor = Predictor.from_json(json.loads(document))
row = {"feature_0": 0.70, "feature_1": -0.20, "feature_2": 1.00}

predictor.margin(row)   # np.float32(-0.44306272)
predictor.output(row)   # np.float32(0.39101142)
```

### Run inference (JavaScript)

There is no `fromFile` — this package does no I/O of its own, on purpose
(D006), so it works the same way in a browser as it does in Node. Read the
artifact however your environment reads files, `JSON.parse` it, and hand the
parsed object to `fromJSON`:

```javascript
import { readFileSync } from "node:fs";
import { fromJSON } from "xgboost-predictor";

const artifact = JSON.parse(readFileSync("artifact.json", "utf8"));
const predictor = fromJSON(artifact);

const row = { feature_0: 0.70, feature_1: -0.20, feature_2: 1.00 };

predictor.margin(row);  // -0.4430627226829529
predictor.output(row);  // 0.39101141691207886
```

Run against the artifact produced by the Python example above, this prints
the same value the Python reference predictor did — `Math.fround` of the
JavaScript result and `np.float32` of the Python result are the same
float32 bit pattern, which is the entire point of this project. Every
number on this page was produced by actually running both sides against
each other, not typed in by hand.

## What is deliberately refused

An unsupported objective or booster raising loudly is a documented feature
of this library, not a gap to work around:

- **`dart`** raises `UnsupportedBoosterError` if the artifact's own dropout
  field (`weight_drop`) is present at either of the two locations it has
  occupied across XGBoost versions. A `dart` model trained with no actual
  dropout is byte-identical to a plain tree ensemble and exports fine —
  correctly, because it *is* one.
- **`gblinear`** raises unconditionally. It has no trees, is deprecated in
  XGBoost 3.3.0 with removal announced, and would need an entirely separate
  predictor implementation in both languages for a booster on its way out.
- **Categorical splits** raise. This format has no representation for them.
- **Multi-class objectives, and any objective whose output arity doesn't
  match a single scalar prediction**, raise — including
  `reg:squarederror` fit with `num_target=2`, which is an in-scope
  objective by name but produces two-column output and is refused on arity
  instead.
- **An artifact from an untested XGBoost version** raises. The supported
  list in `COMPAT.md` is versions actually probed, not a guessed range —
  an unrecognized-field check cannot catch a field that *moved* between
  versions, which is a real, measured failure mode (see `docs/DECISIONS.md`
  D018).
- **A feature-key mismatch at prediction time** raises. Both predictors
  require the input's keys to match the model's feature names exactly — no
  missing key, no extra key. See `COMPAT.md` for what that costs a caller
  whose input doesn't already have that shape.
- **A feature value that is infinite in `float32`** raises at prediction time.
  That is `±Infinity`, and also any finite float64 that narrows to it — `1e39`
  is a legal double whose `float32` is `+inf`. Both predictors compare in
  `float32`, so the two are the same value by the time any comparison happens,
  and refusing only the first gave one mathematical value two behaviours
  (D055). `NaN` is accepted — it is XGBoost's missing-value marker and is
  routed by the model's own `default_left`. Infinity is refused on both sides
  because upstream XGBoost is itself inconsistent about it (see `COMPAT.md`).

Every one of these is backed by a fixture, tested independently on both
sides of the language boundary. If a model or an input hits one of these
refusals, that is this library working as intended.

## More detail

- [`FORMAT.md`](FORMAT.md) — the artifact format specification.
- [`VERIFICATION.md`](VERIFICATION.md) — what is measured, by what method, and
  what is **not** measured. Read the last section if you are on Linux: every
  figure was measured on darwin/arm64 and CI has not yet confirmed them.
- [`COMPAT.md`](COMPAT.md) — compatibility and support policy: feature-key
  strictness and what it costs you, the XGBoost and Node version boundaries,
  and what a caller should expect from the bundled numeric transform.
- [`docs/DECISIONS.md`](docs/DECISIONS.md) — durable engineering decisions, the
  evidence behind each one, and which superseded which.
Every decision that asserts runtime behaviour is mapped to a test that fails when the behaviour is reverted; the map is [`docs/DECISION_COVERAGE.md`](docs/DECISION_COVERAGE.md).

The project's premise is that a plausible wrong number is worse than a crash. The same standard was applied to its authorship: where the model could not establish something, the repository says so.
