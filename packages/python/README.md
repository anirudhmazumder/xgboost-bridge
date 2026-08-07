# xgboost-bridge

Export trained XGBoost models as portable JSON artifacts, with a reference
predictor for validating them. Companion to
[`xgboost-predictor`](https://www.npmjs.com/package/xgboost-predictor) on
npm, which runs inference on those artifacts with zero dependencies in
browser and edge runtimes.

**Why this exists:** the standard way to take an XGBoost model out of
Python — converting it to ONNX and running it elsewhere — fails silently.
Conversion succeeds, inference runs, and the predictions are wrong, with no
exception and no warning. This library exists so that a model it can't
handle raises an error instead of returning a plausible wrong number.

Full documentation, the artifact format specification, and the rationale
behind every design decision live in the source repository:

- [`FORMAT.md`](https://github.com/anirudhmazumder/xgboost-bridge/blob/main/FORMAT.md) — the artifact format.
- [`COMPAT.md`](https://github.com/anirudhmazumder/xgboost-bridge/blob/main/COMPAT.md) — compatibility and support policy.
- [`docs/DECISIONS.md`](https://github.com/anirudhmazumder/xgboost-bridge/blob/main/docs/DECISIONS.md) — engineering decisions and their evidence.

## 1.0 scope

Binary classification, regression, and Cox survival objectives —
`binary:logistic`, `reg:squarederror`, `survival:cox` — with the `gbtree`
booster only. `dart` and `gblinear` raise a specific error at export rather
than being approximated. Multi-class objectives are out of scope for this
release and also raise on export.

## Install

```bash
# Read exported artifacts and run the reference predictor. Requires only numpy.
pip install xgboost-bridge

# Also export models. Requires Python >=3.12: XGBoost 3.3.0 itself does.
pip install "xgboost-bridge[export]"
```

The base package's floor is Python `>=3.10` — reading an artifact and
predicting from it needs nothing but `numpy`. The `export` extra pins
`xgboost==3.3.0`, which requires `>=3.12`, so installing the base package on
3.10 or 3.11 works fine but installing the `export` extra there will fail to
resolve. That is XGBoost's floor, not a packaging bug.

The pin is exact rather than a range, and deliberately so. This library
refuses to export a model produced by an XGBoost version it has not verified
against, because version drift here is silent — XGBoost 3.4.0-dev relocated a
field, and 3.3.0 reads such a model returning wrong predictions with no
warning and exit code 0. A permissive range would resolve a version the
exporter then refuses, so the dependency and the verified list are kept
identical. Widening either requires re-probing.

## Export a model

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
# a model with no feature names cannot be exported.
artifact = export_model(booster, feature_names=feature_names)
document = to_json(artifact)  # a deterministic JSON string
```

The reference predictor reads the same artifact back, useful for confirming
it before it ever reaches JavaScript:

```python
import json
from xgboost_bridge import Predictor

predictor = Predictor.from_json(json.loads(document))
row = {"feature_0": 0.70, "feature_1": -0.20, "feature_2": 1.00}

predictor.margin(row)   # np.float32(-0.44306272)
predictor.output(row)   # np.float32(0.39101142)
```

## Strict feature keys

Prediction input must match the model's feature names exactly — no missing
key, no extra key. A mismatch raises `FeatureKeyMismatchError` rather than
being tolerated.

**What this costs you:** if your input records come from a source that
doesn't guarantee an exact key match — a database row with extra columns, a
form with optional fields, a renamed column upstream — you have to
normalize that shape yourself before calling the predictor. This library
will not silently drop extra keys, and it will not silently treat a missing
key as a "missing value" on your behalf. That normalization work is real,
and it is deliberately pushed onto the caller.

**Why it's still the right call:** under lenient handling, a misspelled or
renamed feature name doesn't fail — it quietly becomes a missing-value
input, and XGBoost's missing-value branches are legitimate model structure.
The result is a confident, plausible, wrong prediction, compounding across
every tree in the ensemble. Strict keys turn that into an error at the call
site instead, which is why this library exists in the first place. Full
rationale, with the specific decisions and measurements behind it, is in
[`COMPAT.md`](https://github.com/anirudhmazumder/xgboost-bridge/blob/main/COMPAT.md).

## What is refused

An unsupported objective or booster raising loudly is a documented feature
of this library. `dart` and `gblinear` raise at export; categorical splits
raise; an artifact from an XGBoost version this library hasn't verified
against raises; multi-output shapes arriving through an otherwise-permitted
objective name (for example `reg:squarederror` with `num_target=2`) raise on
arity. See
[`COMPAT.md`](https://github.com/anirudhmazumder/xgboost-bridge/blob/main/COMPAT.md)
for the complete, current list and the evidence behind each refusal.

## Error messages quote your artifact back

Every exception is an `XGBoostBridgeError` subclass, and most carry structured attributes describing what was wrong — `MalformedTreeError` has `field`, `value`, `expected` and `location`; `NonFiniteFeatureError` has `index` and `value`; `FeatureKeyMismatchError` has `missing_keys` and `extra_keys`. Which attributes exist depends on the exception, so catch the specific class rather than assuming a field is present. The message embeds the offending value verbatim so the failure is legible. If your artifact or prediction input can be influenced by someone else, **do not pass the message straight into an HTTP response body or an HTML template**: escape it, or branch on the structured attributes and compose your own. Ordinary library behaviour, noted because the attributes exist precisely so you never have to display the string.

**AI Disclosure:** Claude Code was used to help implement this project.
