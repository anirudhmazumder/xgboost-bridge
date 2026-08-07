# xgboost-predictor

Zero-dependency XGBoost inference for browser, edge, and Node runtimes. Reads the portable JSON artifacts produced by [`xgboost-bridge`](https://pypi.org/project/xgboost-bridge/) on PyPI.

**Why this exists:** the standard way to take an XGBoost model out of Python — converting it to ONNX and running it elsewhere — fails silently. The conversion succeeds, inference runs, and the predictions are wrong, with no exception and no warning. This package exists so that a model it cannot handle raises an error instead of returning a plausible wrong number.

Zero runtime dependencies. Not "few" — zero.

## Install

```bash
npm install xgboost-predictor
```

## Use

The API takes a **parsed object**, not a path. Consumers do their own I/O, because filesystem access is unavailable in browsers and differs across edge runtimes.

```js
import { fromJSON } from "xgboost-predictor";

const predictor = fromJSON(artifact); // artifact = JSON.parse(...) of an exported file

const row = { feature_0: 0.7, feature_1: -0.2, feature_2: 1.0 };

predictor.margin(row); // the raw margin, float32
predictor.output(row); // after the output transform — a probability, here
```

Both ESM and CommonJS are supported.

## What it refuses

Refusing loudly is the point of the package, not an omission:

- **Feature keys must match exactly** — no missing key, no extra key. Lenient handling would turn a typo into a missing-value path, which is legitimate model structure, so the mistake becomes a confident wrong number rather than an error. You normalize the shape; this package will not guess.
- **`NaN` is the missing value** and routes by the tree's default direction. **A value that is infinite in `float32` raises** — `±Infinity`, and also any finite double that narrows to it, such as `1e39`. This package compares in `float32`, so those are the same value at the point of comparison. Upstream XGBoost is itself inconsistent about infinity, giving different predictions depending on the call path used.
- **A malformed or unrecognized artifact raises** — an unknown field, an unsupported version marker, an out-of-range index, an objective/transform mismatch.

## Error messages quote your artifact back

Every error carries a `code` you can branch on, and most carry structured properties describing what was wrong — `MalformedArtifactError` has `field`, `value`, `expected` and `location`; `NonFiniteFeatureError` has `index`, `feature` and `value`; `FeatureKeyMismatchError` has `missing` and `extra`. Which properties exist depends on the error, so branch on `code` first. The human-readable `message` embeds the offending value verbatim so a developer can see what arrived. If your artifact or prediction input can be influenced by someone else (served from a CDN or a bucket, pasted by a user), **treat `err.message` as untrusted text**: escape it before rendering, or branch on the structured properties and compose your own message. Rendering it into `innerHTML` unescaped would let whoever controls the artifact run script in your origin. This is ordinary library behaviour rather than a defect — the structured properties exist so you never have to parse or display the string.

## Numerical behaviour worth knowing

`sigmoid` and `exp` are **implemented in this package**, not taken from `Math`. IEEE-754 only mandates correct rounding for `+ − × ÷ √`, so `Math.exp` differs between JavaScript engines and C libraries — measured here at up to 2 ULP on 4.2% of sigmoid evaluations. A bundled implementation is the only way the JavaScript and Python predictors can agree bit-for-bit, which they do: **exactly `0.0` difference** across the whole fixture corpus, at both the margin and the final output.

The consequence for you: comparing this package's output against `Math.exp` or against XGBoost's own probability will show last-bit differences. That is deliberate and bounded, not a defect.

## Documentation

- [Artifact format](https://github.com/anirudhmazumder/xgboost-bridge/blob/main/FORMAT.md)
- [Compatibility and support policy](https://github.com/anirudhmazumder/xgboost-bridge/blob/main/COMPAT.md)
- [Engineering decisions and their evidence](https://github.com/anirudhmazumder/xgboost-bridge/blob/main/docs/DECISIONS.md)

MIT licensed.

**AI Disclosure:** Claude Code was used to help implement this project.
