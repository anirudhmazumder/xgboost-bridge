# xgboost-bridge

Read this before changing anything. The invariants below are not style preferences — each one exists because violating it produces *plausible wrong numbers* rather than an error.

## What this is

`xgboost-bridge` exports trained XGBoost models as portable JSON artifacts and runs zero-dependency JavaScript inference in browser and edge environments.

It exists because standard XGBoost→ONNX conversion **fails silently**: the conversion succeeds, inference runs, and the predictions are wrong. No exception, no warning. Every design choice in this repository follows from that failure mode. A crash is an acceptable outcome; a wrong number is not.

| Package | Registry | Role |
|---|---|---|
| `xgboost-bridge` | PyPI | Python export, inspection, reference predictor |
| `xgboost-predictor` | npm | Zero-dependency browser/Node inference |

**1.0 scope:** binary classification, regression, Cox survival.

This is a general-purpose library. **No application-specific vocabulary anywhere — variable names included.** No domain nouns, no dataset names, no borrowed problem framing. If a name would only make sense to someone working on one particular problem, it is the wrong name.

## Layout

```
packages/python/src/xgboost_bridge/   Python package
packages/js/src/                      TypeScript package
fixtures/                             fixture corpus + generators
schema/                               JSON Schema for the artifact format
probes/                               recorded empirical findings
```

Python is a `uv` workspace with members `packages/python` and `fixtures`. JavaScript builds with `tsup`.

## Invariants

### Float32 split precision

XGBoost's engine compares features against thresholds in **float32**. Exported threshold values are the shortest decimal that round-trips in float32 — not a bit-identical float64.

Cast **both sides** of every comparison:

```python
np.float32(value) < np.float32(threshold)
```

```javascript
Math.fround(value) < Math.fround(splitCond)
```

Casting only the sample value is insufficient. It is correct on most rows and wrong on a few — the exact silent-failure signature this project exists to prevent. This is the highest-value invariant in the codebase; it belongs on the first line of the tree walk, not patched in afterward.

This extends beyond the comparison itself. **Anything that reads, stores, or transforms a threshold or a `base_score` value must preserve float32 discipline.** Parsing is part of the numerical core: `JSON.parse` returns float64 unconditionally, and a parser that lands thresholds as unconstrained floats destroys the invariant before the tree walk ever runs. The walk will look correct and produce wrong numbers on a fraction of rows.

### `base_score` is per-objective

`base_score` is stored in a different space depending on the objective. There is no general rule:

- `reg:squarederror` — `boost_from_average` can overwrite a user-supplied value at fit time.
- `binary:logistic` — stored in **probability** space; the margin intercept is `logit(base_score)`.
- Cox / survival — requires `ln(base_score)`.

**Never infer a new objective's space by analogy to an existing one.** Every objective gets its own verification against a real fitted artifact, recorded under `probes/`.

### DART requires two independent detection signals

DART serializes `gradient_booster.name` as `"gbtree"`. Single-signal booster detection silently misclassifies a DART model as a plain tree ensemble. Detect it with two independent signals.

### gblinear determinism

The `shotgun` updater is non-deterministic even at a fixed seed. Pin `coord_descent` anywhere reproducibility matters, fixtures included.

## Design commitments

- **Strict feature keys.** Exact name match — no missing keys, no extra keys. Lenient handling silently converts a typo into a missing-value path, and the error compounds through the ensemble. Rationale is in `COMPAT.md`.
- **No `fromFile` in JavaScript.** Consumers do their own I/O and call `fromJSON`. Keeps the bundle universal across browser, edge, and Node.
- **Fail loudly on anything unrecognized.** Unknown objective, unknown booster, unknown field, out-of-range version marker — raise. Never default, never guess, never skip.
- **Deterministic export.** The same model in produces a byte-identical artifact out. Tested explicitly.
- **Zero JavaScript runtime dependencies.** Non-negotiable. Not "few." Zero.
- **The version marker is the migration mechanism.** Do not reserve structural space in the artifact format for unimplemented features. An unimplemented head has not been designed, so the reserved shape is usually wrong, and the migration happens anyway — having carried dead fields in the meantime.

Durable decisions and their rationale live in `DECISIONS.md`. Add to it rather than relitigating.

## Verification gates

Every one of these must hold before work is considered done.

| Check | Threshold |
|---|---|
| Python suite | All pass; test count never decreases |
| Node suite | All pass; test count never decreases |
| Cross-language parity | **Exactly `0.0`** — not "small," not `1e-15` |
| Python vs XGBoost margin error | ≤ `1e-6`, expected low single-digit `1e-7` |
| `tsc --noEmit` | Clean, run as a step separate from the build |
| JS tests import from `dist/` | Confirmed — never from `src/` |
| Export determinism | Byte-identical across runs |
| JS runtime dependencies | **0** |
| Vocabulary scrub | No application-specific terms, variable names included |

A **nonzero parity number means something is wrong at the bit level.** It is almost always one of two things: a missing float32 cast on one side, or a `base_score` transform applied in the wrong space. Diagnose it. Never accept a tiny nonzero parity and move on — that is precisely the silent failure this project exists to prevent.

## Rules for changes

- **Never loosen, skip, or `xfail` a test.** If a test is in the way, the test is telling you something. Report it; do not disable it.
- **Never add a dependency** without it being an explicit, separate decision. Zero JS runtime dependencies is absolute.
- **Report ambiguity rather than resolving it.** A confident guess about empirical XGBoost behavior propagates into every downstream number.
- Verify float32 boundary tests by deliberately reverting the cast to float64 and confirming those specific tests go red. A boundary fixture that a buggy build still passes is decorative.

## Commands

```bash
uv sync                                    # Python workspace
uv run pytest                              # Python suite
npm --prefix packages/js install           # JS deps
npm --prefix packages/js run build         # bundle to dist/
npm --prefix packages/js run typecheck     # tsc --noEmit, separate step
npm --prefix packages/js test              # build, then run suite against dist/
```
