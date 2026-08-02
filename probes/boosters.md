# Probe — booster types: gbtree, dart, gblinear

Empirical findings on how the three boosters differ in XGBoost's serialized model, how
dart can be distinguished, and whether gblinear is reproducible per updater.

Every claim below is backed by a pasted command and its real output. Anything not
directly measured is labelled **INFERRED**. Two items are surfaced as open decisions
rather than resolved.

## Environment

```
$ uv run python -c "
import sys, xgboost, numpy
print('python', sys.version)
print('xgboost', xgboost.__version__)
print('numpy', numpy.__version__)
"
python 3.12.8 (main, Dec  3 2024, 18:42:41) [Clang 16.0.0 (clang-1600.0.26.4)]
xgboost 3.3.0
numpy 2.5.1
```

Matches the D001 reference pin. All findings below are for `xgboost 3.3.0` exactly.

## Method

Three models fitted on the same synthetic data (400 rows x 6 columns, `numpy`
`default_rng(20260801)`, generic feature names `f0`..`f5`), same seed, same
`objective=binary:logistic`, `eta=0.3`, `max_depth=3`, `nthread=1`, 8 boosting
rounds — one per booster. Serialized with `save_model()` to JSON and diffed
structurally. Scripts lived outside the repository; no fitted binaries were written
into the tree.

---

## Verdicts up front

| Hypothesis | Verdict |
|---|---|
| **H1** — dart serializes `gradient_booster.name` as `"gbtree"` | **REPRODUCED.** And it is worse than stated: the string `dart` appears *nowhere* in the artifact. |
| **H2** — at least two independent detection signals are needed | **PARTIALLY REPRODUCED, AND THE PREMISE FAILS.** The need is confirmed. But **only one independent signal exists inside the artifact.** A second signal exists only at export time, from the live `Booster`, and it is demonstrably unreliable. See the decision below. |
| **H3** — `shotgun` is non-deterministic at fixed seed; `coord_descent` is deterministic | **REPRODUCED, with a material qualification.** `shotgun` was non-deterministic 12/12 trials at `nthread=4` but *fully reproducible* 12/12 trials at `nthread=1`. The non-determinism is thread-parallel, not seed-related. `coord_descent` reproduced bit-exactly at both thread counts. |

---

## 1. `gradient_booster.name` for all three boosters

```
$ uv run python fit.py
=== gbtree ===
gradient_booster keys: ['model', 'name']
gradient_booster.name = "gbtree"

=== dart ===
gradient_booster keys: ['model', 'name', 'weight_drop']
gradient_booster.name = "gbtree"

=== gblinear ===
gradient_booster keys: ['model', 'name']
gradient_booster.name = "gblinear"
```

| Booster requested | `$.learner.gradient_booster.name` |
|---|---|
| `gbtree` | `"gbtree"` |
| `dart` | `"gbtree"` |
| `gblinear` | `"gblinear"` |

**H1 REPRODUCED.** `booster=dart` serializes as `"gbtree"`. A predictor that switches on
`gradient_booster.name` alone reads a dart model as a plain tree ensemble.

It is stronger than H1 claimed — the token `dart` is absent from the entire file, in
both JSON and UBJSON:

```
$ grep -o "dart" dart.json | sort | uniq -c
(no output — the string 'dart' appears nowhere)

$ grep -io "drop" dart.json | sort | uniq -c
   1 drop          # this is the substring of "weight_drop"
```

```
    token 'dart'                 present in model file: False
    token 'rate_drop'            present in model file: False
    token 'skip_drop'            present in model file: False
    token 'normalize_type'       present in model file: False
    token 'one_drop'             present in model file: False
    token 'dart_train_param'     present in model file: False
    token 'learner_train_param'  present in model file: False
```

Same for the binary format:

```
    token b'dart'            in .ubj bytes: False
    token b'rate_drop'       in .ubj bytes: False
    token b'skip_drop'       in .ubj bytes: False
    token b'weight_drop'     in .ubj bytes: True
    token b'normalize_type'  in .ubj bytes: False
```

---

## 2. Structural diff — every field where dart differs from gbtree

Exactly one field, and one exhaustive key census to prove it.

```
########## PATHS PRESENT IN dart BUT NOT gbtree ##########
  $.learner.gradient_booster.weight_drop  =  array[8] scalars preview=[0.7905139, 0.90909094, 0.6993007, 0.7905139]

########## PATHS PRESENT IN gbtree BUT NOT dart ##########
  (none)

########## PATHS IN BOTH, VALUE/SHAPE DIFFERS (dart vs gbtree) ##########
  $.learner.gradient_booster
    gbtree: object{model,name}
    dart  : object{model,name,weight_drop}
```

Every key name appearing anywhere at any depth:

```
  gbtree    key count = 46
  dart      key count = 47
  gblinear  key count = 19
  dart keys NOT in gbtree keys   : ['weight_drop']
  gbtree keys NOT in dart keys   : []
```

`learner.attributes`, `learner.objective`, and `learner_model_param` are byte-identical
across all three boosters:

```
--- gbtree ---                     --- dart ---                      --- gblinear ---
attributes: {}                     attributes: {}                    attributes: {}
base_score: "[5.125E-1]"           base_score: "[5.125E-1]"          base_score: "[5.125E-1]"
boost_from_average: "1"            boost_from_average: "1"           boost_from_average: "1"
num_class: "0"                     num_class: "0"                    num_class: "0"
num_feature: "6"                   num_feature: "6"                  num_feature: "6"
num_target: "1"                    num_target: "1"                   num_target: "1"
objective: {"name": "binary:logistic", "reg_loss_param": {"scale_pos_weight": "1"}}
```

Pasted excerpt, `$.learner.gradient_booster` with the tree array elided:

```json
===== gbtree =====
{
  "name": "gbtree",
  "model": {
    "gbtree_model_param": { "num_parallel_tree": "1", "num_trees": "8" },
    "iteration_indptr": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "tree_info": [0, 0, 0, 0, 0, 0, 0, 0],
    "trees": "<8 trees elided>"
  }
}

===== dart =====
{
  "name": "gbtree",
  "weight_drop": [
    0.7905139, 0.90909094, 0.6993007, 0.7905139,
    0.63572794, 0.3030303, 0.43478262, 0.3030303
  ],
  "model": {
    "gbtree_model_param": { "num_parallel_tree": "1", "num_trees": "8" },
    "iteration_indptr": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "tree_info": [0, 0, 0, 0, 0, 0, 0, 0],
    "trees": "<8 trees elided>"
  }
}
```

The tree representation itself — `trees[]`, `iteration_indptr`, `tree_info`,
`gbtree_model_param` — is structurally identical. dart adds nothing to it and removes
nothing from it.

### 2a. In 3.3.0, "dart" is a *mode of gbtree*, not a booster

This reframes the whole question and was not anticipated by H1/H2.

XGBoost 3.3.0 emits a deprecation warning on `booster=dart`:

```
WARNING: src/learner.cc:341: `booster=dart` is deprecated. Use the tree booster
directly with dropout parameters like `rate_drop`, `skip_drop`, or `one_drop`.
```

`save_config()` on a model trained with `booster=dart` reports the booster as `gbtree`
at *both* levels — not just in the serialized model:

```
--- save_config() for dart
    learner.gradient_booster.name  = "gbtree"
    learner_train_param.booster    = "gbtree"
    dart_train_param = {"normalize_type": "tree", "one_drop": "0",
                        "rate_drop": "0.300000012", "sample_type": "uniform",
                        "skip_drop": "0.100000001"}
```

Note `dart_train_param` is present *even for a plain gbtree model* (with zeros):

```
--- save_config() for gbtree
    dart_train_param = {"normalize_type": "tree", "one_drop": "0",
                        "rate_drop": "0", "sample_type": "uniform", "skip_drop": "0"}
```

The consequences, all measured:

```
--- booster=dart, ALL DEFAULTS (rate_drop=0 skip_drop=0)
    gradient_booster.name   = "gbtree"
    gradient_booster keys   = ['model', 'name']
    weight_drop             = <ABSENT>

--- booster=gbtree, rate_drop=0.3 skip_drop=0.1
    gradient_booster.name   = "gbtree"
    gradient_booster keys   = ['model', 'name', 'weight_drop']
    weight_drop len=8  values=[0.7905139, 0.90909094, 0.6993007, 0.7905139, 0.63572794, 0.3030303, 0.43478262, 0.3030303]

--- booster=gbtree (booster unspecified) rate_drop=0.3
    gradient_booster.name   = "gbtree"
    gradient_booster keys   = ['model', 'name', 'weight_drop']
    weight_drop len=8  values=[0.6688964, 0.6993007, 0.6993007, 0.86956525, 0.90909094, 0.7692308, 0.3030303, 0.43478262]

--- booster=gbtree plain
    gradient_booster keys   = ['model', 'name']
    weight_drop             = <ABSENT>
```

So the mapping is **not** booster-name-based. It is dropout-activity-based:

- `booster=dart` with default `rate_drop=0`/`skip_drop=0`/`one_drop=0` produces **no
  `weight_drop`** and is byte-identical to plain gbtree.
- `booster=gbtree` with `rate_drop=0.3` **does** produce `weight_drop`.

Byte-identity confirmed directly:

```
########## is booster=dart with rate_drop=0 identical to gbtree? ##########
  max|gbtree - dart(defaults)|                        = 0.0
  bit-identical?                                      = True
  raw JSON bytes identical (gbtree vs dart-defaults)? = True
  tree arrays identical?                              = True
```

**The question "is this dart or gbtree?" has no answer in 3.3.0.** The answerable — and
numerically sufficient — question is: **"does this artifact carry per-tree weights that
must be applied?"** That question is decided entirely by `weight_drop`.

---

## 3. Detection signals for dart weighting

### Signal A — presence of the key `$.learner.gradient_booster.weight_drop`

The only signal inside the artifact. Authoritative: it survives a file roundtrip and it,
not the config, determines XGBoost's own predict-time behavior.

```
########## is weight_drop authoritative after a file roundtrip? ##########
  margins bit-identical before/after save+load = True
  max abs diff                                 = 0.0
```

```
  weight_drop survived the roundtrip?  True
  weight_drop after load = [0.7905139, 0.90909094, 0.6993007, 0.7905139, 0.63572794, 0.3030303, 0.43478262, 0.3030303]
```

**Failure mode of Signal A.** A key-presence test has no redundancy. If a producer
(a future XGBoost version, a third-party writer, a lossy transform) drops or renames the
field, the artifact is silently and correctly-shaped as plain gbtree, and there is
nothing else in the file to contradict it. The observed error from that mistake is
**1.27 in margin space** — see Signal B. There is no in-artifact cross-check that can
catch it, because dart contributes no other field.

### Signal B — export-time numerical cross-check against XGBoost's own `predict`

Independent of Signal A: it does not read the field at all. It reconstructs the margin
from `trees[]` ignoring any per-tree weights and compares against
`predict(output_margin=True)`. A large residual proves weights are being applied that
the naive walk is missing.

```
########## export-time numerical cross-check ##########
  plain gbtree
    weight_drop present      = False
    max err IGNORING weights = 4.143380887e-07
    max err APPLYING weights = 4.143380887e-07
  dart, rate_drop=0.3 skip_drop=0.1
    weight_drop present      = True
    max err IGNORING weights = 1.269166159e+00
    max err APPLYING weights = 1.621531536e-07
  gbtree WITH rate_drop=0.3 (3.3.0 deprecation path)
    weight_drop present      = True
    max err IGNORING weights = 1.269166159e+00
    max err APPLYING weights = 1.621531536e-07
  dart, skip_drop=1.0 (weight_drop all 1.0)
    weight_drop present      = True
    max err IGNORING weights = 4.143380887e-07
    max err APPLYING weights = 4.143380887e-07
```

Separation is six orders of magnitude: `1.27e+00` versus `1.6e-07`. This is a genuinely
independent signal — but only where a live `Booster` and XGBoost are importable.

**Failure modes of Signal B.**
1. Available at **export time only**. It cannot run against an artifact read from disk
   with no XGBoost present, which is exactly the JS predictor's situation and the
   `xgboost-bridge` base-install situation under D010.
2. It is blind when every `weight_drop` value is exactly `1.0` — the `skip_drop=1.0` row
   above. In that case there is nothing to detect, because applying and ignoring the
   weights are numerically identical, so the blindness is harmless.
3. It requires prediction rows that actually exercise differing leaves. On a degenerate
   input set the residual could be small by accident. **INFERRED** — not measured; a
   real cross-check should use the full fixture row set, not a handful of rows.

### Rejected candidate — `save_config()` / `dart_train_param`

The obvious second in-artifact signal. **It is not in the artifact, and it lies in both
directions.** `dart_train_param` is not serialized by `save_model` (Section 1 grep), and
it does not survive a roundtrip:

```
########## does dart_train_param survive save_model -> load_model? ##########
  BEFORE save: dart_train_param = {"normalize_type": "tree", "one_drop": "0", "rate_drop": "0.300000012", "sample_type": "uniform", "skip_drop": "0.100000001"}
  AFTER  load: dart_train_param = {"normalize_type": "tree", "one_drop": "0", "rate_drop": "0", "sample_type": "uniform", "skip_drop": "0"}
  identical?  False
```

And on a live `Booster` it is mutable independently of the model:

```
########## failure mode of the config signal ##########
  plain gbtree, before set_param: dart_train_param = {... "rate_drop": "0" ... "skip_drop": "0"}
  plain gbtree, after  set_param: dart_train_param = {... "rate_drop": "0.300000012" ... "skip_drop": "0.100000001"}
  ...but weight_drop in model = <ABSENT>
  => config claims dropout, artifact carries none. Config signal can lie.

  trained-dart, after set_param(0): dart_train_param = {... "rate_drop": "0" ... "skip_drop": "0"}
  ...but weight_drop in model = PRESENT
  => config claims no dropout, artifact carries weights. Config signal can lie both ways.
```

`dart_train_param` reflects *current parameter state*, not training history. It must not
be used to decide whether per-tree weights apply.

### Rejected candidate — `weight_drop` values differing from 1.0

Not independent — it reads the same field as Signal A — and it is **wrong**:

```
--- booster=dart, rate_drop=0.0 skip_drop=1.0 (dropout always skipped)
    weight_drop len=8  values=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    all values == 1.0?      = True

--- booster=dart, 1 boosting round
    weight_drop len=1  values=[1.0]
    all values == 1.0?      = True
```

A dart model can legitimately carry an all-ones `weight_drop`.

### Rejected candidate — key ordering

`gbtree`/`dart` serialize `name` before `model`; `gblinear` serializes `model` before
`name`. Not a signal: JSON object key order is not semantically meaningful and no parser
should depend on it. Recorded only so it is not mistaken for one.

### Rejected candidate — `weight_drop` arity

`len(weight_drop) == num_trees` held in every case measured:

```
  rounds=  0  num_trees=  0  len(weight_drop)=<ABSENT>  arity_match=n/a
  rounds=  1  num_trees=  1  len(weight_drop)=1  arity_match=True
  rounds=  2  num_trees=  2  len(weight_drop)=2  arity_match=True
  rounds=  5  num_trees=  5  len(weight_drop)=5  arity_match=True
  rounds= 13  num_trees= 13  len(weight_drop)=13  arity_match=True
```

Including when trees outnumber rounds:

```
########## dart + num_parallel_tree (trees != rounds) ##########
  num_parallel_tree=3, rounds=4
  num_trees          = 12
  num_parallel_tree  = 3
  len(weight_drop)   = 12
  iteration_indptr   = [0, 3, 6, 9, 12]
  weight_drop        = [0.9756098, 1.0, 1.0, 1.0, 0.9756098, 0.9756098, 1.0, 0.9756098, 1.0, 0.24390244, 0.24390244, 0.24390244]
```

`weight_drop` is indexed **per tree**, not per boosting round. This is a useful
*validation* assertion (a length mismatch should raise, per D007) but it is a consequence
of Signal A, not an independent detector: it cannot fire when the field is absent.

> **DECISION NEEDED: the "two independent detection signals" invariant cannot be
> satisfied from the artifact alone in XGBoost 3.3.0.**
>
> **Context:** `weight_drop` is the *only* dart-distinguishing field anywhere in the
> serialized model (exhaustive key census, Section 2). The string `dart` is absent from
> both JSON and UBJSON. `dart_train_param` is not serialized and lies in both directions
> when read from a live `Booster`. A genuinely independent second signal exists only as
> an export-time numerical cross-check requiring XGBoost in-process.
>
> **Options:**
> A) Two signals at export time — key presence (A) plus numerical cross-check (B) — and
>    one signal plus a hard arity assertion when reading an artifact. Honest about the
>    asymmetry; the JS reader still has single-signal detection.
> B) Have the exporter write its own redundant marker into the `xgboost-bridge`
>    artifact (e.g. an explicit per-tree weights array, always present, all-ones for
>    plain gbtree). Detection becomes structural and unconditional on both sides; costs
>    an always-present array and is a Phase 3 format question.
> C) Refuse to export any model carrying `weight_drop` for 1.0. dart is not in the
>    stated 1.0 scope (binary classification, regression, Cox survival). Zero silent-
>    failure surface; costs dart support.
>
> **Lean:** C for 1.0 scope, plus A's detection logic used as the *rejection* test — so
> a dropout-weighted model raises rather than being silently mis-predicted. If dart is
> in scope, B, because it is the only option that gives the JS reader real redundancy.
>
> **Blocks:** the Phase 3 artifact format's tree-weight representation, and whether the
> exporter's booster dispatch raises or converts on `weight_drop`.

---

## 4. dart's serialized dropout / weight representation

**There is no serialized dropout configuration.** `rate_drop`, `skip_drop`, `one_drop`,
`normalize_type`, and `sample_type` are **not** in the model file (Section 1 greps, JSON
and UBJSON). They are training-time only.

What *is* serialized is the *outcome*: `$.learner.gradient_booster.weight_drop`, a flat
array of `num_trees` floats, one multiplier per tree, in tree order.

`normalize_type` changes the values but not the representation:

```
--- booster=dart, normalize_type=forest
    weight_drop len=8  values=[0.591716, 0.591716, 0.591716, 0.7692308, 0.7692308, 0.7692308, 0.7692308, 0.7692308]
--- booster=dart, normalize_type=tree (default)
    weight_drop len=8  values=[0.7905139, 0.90909094, 0.6993007, 0.7905139, 0.63572794, 0.3030303, 0.43478262, 0.3030303]
```

`one_drop=1` likewise changes only the values:

```
--- booster=dart, one_drop=1 rate_drop=0.3
    weight_drop len=8  values=[0.4786378, 0.7155635, 0.34417105, 0.40444893, 0.7692308, 0.7155635, 0.20222446, 0.43478262]
```

### What a predictor needs

Measured by reconstructing the margin from the artifact with a float32-cast tree walk
(both sides of every comparison cast), then comparing against
`predict(output_margin=True)`:

```
########## what does weight_drop multiply? reconstruct the margin ##########
  base_score (probability space) = 0.5125
  logit(base_score)              = 0.0500104205746612
  weight_drop                    = [0.7905139, 0.90909094, 0.6993007, 0.7905139, 0.63572794, 0.3030303, 0.43478262, 0.3030303]
  weight_drop applied          max|recon - xgb_margin| = 1.569380248e-07
  weights ignored (all 1.0)    max|recon - xgb_margin| = 1.269166159e+00

  xgboost margins : [ 0.9354988 -1.6699885 -1.3686793 -1.9872636  1.7032771]
```

The rule, confirmed to `1.57e-07` (inside the `<= 1e-6` gate):

```
margin(x) = logit(base_score) + sum_i  weight_drop[i] * leaf_i(x)
```

A predictor therefore needs exactly one thing beyond the plain gbtree walk: the per-tree
multiplier, applied to each tree's leaf value before summation. No dropout simulation, no
RNG, no `normalize_type` handling — the normalization is already baked into the numbers.

**Ignoring `weight_drop` costs 1.27 in margin space.** Not an error, not a warning — a
plausible wrong probability. This is precisely the silent-failure signature this project
exists to prevent.

---

## 5. dart predict-time determinism

20 repeated `predict()` calls on the same fitted dart model and the same 32 input rows,
comparing bit-patterns:

```
########## dart predict() repeatability, 20 calls ##########
  predict(dm, output_margin=True)
     distinct bit-patterns across 20 calls = 1
     max |call_i - call_0|                 = 0
     every call bit-identical to call_0    = True
  predict(dm)  [probability]
     distinct bit-patterns across 20 calls = 1
     max |call_i - call_0|                 = 0
     every call bit-identical to call_0    = True
  inplace_predict(rows)
     distinct bit-patterns across 20 calls = 1
     max |call_i - call_0|                 = 0
     every call bit-identical to call_0    = True
  predict(dm, iteration_range=(0,4), output_margin=True)
     distinct bit-patterns across 20 calls = 1
     max |call_i - call_0|                 = 0
     every call bit-identical to call_0    = True
  predict(dm, training=True, output_margin=True)
     distinct bit-patterns across 20 calls = 15
     max |call_i - call_0|                 = 0.94517579674720764
     every call bit-identical to call_0    = False
  inplace_predict(rows, iteration_range=(0,4))
     distinct bit-patterns across 20 calls = 1
     max |call_i - call_0|                 = 0
     every call bit-identical to call_0    = True
```

```
  --- does training=True change the VALUE vs training=False? ---
     max|training=False - training=True| = 0.7025803327560425
     bit-identical                       = False
```

**With the default `training=False`, dart prediction is bit-exactly deterministic** —
1 distinct bit-pattern across 20 calls, max difference exactly `0`, for `predict`,
`inplace_predict`, margin output, probability output, and truncated `iteration_range`.

**With `training=True`, dart prediction is non-deterministic** — 15 distinct bit-patterns
across 20 calls, max spread `0.945` in margin space. That is live predict-time dropout,
and it also shifts the value away from `training=False` by `0.703`.

`iteration_range` truncates without rescaling: the retained trees keep their full-model
`weight_drop` values.

```
  --- iteration_range truncation: does it rescale weight_drop? ---
     weight_drop              = [0.7905139, 0.90909094, 0.6993007, 0.7905139, 0.63572794, 0.3030303, 0.43478262, 0.3030303]
     margin[0] full (8 trees) = 0.9354987740516663
     margin[0] first 4 trees  = 0.7438511252403259
```

`0.7438511252403259` is consistent with summing only the first four
`weight_drop[i] * leaf_i` terms on top of `logit(base_score)` — no renormalization over
the retained subset. **INFERRED** from the truncation arithmetic; not separately
reconstructed term-by-term.

> **DECISION NEEDED (scope): dart is deterministically reproducible at inference, but
> only on the `training=False` path.**
>
> **Context:** `training=False` is the default and is bit-exact across 20 calls.
> `training=True` applies live dropout and is non-reproducible by construction — 15
> distinct results in 20 calls, `0.945` spread. A ported predictor can only ever
> reproduce the `training=False` path; it has no access to XGBoost's RNG state.
>
> **Options:**
> A) Define the artifact's contract as the `training=False` path only, state it in
>    `COMPAT.md`, and hold parity to exactly `0.0` against
>    `predict(training=False)`. `training=True` is out of contract.
> B) Exclude dart from 1.0 entirely (see the Section 3 decision), which makes this moot.
>
> **Lean:** no lean — this rides on the Section 3 scope decision. Surfacing, not
> resolving. Note that if dart *is* supported, the reference predictor must never be
> validated against `predict(training=True)` or parity will be nonzero for reasons that
> have nothing to do with float32 or `base_score`.

---

## 6. gblinear determinism per updater

12 trials per configuration. Each trial refits from scratch on identical data with
identical `seed=20260801`, then compares (a) the serialized `weights` array bit-exactly
and (b) the `float32` margin output bit-exactly. Both thread counts tested, because
`nthread` turned out to be the deciding variable.

```
########## gblinear determinism per updater ##########
--- updater=shotgun  nthread=4  trials=12  seed=20260801
    distinct weight vectors (bit-exact) = 12 / 12
    distinct margin vectors (bit-exact) = 12 / 12
    max |weights_i - weights_0|         = 0.010527699999999918
    max |margin_i  - margin_0|          = 0.027578353881835938
    REPRODUCIBLE = False
    example of two differing weight vectors:
      A: [2.0945485, -1.1851498, -0.060741328, -0.11755441, -0.049389508, 0.09993565, -0.061605595]
      B: [2.0999956, -1.196069, -0.060173344, -0.116908535, -0.0558972, 0.098485895, -0.061387926]

--- updater=shotgun  nthread=1  trials=12  seed=20260801
    distinct weight vectors (bit-exact) = 1 / 12
    distinct margin vectors (bit-exact) = 1 / 12
    max |weights_i - weights_0|         = 0
    max |margin_i  - margin_0|          = 0
    REPRODUCIBLE = True
    weights_0 = [2.0998752, -1.1958269, -0.06036332, -0.11630667, -0.05826873, 0.10046119, -0.06128593]

--- updater=coord_descent  nthread=4  trials=12  seed=20260801
    distinct weight vectors (bit-exact) = 1 / 12
    distinct margin vectors (bit-exact) = 1 / 12
    max |weights_i - weights_0|         = 0
    max |margin_i  - margin_0|          = 0
    REPRODUCIBLE = True
    weights_0 = [2.0998752, -1.1958269, -0.06036332, -0.11630667, -0.05826873, 0.10046119, -0.06128593]

--- updater=coord_descent  nthread=1  trials=12  seed=20260801
    distinct weight vectors (bit-exact) = 1 / 12
    distinct margin vectors (bit-exact) = 1 / 12
    max |weights_i - weights_0|         = 0
    max |margin_i  - margin_0|          = 0
    REPRODUCIBLE = True
```

| Updater | `nthread` | Trials | Distinct weight vectors | Distinct margins | Reproducible |
|---|---|---|---|---|---|
| `shotgun` | 4 | 12 | **12 / 12** | **12 / 12** | **No** |
| `shotgun` | 1 | 12 | 1 / 12 | 1 / 12 | Yes |
| `coord_descent` | 4 | 12 | 1 / 12 | 1 / 12 | Yes |
| `coord_descent` | 1 | 12 | 1 / 12 | 1 / 12 | Yes |

**H3 REPRODUCED**, with a qualification that matters for how the invariant is worded.
`shotgun` at `nthread=4` produced **12 distinct weight vectors in 12 trials** at a fixed
seed — maximally non-deterministic, no ambiguity, no need to hunt for a rare
disagreement. Drift is `1.05e-02` in weight space and `2.76e-02` in margin space: small
enough to look like rounding, large enough to be a wrong prediction.

The qualification: `shotgun` at `nthread=1` was reproducible in all 12 trials. The
non-determinism is **thread-parallel** (racing coordinate updates), not seed-related.
This does **not** license relaxing the invariant. Two reasons: it is 12 trials on one
machine and one platform, and it makes reproducibility depend on a thread count that
callers control and libraries change defaults for. **Pin `coord_descent`; do not rely on
`nthread=1` as a substitute.** `coord_descent` was bit-exact at both thread counts, which
is the stronger property.

Incidentally, `shotgun`@`nthread=1` and `coord_descent` converged to the *same* weight
vector to the last bit on this data. Observed, not general — do not build on it.

---

## 7. gblinear serialized structure

```
########## gblinear serialized structure ##########
  gradient_booster.name             = "gblinear"
  gradient_booster.model keys       = ['boosted_rounds', 'weights']
  boosted_rounds                    = 8
  num_feature                       = 6
  len(weights)                      = 7
  weights                           = [2.0998752, -1.1958269, -0.06036332, -0.11630667, -0.05826873, 0.10046119, -0.06128593]
  base_score                        = [5.125E-1]
```

Full pasted excerpt:

```json
===== gblinear : $.learner.gradient_booster =====
{
  "model": {
    "boosted_rounds": 8,
    "weights": [
      2.0998752, -1.1958269, -0.06036332, -0.11630667,
      -0.05826873, 0.10046119, -0.06128593
    ]
  },
  "name": "gblinear"
}
```

Relative to gbtree, gblinear has an entirely different `model` object — no trees, no
`cats`, no `iteration_indptr`, no `tree_info`, no `gbtree_model_param`:

```
########## PATHS PRESENT IN gblinear BUT NOT gbtree ##########
  $.learner.gradient_booster.model.boosted_rounds  =  8
  $.learner.gradient_booster.model.weights  =  array[7]

########## PATHS IN BOTH, DIFFER (gblinear vs gbtree) ##########
  $.learner.gradient_booster.model
    gbtree  : object{cats,gbtree_model_param,iteration_indptr,tree_info,trees}
    gblinear: object{boosted_rounds,weights}
  $.learner.gradient_booster.name
    gbtree  : "gbtree"
    gblinear: "gblinear"
```

### Where the bias lives

`len(weights) == num_feature + 1`, verified across three widths:

```
########## gblinear weights length vs num_feature, several widths ##########
  num_feature=  2  len(weights)=3  (= num_feature + 1? True)
  num_feature=  6  len(weights)=7  (= num_feature + 1? True)
  num_feature= 11  len(weights)=12  (= num_feature + 1? True)
```

`weights[:num_feature]` are the per-feature coefficients; `weights[num_feature]` — the
**last** element — is the bias.

### How the bias relates to `base_score`

They are **separate and both required**. The bias does *not* absorb `base_score`:

```
  bias  = weights[-1]  = -0.06128593161702156
  logit(base_score)    = 0.0500104205746612

  dot(w,x) + bias  (base_score NOT added)    max err = 5.001067005e-02
  dot(w,x) + bias + logit(base_score)        max err = 4.498703290e-07

  xgb margins = [ 0.88480353 -2.48027182 -1.26669443 -4.01054239  0.78860223 -2.76327944]
```

The residual from omitting `base_score` is `5.001067e-02`, which is `logit(base_score)`
to five figures. Confirmed rule:

```
margin(x) = logit(base_score) + weights[num_feature] + sum_i weights[i] * x_i
```

`base_score` is in **probability** space here, consistent with the `binary:logistic`
finding in `CLAUDE.md` — `logit()` is required, and this was verified against a real
fitted artifact rather than assumed by analogy.

Accumulating in float32 is *tighter* than float64, supporting the float32 discipline of
D004:

```
########## gblinear reconstruction with float32 discipline ##########
  float32 accumulation, bias + logit(base_score) + sum(w_i*x_i)
  max err = 2.384185791e-07
  float64 accumulation, same formula
  max err = 5.945697747e-07
```

`2.38e-07` versus `5.95e-07` — both inside the `<= 1e-6` gate, but the float32 path is
2.5x closer, which is what one expects if XGBoost accumulates the linear term in float32.
Both are single-digit `1e-7`, matching the expected band.

`boosted_rounds` is `8` and appears to carry no predictive weight — the weights are the
converged model. **INFERRED**; not separately tested by mutating it.

---

## 8. Out-of-scope observations

1. **Both `dart` and `gblinear` are deprecated in XGBoost 3.3.0.** Pasted verbatim:

   ```
   WARNING: src/learner.cc:341: `booster=dart` is deprecated. Use the tree booster
   directly with dropout parameters like `rate_drop`, `skip_drop`, or `one_drop`.

   WARNING: src/learner.cc:824: `booster=gblinear` is deprecated and support will be
   removed in a future release.
   ```

   Neither is in the stated 1.0 scope, but both appear in this probe's brief. If either
   is supported, the version-support boundary (D001) becomes load-bearing: `gblinear`
   support is announced for removal, so an artifact produced today may not be
   re-verifiable against a future XGBoost. Worth an explicit `COMPAT.md` line.

2. **`booster=gblinear` silently ignores tree parameters.** Passing `max_depth` produced
   only a warning, not an error:

   ```
   WARNING: src/learner.cc:793:
   Parameters: { "max_depth" } are not used.
   ```

   Not this library's bug, but it is the same silent-acceptance pattern the project
   exists to guard against, and it argues for the exporter validating that the parameters
   it received are consistent with the booster it found.

3. **`base_score` is `[5.125E-1]` — a JSON *string* containing a JSON array**, not a
   number:

   ```
   "base_score": "[5.125E-1]",
   ```

   Identical for all three boosters. It needs two parses, and the `E-1` exponent form
   plus the array wrapper are both easy to get wrong. Per D004 this is inside the
   numerical core. Flagging it because it is a parsing trap on the `base_score` path and
   properly belongs to the `base_score` probe, not this one.

4. **`learner.attributes` is `{}` for all three.** No booster identity hides there.

5. **`split_type` and the `cats*` arrays exist on every gbtree/dart tree** and were empty
   (`array[0]`) throughout, because this probe used only numeric features. Categorical
   support is a separate scope question; noting that the fields are unconditionally
   present so a strict "no unknown fields" reader (D007) must expect them.

---

## Summary of measured constants

| Quantity | Value |
|---|---|
| `gradient_booster.name` — gbtree / dart / gblinear | `"gbtree"` / `"gbtree"` / `"gblinear"` |
| dart-only fields in the artifact | `weight_drop` — one field, exhaustively confirmed |
| dart margin error if `weight_drop` ignored | `1.269166159e+00` |
| dart margin error if `weight_drop` applied | `1.621531536e-07` |
| dart `predict()` determinism, `training=False`, 20 calls | 1 distinct result, max diff `0` |
| dart `predict()` determinism, `training=True`, 20 calls | 15 distinct results, max diff `0.945` |
| `shotgun` @ `nthread=4`, 12 trials | 12 distinct weight vectors — non-deterministic |
| `shotgun` @ `nthread=1`, 12 trials | 1 distinct weight vector |
| `coord_descent` @ `nthread=1` and `4`, 12 trials each | 1 distinct weight vector — deterministic |
| gblinear `len(weights)` | `num_feature + 1`, bias last |
| gblinear margin error, float32 accumulation | `2.384185791e-07` |
