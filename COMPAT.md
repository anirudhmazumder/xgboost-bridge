# Compatibility and support policy

This document describes what `xgboost-bridge` (PyPI) and `xgboost-predictor`
(npm) commit to supporting, and what a caller pays for that in return. It is
derived from the decisions recorded in `docs/DECISIONS.md`; where this file and
`docs/DECISIONS.md` disagree, `docs/DECISIONS.md` is the source of truth and this file
has a bug.

The project is pre-1.0. Everything below is subject to change before a 1.0
release, and anything not explicitly settled here is called out as such
rather than guessed at.

## Strict feature keys

Prediction input must match the model's feature names exactly: no missing
keys, no extra keys. Any mismatch raises an error rather than being
tolerated. (D005)

**What this costs the caller:** if your input records come from a source
that doesn't guarantee an exact key match — a database row with extra
columns, a form with optional fields, a renamed column upstream — you must
normalize that shape yourself before calling the predictor. This library
will not silently drop extra keys, and it will not silently treat a missing
key as "missing value" on your behalf. That normalization work is real, and
it is deliberately pushed onto the caller.

**Why it's still the right call:** a misspelled or renamed feature name
under lenient handling doesn't fail — it quietly becomes a "missing value"
input, and XGBoost's missing-value branches are legitimate model structure.
The result is a confident, plausible, wrong prediction, and the error
compounds across every tree in the ensemble. Strict keys turn that into an
error at the call site instead, which is the whole reason this library
exists in the first place.

## Supported model surface (1.0 scope)

1.0 targets binary classification, regression, and Cox survival objectives:
`reg:squarederror`, `binary:logistic`, `survival:cox`. Multi-class objectives
(`multi:softmax`, `multi:softprob`) are explicitly out of scope for 1.0: they
raise on export, and the artifact format reserves no fields or shape for
them. Adding multi-class support later is expected to require a new artifact
format version rather than filling in a pre-reserved slot. (D003)

**Booster: `gbtree` only.** This is the only booster supported for 1.0.
(D016)

- `dart` is refused. The exporter raises if `weight_drop` is present at
  **either** known on-disk location — `gradient_booster.weight_drop` or
  `gradient_booster.model.weight_drop`. Both paths are checked because the
  field's location itself changed between XGBoost versions (see the version
  boundary section below); checking only one path lets a dart model through
  unrefused. A model trained as `dart` with `rate_drop=0`/`skip_drop=0` is
  byte-identical to `gbtree` and exports fine — that is a feature of this
  decision, not a gap: that model produces correct predictions precisely
  because it is indistinguishable from a plain tree ensemble, and every
  caller with actual dropout activity gets a loud error instead of a quiet
  wrong number.
- `gblinear` is refused outright. It is deprecated in XGBoost 3.3.0 with
  removal explicitly announced, and it is a wholly separate inference path
  with no trees — supporting it would mean a second predictor
  implementation, in two languages, for a booster on its way out.

**Categorical splits are refused.** The exporter raises if any of the
following is true: `split_type` contains a categorical marker, the
`categories_nodes` array is non-empty, or `learner.feature_types` contains
`'c'`. All three are checked even though one signal would do, because this
is a refusal test and redundancy there is free. (FORMAT.md §11)

**Output arity is checked, not just the objective name.** Export requires
**all** of: the objective is in the supported set above, `num_target ==
"1"`, `num_class ∈ {"0", "1"}`, and `size_leaf_vector == "1"` for **every
tree** — a model with zero trees passes this last check vacuously. All four
gate fields, `objective.name` included, are JSON strings and are compared
as strings; an integer comparison silently never fires (`num_class == 0` is
`False` against the string `"0"`). Anything else raises. (D017, amended by
D037)

**Why `num_class` admits `"1"`:** verified across all three in-scope
objectives that `num_class=1` produces the *same* single-output model as
`num_class="0"` — trees byte-identical, margins bit-identical 400/400,
`predict()` shape `(400,)` — and that requiring `"0"` rejected all three. A
false rejection reads as correct strictness and is not, which is worse.
Relaxing to `{"0", "1"}` admitted nothing extra across a 23-shape table with
zero false acceptances. (D037, `probes/arity_gate.md`)

**Why `size_leaf_vector` is checked per tree:** the field exists only inside
each tree's `tree_param`, never at the model level — a zero-tree model has
zero occurrences of it anywhere in the document. Without the vacuous-pass
rule for zero trees, the gate would reject every zero-round model that this
format is otherwise required to export. (D037)

*Why arity is checked separately from the objective name:* an
objective-name allow-list alone has a hole. `reg:squarederror` with
`num_target=2` is still an in-scope objective by name, but it produces a
two-element `base_score`, an interleaved `tree_info`, and `(N, 2)` margins —
multi-output predictions arriving through a permitted door. A scalar
predictor accepts that shape and returns a confident, wrong number for every
row. The arity checks close that hole; an unsupported objective or booster
raising loudly is a documented feature of this library, not an omission.

Any objective or booster type this library does not recognize raises loudly
at export or at load time — it does not default to a nearby-looking
objective and does not degrade to a best-effort prediction. The same applies
to unrecognized artifact fields and out-of-range format version markers.
(D007) If you are relying on an objective or booster not named above, treat
it as unsupported until it is added and verified against a real fitted
model, not as "probably fine by analogy."

## `objective` is not operative

The artifact's `objective` field is retained for inspection, and it is
cross-checked against `output_transform` at export time — that is its only
role. **No predictor branches on it, in either language.** A reader should
not assume this field drives prediction behavior; the field that actually
determines the output transform is `output_transform` itself, and the
transform applied to the running score's intercept has no runtime
representation at all. Both language test suites assert that nothing
branches on `objective`, so this cannot silently regress. (D028)

## `base_score` is clamped before the logistic intercept transform

For `binary:logistic`, XGBoost clamps `base_score` to `[f32(1e-6), f32(1 -
1e-6)]` before deriving the margin intercept, but **stores the value
unclamped**. This library's exporter reproduces the clamp; applying the
intercept recipe to the stored value without it is wrong by up to `13.8` in
margin space. (D035)

**What this means for a caller:** if you pass an extreme `base_score` —
very close to `0` or to `1` — and hand-derive `logit(base_score)` yourself
to compare against this library's exported `intercept`, expect a large
difference. This library's value is the one that matches XGBoost's own
observed zero-tree margin for the same configuration; the unclamped `logit`
of the stored value does not.

## XGBoost version support boundary

The empirical behaviors this library depends on — float32 threshold
representation, per-objective `base_score` space, DART detection, gblinear
determinism, and the output-transform clamp constants (D032) — were
established against **XGBoost 3.3.0**, and that is the version verification
runs against by default. (D001)

**The support policy is an enumerated list of versions actually probed, not
a range.** (D018) The exporter raises unless the artifact's version marker
matches a version on that list. Today the list is:

- **XGBoost 3.3.0** — the reference version. Verified.

A second, drift-detection pass has run against the newest XGBoost build
obtainable, and its result is recorded here rather than left pending:

- The newest **released** XGBoost as of the pass is **3.3.0 itself** — the
  reference pin. There is no newer release to probe.
- The newest **available build** of any kind was a `master` nightly:
  `xgboost.__version__` reports `3.4.0-dev`, built from upstream commit
  `e787a447de12c15bdf06f65ddbf79b056743113d`. (`probes/version_drift.md`)
- That build is **not** on the supported list. It is not merely untested —
  it was probed specifically and found to change behavior in a way that
  silently breaks predictions across the version boundary: it relocated
  `weight_drop` from `gradient_booster.weight_drop` to
  `gradient_booster.model.weight_drop`. XGBoost 3.3.0 loads an artifact
  written by that build and returns predictions with max error `1.26`,
  **zero rows correct**, zero warnings, exit code `0` — and then silently
  drops the field on re-save. Everything else measured (float32 threshold
  grammar, the comparison operator, `base_score` storage and transforms,
  tree structure, gamma pruning) was byte-identical or numerically
  equivalent between 3.3.0 and this build.
- The nightly build's own artifact version marker is `[3, 4, 0]` — the
  plain release triple, with no `dev` discriminator. A reader cannot tell
  from the marker alone whether a `[3, 4, 0]`-marked artifact came from this
  nightly or from an eventual real 3.4.0 release.

**Why an enumerated list rather than a range, structurally:** unrecognized-
field detection (D007) catches an artifact that gains a new field, but it
cannot catch a field that moves or disappears, because a missing optional
field is indistinguishable from a field that was never there. The
`weight_drop` relocation above is exactly that case — nothing in the 3.4.0-
dev artifact is an *unrecognized* field, so field-level strictness alone
does not defend against it. An untested version is therefore treated as an
unrecognized input, refused by the version marker check, independent of
whether unrecognized-field detection would also have caught it.

The Python `export` extra's dependency spec currently allows
`xgboost>=3.3,<4` (see `packages/python/pyproject.toml`); that is what
pip/uv will resolve, not a compatibility claim. The compatibility claim is
the enumerated list above.

## Two Python floors

There are two different `requires-python` floors in this repository, and
they exist for different reasons: (D013)

- The **published `xgboost-bridge` package** declares `requires-python =
  ">=3.10"`. Its mandatory runtime dependency is `numpy` only, which
  supports 3.10, so reading an exported artifact and running the reference
  predictor work on 3.10+.
- The **workspace root and the fixture package** declare `requires-python =
  ">=3.12"`, and the `export` extra depends on `xgboost>=3.3,<4`, which
  itself requires Python `>=3.12`.

**The resolver failure this causes if you miss it:** installing the base
package on Python 3.10 or 3.11 works fine. Installing
`xgboost-bridge[export]` on Python 3.10 or 3.11 does not — pip/uv will fail
to resolve the environment, because the `xgboost` dependency pulled in by
the `export` extra requires `>=3.12` even though `xgboost-bridge` itself
claims to support `>=3.10`. If you hit an unresolvable-dependency error
installing the `export` extra, check your Python version before assuming
the package metadata is wrong — it isn't; exporting genuinely requires
3.12+, only reading and predicting do not.

## Node.js version support

`packages/js/package.json` declares `engines.node: ">=20"`. That is an
advisory to npm and to tooling that reads `engines`; **`xgboost-predictor`
does not itself check the Node version at runtime and cannot refuse an
unsupported one the way the Python exporter refuses an unsupported XGBoost
version** — there is no version-marker equivalent to gate on inside a JS
runtime.

What is actually verified, as opposed to merely declared: the Node test
suite (92 tests), the full fixture-corpus comparison against XGBoost, and
the cross-language parity harness have all been run, and pass with the
same result, on **Node 20.19.0, 24.7.0, and 24.18.0**. That is the set of
versions this compatibility policy can make a claim about. A Node runtime
outside that set is not known to work and is not known to fail — it is
simply untested, in the same sense an unprobed XGBoost version is untested
(D018's reasoning, applied to the other side of the boundary, informally:
there is no code-level refusal here, only an absence of evidence).

## Upstream hazards this library documents rather than papers over

Two behaviors originate in XGBoost itself, not in this library, and are
recorded here because a caller comparing this library's output against
XGBoost's needs to know about them.

**`±inf` at predict time behaves differently depending on how XGBoost is
called.** A non-finite feature value **raises** when passed through
XGBoost's `DMatrix` construction path with the default `missing=` setting,
but is treated as an **ordinary comparable value** — not a missing value —
when passed through `inplace_predict`. The same input can therefore produce
two different predictions from the same fitted XGBoost model depending only
on which call path was used to get a prediction out of it.
(`probes/tree_structure.md` §6.1) This library does not try to reproduce
both behaviors. It picks one — a non-finite feature value **raises** — and
pins that choice with a fixture. (D022) Surfacing exactly this class of
call-path-dependent divergence is why this library exists; it is recorded
here as an upstream hazard rather than a defect in this library.

**This library's probability/hazard output is not guaranteed to match
XGBoost's bit-for-bit, by design.** Margins are float32 throughout this
library's prediction path, and the final output-space transform — margin to
probability for `binary:logistic`, margin to hazard ratio for
`survival:cox` — is evaluated under **float32 semantics** as well:
`np.float32(...)` wraps every intermediate in Python, `Math.fround(...)`
wraps every intermediate in JavaScript. This transform also reproduces
XGBoost's own clamps at the output — see the next section. (D032, FORMAT.md
§5.1, §5.2)

**Why not float64, which an earlier version of this document specified.**
XGBoost transforms in float32 **with clamps**. A float64 transform is not
off by a ULP in the tail there — it is qualitatively wrong: **relative
error `1.0`** below the logistic clamp floor, and finite-versus-`inf` for
Cox. Measured: `400/400` bit-exact for float32-throughout against `236/400`
for float64-then-narrow, and on all `164` rows where the two hypotheses
disagree, XGBoost matches float32 `164/164` and float64 `0/164`. Divergence
from upstream that is itself silent is exactly what this library exists to
surface, so upstream is matched in the tail rather than diverged from
quietly. (D032, superseding D026's precision requirement)

The margin computation itself is unaffected by any of this and stays in the
low `1e-7` range against XGBoost.

## The output transform is bundled, not the platform's

The margin-to-probability transform for `binary:logistic` and the
margin-to-hazard-ratio transform for `survival:cox` are implemented from
scratch in both packages, evaluated under float32 semantics (previous
section). Neither package calls a platform transcendental on the prediction
path — no `Math.exp`, no `math.exp`, no `np.exp`. (D030, amended by D032 for
evaluation precision)

**Why this exists, briefly, because it changes what a caller will observe:**
even with bit-identical float32 inputs on both sides, calling a platform
`exp` would not give a bit-identical *result*. IEEE-754 mandates correct
rounding only for `+ − × ÷ √` and fused multiply-add; `exp` is not required
to be correctly rounded, and no two `libm` implementations agree with each
other. Measured on one platform pair, V8 against Apple `libm`: **4.2%** of
sigmoid evaluations and **9.6%** of `exp` evaluations differed, by up to
**2 ULP**. Cross-language parity in this library is required to be exactly
`0.0`, with no tolerance, at both the margin and the final output — and no
honest tolerance number exists to paper over this, because the 2-ULP figure
was measured on exactly one platform pair. glibc is a third `libm`
implementation, and recent glibc is correctly rounded where V8's is not, so
a Linux runner would show a different divergence than the pair actually
measured, and neither measurement bounds the other. The resolution is to
stop calling any platform `exp` at all: both packages implement `sigmoid`
and `exp` from `+ − × ÷` and exact power-of-two scaling, narrowing every
intermediate to float32, so Python and JavaScript execute an identical
sequence of operations under float32 semantics and agree bit-for-bit by
construction. The full argument, including why a tolerance was rejected on
evidence rather than principle, is recorded in D026's 2026-08-02
correction, in D030, and in D032's precision correction; it is not repeated
here.

**What this means for a caller:**

- **This is a bundled implementation, not the platform's.** If you compute
  `scipy.special.expit(margin)` or `1/(1+Math.exp(-margin))` yourself and
  compare it against this library's output, expect differences in the last
  bit or two of the result **inside the clamped range**. That is expected
  and intentional — this is where to find out why, rather than filing it as
  a bug.
- **Outside the clamped range, this library reproduces XGBoost's clamps and
  a naive float64 computation will not.** `binary:logistic` floors at
  margin `f32(-88.7)`, returning exactly `3.006635794144578e-39` and never
  `0.0`; a float64 sigmoid on the same margin has **relative error `1.0`**
  against that floor. `survival:cox` has no clamp and returns `+inf` above
  margin ≈ `88.72`, where a float64 `exp` returns a large but finite
  number. (D032)
- **The clamp constants are XGBoost internals, not part of any published
  specification, and are version-sensitive** in the same way `weight_drop`
  is (see the version boundary section below). They are re-verified
  whenever the tested-version list widens. (D032)
- **This library's probability or hazard-ratio output also differs from
  XGBoost's own output** by roughly 1–2 ULP inside the clamped range, by
  construction, because XGBoost performs its output-space transform in C++
  `libm` and this library does not call `libm` at all. This is expected and
  is not a regression; see the accuracy-gate section below for how it is
  measured.
- **The margin-level comparison against XGBoost is unaffected** and stays in
  the low `1e-7` range described above. Only the post-transform output —
  probability or hazard ratio — is involved.
- **The JavaScript package's zero-runtime-dependency guarantee is
  untouched.** The bundled transform is written from plain arithmetic
  (`+ − × ÷` and exact power-of-two scaling), not a library, so it adds no
  entry to `dependencies`. (D009, D030)

## The Python-vs-XGBoost accuracy gate is relative; cross-language parity is exact and separate

Two different comparisons exist here, and this document keeps them apart —
conflating them is how a tolerance leaks into the parity gate. (D033)

- **Python vs XGBoost, margin:** **absolute**, ≤ `1e-6`. Measured `0.0`
  bit-exact at every sweep configuration probed; treat any regression from
  that as a defect to diagnose, not headroom to spend.
- **Python vs XGBoost, output** (probability or hazard ratio): **relative**,
  ≤ `1e-6`, computed against XGBoost's value. Explicit rules for the rows
  that otherwise fall out of a naive ratio comparison: `±inf` must match as
  bit patterns and is never divided; `NaN` on either side is always a
  failure, never silently skipped, because `NaN` compares unequal to
  everything including itself; where XGBoost's value is exactly `0.0` or
  `-0.0`, the comparison is bit-pattern equality rather than a ratio. The
  harness reports **max** relative error and the row that produced it,
  never a mean.
- **Cross-language parity** (Python vs JavaScript, checked at both the
  margin and the output) is a **separate gate with no tolerance at all** —
  exactly `0.0`, bit-identical, at both measurement points. It is not the
  same check as the accuracy gate above and carries none of its tolerance.

**Why the output gate is relative rather than absolute.** `survival:cox`'s
output is a hazard ratio spanning `2.85e-04` to `7.56e+08` — an absolute
bound there is meaningless. Measured absolute error against XGBoost reaches
`6.96e+23` (with rows at `+inf`) while the max relative error is `5.7e-08`.
The mirror-image failure hits `binary:logistic`: it would pass an absolute
`1e-6` bound trivially in the clamp tail while being relatively `100%`
wrong there, the exact case a float64 transform produces. (D033)

## Measured accuracy of the two shipped predictors, on the full corpus

The gates above are bounds. This is what the two predictors actually
measured against them, on all 23 corpus fixtures (289 value rows plus 10
rows both sides are expected to refuse), as recorded in `docs/DECISIONS.md`
D047–D049:

- **Margin, bit-exact vs XGBoost:** **289/289** on both the Python and the
  JavaScript predictor.
- **Output, bit-exact vs XGBoost:** 283/289 on both predictors; the six
  divergences are the same `(fixture, row)` pairs on both sides, at max
  **relative** error `9.56e-08` (Python) and `9.555893664308718e-08`
  (JavaScript) — both far inside the `1e-6` gate, and both are `libm`
  rounding differences inside the bundled `exp`, expected by construction
  rather than a defect (see the previous section).
- **`survival:cox` rows that overflow to `+inf`:** 2/2 bit-exact.
- **`binary:logistic` rows at the clamp floor:** 21/21 bit-exact at bit
  pattern `0x0020bd47`, never `0.0`.
- **Rows both predictors are expected to refuse:** 10/10 raise, on both
  sides, with the same error kind.
- **Cross-language parity, Python vs JavaScript, at both measurement
  points:** exactly `0.0` — **0 margin-point mismatches, 0 output-point
  mismatches, 0 refusal disagreements**, across all 299 compared rows.

None of these numbers is a claim about a hypothetical future model — they
are what running both predictors against this corpus, today, actually
produced. A caller comparing this library's output against XGBoost's own
should expect exactly this shape of result: bit-exact margins, and output
differing from XGBoost by at most one `libm` rounding step, on a small,
specifically-identified set of rows, never as a systematic drift.

## What this document does not yet cover

The following are governed by decisions not yet made, evidence not yet
gathered, or not yet recorded, and are intentionally absent above rather
than guessed at:

- Release and publishing mechanics. `.github/workflows/release.yml` defines
  the intended PyPI (trusted publishing via OIDC) and npm (`--provenance`)
  jobs, but that workflow triggers only on a manual `workflow_dispatch` and
  gates its publish jobs on a deployment environment that does not exist
  yet in this repository's settings — it cannot fire by pushing or tagging,
  and it has never been executed. No version of either package has been
  published under either name as of this writing.
- AI-authorship disclosure text — deferred to the 1.0 announcement per D012,
  not part of this compatibility policy.
