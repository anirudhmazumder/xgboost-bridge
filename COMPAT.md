# Compatibility and support policy

This document describes what `xgboost-bridge` (PyPI) and `xgboost-predictor`
(npm) commit to supporting, and what a caller pays for that in return. It is
derived from the decisions recorded in `DECISIONS.md`; where this file and
`DECISIONS.md` disagree, `DECISIONS.md` is the source of truth and this file
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
"1"`, `size_leaf_vector == "1"`, and `num_class == "0"`. Anything else
raises. (D017)

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

## XGBoost version support boundary

The empirical behaviors this library depends on — float32 threshold
representation, per-objective `base_score` space, DART detection, gblinear
determinism — were established against **XGBoost 3.3.0**, and that is the
version verification runs against by default. (D001)

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
library's prediction path, but the final output-space transform — margin to
probability for `binary:logistic`, margin to hazard ratio for
`survival:cox` — is computed in **float64 on both sides**, in both Python
and JavaScript, and the result is not narrowed back to float32. Widening is
necessary: JavaScript has no float32 `exp`, and `Math.fround(Math.exp(x))`
is a float64 exponential rounded once, which is not the value a genuine
float32 exponential would produce. A float32 output transform is therefore
not reproducible in the JavaScript runtime at all, and specifying one would
make exact cross-language parity at the output stage unreachable by
construction. (D026, FORMAT.md §5.1)

Widening to float64 is **necessary but not sufficient** for exact
cross-language agreement at the output stage — see the next section for why,
and for what closes the remaining gap. The margin computation itself is
unaffected by any of this and stays in the low `1e-7` range against
XGBoost.

## The output transform is bundled, not the platform's

The margin-to-probability transform for `binary:logistic` and the
margin-to-hazard-ratio transform for `survival:cox` are implemented from
scratch in both packages. Neither package calls a platform transcendental on
the prediction path — no `Math.exp`, no `math.exp`, no `np.exp`. (D030)

**Why this exists, briefly, because it changes what a caller will observe:**
widening the transform to float64 on both sides (previous section) delivers
bit-identical *inputs* to `exp`, but that is not sufficient for a
bit-identical *result*. IEEE-754 mandates correct rounding only for
`+ − × ÷ √` and fused multiply-add; `exp` is not required to be correctly
rounded, and no two `libm` implementations agree with each other. Measured
on one platform pair, V8 against Apple `libm`: **4.2%** of sigmoid
evaluations and **9.6%** of `exp` evaluations differed, by up to **2 ULP**.
Cross-language parity in this library is required to be exactly `0.0`, with
no tolerance, at both the margin and the final output — and no honest
tolerance number exists to paper over this, because the 2-ULP figure was
measured on exactly one platform pair. glibc is a third `libm`
implementation, and recent glibc is correctly rounded where V8's is not, so
a Linux runner would show a different divergence than the pair actually
measured, and neither measurement bounds the other. The resolution is to
stop calling any platform `exp` at all: both packages implement `sigmoid`
and `exp` from correctly-rounded primitives, so Python and JavaScript
execute an identical sequence of IEEE-754 double operations and agree
bit-for-bit by construction. The full argument, including why a tolerance
was rejected on evidence rather than principle, is recorded in D026's
2026-08-02 correction and in D030; it is not repeated here.

**What this means for a caller:**

- **This is a bundled implementation, not the platform's.** If you compute
  `scipy.special.expit(margin)` or `1/(1+Math.exp(-margin))` yourself and
  compare it against this library's output, expect differences in the last
  bit or two of the result. That is expected and intentional — this is
  where to find out why, rather than filing it as a bug.
- **This library's probability or hazard-ratio output also differs from
  XGBoost's own output** by roughly 1–2 ULP, by construction, because
  XGBoost performs its output-space transform in C++ `libm` and this
  library does not call `libm` at all. This is expected, is bounded well
  inside the `1e-6` gate, and is not a regression.
- **The margin-level comparison against XGBoost is unaffected** and stays in
  the low `1e-7` range described above. Only the post-transform output —
  probability or hazard ratio — is involved.
- **The JavaScript package's zero-runtime-dependency guarantee is
  untouched.** The bundled transform is written from plain arithmetic
  (`+ − × ÷` and exact power-of-two scaling), not a library, so it adds no
  entry to `dependencies`. (D009, D030)

## What this document does not yet cover

The following are governed by decisions not yet made, evidence not yet
gathered, or not yet recorded, and are intentionally absent above rather
than guessed at:

- Release and publishing mechanics (PyPI trusted publishing, npm
  provenance) — no publish workflow exists in this repository yet.
- Two evidence gaps are currently blocking and under active probe
  (`FORMAT.md` §14): the exact objective-to-output-transform pairing and
  XGBoost's internal output-transform precision for `binary:logistic` and
  `survival:cox`, and whether `num_class` can legitimately be `"1"` (as
  opposed to `"0"`) for a single-output binary model. Neither is resolved
  here; this document will be updated once each lands.
- AI-authorship disclosure text — deferred to the 1.0 announcement per D012,
  not part of this compatibility policy.
