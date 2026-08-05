# Decisions

Durable engineering decisions and their rationale. Append; do not rewrite history. If a decision is reversed, add a new entry that supersedes the old one and mark the old one superseded.

Format: what was decided, what forced it, what it costs.

---

## D001 — Pin XGBoost 3.3.0 as the reference version, and track drift separately

*2026-08-01*

The empirical behaviors this library depends on — float32 threshold representation, per-objective `base_score` space, DART's `gradient_booster.name` value, gblinear updater determinism — were established against XGBoost 3.3.0. Verification runs against `xgboost==3.3.0` exactly.

A separate verification pass runs against the newest XGBoost available, and **records the resolved version number explicitly** rather than the word "latest."

**Why:** With the version pinned, a behavior that fails to reproduce is a real signal about the library rather than an artifact of a version bump. Recording the resolved version number matters because "latest" is not reproducible after the fact, and detecting drift is the entire purpose of the second pass.

**Cost:** Two verification matrices instead of one.

---

## D002 — `.gitignore` carries standard Python/JavaScript ignores only

*2026-08-01*

Tracked `.gitignore` contains `__pycache__/`, `*.py[cod]`, `.venv/`, `.pytest_cache/`, `*.egg-info/`, `node_modules/`, `dist/`, `.claude/settings.local.json` — nothing else.

**Why:** Contributors need working ignore rules from a fresh clone. A `.gitignore` that ignores itself ships an empty ruleset, and build and cache artifacts flood `git status` on first sync.

---

## D003 — Multi-class is out of scope for 1.0, and no structural space is reserved for it

*2026-08-01*

`multi:softmax` and `multi:softprob` raise on export. The artifact format reserves **no** fields, no per-class grouping slot, and no placeholder shape for them.

Multi-class behavior is still investigated empirically, for one narrow purpose: confirming the chosen tree representation *could* be extended to per-class grouping without restructuring what already exists. That is an extensibility check, not a commitment.

**Why:** Reserving space designs a shape against a feature nobody has thought through. The usual outcome is that the reserved shape is wrong and the migration happens anyway, having carried unused fields in the interim. The version marker is the migration mechanism.

**Cost:** Adding multi-class later increments the format version.

---

## D004 — Threshold and `base_score` handling is one indivisible numerical core

*2026-08-01*

Every code path that **reads, stores, or transforms** a split threshold or a `base_score` value belongs to the numerical core and is reviewed as such — not only the comparison operators. This explicitly includes artifact parsing on both sides.

**Why:** A narrower rule scoped to "the comparison" leaves a hole. `JSON.parse` returns float64 unconditionally, and a Python parser that lands thresholds as unconstrained floats destroys float32 discipline before the tree walk ever executes. The walk then reads as correct while producing wrong predictions on a fraction of rows — the exact failure this project exists to prevent, reintroduced one layer upstream.

---

## D005 — Strict feature keys

*2026-08-01*

Prediction input must match the model's feature names exactly: no missing keys, no extra keys. Any mismatch raises.

**Why:** Lenient handling turns a misspelled feature name into a missing-value path. XGBoost's missing-value branches are legitimate model structure, so the mistake produces a confident, plausible, wrong prediction instead of an error — and the error compounds across the ensemble.

**Cost:** Callers with loosely-shaped input must normalize it themselves. Documented in `COMPAT.md`.

---

## D006 — No `fromFile` in the JavaScript package

*2026-08-01*

The JavaScript API accepts a parsed object via `fromJSON`. Consumers perform their own I/O.

**Why:** Filesystem access is not available in browsers and differs across edge runtimes. Shipping a file loader either pulls in a Node dependency or forces a runtime split in the bundle. Neither is acceptable against a zero-dependency, universal-bundle constraint.

---

## D007 — Fail loudly on anything unrecognized

*2026-08-01*

Unknown objective, unknown booster type, unknown artifact field, out-of-range version marker — all raise. Nothing defaults, nothing is skipped, nothing is inferred by analogy.

**Why:** This library exists because the standard conversion path fails silently. A crash is recoverable; a wrong number that looks right is not.

---

## D008 — Export is deterministic and tested as such

*2026-08-01*

The same model exported twice produces byte-identical output. Enforced by an explicit test, not by convention.

**Why:** Byte-identical output makes artifacts diffable, cacheable, and content-addressable, and it turns any accidental nondeterminism — dict ordering, float formatting, timestamp leakage — into a test failure rather than a mystery.

---

## D009 — Zero JavaScript runtime dependencies; near-zero dev dependencies

*2026-08-01*

`dependencies` is empty and stays empty. Dev dependencies are `tsup` and `typescript` only; tests run on Node's built-in `node:test` runner rather than a third-party framework.

**Why:** Runtime dependencies are non-negotiable — every one is a supply-chain and bundle-size liability in an edge deployment. Dev dependencies are a weaker constraint, but a built-in test runner costs nothing and removes an entire class of version-drift maintenance.

---

## D010 — Python runtime dependency is `numpy` only; XGBoost is an optional extra

*2026-08-01*

`xgboost-bridge` depends on `numpy`. XGBoost moves to an `export` extra.

**Why:** Reading a fitted model requires XGBoost. Inspecting an artifact and running the reference predictor do not. Making XGBoost mandatory would force a large native dependency on consumers who only ever read exported artifacts.

**Cost:** Two install paths to document. `pip install xgboost-bridge[export]` is required to export.

---

## D011 — JavaScript tests import from `dist/`, never from `src/`

*2026-08-01*

The test suite imports the built bundle. `tsc --noEmit` runs as a step separate from the build.

**Why:** Testing `src/` verifies code that is not what ships. Bundler configuration errors — wrong entry point, broken export map, dropped type declarations, an accidentally externalized module — are invisible to a source-level test suite and break every consumer. Keeping typecheck separate from bundling means a type error cannot be masked by a successful build.

---

## D012 — AI-authorship disclosure is deferred to the 1.0 announcement

*2026-08-01*

This project is AI-authored under human direction. The disclosure lands with the 1.0 announcement rather than in the README now.

**Why:** Recorded here so the commitment exists and is dated, independent of when the announcement copy is written.

---

## D013 — Two Python floors: `>=3.10` published, `>=3.12` for development

*2026-08-01*

`xgboost-bridge` declares `requires-python = ">=3.10"`. The workspace root and the fixture package declare `>=3.12`.

**Why:** XGBoost 3.3.0 requires Python `>=3.12`, so anything that fits a model or generates fixtures needs 3.12. The published package's own runtime path needs only `numpy`, which supports 3.10 — and consumers who read artifacts and predict, without ever exporting, should not be forced to upgrade.

**Cost:** `pip install xgboost-bridge[export]` requires 3.12+ even though the base package installs on 3.10. Must be stated in `COMPAT.md`; it is otherwise a confusing resolver failure.

---

## D014 — JavaScript tests are plain JavaScript

*2026-08-01*

Test files under `packages/js/test/` are `.js`, executed directly by `node --test`. There is no test transpile step.

**Why:** Node 20 cannot execute TypeScript. A transpile step emits a build artifact into the source tree and buys no type checking, because the test's job is to exercise the *runtime bundle* — the entry point, the export map, what a consumer actually receives. Type-checking test sources against `dist/*.d.ts` would additionally couple typecheck to build order, which D011 exists to prevent.

**Revisit:** Worth reconsidering once there is a real public API surface to type against. The benefit is empty while `dist/` contains only error classes.

---

## D015 — The artifact stores a derived margin intercept, not `base_score`

*2026-08-01* — **still stands; amended by D035** (clamp before transform) and its export assertion **SUPERSEDED by D034** (independent oracle).

The artifact carries exactly **one operative numeric intercept**: the margin-space intercept, already transformed, as a float32 value. Predictors add it and never transform it.

The original `base_score` is retained alongside it in a structurally separate provenance block that **no predictor reads, in either language**. Export asserts that the two agree; that assertion lives only in Python and costs nothing at runtime. The schema and `COMPAT.md` must state explicitly which field is operative.

**Why:** The per-objective link space is the single largest source of silent wrongness in this project's history. Under this decision it leaves the artifact entirely — `binary:logistic`'s float32 `1/p − 1` form, Cox's `ln`, and regression's identity all collapse into one float32 field, and the link-space transform becomes an export-time concern with no runtime representation at all.

The transform is not merely delicate, it is *specifically* delicate: the textbook `log(p/(1-p))` breaches the `1e-6` margin gate (measured `7.63e-06` at `base_score=0.987654` with 100 trees, and `1.91e-06` at `0.48`). Reproducing it requires the exact float32 expression, not generic float32 discipline. Implementing that once, in one language, is categorically safer than mirroring it in two.

**This does NOT collapse the output transform.** Margin → probability via sigmoid still exists in the JavaScript predictor for `binary:logistic`. The intercept transform and the output transform are separate concerns and the spec keeps them distinct.

**Cost:** The artifact no longer round-trips to a `base_score` a predictor could use. Accepted; provenance covers inspection.

---

## D016 — `gbtree` is the only supported booster for 1.0

*2026-08-01*

The exporter raises on anything that is not a plain tree ensemble. Specifically:

- Raise if `weight_drop` is present at **either** known JSON path (`gradient_booster.weight_drop` or `gradient_booster.model.weight_drop`).
- Raise on `gblinear`.

**Why, dart:** The two-signal detection rule cannot be satisfied from the artifact alone — exactly one in-artifact signal exists (`weight_drop`), confirmed by an exhaustive key census. The invariant that motivated dart support is the invariant that refuses it. Dart is also the only booster with a version ceiling, and it is the silent one (see D018).

A model trained as `dart` with `rate_drop=0`/`skip_drop=0` is byte-identical to `gbtree` and exports fine. That is a **feature of this decision, not a gap**: those users get correct predictions precisely because the model is indistinguishable from a tree ensemble, and everyone with actual dropout gets a loud error.

**Why, gblinear:** Deprecated in 3.3.0 with removal explicitly announced, and it is an entirely separate inference path with no trees — a second predictor implementation, in two languages, for a booster slated for removal.

**Why this is the cheap direction to be wrong in:** Adding a booster later is purely additive and does not migrate the format, unlike D015. One booster, one code path, for 1.0.

---

## D017 — The export gate checks output arity, not just the objective name

*2026-08-01* — **still stands, amended by D037**: `num_class` admits `"1"`, `size_leaf_vector` is per-tree, and all fields compare as strings.

Export requires **all** of: objective in the supported set, `num_target == "1"`, `size_leaf_vector == "1"`, and `num_class == "0"`. Anything else raises.

**Why:** An objective-name allow-list has a hole. `reg:squarederror` with `num_target=2` is an in-scope objective that produces `tree_info=[0,1,0,1,...]`, a two-element `base_score`, and `(N,2)` margins. A scalar predictor accepts it and returns confident wrong numbers — the multi-class failure signature arriving through a permitted door. The arity assertions close it with zero measured false positives.

---

## D018 — An explicit, enumerated version ceiling

*2026-08-01*

The exporter asserts the XGBoost artifact version marker is one that has actually been probed, and raises above it. `COMPAT.md` records **the list of versions actually tested**, not a guessed range. An untested version is an unrecognized input.

**Why:** XGBoost 3.4.0-dev relocated `weight_drop` from `gradient_booster.weight_drop` to `gradient_booster.model.weight_drop`. XGBoost 3.3.0 loads such an artifact and returns predictions with `max err 1.26`, **0 rows correct, zero warnings, exit code 0** — then silently drops the field on re-save. The project's exact failure signature, occurring inside XGBoost, between adjacent versions.

The structural point: D007's unrecognized-field rule catches *additions* and **cannot** catch relocations or removals, because a missing optional field is not an unknown field. Unknown-field detection is therefore not a substitute for a version bound. Only an explicit upper bound defends against this class.

---

## D019 — Redundant safeguards are untested safeguards

*2026-08-01*

Float32 narrowing happens at more than one site: when threshold and leaf values are read, and after every accumulator addition. Both are kept — together they mirror XGBoost's actual float32 accumulator.

But the redundancy has a cost that must be paid explicitly: **each narrowing site is verified in isolation.** The revert-and-confirm-red methodology is applied to one site at a time, never to the pair. If a site cannot be made to fail on its own, that is reported as such rather than covered by a test that proves nothing.

**Why:** Narrowing after every addition absorbs the effect of narrowing leaf values on read, so a suite that only ever reverts both at once pins neither. A later refactor could delete one site and no test would notice.

---

## D020 — `learner.attributes` is excluded except by explicit whitelist

*2026-08-01*

Nothing under `learner.attributes` reaches the artifact unless it is on an explicit whitelist.

**Why:** It is the only nondeterministic surface observed in the serialized model — early stopping writes `best_score` as a full-precision string. Determinism by construction beats determinism by hope, and D008 requires byte-identical export.

---

## D021 — A model with no feature names cannot be exported

*2026-08-01*

If `feature_names` is empty — the case when a model was fit from a bare array — export raises and requires the caller to supply names explicitly.

**Why:** D005's strict-key policy has nothing to be strict about without names. A strict-key policy with no keys to check reads as enforced and is not, which is worse than having no policy: the caller believes a typo will be caught.

---

## D022 — `±inf` at predict time raises

*2026-08-01*

Non-finite feature values raise. Pinned by a fixture, not left to convention.

**Why:** Consistent with fail-loudly. Upstream is genuinely inconsistent here — `±inf` raises through `DMatrix` but is treated as an ordinary comparable value through `inplace_predict`, so the same input yields two different predictions depending on the call path. `COMPAT.md` documents this as an upstream hazard; surfacing exactly this class of divergence is why the library exists.

---

## D023 — Early stopping is resolved empirically, not by design

*2026-08-01*

An early-stopped model serializes `best_iteration` while **all** trees remain in `trees[]`. Which tree count applies is not decided here. It is measured in Phase 4 against XGBoost's own `predict()`, since fixtures carry that output as ground truth. The finding is recorded with whether it is version-dependent. If it is ambiguous, export raises.

**Why:** Guessing which trees XGBoost actually uses would propagate into every early-stopped artifact. This is precisely the kind of empirical question that gets measured rather than reasoned about.

---

## D024 — CI actions are SHA-pinned before release

*2026-08-01*

Workflow actions are currently pinned to major-version tags. They move to full commit SHAs before the 1.0 release.

**Why:** A mutable major tag is a supply-chain hole in a workflow that will eventually hold publish credentials. Acceptable pre-1.0 while nothing is published; not acceptable once it is.

---

## D025 — The accumulation recipe is normative

*2026-08-01*

Both predictors implement exactly this, and it is treated as specification rather than as an implementation detail:

1. Narrow thresholds **and** leaf values to float32 when read.
2. Initialize the accumulator with the float32 intercept — **before any tree**.
3. Walk trees in serialized `trees[]` order. Leaf iff `left_children[i] == -1`.
4. Route with both sides cast: strictly `<` goes left, and **equality goes right**.
5. Narrow the accumulator to float32 **after every single addition**.

Measured: 5000/5000 bit-exact against `predict(output_margin=True)`, max abs error `0.0`. Independently reproduced Python-vs-JavaScript at 500/500 bit-identical, max difference `0.0`.

**Why:** Every deviation was measured and every one is worse — adding the intercept last scores 747/5000 and breaches the `1e-6` gate at `1.34e-05`. The order of operations is not stylistic; it is the difference between hitting exactly `0.0` parity and not.

**Trap worth recording for fixture design:** at `base_score=0.5` every wrong variant scores 5000/5000. A corpus built on `0.5` validates a broken implementation. Cox has the same trap at its estimated default, where the intercept is exactly `0.0` and intercept placement stops mattering.

---

## D026 — The output transform runs in float64 on both sides, and parity is measured twice

*2026-08-01* — **precision requirement SUPERSEDED by D032.** The two-measurement-point rule and the float32-margin rule still stand; the float64 output transform does not.

Margins are float32 throughout. The **output** transform — margin to probability or hazard ratio — widens to float64 on both sides and the result is not narrowed back.

Cross-language parity is therefore measured at **two** points, both required to be exactly `0.0`: at the margin, and at the final output after the transform.

**Why float64:** JavaScript has no float32 `exp` or `log`. `Math.exp` is float64, and `Math.fround(Math.exp(x))` is a float64 exp rounded once — not the same value a genuine float32 exp would produce. A float32 output transform is therefore **not reproducible in the JavaScript runtime at all**, so specifying one would make exactly-`0.0` parity unreachable at the output stage by construction. Widening is reproducible on both sides: `np.float64` of a float32 is exact, and a JS number already is float64.

**Accepted consequence:** if XGBoost computes its own transform in float32 internally, this library's probability output differs slightly from XGBoost's. Bounded by the `1e-6` gate and accepted — **cross-language reproducibility wins over matching XGBoost bit-for-bit at the output stage.** The margin comparison is unaffected and stays in the low `1e-7` range.

**Why two measurement points:** a margin-only parity check passes while a transform-precision mismatch ships. That is this project's failure mode relocated one stage downstream, which makes it exactly as dangerous and slightly harder to see.

### Correction, 2026-08-02 — the original reasoning was wrong, the conclusion survives

As first written, this entry claimed that because both languages enter the transform with bit-identical float64 inputs and identical IEEE-754 double semantics, widening was sufficient for a bit-identical result. **That claim is false and was measured false.**

IEEE-754 mandates correct rounding only for `+ − × ÷ √` and fused multiply-add. `exp` is not required to be correctly rounded and no two `libm` implementations agree. Measured on one platform pair, V8 against Apple `libm`: **4.2%** of sigmoid evaluations and **9.6%** of `exp` evaluations differ, by up to **2 ULP** (worst sigmoid case at margin `0.9417615532875061`: `0.7194553455999664` versus `0.7194553455999666`). Python's `np.exp` and `math.exp` agreed on 6009/6009, locating the split between runtimes rather than within one.

Widening to float64 is still correct and still required — float32 is *unreproducible* in JavaScript, not merely divergent, so this was never a precision-width problem and widening could not have fixed it. Both parity measurement points also stand at exactly `0.0`, with no tolerance. What changed is *how* the second one is attained: see D030. The error here was mine, and it is recorded rather than quietly overwritten because the reasoning failure — inferring bit-identity from shared semantics — is the kind that recurs.

---

## D027 — Dead nodes are neutralized in place, neither compacted nor carried

*2026-08-01*

The exporter walks each tree from the root, then overwrites every unreachable node with canonical safe values (`split_indices=0`, `node_values=0.0`, both children `-1`, `default_left=0`). Array lengths are unchanged and **no index is renumbered**.

**Why not compact:** remapping every child reference to new indices is a correctness risk — a remapping bug silently reroutes a live sample — and the payoff is only size.

**Why not carry verbatim:** the reader could then never assert that a `split_indices` value is in range, because `2147483647` is legitimate in a legitimately pruned model. That exception would weaken validation for *every* artifact to accommodate nodes the walk never visits.

Neutralizing is the reachability walk without the renumbering: strictly less risk than compaction, strictly better validation than carrying.

**Cost, accepted:** heavily-pruned trees keep their dead slots, which matters for browser delivery — at `gamma=1e9` a 59-node tree has 58 dead nodes. Compaction is a v1.1 optimization and does **not** require a format change; an artifact with no dead slots is already valid.

**Verification is mandatory.** A neutralization that clears a *live* node is silent wrongness. Export walks each neutralized tree against XGBoost's `predict()`, and the corpus must include a pruned model where a neutralized node would have been visited had the reachability walk been wrong. Reachability is the definition; the `split_indices == 2147483647` marker is asserted to agree and raises if it does not.

---

## D028 — `objective` is non-operative metadata, enforced by test

*2026-08-01*

`objective` stays in the artifact. Its only role is an export-time cross-check against `output_transform`. **No predictor branches on it, in either language**, and the test suites assert this in both.

`COMPAT.md` and the JSON Schema description both state that it is non-operative.

**Why keep it:** one string, and it preserves inspectability — a human reading an artifact can tell what produced it.

**Why the test:** without one, a future contributor adds `if objective == ...` to a prediction path and the field quietly becomes a second source of truth about behavior that `output_transform` already determines. Two fields that must agree, where only one is validated, is how a silent divergence starts.

---

## D029 — The vocabulary scrub is executable, not a manual grep

*2026-08-01*

The scrub runs as a test in the Python suite over all tracked source and documentation. Ambiguous English words are not matched bare; a term is matched only in a form that indicates domain use.

**Why:** the first scrub pattern flagged the phrase "incidental churn on master" — ordinary English, not the modelling task. A check whose output has to be interpreted by hand is a check that gets waved through once it is inconvenient, and "the scrub had a false positive" is indistinguishable from "the scrub found something" in a hurried review.

---

## D030 — Both packages bundle their own `sigmoid` and `exp`

*2026-08-02* — **still stands, amended by D032**, which changes the evaluation precision to float32 and adds XGBoost's clamps. The decision to bundle rather than call `libm` is unchanged.

Neither package calls a platform transcendental on the prediction path. No `Math.exp`, no `math.exp`, no `np.exp`. Both implement the transform from correctly-rounded primitives so that the two languages execute an identical sequence of IEEE-754 double operations and agree bit-for-bit **by construction**.

**Why not simply state a tolerance instead.** A tolerance has to be a number, and no honest number exists. The 2-ULP figure in D026 was measured on exactly one platform pair. glibc's `exp` is a third implementation, and recent glibc is correctly rounded where V8's fdlibm-derived port is not — so a Linux CI runner produces a *different* divergence than a macOS laptop, and **neither measurement bounds the other**. Publishing a bound measured on one pair and applying it to every platform a consumer runs on is a silently generalized numerical claim, which is the exact failure this project exists to prevent, appearing in the verification gate itself. The option was rejected on evidence, not on principle.

**Why it is worth the work.** Not because 2 ULP matters — `3e-16` at a probability changes no decision anyone makes. Because **exact equality is a tripwire and any tolerance is a band real bugs hide inside.** A gate set at 2 ULP passes a genuine 1-ULP defect forever. The methodological claim of this project is that exactness detects what tolerance conceals.

**The risk this introduces, stated plainly.** Novel numerical code is new silent-wrongness surface, in the one place the project previously had none. And **bit-identical wrong is still wrong** — invisible to the parity harness precisely because both sides agree perfectly. Hence the validation rule below, which is not negotiable.

**Validation.** Each side is validated **independently** against a high-precision reference (`mpmath`, 50 digits), never against the other side. Cross-language agreement is a separate check and is **not** evidence of correctness: two identical implementations agreeing proves only that the same code was written twice. At least `1e6` sample points per objective across the full representable range, covering overflow and underflow boundaries, the subnormal transition, `±inf`, `NaN`, exact `0.0` and `-0.0`, and margins that saturate sigmoid at exactly `0` and `1`. **Max** ULP error is reported, never mean.

**Implementation constraints** — bit-identity is losable by accident: only `+ − * /` and exact power-of-two scaling; every operation a separate statement with an explicit named intermediate, so no runtime can contract a fused expression into an FMA; no vectorization; argument-reduction constants split hi/lo and written as literal float64 bit patterns in both languages rather than decimal strings each parser rounds independently, with a test asserting both sides' constants parse to identical bits.

**Accuracy is not a design input.** ~1 ULP against a correctly-rounded reference versus `libm`'s ~0.5 is a relative difference near `1e-16` — sixteen orders below the `1e-6` gate.

**Consequence for the XGBoost comparison.** XGBoost transforms in C++ `libm`, so this library's probability output diverges from XGBoost's by roughly 1–2 ULP **by construction**. Expected, irrelevant at `1e-6`, and not to be read as a regression. The margin-level comparison is unaffected.

This is now the most dangerous code in the repository. Phase 5's adversarial-fixture treatment applies to it in full, including D019's revert-and-confirm-red methodology.

---

## D031 — `mpmath` is a test-only dependency

*2026-08-02*

`mpmath` enters the workspace dev dependency group solely as the high-precision reference for validating the bundled transform (D030). It is **not** a runtime dependency of `xgboost-bridge`, does not appear in the published package's `dependencies` or in any extra, and has no JavaScript counterpart — the zero-runtime-dependency rule for `xgboost-predictor` is untouched.

**Why a dependency at all,** given that adding one requires an explicit decision: validating a hand-written transcendental needs a reference more accurate than the thing being tested, and neither the standard library nor `numpy` provides arbitrary precision. Writing our own 50-digit reference to check our own transform would be circular.

---

## D032 — The output transform runs in float32 and reproduces XGBoost's clamps

*2026-08-02* — **supersedes the precision requirement in D026 and amends D030**

The bundled `sigmoid` and `exp` are evaluated under **float32** semantics: `np.float32(...)` per intermediate in Python, `Math.fround(...)` per intermediate in JavaScript. They reproduce XGBoost's measured clamps.

**Why D026's premise dissolved.** D026 chose float64 because JavaScript has no float32 `exp`. That was true of `Math.exp` and stopped being true the moment D030 made the transform ours: a transform built from `+ − × ÷` and exact power-of-two scaling can be evaluated under float32 semantics deterministically in both languages.

**Why float32 arithmetic can be simulated exactly.** A float64 operation followed by narrowing to float32 is *exact* for `+ − × ÷`, because float64 carries more than twice float32's significand and double-rounding cannot occur for those four operations. This is a property of the IEEE-754 formats, not an empirical result. It does not extend to `exp` — which is exactly why `exp` must be built from the four operations rather than called.

**Why float64 was not merely imprecise but wrong.** XGBoost transforms in float32 with clamps. Verified: 400/400 bit-exact for float32-throughout versus 236/400 for float64-then-narrow, and on all 164 rows where the two hypotheses disagree XGBoost matches float32 **164/164** and float64 **0/164**. In the tail float64 is qualitatively wrong — relative error `1.0` below the logistic clamp floor, and finite-versus-`inf` for Cox. Divergence from upstream that is itself silent is the thing this library exists to surface, so upstream is matched in the tail rather than diverged from quietly.

**Clamps, measured.** `binary:logistic` floors at margin `f32(-88.7)`, returning exactly `3.006635794144578e-39` and never `0.0`; the sole float32 input producing those bits is `-88.69999694824219`, found by exhaustive scan of all 262145 float32 values in `[-90, -88]`. `survival:cox` has no clamp and returns `+inf` above margin ≈ `88.72`. Consequence for fixtures: sigmoid saturation at exactly `1` is reachable, at exactly `0` is **not** — the clamp prevents it.

**Clamp constants are XGBoost internals and version-sensitive** in the same way `weight_drop` proved to be. They fall under the D018 version ceiling and are re-probed whenever the tested version list widens.

Cross-language parity stays exactly `0.0` at both measurement points; there is no `libm` in the path. Bit-exactness with XGBoost at the output remains unreachable and is not a goal — its own `expf` is not correctly rounded (an mpmath-exact reference scores 1600/2500 against it).

---

## D033 — The Python-vs-XGBoost output gate is relative; the margin gate stays absolute

*2026-08-02* — **supersedes the absolute output-gate row**

- Margin: **absolute** ≤ `1e-6`. Currently `0.0` bit-exact at all 16 measured sweep configurations; a regression from that is a defect to diagnose, not headroom.
- Output: **relative** ≤ `1e-6`, against XGBoost's value.

Explicit rules for the rows that otherwise vanish from the comparison: `±inf` must match as bit patterns and is never divided; NaN is always a failure on either side, because NaN compares unequal to everything including itself and a naive harness silently *skips* exactly those rows; where XGBoost's value is `0.0` or `-0.0`, require bit equality rather than a ratio. The harness reports **max** relative error and the row that produced it, never a mean.

**Why:** an absolute output bound flags the wrong objective and misses the real defect. Cox output is a hazard ratio spanning `2.85e-04` to `7.56e+08`, so absolute error there is meaningless — measured `1.94` to `6.96e+23` plus `+inf` rows — while its relative error is `5.7e-08`. Meanwhile logistic passes an absolute gate trivially while being relatively 100% wrong below the clamp.

**This concerns only the accuracy gate.** Cross-language parity is exact equality and no tolerance question touches it. The two are kept separate in the harness and in `CLAUDE.md`, because conflating them is how a tolerance leaks into the parity gate.

---

## D034 — Every check needs an oracle independent of what it checks

*2026-08-02* — **supersedes D015's export assertion**

Export validates the derived intercept against **XGBoost's own observed zero-tree margin**, not against a re-derivation of the export recipe.

**Why the original check was decorative.** It compared the stored `base_score` against a re-derivation using the same recipe. Both sides ran the same code, so **an error in the recipe could not make it fire** — and D035's clamp defect is exactly such an error, which that check passed silently. A validation whose oracle shares the defect it looks for provides no information.

Standing rule for this codebase. Before accepting any validation, state what its oracle is and why the oracle cannot share the defect. If the answer is "it compares our value to our other value," delete or replace the check.

- Numerical implementations validate against a high-precision reference (`mpmath`, 50 digits), **per side, independently**. Cross-language agreement is a separate check and is never evidence of correctness: two identical implementations agreeing proves only that the code was written twice.
- Export-side assertions validate against XGBoost's observed output.
- Redundant safeguards are untested safeguards. If removing either of two overlapping protections breaks no test, neither is pinned. Test each independently or collapse them to one.

---

## D035 — `base_score` is clamped before the logistic intercept transform

*2026-08-02* — **amends D015**

For `binary:logistic`, clamp `p` to `[f32(1e-6), f32(1 - 1e-6)]` **before** applying `-log(f32(f32(1/p) - 1))`. Cox and regression are not clamped, verified at `1e-38` and `1e38`.

**Why:** XGBoost clamps `base_score` before deriving the intercept but **stores it unclamped**. Applying the recipe to the stored value is wrong by up to **13.8 in margin space**. Independently verified: at `base_score=1e-12` the unclamped recipe gives `-27.631021` against XGBoost's `-13.815510`; at `1e-7`, `-16.118095` against `-13.815510`; at `0.9999999`, `15.942385` against `13.745160`. Clamping first reproduces XGBoost exactly in all cases. Additionally `base_score = 1 - 1e-10` stores as `[1E0]`, where the unclamped recipe raises `math domain error`.

---

## D036 — `boost_from_average` selects the intercept space

*2026-08-02*

Export reads `learner.learner_model_param.boost_from_average`. With **zero trees and `boost_from_average == "1"`**, XGBoost emits the **raw `base_score`** as the margin, with no link transform. Every model with at least one tree applies the transform.

Verified: logistic default gives margin `0.5` where the link gives `-0.0`; Cox default gives `0.5` where the link gives `-0.693147`. Flipping that one string moves the margin between `0.5` and `-0.0`.

The field is not carried in the artifact — it is an export-time input whose effect is already baked into `intercept`. Carrying it would invite a predictor to branch on it.

**Consequence for the required signed-zero fixture:** it must pass `base_score = 0.5` **explicitly**. Built from the default it lands in the raw space and tests nothing, while looking exactly like a signed-zero fixture.

---

## D037 — Arity gate: `num_class ∈ {"0","1"}`, per-tree `size_leaf_vector`, string comparison

*2026-08-02* — **amends D017**

Require `num_target == "1"`, `num_class ∈ {"0", "1"}`, and `size_leaf_vector == "1"` for **every tree**, with a zero-tree model passing vacuously. All four gate fields, `objective.name` included, are JSON strings and are compared as strings.

**Why `"1"` is admitted:** independently verified across all three in-scope objectives that `num_class=1` yields the *same* single-output model — trees byte-identical, margins bit-identical 400/400, `predict()` shape `(400,)` — and that requiring `"0"` **rejects all three**. A false rejection is worse than over-strictness because it reads as correct strictness and is not. Relaxing admitted nothing extra across a 23-shape table with zero false acceptances.

**Why per-tree:** `size_leaf_vector` exists only in each tree's `tree_param`, never in `learner_model_param`. A model-level comparison has no referent, and a zero-tree model has zero occurrences — without the vacuous-pass rule the gate rejects every zero-round model.

**Why string comparison:** `num_class == 0` is `False` against `"0"`. An integer comparison silently never fires, which would disable the gate rather than trip it.

Zero-round models serialize `"trees": []` — present and empty, never absent, verified on all three objectives.

---

## D038 — Export raises on an early-stopped model whose tree count is ambiguous

*2026-08-02* — **resolves D023**

The exporter raises when `iteration_indptr[best_iteration + 1] != len(trees)`. The error directs the caller to slice the model explicitly — `bst[0:best_iteration + 1]` — and re-export.

**Why raising is the only correct answer: the tree count is not a property of the model.** Measured on **one file on disk, loaded two ways**:

| Loaded as | vs walk over all trees | vs walk over `best_iteration+1` iterations |
|---|---|---|
| bare `Booster` | **2500/2500** | 0/2500 |
| sklearn estimator | 0/2500 | **2500/2500** |

Max absolute divergence between the two readings: `1.55`. **No field in the artifact distinguishes them.** An exporter that picks either reading is silently wrong for half its callers, and the wrongness is a plausible number on every row. `iteration_range=(0,0)` compounds it — explicitly passed, it means "all trees" through `Booster.predict` and "`best_iteration+1` iterations" through the estimator. Same argument, same model, two answers.

**Why the predicate is not simply "`best_iteration` is present."** With `early_stopping_rounds` set but never fired, `best_iteration=4` and `num_boosted_rounds=5`, and both readings agree 2500/2500. Refusing on presence alone would reject models that are unambiguous. The predicate keys on whether the trees actually extend past the best iteration.

**Truncation, if ever implemented, must be in iterations rather than trees.** `iteration_indptr` is authoritative. At `num_parallel_tree=3` with `best_iteration=7`, truncating to `trees[0:24]` is bit-exact 2500/2500 while truncating to `trees[0:8]` scores 0/2500 at max error `2.19`. A tree-based truncation is correct at `num_parallel_tree=1` and wrong above it — the classic shape of a defect that passes every test written against the default.

**`best_iteration` lives only at `learner.attributes.best_iteration`**, as the JSON string `"7"`, and is absent from every model param. Reading it at export needs no D020 whitelist entry; emitting it would. It is not emitted — the artifact has no ambiguous tree count to record, because ambiguous models are refused.

**Not version-dependent.** Re-measured on XGBoost 3.4.0: every number byte-identical, 50 key paths identical, no relocation. The behavior is API-path-dependent, which is worse — a version ceiling cannot defend against it, only refusal can.

**Unmeasured and recorded as such:** the `save_best=True` callback. If it trims the model, the predicate is satisfied and export proceeds — which is safe, because a satisfied predicate means both readings coincide.

---

## D039 — The `base_score` clamp bounds are pinned, and the earlier approximation was already exact

*2026-08-02* — **confirms D035; no numeric change**

The logistic clamp bounds, pinned to adjacent float32 pairs by exhaustive search rather than bracketed:

| Bound | Last value inside | First value outside | Saturated intercept |
|---|---|---|---|
| Lower | `1.0000001111620804e-06` (`0x358637BE`) | `1.0000002248489182e-06` (`0x358637BF`) | `-13.815509796142578` (`0xC15D0C54`) |
| Upper | `0.9999988675117493` (`0x3F7FFFED`) | `0.999998927116394` (`0x3F7FFFEE`) | `13.745160102844238` (`0x415BEC2D`) |

Over a 226-value sweep the pinned clamp and D035's approximate `[f32(1e-6), f32(1 - 1e-6)]` scored **226/226 each, value for value**, against **52/226** for no clamp. **The approximation was already exact.** The source literal is pinned only to an equivalence class — 8 admissible values for the lower bound, 2 for the upper — and every member gives bit-identical intercepts on every float32 input, so the choice has no observable consequence.

Measured behaviour at the extremes, which the exporter must reproduce or refuse rather than guess: `0.0`, `-0.0`, the minimum subnormal, and `1.0` are all **accepted** and clamp to a saturated intercept. Negative values and values `> 1` **raise** at fit *and* at load. `nan` and `inf` **raise** in the JSON parser. `-inf` is **accepted and silently stored as `[0E0]`** — the one asymmetry, and worth knowing because it is the only non-finite input that does not announce itself.

Cox and regression are unclamped, confirmed across 34 values from `1.4e-45` to `3.4e38` and 25 values spanning `±3.4e38`.

---

## D040 — Both intercept logarithms are float32 logarithms

*2026-08-02* — **amends D015 and D035**

`np.log` applied to a float32, for both `survival:cox` and `binary:logistic`. **Not** `np.float32(math.log(float(x)))`.

**Why this was nearly missed, and the methodological rule it produces.** The two routes disagree on only **0.055%** of float32 inputs. Two independent sweeps — 79 values and 1432 values — both concluded "no difference," and one explicitly recorded logistic as unaffected.

Isolating the disagreeing inputs first and *then* asking XGBoost settles it immediately. Of 13,421,774 float32 values in `[0.4, 1.2]`, 7387 make the routes differ. On a 120-value sample of those, Cox: float32 log **120/120**, float64 route **0/120**. On 75 disagreeing values for logistic: **75/75** versus **0/75**.

**The rule: a sample that does not deliberately target the inputs where two candidate implementations diverge cannot distinguish them, and its silence is not evidence of equivalence.** Find the disagreeing inputs, then consult the oracle. This is the independent-oracle principle of D034 applied to sampling rather than to comparison.

**Cost of getting it wrong:** because D034 requires the derived intercept to match XGBoost's observed margin bit-for-bit, the float64 route makes export raise spuriously on roughly one model in two thousand, with nothing to indicate the rule rather than the model is at fault.

---

## D041 — Two error classes for two different failures

*2026-08-02*

`MalformedTreeError` means a structure that contradicts the evidence the reader was built from: unequal array lengths, an absent field, a child index that does not point forward, a non-finite threshold, a dead-node marker disagreeing with reachability, a `split_type` array shorter than the node count.

`UnsupportedModelShapeError` means a well-formed model whose **output arity** this version declines to support: `num_target`, `num_class`, `size_leaf_vector`, and feature-name count or uniqueness.

**Why separate them.** They call for different responses. An arity refusal is a scope statement — the model is fine and a later version may support it. A malformed-structure refusal says the reader's assumptions have already been violated, so continuing would walk a structure under premises known to be false. Collapsing both into one class tells a caller "unsupported" when the truth is "unrecognized," and those warrant different bug reports.

---

## D042 — Concurrent agents must not run the shared suite while another is mid-write

*2026-08-02*

An agent working on one module runs only its own test file during development, and the full suite once at the end. Where two agents' work must both be verified against the whole suite, they are serialized.

**Why:** two agents editing disjoint modules still share one test suite. One reported an intermittent failure in the other's test file at roughly 1 run in 25 — a test that is fully deterministic, with a hand-built fixture and no randomness or shared state. Re-measured after both finished: **0 failures in 40 isolated runs and 24 full-suite runs.** The observation was a torn read — the suite imported a module mid-write, so a test asserting a guard ran against a version that did not yet have it.

The cost of misdiagnosing this is high in both directions: chasing a nonexistent race, or dismissing a real flake as "probably concurrency." Neither is acceptable in a project whose entire premise is that intermittent wrongness is the dangerous kind.

---

## D043 — The intercept oracle is case-dependent, and a non-finite intercept is refused

*2026-08-02* — **refines D034 and D036**

**The oracle.** For a model with at least one tree, refit with zero boosting rounds and `base_score` passed explicitly, and compare against that margin. For a model that **already has zero trees**, compare against the source model's own margin.

**Why the second case cannot use the first rule.** Passing `base_score` explicitly flips `boost_from_average` to `"0"` and puts the refit in link space. A zero-tree model with `boost_from_average == "1"` is correct precisely *because* its margin is the raw `base_score` (D036). Applying the refit rule there compares a raw-space intercept against a link-space oracle and **rejects a correct intercept** — specifically the one configuration the signed-zero fixture requires. Reading the source model's own margin is still XGBoost's observed output rather than a re-derivation, so D034's independent-oracle property is preserved. D034 as first written did not cover this cell.

**Non-finite intercepts are refused at export.** `survival:cox` with `base_score = 0.0` derives `-inf` (bits `0xFF800000`); with any negative `base_score` it derives `NaN` (`0x7FC00000`). XGBoost accepts both with no error and no warning.

The derivation **reproduces** them bit-for-bit, because the oracle above would otherwise disagree; the refusal therefore belongs at export, not in the derivation. Worth noting why the oracle alone is not enough: a bit-pattern comparison happily matches `NaN` against `NaN`, so what catches this is the finiteness requirement of §9.3, not the equality check.

**Two probe reports are superseded on specific points**, recorded here rather than edited into the reports, which stand as records of what was measured when:

- `probes/base_score_clamp.md` §8 concludes the logistic logarithm question is "not decidable" from 0 disagreements over 1432 values. That is a sampling artefact. A scan of *consecutive* float32 values in `[0.45, 0.55]`, where the logistic intercept's magnitude is small, finds **3621 disagreements in 2,516,584 values**, and XGBoost matches the float32 route 8/8 and the float64 route 0/8. D040 holds. The probe made the same sampling error it correctly identified in another report.
- `probes/base_score.md` §9 step 3 states the logistic recipe with no clamp and a float64 logarithm. D035 and D040 supersede it. A reader following §9 alone reproduces the 13.8-in-margin-space defect.

`SUPPORTED_OBJECTIVES` is currently defined in both `objectives.py` and `validate.py`, with a test pinning their agreement so they cannot drift silently. Collapsing them is a module-dependency question deferred rather than decided.

---

## D044 — Fixture ground truth is stored as float32 bit patterns

*2026-08-02*

Each fixture records XGBoost's expected margin and expected output as **uint32 hex bit-pattern strings** (`"0x3f800000"`), with a decimal rendering alongside for human inspection only. Consumers compare the bit patterns; the decimals are never the comparison.

**Why not JSON numbers.** JSON has no representation for `±inf`, and `survival:cox` genuinely returns `+inf` above margin ≈ `88.72` — measured on 734/2500 rows of a real model. Any numeric encoding therefore needs a special case for the exact values that matter most, and a special case in the ground truth is the last place one belongs.

Bit patterns handle `+inf`, `-inf`, `NaN`, and `-0.0` with no special cases at all, and they make the correct comparison the *only* convenient one. A decimal ground truth invites `==`, under which `-0.0 == 0.0` is `True` and two different artifacts compare equal.

**Cost:** a fixture is less readable at a glance, which the decimal field mitigates. The decimal is explicitly non-normative: if the two ever disagree, the bit pattern is right and the fixture is regenerated.

**Addendum, 2026-08-02.** Two encoding details D044 did not cover, settled by the generator:

- A **missing feature value in `rows`** is JSON `null`, converted to `NaN` on read. `rows` is input, not ground truth, so the bit-pattern rule does not apply, and standard JSON has no `NaN` literal. `null` keeps every fixture parseable by a plain `JSON.parse` with no custom grammar — which matters, because the JavaScript predictor must read these files with zero dependencies.
- A **non-finite value in the non-normative decimal fields** is the string `"inf"`, `"-inf"`, or `"nan"`. Same reason. These fields are for human inspection only; the bit pattern governs.

Each fixture records the convention in `meta.nan_encoding` rather than leaving a reader to infer it.

**Empirical note for anyone generating pruned models:** `tree_method="hist"` does not produce dead nodes at any `gamma`. It declines to grow a losing split rather than growing and then pruning one, so `num_deleted` stays `0`. `tree_method="exact"` is required to obtain a genuinely pruned tree, which is also what `probes/tree_structure.md` used. Discovered by measurement after several `gamma` values produced no deletions.

---

## D045 — Infinite feature values are refused up front, not lazily

*2026-08-05* — **implements D022, which was specified but unimplemented**

`walk_margin` validates the **entire** input row before walking, and raises `NonFiniteFeatureError` on any `±inf`. `NaN` is accepted: it is the missing value and routes by `default_left`.

**This was a live gap, not a hardening exercise.** D022 recorded the decision on 2026-08-01 and `FORMAT.md` §9.3 restated it, but nothing implemented it. An adversarial fixture pass noticed. Verified before fixing: `walk_margin` with `+inf` in a feature returned `np.float32(13.74516)` — an ordinary-looking number. A specified refusal that no code performs is worse than an unspecified one, because the decision record makes it look handled.

**Why the whole row rather than only the nodes visited.** A lazy check makes the same invalid input raise or not depending on which branches that particular tree takes — the outcome becomes a property of the model instead of the input. Cost is `O(features)` against an `O(depth × trees)` walk.

Both variants were reverted and confirmed red: removing the guard turns 4 tests red; narrowing it to the first column only — the plausible lazy version — turns 2 red, including the test that exists specifically for a column no node reads.

**A note on the threshold-side cast, from the adversarial measurements.** Of the six deliberately-broken walk variants, "cast the threshold but not the sample" produced only 1 wrong row in 184, while "cast the sample but not the threshold" produced 96. That asymmetry is expected and worth recording: this project's own emission rule (D044, `float(np.float32(x))` at full float64 precision) makes every stored threshold recover its exact float32 at any width, so there is no sub-ULP gap left for a mis-cast sample to exploit. The threshold-side cast is therefore defence for artifacts this exporter did **not** produce — hand-edited, third-party, or a future format revision — rather than for our own. It stays, and the reasoning is recorded so nobody later removes it as dead weight after measuring only against our own corpus.
