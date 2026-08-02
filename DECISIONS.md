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

*2026-08-01*

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

*2026-08-01*

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

*2026-08-01*

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

*2026-08-02*

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
