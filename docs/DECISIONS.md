# Decisions

Durable engineering decisions and their rationale. Append; do not rewrite history. If a decision is reversed, add a new entry that supersedes the old one and mark the old one superseded.

Format: what was decided, what forced it, what it costs.

---

## General principles

Two rules generalize past the entries that produced them, and both are here because
each was learned by getting it wrong in a way that looked correct. They are the same
shape with a different noun.

**1. Every check needs an oracle independent of what it checks.** Before accepting any
validation, state what its oracle is and why the oracle cannot share the defect. If the
answer is "it compares our value to our other value," the check is decorative — delete
it or replace it. Found by D034: an export assertion compared a derived intercept
against a re-derivation of the same recipe, so a recipe error could not make it fire,
and it passed the real clamping defect of D035. Applied since to sampling (D040 — a
sample that does not target the disagreeing inputs cannot distinguish two
implementations), to the parity comparison layer (D047 — a check that cannot tell
"equal" from "both unparseable" is not a check), and to the intercept itself (D053 —
once the value comes *from* the oracle there is nothing left to compare, so the check
is retired rather than kept for reassurance).

**2. A gate and the thing it gates must not share a trust domain.** Ordering is not a
control when the credential does not depend on it. Found by D054: both publish
workflows ran the full test suite, two build scripts and `npm ci` *before* the publish
step, inside a job that already held `id-token: write` — and one of them documented
that ordering as a safeguard. It was not one. The OIDC request variables are present
for the job's whole duration, so any dependency executing an install script could have
minted a publishing token at step one, and the gate would still have passed afterwards.
The fix is structural, exactly as it is for rule 1: put the work and the credential in
different jobs, so the job that can publish runs nothing that could want to.

Stated generally: **a safeguard placed inside the blast radius of what it guards is not
a safeguard.** Rule 1 is that statement about *information* — an oracle downstream of
the defect cannot see it. Rule 2 is the same statement about *authority* — a check
inside the credential's scope cannot constrain it. When adding either kind of
protection, name what it is outside of.

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

*2026-08-01* — **discharged 2026-08-06; see D059**

This project is AI-authored under human direction. The disclosure lands with the 1.0 announcement rather than in the README now.

**Why:** Recorded here so the commitment exists and is dated, independent of when the announcement copy is written.

> **Discharged.** The disclosure is now in all three READMEs, as one line, ahead of the
> announcement rather than with it. D059 records the wording and why it is short.

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

An early-stopped model serializes `best_iteration` while **all** trees remain in `trees[]`. Which tree count applies is not decided here. It is measured against XGBoost's own `predict()` while the exporter is built, since fixtures carry that output as ground truth. The finding is recorded with whether it is version-dependent. If it is ambiguous, export raises.

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

This is now the most dangerous code in the repository. The adversarial-fixture treatment applies to it in full, including D019's revert-and-confirm-red methodology.

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

*2026-08-02* — **amends D015 and D035**; **narrowed by D053**

> **Narrowed as a generalisation, 2026-08-06.** Everything measured below holds, on
> darwin/arm64. It is not a fact about XGBoost: XGBoost calls the platform's `logf`, and its
> own intercept differs between darwin/arm64 and linux/x86_64 by 1 ULP on 29 of 58
> discriminating inputs. `np.log` of a float32 reproduces XGBoost on one of those platforms,
> not on both, and no fixed recipe reproduces it on both. The intercept is therefore read out
> of the engine rather than computed — see **D053** and `probes/platform_log.md`. The
> float32-versus-float64 route finding below still governs `derive_intercept`, which
> documents the engine's behaviour and is no longer on the export path.

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

---

## D046 — Bundled transform accuracy, and how its ULP bound must be measured

*2026-08-05* — **implements D030 and D032**

`transform.py` provides `exp_f32`, `sigmoid_f32`, and `identity_f32`, built from `+ − * /` and exact power-of-two scaling only. Measured against `mpmath` at 50 digits:

| Function | Max ULP | Correctly rounded |
|---|---|---|
| `exp_f32` | **1** | 923049/1000000 |
| `sigmoid_f32` | **2** | 765376/1000000 |

Independently re-measured during review: `exp` 1 ULP over 60k points; `sigmoid` 2 ULP over 90k points above the clamp floor, with the reported worst inputs reproducing at exactly 2. The platform float32 exponential is in the same class — max 1 ULP, correctly rounded 99.55% — so the bundled version is not markedly worse, which is the only thing accuracy needed to establish. A compensated accumulation was tried and rejected: it does not lower the max ULP and costs six extra operations, and every extra operation is another way for the hand-written JavaScript port to diverge.

**A ULP figure for `sigmoid_f32` is only meaningful above the clamp floor.** Below `f32(-88.7)` the implementation deliberately returns `3.006635794144578e-39` while an unclamped reference keeps falling, so comparing them reports the clamp as if it were error — measured as 1,560,434 ULP at margin `-89.99927` on a first attempt at verification. Below the floor the correct check is a **predicate** (does it return exactly the floor bit pattern, and never `0.0`), not a distance. Recorded because the naive sweep produces an alarming number that means nothing.

**The sampling flaw worth recording.** The first 1e6-point sweep reported max ULP 1 for `sigmoid_f32` — wrong. Targeted boundary bands had consumed the whole point budget, leaving the random fill empty, while a 40,000-point uniform draw found 2 ULP at a rate of `3e-4`. Bands are now capped and a non-empty fill is asserted by a test whose only job is that. This is D040's lesson recurring in a different guise: a sweep that does not sample where the answer varies cannot find it, and a large point count is not evidence of coverage.

**One structural check exists because no ULP measurement could replace it.** A drift to float64 intermediates would score *better* against `mpmath`, so accuracy cannot detect it. An AST scan therefore requires at most one arithmetic operation per statement and every one wrapped in `np.float32(...)`, and a token scan rejects `**`, `math.pow`, and every platform exponential by name.

Constants are pinned twice: once as integer bit patterns, so the JavaScript port can assert the same integers, and once against `mpmath` for meaning (`_INV_LN2 == round(1/ln2)`, `_Cn == round(1/n!)`). The second check caught a transcription typo in the floor constant that the first would have happily pinned.

---

## D047 — The reader narrows structurally, and the divergence from XGBoost is pinned as a set

*2026-08-05*

`Predictor.from_json` loads `node_values` into read-only `dtype=np.float32` arrays at parse time, per FORMAT.md §9.2, and validates the artifact against §13 on load.

**Measured against the full corpus, ordinary and adversarial, 23 fixtures:**

| | Result |
|---|---|
| Margin, bit-exact vs XGBoost | **289/289** |
| Output, bit-exact vs XGBoost | 283/289, max **relative** error `9.56e-08` |
| Cox `+inf` output rows | 2/2 bit-exact |
| Logistic clamp-floor rows | 21/21 bit-exact at `0x0020bd47`, never `0.0` |
| Refusal fixture | 10/10 raise |

The six output divergences are `libm` differences inside the bundled `exp`, expected by construction (§5.2: bit-exactness with XGBoost at the output is unreachable, because its own `expf` is not correctly rounded) and far inside the `1e-6` relative gate of D033.

**They are pinned as an exact set of `(fixture, row)` pairs plus the count, not absorbed into a tolerance.** Movement in *either* direction fails — including an improvement. That keeps the gate a tripwire rather than a band a future defect could hide inside, which is the distinction D033 rests on. All six fall on the two objectives that use the bundled `exp`; the `identity` fixtures diverge on nothing, which is the right shape for the finding.

**The structural-narrowing site cannot be pinned by a prediction, and that is disclosed rather than papered over.** Reverting `node_values` to un-narrowed Python floats turns **23 tests red and zero prediction tests red**. The reason is arithmetic, not thin coverage: `walk_margin` narrows both operands, and `np.float32(np.float64(x)) == np.float32(x)`, so narrowing at parse is idempotent with narrowing at comparison. What differs is every *other* consumer of the array — a re-serializer, an inspection utility, an arithmetic transform — which is precisely §9.2's argument for making it structural. The pin is therefore structural too: dtype, read-only-ness, and a hand-edited artifact read back through the public view where the un-narrowed float64 is a visibly different number. Same reasoning applies to the intercept.

**D028's no-branching-on-`objective` rule needs both of its checks.** A no-op `if objective == ...` inserted into the prediction path turns only the source-level check red; an obfuscated behavioural branch (`getattr(self, "_" + "objectiv" + "e")`) turns nine behavioural tests red while the source check stays green. Neither is redundant, which under D019 is the test for whether both should exist.

**One check beyond §13's enumerated list: a cycle in the reachable subgraph.** Every child index can be in range and still form a cycle, and `walk_margin` then never terminates — measured, a subprocess was killed at a 5-second timeout having printed nothing. A non-terminating predictor is a worse outcome than a raise. Confined to the reachable subgraph so §13's rule that unreachable nodes must *not* raise is untouched.

**Corrected in FORMAT.md §16:** the worked example printed `sigmoid(margin) = 0.5696602593994496`, a float64 result, where §5.1 requires float32 evaluation and the correct decimal is `0.5696602463722229`. Both narrow to bit pattern `0x3f11d541`, so nothing was contradictory — but an implementer comparing decimals against the wrong one concludes they have a bug, and the JavaScript port is written from that example next.

---

## D048 — The JavaScript predictor, and what the absorption mirror reveals

*2026-08-05*

`packages/js/src/{transform,artifact,predict}.ts`. Zero runtime dependencies, `fromJSON` only, `node_values` loaded into a `Float32Array` at parse time.

| Check | Result |
|---|---|
| Node suite | **92 pass** / 0 fail / 0 skipped / 0 todo, on Node 20.19.0, 24.7.0 and 24.18.0 |
| Corpus margin, bit-exact vs XGBoost | **289/289** across all 23 fixtures |
| Corpus output, bit-exact vs XGBoost | 283/289, max relative `9.555893664308718e-08` |
| `expF32` / `sigmoidF32` max ULP vs mpmath | **1** / **2** |
| Argument-reduction constants | all **16** bit-identical to Python's pinned integers, both encodings |
| Runtime dependencies | **0** |

**The transform was validated against an mpmath reference table generated in Python, not against the Python implementation.** That distinction is the whole point: agreeing with `mpmath` is evidence of correctness, whereas agreeing with the other language proves only that the same code was written twice. The table generator deliberately duplicates the exact-rounding oracle rather than importing it from the Python test suite, so the JavaScript oracle does not depend on Python test internals.

**The 6 output divergences are the same 6 `(fixture, row)` pairs, with the same max relative error, as the Python side recorded in D047.** Both sides also pin them as an exact set rather than a tolerance. That is the strongest available pre-harness signal that cross-language parity is exactly `0.0`; the parity harness confirms it formally.

**The absorption pattern is mirrored between the languages, and this is worth understanding.** In JavaScript, parse-time narrowing absorbs three of the five comparison-site casts — reverting the threshold cast, the leaf cast, or the intercept cast *alone* turns zero tests red. In Python it went the other way: NEP 50 weak-scalar promotion made the *parse-time* site the absorbed one (D047). So neither language can pin all five sites, they cannot pin the same ones, and which site looks redundant is a property of the host language's promotion rules rather than of the algorithm. Both structural sites are pinned in isolation on the JavaScript side (5 and 2 tests). The absorbed casts stay, per D045's reasoning, and the asymmetry is recorded so nobody "simplifies" one language to match the other.

**One measurement settles why FORMAT.md §5.5 demands constants as bit patterns.** Flipping `LN2_LO` by a single bit pattern left **every ULP test green** and turned only the bit-pattern comparison red. Accuracy cannot detect a mis-transcribed constant; only the integer comparison can.

**Two error codes were added to the `ErrorCode` union** — `MALFORMED_ARTIFACT` and `NON_FINITE_FEATURE`. The implementing agent could not edit that file and correctly refused to reuse a wrong code, because a caller switching on `code` would then handle a malformed artifact as "unrecognized field". It used a cast and flagged the debt; the union now carries both and the casts are gone.

**The `npm test` script was not portable across the engine range it declares.** `node --test test/` (bare directory) fails on Node ≥ 22; `node --test "test/*.test.js"` (quoted glob) fails on Node 20, which cannot expand a glob itself. `engines.node` says `>=20`, so both spellings were broken somewhere in the supported range. The fix is an **unquoted** glob: npm runs scripts through a shell, so the shell expands it and Node only ever receives explicit paths. Verified on 20.19.0, 24.7.0 and 24.18.0. This surfaced only because the suite was run on more than one Node.

**Two §13 refusals are not expressible in JavaScript**, recorded rather than papered over: `format_version: 1.0` is indistinguishable from `1` after `JSON.parse`, and `-0` passes `Number.isInteger` in an index array (with no numeric consequence, since it is `=== 0` as an index). The Python reader rejects the first; the JavaScript reader cannot. Every other spelling — `"1"`, `true`, `null`, `1.5`, `0`, `2` — is rejected on both sides.

---

## D049 — Cross-language parity is exactly 0.0, at both measurement points

*2026-08-05* — **satisfies the headline gate**

```
rows compared                 299   (289 value + 10 refused)
margin-point mismatches         0
output-point mismatches         0
refusal disagreements           0
input-bit disagreements         0
PARITY: 0.0 at both measurement points, on bit patterns.
```

All 23 fixtures, ordinary and adversarial. Verified on Node 20.19.0, 24.7.0 and 24.18.0; two consecutive runs byte-identical.

**What this gate does not establish.** Cross-language agreement is not evidence of correctness — two identical implementations agreeing proves only that the code was written twice, and both sides being equally wrong is invisible here. Correctness lives in separate gates: against XGBoost's own recorded output (289/289 bit-exact at the margin) and against `mpmath` per side (D046, D048). The harness answers exactly one question, and `test_the_harness_carries_no_tolerance_of_any_kind` scans both its files for `1e-6`, `isclose`, `approx`, `atol`, `rtol` so the accuracy bound cannot leak in even as a fallback.

**Transport is measured, not assumed.** Values cross the boundary as uint32 hex bit patterns. Eleven probe values also cross as plain JSON numbers *in the same round trip*, and that control **loses four of them**: `-0.0` arrives as `0`, and `+inf`, `-inf`, `NaN` all arrive as `null`. So the encoding choice is demonstrated rather than argued. A transport that quietly normalized a value would let the harness report perfect parity while hiding a real difference.

**Signed zero is why the margin point cannot be dropped.** The signed-zero fixture's margin is `0x80000000` on both sides, but its *output* is `0x3f000000` — sigmoid maps both signed zeros to one half, so the sign is observable at the margin and nowhere after it. Injecting `0x80000000 → 0x00000000` yields margin-point mismatches and **zero** output-point mismatches, which is the asymmetry a single-point harness would miss entirely.

**Nine of the 24 tests exist to prove the harness can fail**, because a parity harness that has never been seen to fail is not evidence of anything: one ULP at each point separately, negative-zero normalization, a side that stops refusing, a side that refuses with a different error, an injected `objective` branch, a side fed different inputs, a malformed bit-pattern token, and a silently dropped fixture. Each asserts the point name, fixture and row, and that the *other* point stays clean.

**A real defect the injection work found in the harness itself.** The first injection helper emitted `0x-40f572f5`, because JavaScript's `& 0xffffffff` yields a signed int32, and the comparison string-compared it happily. **Two malformed tokens that agreed would have been credited as parity.** Tokens are now validated against the uint32 pattern and against value/refusal exclusivity before any comparison, and five malformed spellings are pinned. This is the independent-oracle principle applied to the comparison layer: a check that cannot distinguish "equal" from "both unparseable" is not a check.

**Refusal kind is part of the gate**, not merely refusal-versus-value: both sides name `NonFiniteFeatureError`, and a one-sided rename fails. Stricter than the brief required, and kept.

**Operational consequence, worth knowing.** `uv run pytest` now requires `node` and a current `packages/js/dist/`, and the harness *refuses* a bundle older than `packages/js/src/` rather than measuring stale code. That is intended — a parity number measured against a stale bundle describes code that is not the source — but it introduces an ordering relationship between the two suites that did not previously exist.

---

## D050 — Schema, docs, release configuration, and the licence that was missing

*2026-08-05*

**JSON Schema** at `schema/xgboost-bridge-v1.schema.json`, draft 2020-12, `additionalProperties: false` at every level because the format raises on unrecognized keys and the schema must agree. All **23** fixture artifacts validate. Six mutations are pinned as rejections — an eighth top-level key, a missing required key, `format_version: 2`, a sixth tree key, a numeric `provenance.base_score`, and an empty `feature_names` — because a schema that accepts everything passes a corpus exactly as well as a correct one does.

The JavaScript side asserts the same structural invariants by hand rather than gaining a schema-validation dev dependency, and additionally cross-checks the schema's own key sets against what the fixtures carry, so schema and corpus cannot drift apart unnoticed.

Two `description` fields are required content rather than decoration: `objective` is documented as non-operative metadata that no predictor branches on (D028), and `intercept` as the single operative numeric value, with `provenance.base_score` read by nothing (D015).

**A LICENCE file was missing entirely**, while `packages/js/package.json` and `packages/python/pyproject.toml` both declared `MIT`. Two manifests asserting a licence that the repository does not contain is a real release blocker, not a formality — it is also the kind of gap that no test would ever surface. MIT text added.

**Release configuration exists and cannot fire.** `workflow_dispatch` only — no push, tag, or release trigger anywhere in the file — both jobs gated on an `environment: release` that does not exist, and neither has working credentials. Recorded honestly in the workflow itself that GitHub auto-creates a referenced environment without protection unless a human configures a reviewer rule, so that gate is only real once someone does. All four CI actions are now SHA-pinned to commit hashes resolved from the GitHub API (D024), none invented.

**CI now needs Node in the Python job.** The parity harness shells out to `node` against a current `packages/js/dist/`, so `pytest` no longer runs standalone. Verified the requirement was real by deleting `dist/` and watching 20 parity tests fail rather than assuming.

**That guard then caught me.** After editing `packages/js/src/types.ts` I ran the suite without rebuilding, and 23 tests failed — the staleness refusal from D049 working as designed. A parity number measured against a stale bundle describes code that is not the source, and this is the first time the guard fired against a real mistake rather than a test.

**One correction to my own earlier work.** I had reported the stale `types.ts` header comment as fixed when the JavaScript predictor landed. It was not: the sentence wraps across two lines and my single-line replacement silently matched nothing, because I did not assert the substitution took effect. Fixed now, with an assertion.

---

## D051 — Audit findings: three code defects and four packaging defects, and how an install simulation found them

*2026-08-05*

Three independent adversarial audits ran against a fully green build (882 Python tests, 106 Node, parity `0.0`). **None could produce a wrong number from the numerical core** — the arithmetic held on every input class attacked, including a 3.4-million-value exhaustive sweep of the sigmoid subnormal band and a 10,000-row differential fuzz over 400 generated artifacts. The defects were all at edges the tests do not reach.

### The finding that mattered most, and why no test could see it

**`pip install "xgboost-bridge[export]"` was broken out of the box.** The extra declared `xgboost>=3.3,<4`; xgboost 3.4.0 is released on PyPI, so a clean install resolved 3.4.0 and the *first* `export_model` call raised `UnsupportedVersionError`. Confirmed by building the wheel and installing it into a fresh virtual environment.

All 882 tests passed throughout, because the `uv` workspace pins `xgboost==3.3.0`. **Every test in this repository runs against the source tree; nothing was ever tested as an installed package.** That makes an entire class of defect structurally invisible — a broken dependency spec, a file that never ships, a module that imports only because `src/` happens to be on the path.

Fixed by pinning the extra to `xgboost==3.3.0`, matching the enumerated ceiling of D018 exactly. Not `<3.4`: 3.3.1 would also be untested and would also raise, so the dependency spec now equals the tested list. Widening it requires a probe — including the version-sensitive clamp constants of D032 — which is scope, not a fix.

### Code defects, all fixed

1. **The mandatory neutralization self-check existed nowhere.** FORMAT.md §8.3 and D027 both state it as a MUST; `export.py` never walked a tree or called `predict`. The uncovered failure is the one §8.3 names — a neutralization that clears a *live* node — and reading `base_weights` instead of `split_conditions` (off by `5.10` in margin) would also have passed everything. Aggravating: no export-level test had ever seen a tree with a dead node, because the fitting helper never set `gamma`. Now implemented, with a sample that is provably complete rather than probabilistic: one boundary row per live leaf, verified against XGBoost's own `pred_leaf`. Measured detection over **every single-live-node corruption on four models: 421 detected, 0 missed.**
2. **`predict.py` coerced feature values with bare `float()`**, so the string `"nan"` was accepted and routed down the missing-value branch, giving a margin bit-identical to a real `NaN`. That is D005's failure mode one level down: the key set was compared exactly, then the values went to Python's most lenient constructor. A caller reading rows from CSV got a plausible prediction with no error. Now only real numbers are accepted, `bool` refused explicitly, everything else raising `InvalidFeatureValueError`.
3. **The JavaScript `Predictor` constructor turned a prototype-chain lookup into a boxed `Number`.** `OUTPUT_FUNCTIONS` was a frozen *ordinary* object, so `outputTransform: "constructor"` resolved through `Object.prototype`, and the public constructor validated nothing. The result serialized as the right number and arithmetic on it gave the right number, but `Object.is` against it failed — exactly this project's failure signature. Now a null-prototype table plus an own-property guard, matching Python's `MappingProxyType`.

### Packaging defects, all fixed

- **Neither distribution shipped the licence**, while both manifests declared MIT. D050 recorded this as closed; it had been closed at the repository root only, and the wheel carried `License-Expression: MIT` with no `License-File`. Now `license-files` in the Python manifest and `LICENSE` in both package directories, verified present in a built wheel.
- **The npm tarball shipped no README and no LICENSE** — `files: ["dist"]` excluded both, so the npm page would have rendered empty. Now 9 files including both.
- **`packages/js/package.json` had no `repository` field**, which npm requires for `--provenance`; the publish step would have failed on that before authentication.

### Measured install behaviour, both registries

| | |
|---|---|
| Base wheel install, numpy only, no repo, no xgboost | **289/289 margin bit-exact** against XGBoost's recorded ground truth |
| Wheel at the declared floor, Python 3.10.20 + numpy 1.24.4 | 289/289 margin bit-exact; transform max ULP unchanged at 1 and 2 |
| `[export]` extra after the fix | resolves 3.3.0, export succeeds, margin bit-exact |
| npm tarball, consumer project, ESM and CJS | both work; signed-zero margin `0x80000000` exact; **0** runtime dependencies pulled in |

The numpy floor result matters beyond packaging: it shows the float32 discipline does **not** depend on NEP 50 promotion, despite FORMAT.md §10.1 discussing it at length.

### The export self-check's cost, measured rather than assumed

Completeness was kept over a bounded sample. Measured: 1000 trees at depth 6, **0.05s**; 300 trees at depth 12 over 60 features, **1.31s**. Export is a one-time offline operation, so ~1s buys a check with no probabilistic gap — which is what "loud failure over silent wrongness" and "exact over tolerant" both point at.

### Findings recorded but not fixed

Deliberately deferred, each with a reason: the **JSON Schema enforces roughly a third of what FORMAT.md §13 requires** (10 of 11 wrong-but-well-formed artifacts validate) — both shipped readers reject 10 and 9 of those respectively, so no wrong number reaches a consumer through them, but a third party validating against the published schema and then walking the artifact themselves is exposed. The **generators' self-checks never run in CI**, including the transform's independent mpmath oracle; the corpus was verified to regenerate byte-identically, so nothing is currently wrong, but a future divergence would not be caught. The **vocabulary scrub's camelCase blind spot is fixed** — see D052; it was listed here as deferred and is no longer. `dist/index.cjs` is **never executed by any test** — it works, verified by hand, but nothing would notice if it stopped. Both **`Predictor` classes are mutable at runtime** while their docstrings imply otherwise. `_version.py` and the Python manifest carry **two unlinked version literals**, so a one-sided bump would silently mislabel the provenance of every artifact from that release.

## D052 — Independent-worktree audit: what an outside reader found that no internal check could

*2026-08-06*

An audit ran from a separate worktree against `7229719`, deliberately not starting from
this repository's own vocabulary scrub or `DECISION_COVERAGE.md` — both were written by
the system that wrote the code, so both encode its blind spots. Six items were acted on.
The pattern connecting most of them is that **the executable checks were correct and the
prose around them was not**, which is D022's failure mode one level out: the record made
it look handled.

### The schema `$id` named a domain nobody owned

`schema/xgboost-bridge-v1.schema.json` declared `$id: https://xgboost-bridge.dev/...`.
That domain returns NXDOMAIN. A JSON Schema `$id` is an identifier and is not required to
resolve, so nothing broke and no test could fire — but the repository is public, and an
unregistered domain in a published schema is registrable by anyone, who could then serve a
different document at this project's canonical identifier.

Moved to `https://raw.githubusercontent.com/anirudhmazumder/xgboost-bridge/main/schema/…`,
which is bound to the account that holds the repository. Pinned by
`test_schema::test_schema_id_is_hosted_where_the_project_actually_controls_the_namespace`,
on the host prefix rather than the exact string so the path stays free to change with the
format version.

### `COMPAT.md` documented a withdrawn defect as current behaviour

`COMPAT.md` stated in two places that the export extra allowed `xgboost>=3.3,<4`, and told
the reader to check `packages/python/pyproject.toml` to confirm it. The manifest says
`xgboost==3.3.0`. The range named is the *exact specifier D051 records as a shipped
defect* — the one that resolved 3.4.0 and made the first `export_model` call raise.
`packages/python/README.md` had it right, so the two user-facing documents contradicted
each other, and a reader following COMPAT.md's own instruction to verify would have found
the opposite of what it claimed.

**Why every check passed for the whole period:**
`test_the_export_extra_pins_exactly_the_enumerated_version_ceiling` pins the manifest.
Nothing pinned the sentence describing the manifest. D051 fixed the code and the decision
record, and left the compatibility document stating the defect.

So the prose is pinned too, by the same comparison:
`test_export::test_user_facing_docs_state_the_same_xgboost_specifier_as_the_manifest`
reads every PEP 440 specifier in `COMPAT.md`, `README.md` and `packages/python/README.md`
and requires it to equal what the manifest declares. `docs/DECISIONS.md` and `probes/` are
excluded on purpose — a historical record has to stay free to quote a specifier that was
withdrawn, which is what this very entry does.

The test's docstring states what it does **not** catch, rather than implying completeness:
a bare version named without an operator is prose it does not read. That limit is written
down because the finding below is what happens when a check's description outruns it.

### The vocabulary scrub overstated its own coverage

The guard was `(?<![A-Za-z])` — not preceded by a letter. Correct for a term at the start
of an identifier segment or after an underscore, and structurally blind to one in the
middle of a camelCase identifier, where the preceding character is *always* a letter.
Measured against the real compiled patterns, four such spellings passed a clean scrub —
a domain noun behind a `get`/`num`/`load` prefix, and an ambiguous term followed directly
by a capitalised suffix. Worse, the comment above `AMBIGUOUS_IDENTIFIER_RE` cited that
last form as covered while the pattern required a `_` or `-` separator and could not match
it. The concrete spellings are in `must_flag`, which is the one place they belong.

**A check that claims coverage it does not have is worse than one that admits a gap**, and
this one was documented as a known limitation under "Findings recorded but not fixed" —
which is how it survived. A recorded gap reads as a managed gap.

Fixed with a second, case-**sensitive** entry form: preceded by a lowercase letter or
digit, term capitalized, so a real camelCase hump matches and ordinary English does not.
The case sensitivity is load-bearing — a global `re.IGNORECASE` would make the capitalized
alternation match a lowercase run and flag `impatient`, the exact false positive D029 says
teaches reviewers to ignore the check. Both are pinned in
`test_scrub_detects_what_it_claims_to_detect`: six camelCase violations in `must_flag`,
`impatient` and `reallocation` in `must_not_flag`.

Turning the guard on flagged exactly one line in the repository — the sentence in this
file that disclosed the gap, which cited the identifiers as examples. **No application
vocabulary existed then or exists now**; an independent sweep by the auditor found none
either. The scrub was measuring nothing, correctly, for the wrong reason.

### Publishing asymmetry: one path verified the artifact, the other published it unread

`publish-javascript` re-ran typecheck, build and the suite before publishing, with a
comment explaining that skipping them because "CI already passed" trusts a different
commit's result. `publish-python` built a wheel and uploaded it with **no test step at
all**. The reasoning was written down in one job and not applied to the other.

Both now run their suite, and both now run their clean-install script — the wheel and the
tarball exercised as installed packages from outside the repository, before upload rather
than after. A PyPI version and an npm version are each permanent once taken; the source
tree passing says nothing about the distribution, which is the whole finding of D051.

Both workflows also moved from `npm install` to `npm ci`. A lockfile that is present and
unenforced lets a publish job resolve dependencies that were never tested.

### Publishing metadata

The Python manifest had no `[project.urls]` and no `authors`, so the PyPI page would have
carried no link back to the source, the format specification, or the evidence. `authors`
records a name and no address deliberately: a release is permanent per version and an
email published to the index cannot be withdrawn from it or from its mirrors.

### Not acted on, and why

The `esbuild` advisory (GHSA-g7r4-m6w7-qqqr) does not apply. **Corrected 2026-08-06,** after a
security audit re-derived it: the original entry got three things wrong and reasoned from the
wrong facts. It is rated **Moderate** (CVSS 5.3), not low; the **Windows qualifier is not in
that advisory at all** — it is that esbuild's dev server sets `Access-Control-Allow-Origin: *`
by default, which is platform-independent; and the lockfile has **esbuild 0.27.7**, while the
advisory affects `<= 0.24.2` and was fixed in 0.25.0, so there was never a risk here to accept.

The one clause that held is that `tsup` is a dev dependency and is in no shipped artifact.

The methodological point outlives the corrections. The entry dismissed the advisory using
properties *of the advisory* — its severity, its platform — rather than the two facts that
actually settle it: **the pinned version is past the fix**, and **this project never starts an
esbuild dev server**, which is the argument that survives a version bump. Zero hits for
`--serve`, `serve(`, `devServer` or `watch` outside `node_modules`; `tsup.config.ts` has no
`watch`; every workflow runs one-shot `tsup`.

And looking the package up in an advisory database was not the same as reading it. The audit
that produced this entry did not open `esbuild/install.js`, which contains the install-time
execution path that mattered: an integrity-free fallback that fetches a tarball over plain
`https.get`, unpacks it by hand, and `chmod 0755`s the result, plus an `ESBUILD_BINARY_PATH`
override that rewrites `lib/main.js`. Dormant, because all 26 `@esbuild/*` platform packages
are sha512-locked and resolve on both CI platforms — but it was live during every `npm ci`,
including inside the jobs that held publishing credentials. That is addressed under D054. The institutional email
in commit metadata is the owner's to decide and would require a history rewrite on an
already-public repository.

### The historical record is exempt — and the exemption arrived one step too late

D052 as first committed treated `docs/DECISIONS.md` two ways in the same change.
The specifier check **excluded** it, so a decision entry could quote a dependency range
that had been withdrawn — correct, and the reason was written down. The vocabulary scrub
**did not** exclude it, flagged this file for naming the identifiers the fixed blind spot
used to miss, and the prose was rewritten to satisfy the check. Same file, same category
of content, opposite rule, one commit.

Settled once, in `packages/python/tests/_policy.py`, and applied by both:

> **The historical record is exempt from any check that would otherwise edit it.**

A decision entry and a probe transcript are evidence. Their value is that they say what
was true at the time — including the wrong value, the withdrawn specifier, and the
identifier that used to slip. A check that forces evidence to be rewritten to stay green
does not make the repository more correct; it makes the record less accurate, and it does
so invisibly, because the edit looks like a passing build.

Scope is `docs/DECISIONS.md` and everything under `probes/`, **prose scanning only**.
Nothing is exempt from a behavioural test, and neither check that consults the policy can
affect a prediction. `FORMAT.md`, `COMPAT.md`, `VERIFICATION.md` and both package READMEs
stay fully in scope for both checks — they are documents a user acts on, not records of
what was once believed. Measured, not assumed: a domain term added to `docs/DECISIONS.md`
and to `probes/accumulation.md` leaves the scrub green; the same term in `COMPAT.md` or in
`packages/js/src/predict.ts` fails it.

`test_export::test_the_pinned_documents_are_not_part_of_the_historical_record` is where
the two checks meet. Without it, the specifier check's document list is an independent
allow-list that merely happens to agree with the scrub's exemption; adding a historical
file to it would demand that a superseded entry be edited to match the current manifest,
and nothing would have objected.

**The rewrites this exemption would have made unnecessary are deliberately kept.** Under
the rule now in force, the earlier sentence naming the four camelCase identifiers, and the
first draft of this entry, would both have passed untouched. Reverting them to prove the
point would be a third edit to the same record for the convenience of the tooling, which
is the behaviour the rule exists to stop. The concrete spellings live in `must_flag`,
which is where an executable check belongs, and this paragraph is the record that the
exemption arrived after the edits rather than before them.
## D053 — The intercept is read out of the engine, because upstream is not portable

*2026-08-06* — **supersedes D015's derivation as normative, narrows D040, and retires the export-time intercept comparison of D034**

`export_model` no longer computes the intercept from `base_score`. It reads XGBoost's own
value via `objectives.observe_intercept`. `objectives.derive_intercept` still documents how
the engine reaches that value, is still fully tested, and is **off the export path**.

### The measurement

XGBoost derives this intercept with the platform's `logf`. IEEE-754 requires correct
rounding only for `+ − × ÷ √` and fma; `logf` is not covered. Measured on 58 inputs chosen
because they discriminate the candidate routes — agreement with XGBoost's observed intercept:

| route | darwin/arm64 | linux/x86_64 |
|---|---|---|
| `np.float32(np.log(f32))` — what the exporter used | **58/58** | 36/58 |
| `np.float32(math.log(f64))` | 10/58 | 39/58 |
| `Decimal.ln()` at 40 digits | 10/58 | 39/58 |
| `mpmath` at 60 digits, correctly rounded | 10/58 | 39/58 |

The 10 are control values where every route agrees. **No route is exact on both platforms,
and the correctly-rounded route is exact on neither** — XGBoost is not correctly rounded
either, so implementing a correctly-rounded logarithm would be equally wrong.

Diffing XGBoost's own column across platforms: it differs on **29 of 58**, always by exactly
1 ULP, darwin higher on 15 and lower on 14. Not a systematic offset — last-place rounding.
`reg:squarederror` takes the identity link and is bit-exact everywhere by every route, which
localises this to the logarithm. Full evidence in `probes/platform_log.md`.

### Why this resolves the way it does

"Bit-exact against XGBoost" and "platform-independent" are not a choice of recipe here —
they are incompatible, because upstream itself is not reproducible across platforms for
these two objectives. A 1-ULP intercept error is **silent**: it shifts every margin the
model produces, with no error raised. The ordering says match upstream, or refuse the case
entirely, where divergence would be silent. Reading the value out of the engine satisfies
that on every platform by construction; deriving it guarantees a spurious refusal somewhere.

The previous behaviour was not *unsafe* — the export gate refused rather than emitting a
wrong number, which is the intended failure mode. It refused **valid models**, on 13 of 50
ordinary `binary:logistic` `base_score` values on Linux, and it meant the suite could only
pass on one platform.

Inference is untouched: the intercept is a stored field, neither predictor computes a
logarithm, and parity remains exactly `0.0` at both measurement points.

### What checks it now, and why the old check had to go

The intercept comparison is retired rather than relocated. Comparing the engine's value
against the engine's value is the decorative check the independent-oracle principle rejects,
and keeping it would have been a check that structurally cannot fire.

What stands behind the intercept is `export._verify_against_source_margin`: the assembled
artifact must reproduce XGBoost's own `predict(output_margin=True)` bit-for-bit on rows
chosen to reach specific leaves. That oracle is independent of the value's provenance, and
it is **strictly stronger** than what it replaces — it covers the trees and the accumulation
as well, and it validates the value after emission rather than before.

It had teeth for tree defects already; it had **none for an intercept error**, which is now
the failure it is the last line against. Two tests perturb the artifact's intercept by one
ULP in each direction and require `MarginMismatchError`. Neutering the check turns five
tests red, two of them these.

One honest gap, stated rather than glossed: for a **zero-tree** model the margin *is* the
intercept, so that check is tautological there. There is no derivation left to be wrong
about — the artifact's only number is the engine's own answer, copied — but it is not
verification and is not presented as such.

### What this does *not* claim

D040 stands as a measurement and is **narrowed as a generalisation**: `np.log` of a float32
is the route XGBoost takes *on darwin/arm64*, not the route XGBoost takes. Its own
methodological lesson was then walked into one level up — a first pass at this investigation
used 13 hand-picked values, found every route in agreement, and concluded that XGBoost was
correctly rounded and numpy was the mover. Both halves were false. The rule earns restating:
**a sample that does not deliberately target the disagreeing inputs cannot distinguish two
implementations, and its silence is not evidence of equivalence.**

Only two platforms are measured. A third may agree with neither, and the design is
indifferent to that by construction — which is the point of observing rather than deriving.

### A tolerance appears in this entry, and it is not drift

A future reader will find the phrase "within 1 ULP" below and should not read it as a
loosened gate. Two different things are being measured, and only one of them protects the
artifact:

| | before | after |
|---|---|---|
| **The artifact's intercept**, which every prediction depends on | derived, then required bit-equal to the engine — *before* emission | taken from the engine, and required bit-equal to the engine's `predict` on leaf-reaching rows *after* emission |
| **`derive_intercept`**, a reference function that determines no shipped number | required bit-equal to the engine | required within 1 ULP of the engine |

The gate that protects the artifact was made **stronger**, in two ways: it now validates the
value after emission and serialization rather than before, and its oracle covers the trees and
the accumulation as well as the intercept. A single-bit intercept error fails it, pinned in
both directions.

The tolerance sits on `derive_intercept`, and the reason it must is not convenience: that
function is **provably unable to be exact on both platforms**, because the thing it targets —
XGBoost's `logf` — gives two different answers on the two platforms measured, and neither is
the correctly-rounded one. Requiring bit-equality there does not buy strictness; it buys a
suite that can only pass on whichever machine the constants were recorded on, which is the
defect this entry exists to fix. The bound is 1 ULP rather than a percentage precisely so it
stays a tripwire: a wrong space, a missing clamp, or an unsnapped `base_score` moves the
result by far more than one float32 step and still fails.

Two further re-pinnings in the same spirit, neither of them a relaxation of what is caught:
the Cox NaN test now asserts quiet-NaN-and-refused instead of a **sign bit that IEEE-754
leaves unspecified** for `log` of a negative, and it additionally asserts the refusal, which
it did not before. The snapping test moved from `0.7`, where the claim was a 1-ULP
coincidence, to values where the routes separate by 10, 110 and 116804 ULP, and it now
asserts its own discriminating power so it cannot silently stop testing anything.

The general rule this yields: **when a check compares our value against an upstream value
that is itself not reproducible, exactness is not available and pretending otherwise pins the
platform rather than the behaviour.** Move the exact requirement onto something that *is*
reproducible — here, the engine's own output on the machine doing the export — and bound the
non-reproducible comparison explicitly.

### Test consequences

All 18 first-run Linux failures were this defect in three shapes, and each was re-pinned to
a claim that holds on both platforms rather than loosened:

- **Thirteen** pinned one libm's answer as XGBoost's. Now: the shipped value is exact
  against the engine, and the recipe is asserted within 1 ULP of it. The shipped gate is
  *stronger* than before, since it covers emission; the recipe's bound is weaker, on a
  function that determines no shipped number.
- **Two** pinned a NaN's sign bit, which differs by platform (`0x7FC00000` versus
  `0xFFC00000`) and which IEEE-754 leaves unspecified for `log` of a negative. Now pinned as
  a class — quiet NaN, not finite, refused — plus the refusal itself, which was not asserted
  before.
- **Three** were 1-ULP claims that a 1-ULP platform difference erases. `base_score` snapping
  was pinned at `0.7`, where the snapped and unsnapped routes separate by exactly one step.
  Re-pinned at `0.99`, `0.999` and `0.999999`, where they separate by 10, 110 and 116804,
  and the test now asserts its own discrimination so it cannot silently stop testing.

Two tests that pinned *counts* — which `base_score` values the textbook logit gets right,
and how many values each log route matches — now report those counts and assert the failure
signature instead. The counts are properties of a libm, and pinning them is what made the
tests darwin-only.

Python suite 960 → 976 for this change, and 977 with the concurrently-merged audit work of D052.

## D054 — Security audit: the credential outlived the gate, and three checks read wider than they reached

*2026-08-06* — **corrects D052's esbuild assessment; supersedes nothing numerical**

A read-only security audit ran against `c6eff20` from an independent context, covering supply
chain, workflow permissions, environment gating, untrusted-input handling in both readers, and
the built distributions re-derived rather than taken from the packaging scripts' own output.
Fifteen items; the ones acted on are below. What it verified as sound is recorded at the end,
because that list is as useful as the findings.

### The finding that mattered: gate ordering is not a control

Both publish workflows were a single job that declared `id-token: write` and then ran `npm ci`,
`uv sync`, the suites and two build scripts *before* the publish step. `release-testpypi.yml`'s
own header called that a safeguard: "the full gate runs BEFORE anything is published."

It was not one. `ACTIONS_ID_TOKEN_REQUEST_URL` and its token are present for a job's entire
duration, so a dependency executing an install script during `npm ci` could have minted a
publishing token and uploaded its own artifact at step one — and the gate would still have
passed afterwards. Step order constrains nothing when the credential does not depend on it.

**This is the independent-oracle principle with a different noun.** A gate and the thing it
gates must not share a trust domain; when they do, the gate cannot constrain what holds the
credential, exactly as a check cannot catch a defect its oracle shares.

Both workflows are now three jobs: `verify` holds no credential and cannot publish, `publish`
holds `id-token: write` and does nothing but download an artifact and call the publisher, and
the TestPyPI round trip runs credential-free afterwards. The publisher action is third-party
code inside the credentialed job — unavoidable, it is what uploads, and it is SHA-pinned.

Also: `release.yml`'s two jobs declared only `id-token: write` with no top-level `permissions`
block, so `contents` was `none` and `actions/checkout` would have failed on the first ever run.
Fixed, and `publish-javascript` now `needs: publish-python`, so two irreversible uploads cannot
disagree about what 1.0 is.

### The environment gate did not exist, and one file said it did

`gh api .../environments` returns `total_count: 0`. GitHub creates an environment on first
reference **with no protection rules**, so the first dispatch would have published unreviewed.
`release.yml` was honest about this; `release-testpypi.yml:17` listed "`environment: testpypi`
carries a required reviewer" among the reasons it "cannot fire accidentally". That was a spec
contradicting the evidence, in the one file that performs an irreversible upload.

Rewritten to separate what the file configures from what repository settings must supply, and
to record that the Trusted Publisher entry must itself name the environment — an entry with a
blank environment field accepts a token minted from any environment, or none, so the binding
has to exist on both sides to mean anything. Creating the environment and the publisher remains
the owner's, and is not done.

### Three checks whose message read wider than their reach

The pattern connecting them is D022's, one level out: the executable check was correct and the
prose around it was not, so the record made it look handled.

- **The dependency count read one field.** `ci.yml` counted `dependencies` while its comment
  claimed to guard against `package.json` being edited at all; a `peerDependencies` or
  `optionalDependencies` entry passed silently. Now every install-time field.
- **The sdist was never opened.** `tools/clean_install_python.sh` located the sdist, printed
  its name, and checked only the wheel — so hatchling's default scope shipped all eleven test
  modules inside the tarball. Never a wrong number, because `packages` scopes the wheel build
  and an sdist install still installs no tests. Now scoped in the manifest and inspected by the
  script. `.gitignore` still ships: hatchling includes it regardless of `include` and of an
  explicit `exclude`, measured rather than assumed, and the comment says so.
- **"No source files in the tarball" was true only on a technicality.** `sourcemap: true` puts
  all four source files verbatim inside `dist/*.map`, about 180 kB of the 294 kB published. The
  check tested for `src/` *paths*. No local-path leak — `sources` is relative, and a grep of
  every `dist/` file for the owner's directory names and email finds nothing — and the
  repository is public MIT, so shipping source is defensible. The message now says what it
  checks.

### Install-time code execution removed from the publish path

`esbuild` and `fsevents` are the only two packages with install scripts, and no workflow passed
`--ignore-scripts`, so `esbuild/install.js` ran during every `npm ci` — including inside the
jobs that held publishing credentials. Now `--ignore-scripts` in all three workflows and in
`tools/clean_install_js.sh`, **verified** by moving `node_modules` aside and confirming a clean
`npm ci --ignore-scripts` still builds, type-checks and passes 114 tests: the binary comes from
a locked, hashed platform package, so the script had nothing to do.

Two unpinned links on that path are now pinned: `[build-system] requires = ["hatchling"]`, which
built the PyPI wheel from whatever version the index served that day, and
`npm install typescript@5` inside the consumer-compile check.

### Untrusted input: one contract breach, one uncatchable hang

- **A JSON integer too large for a float64 escaped as `OverflowError`.** `1e400` was already
  refused — it parses as a float and arrives as `inf`. A 401-digit integer *literal* is also
  valid JSON, is under CPython's 4300-digit parse ceiling, and arrives as a Python `int`;
  `np.float32` of it raises before any refusal in `predict.py` runs, and `np.errstate` does not
  cover it because it is a Python conversion error rather than a numpy warning. A caller
  following the documented contract and writing `except XGBoostBridgeError` got an unhandled
  exception. Now structured at both cast sites, pinned by five tests including one asserting the
  input still overflows an unguarded cast — otherwise a future numpy returning `inf` would make
  the guard pass for the wrong reason. **The JavaScript reader never had this**, because
  `JSON.parse` returns float64 unconditionally and the same input arrives as `Infinity`; that
  asymmetry is the parse-precision seam the format already cares about, appearing from the other
  side.
- **The public `Predictor` constructor did not establish termination.** `fromJSON` runs the
  cycle check, but the constructor is exported and a hand-built `LoadedArtifact` with a cyclic
  child set did not throw — it spun, and the process had to be killed. Not reachable from
  `fromJSON`, and not from handing it a parsed artifact, so it is an API footgun rather than an
  untrusted-input path; but it is the only malformed input in this package whose consequence a
  caller cannot catch, which is why `artifact.ts` refuses it at load. The constructor now runs
  the same check. That repeats work `fromJSON` already did — one extra O(nodes) pass on a path
  that is not the hot one, chosen over a flag recording whether validation happened, which is
  the kind of coupling that goes stale. The test carries an explicit timeout, because its
  regression mode is the hang it exists to prevent and an untimed test would stall CI rather
  than fail it.

### Smaller items

`probe-platform.yml`'s comment claimed the dispatch input was "validated against the actual
directory listing"; the guard was `[ -f "probes/$X" ]`, which accepts `../`. It granted no
privilege — dispatch already requires write access, the job has `contents: read`, no secret and
no environment — but an asserted validation that does not exist is what stops the next reader
from checking. Now matched against the listing itself. The `env:` indirection was already the
correct pattern against injection and is unchanged.

`.claude/worktrees/` was excluded only by `.git/info/exclude`, which is per-clone and does not
travel; moved into the tracked `.gitignore`. The two private orchestration files stay where they
are, deliberately — an entry in a tracked `.gitignore` would itself be evidence they exist.

Two tracked agent files instructed a contributor with beliefs this repository records as
measured-false: `numeric-core.md` gave the logistic intercept as `logit(base_score)`, which
`CLAUDE.md` marks superseded, and `numeric-reviewer.md` carried a garbled remnant of the
two-signal dart premise. Both corrected, with the D053 engine-read rule added.

### Left to the owner, and not done

A required-reviewer rule on the `testpypi` and `release` environments; the PyPI and TestPyPI
Trusted Publisher entries, each naming its environment; branch protection or a ruleset on `main`
(without one, `workflow_dispatch`-only is a real control against *accidental* firing but not
against a compromised account, since anyone with write access can edit a workflow and dispatch
it); and npm Trusted Publishing, which would remove the planned `NODE_AUTH_TOKEN` and make the
npm side symmetric with PyPI's. No secret exists in the repository today — verified, zero.

### Verified sound, and worth recording as such

No prototype-pollution sink in either reader: `__proto__` at every level is refused, and
`Object.prototype` is unpolluted after every attempt. `OUTPUT_FUNCTIONS` is `Object.create(null)`
and frozen, so `"constructor"` and `"valueOf"` are refused as transform names. Cycles, 2-cycles,
and a back-edge at the end of a 100,000-node chain are all refused in milliseconds; a 200,000-node
chain and a 200,000-node overlapping DAG both load and walk without amplification. Allocation is
bounded by actual parsed length everywhere — no length field drives an allocation. `elementAt`
converts an out-of-bounds typed-array read into a raise rather than letting `undefined` become
`NaN` and route down the missing-value branch. An unreachable node with out-of-range links is
refused, which is *stricter* than FORMAT.md §13 requires, and in the right direction.

Zero JS runtime dependencies confirmed by walking the lockfile rather than trusting the field:
100 entries, 99 `dev: true`, the hundredth is the root. All 99 resolve to `registry.npmjs.org`
with sha512 integrity; `uv.lock`'s 17 registry packages are sha256-hashed on sdist and every
wheel. No package from the recent `chalk`/`debug`/`cross-spawn` incident families. Nothing is
fetched at consumer install time by either package, and neither has any lifecycle script. All
four pinned action SHAs were dereferenced against upstream and match the tags their trailing
comments claim. No `pull_request_target`, no `workflow_run`, no cache configured anywhere. The
Python package contains no `eval`, `exec`, `pickle`, `subprocess`, `open`, or network import,
and executes nothing at import time. No secret, token or key anywhere in the tree or in any of
464 objects across all five refs. `CLAUDE.local.md` and `PLAN.md` are absent from history, from
all tracked content, from `.gitignore`, and from all three built distributions.

## D055 — A feature value infinite *in float32* is refused, the same as an explicit infinity

*2026-08-06* — **narrows D022's refusal to cover the case it missed**

`walk_margin` and the JavaScript row reader now narrow each feature value to float32 and
refuse an infinite **result**. Previously both tested the float64 for `±inf`, so a finite
float64 that becomes infinite through this library's own required narrowing went through.

### The inconsistency

`1e39` is a legal float64. `f32(1e39)` is `+inf`. The walk then compared `inf` against
thresholds, routed consistently, and returned a number — while an explicitly infinite input
raised. **Same mathematical value by the time any comparison happens, two behaviours, and no
error either way.** Both predictors agreed on those rows, so it was never a cross-language
defect; it was a hole in the refusal, which is a worse thing for this library to have.

The evidence that it was unintentional rather than chosen: numpy emitted
`RuntimeWarning: overflow encountered in cast` **1,160 times** in a 100,000-row parity run, on
exactly the rows being accepted. A deliberate behaviour does not warn about itself. Nothing in
`FORMAT.md`, `COMPAT.md` or D022 mentioned the case, in either direction.

### Why refuse rather than accept

Value ordering rank 1, loud failure over silent wrongness. This library compares in float32;
that is not an implementation detail but the central invariant. A value that cannot be
represented as a finite float32 has no defined position relative to any threshold, and routing
it as `+inf` is a decision no measurement supports — XGBoost is *itself* inconsistent about
infinite features (raising through `DMatrix`, comparing ordinarily through `inplace_predict`),
which is exactly why D022 refused them in the first place. That reasoning covers `1e39`
identically; the original rule simply did not reach it.

Narrowing the set of accepted inputs toward loud failure is not gate-weakening. Nothing that
previously produced a *number* now produces a different number: it produces a raise.

### Implementation, and the two things that had to be got right

`np.isinf(np.float32(value))`, not `not np.isfinite(...)` — `NaN` narrows to `NaN` and must stay
**accepted**, because it is the missing value and routes by the tree's default direction. In
JavaScript, `Math.fround(value) === Infinity || === -Infinity`, for the same reason.

And narrow-then-test rather than a magnitude bound: `abs(value) > f32_max` would reject
`3.4028234663852886e38`, the largest finite float32, which is an ordinary input. Both
directions are pinned — six values refused, three accepted, per language.

The `np.errstate(over="ignore")` around the check is deliberate: the cast warns on exactly the
inputs now being refused, and emitting a numpy warning *and* raising is two reports of one
problem when only the raise is actionable. Pinned by a test asserting no `RuntimeWarning`
escapes. The 100,000-row run now reports **`python-side warnings none`**, where it previously
reported 1,160.

### What it cost, measured

Nothing in the corpus. Checked explicitly rather than assumed: **0 of 299 fixture rows** carry a
value that is finite as float64 and infinite as float32, so no recorded ground truth is
invalidated and no fixture needed regenerating. Corpus parity stays `0.0` with the same 10 rows
refused by both sides; the 100,000-row run stays `0.0` with refusal agreement intact, the newly
refused rows now counting as refusals rather than as matching values.

Suites: Python 982 → 992, Node 114 → 124.

Decided before rc deliberately. Refusal semantics are the published contract, and narrowing them
after a release is a breaking change.

## D056 — The npm package ships sourcemaps, and that is a choice rather than a default

*2026-08-06*

`sourcemap: true` stays in `tsup.config.ts`. The security audit surfaced it as an undecided
default: `dist/*.map` carries all four TypeScript sources verbatim in `sourcesContent`, 180 kB of
the 293 kB published, and nothing recorded whether that was intended.

### What was measured

| | with maps | without |
|---|---|---|
| Tarball, compressed | **79.8 kB** | 31.2 kB |
| Unpacked | 292.9 kB | 113.4 kB |
| Files | 9 | 7 |

So the cost is 48.6 kB compressed, 179.5 kB on disk.

### Two facts that decide it

**The bundle is not minified.** `tsup.config.ts` sets no `minify`, so `dist/index.js` ships as
readable JavaScript with every identifier intact — it is the TypeScript with types stripped and
modules concatenated. "Source visible to anyone who fetches the package" is therefore already
true without the maps, and the repository is public MIT besides. Source *visibility* is not a
consequence of this decision; only size is.

**esbuild strips the comments.** 73 comment lines survive in `dist/index.js`; the load-bearing
rationale does not — the note explaining that the threshold-side `Math.fround` is idempotent
given the `Float32Array` (D045) is in the source and absent from the bundle. `sourcesContent` is
the only place a consumer can read it.

That is decisive *for this library specifically*. Its users' characteristic problem is a last-bit
disagreement with XGBoost, and the comments are where the measurements that explain such a
disagreement live — the platform-`logf` story of D053, the bundled-transform story of D032, the
equality-routes-right rule. A package whose value proposition is "we can tell you why the number
is what it is" should ship the reasoning next to the code.

The size objection is also weaker than it looks: a `.map` is fetched only when devtools are open
(`sourceMappingURL`, confirmed present in both bundles), and bundlers do not emit `.map` into
production output. The 48.6 kB is install-size for a zero-dependency package, not runtime payload
for an end user.

### What changes

Only that it is now decided and enforced. `tools/clean_install_js.sh` previously printed "no
source or test files in the tarball" while the source shipped inside the maps — true of `src/`
*paths* only. The message now says what it checks, and the check additionally asserts the maps
are present, carry `sourcesContent`, and contain no absolute path — so the properties that make
this decision safe are pinned rather than re-verified by hand each release.

Revisit if the bundle ever becomes minified, which would make the maps load-bearing for
debuggability rather than merely useful, or if `sourcesContent` ever starts carrying something
other than the four source files.

## D057 — Versions bumped to `1.0.0rc1` / `1.0.0-rc.1`, and the Python pair is now one literal

*2026-08-06* — **closes the two-unlinked-literals item deferred in D051 and D052**

`1.0.0rc1` on PyPI (PEP 440) and `1.0.0-rc.1` on npm (semver), so `1.0.0` stays unspent on both
indexes. Neither rc version can ever be reused: PyPI and TestPyPI refuse re-upload of a filename
even after deletion, and npm forbids republishing a version after unpublish. Spending an rc
spends only that rc.

### The version now has one home per package

`packages/python/pyproject.toml` no longer carries a copy. It declares `dynamic = ["version"]` and
hatchling reads `src/xgboost_bridge/_version.py`, which is the module that stamps
`provenance.exporter_version` into every artifact. D051 and D052 both recorded the two literals as
a deferred risk — "a one-sided bump would silently mislabel the provenance of every artifact from
that release" — and a hand bump is exactly the operation that invites it, so it is closed here
rather than performed carefully once more.

### Proving the bump moved nothing numeric

Asserted rather than assumed, because "only the version changed" is the kind of claim that is
usually true and occasionally not. Every fixture was captured before and after:

| | fixtures changed |
|---|---|
| Margin bit patterns (299 across 23 fixtures) | **0** |
| Output bit patterns (299) | **0** |
| Margin and output decimals | **0** |
| `intercept` | **0** |
| `node_values` — thresholds and leaves, 3258 values | **0** |
| `provenance.exporter_version` | **23 of 23** |
| Whole file bytes | 23 of 23 |
| **Canonical content with `exporter_version` removed** | **0 of 23** |

That last row is the proof: with one field deleted, all 23 regenerated fixtures are byte-identical
to their predecessors. Corpus parity stays `0.0`, the 100,000-row run stays `0.0`.

### A gap the bump exposed

The full suite passed with all 23 fixtures stamped `0.1.0.dev0` while the installed package
reported `1.0.0rc1`. **Nothing checked fixture provenance against the exporter under test** — so
the stale-provenance case D052 called a risk was live, and observable, and silent. Two tests now
close it: every fixture's `exporter_version` must equal `__version__`, and the corpus must carry a
single value (a split corpus is how half the fixtures end up describing one exporter and half
another, which is what a partial regeneration produces). Both were confirmed to fail on a
deliberately staled fixture and pass after regeneration.

This also makes a future bump auditable rather than trusted: the suite goes red until the corpus
is regenerated, so a version cannot ship with fixtures claiming the previous one.

Suites: Python 992 → 994, Node 124.

## D059 — The AI-authorship disclosure is one line, identical in all three READMEs

*2026-08-06* — **discharges D012**

```
**AI Disclosure:** Claude Code was used to help implement this project.
```

Last line of `README.md`, `packages/python/README.md` and `packages/js/README.md`. Byte-identical
in all three — verified by hash, not by eye.

### Why identical matters more than the wording

The two package READMEs are **frozen into the wheel and the npm tarball at publish time** and
render on the PyPI and npm project pages permanently for that version. A wording difference
between them is not correctable after release; it can only be superseded by a new version. So
sameness is the property under test, and it is checked by a test rather than by review.

### Why one line rather than the section it replaces

Both package READMEs previously carried a three-paragraph section, and both ended by linking to
`README.md#how-this-was-built` for "full disclosure and the supporting links". That anchor was
removed when the root section was, so **two published-facing documents were pointing at a heading
that no longer existed** — the reason this entry checks for dangling references rather than
assuming there were none.

The longer version was also redundant with the repository itself. `CLAUDE.md` is at the root, the
agent definitions are tracked under `.claude/agents/`, and every commit carries a
`Co-Authored-By` trailer. Anyone who wants the fuller record has it without a README paragraph
summarising it, and a summary that drifts from what it summarises is worse than a pointer.

### What was cleaned up alongside

The root README kept an orphaned sentence after its section was removed — *"The same standard was
applied to its authorship: where the model could not establish something, the repository says
so"* — referring to a section that was no longer there. Removed.

All links in all three READMEs re-checked: **0 broken**, counting relative paths, absolute
`blob/main/` URLs against the working tree, and in-document anchors against their own headings.
The four remaining external links are the PyPI and npm project pages, which 404 until first
publish and are expected to.

## D058 — Seventeen findings from two adversarial sweeps: the recurring class, worked

*2026-08-06* — **narrows D021's reachability prose, resolves a contradiction in FORMAT.md §13, inverts the shared-subtree allowance, corrects D048's stale note and D051's mutability item**

Two read-only sweeps, one per language, hunting a single defect class: **a validator
establishes less than its consumers assume.** Sibling forms: documented behaviour with no
pinning test; a check whose comment reads wider than its reach. Four instances were already
known — `walk_margin` refusing nothing where D022 said it refused, a dependency count reading
one manifest field, a packaging script printing the sdist's name without opening it, and a
constructor establishing termination but nothing else.

### The two that could produce a wrong number

**A DAG was accepted as a tree, in both readers, returning the same plausible margin.** Child
indices in range, reachable subgraph acyclic, every node consistently internal or leaf — and two
parents pointing at one child. Because *both* languages accepted it and agreed on the number,
cross-language parity could never have caught it. Agreement is not correctness; that is the
premise of this project's gate structure, demonstrated against itself.

Both readers now refuse it, counting in-edges during the traversal that already runs.
`trees.extract_trees` refuses it too, where the consequence was different and worse: export's
self-check enumerates root-to-leaf *paths*, which equals the leaf count only for a tree, so a
hand-built forward-pointing diamond chain made `export_model` **hang** — 18 nodes 0.006 s, 26
nodes 0.265 s, 34 nodes 14.18 s, 60 never returned. A hang is the one failure a caller cannot
catch.

Measured before deciding: maximum in-degree across all 582 fixture trees and 3258 nodes is **1**,
so nothing this exporter produces is affected. This **inverts a considered position** — a Python
test asserted the permissive behaviour and checked the DAG's exact margin bits, and
`artifact.ts` documented it as deliberate. Both are superseded, and the test is inverted rather
than deleted, with a companion asserting an ordinary tree still loads so a `>= 1` typo cannot
pass.

**The JavaScript `Predictor` constructor did not establish finiteness.**
`new Predictor({...loaded, intercept: Infinity}).output(row)` returned **`1`** — a plausible
probability. `fromJSON` always enforced it; the constructor is exported and validated only
`outputTransform`. Empty `featureNames` was reachable the same way, which **voids D021** in that
decision's own words: "a strict-key policy with no keys to check reads as enforced and is not."
Duplicates supplied one column twice. All four are now established there.

### The asymmetry in refusal semantics

A loaded JavaScript model was **mutable**: `p.intercept = 100` changed every prediction,
`p.trees[0].splitIndices[0] = 1` rerouted a split, and `readonly` on `LoadedTree` is compile-time
only. The Python reader refused each equivalent. Refusal semantics are the published contract, so
this was a difference between the two packages' contracts rather than a considered difference —
and tightening it after 1.0.0 would be a breaking change. The instance and each tree object are
frozen now. Element writes into a typed array remain possible, because `Object.freeze` on a
`Float32Array` with elements throws; that residue is documented rather than claimed away.

D051 recorded "*both* `Predictor` classes are mutable while their docstrings imply otherwise."
Half of that was already false — Python's docstring was accurate. The other half was worse than
recorded, because the Python one *could* be defeated: `node_values.flags.writeable = False` on an
array that owns its buffer can simply be set back to `True`. Handing out a view of a read-only
base makes the claim true, since numpy refuses to re-enable the flag when the base is not
writeable.

### Freezing broke three mutation-based proofs, and the replacements are stronger

Two tests and the parity emitter proved `objective` and `provenance` non-operative by
overwriting them after construction. Freezing made that throw. Each now **constructs** a
differently-labelled predictor instead — which is not merely equivalent: mutating a field after
construction cannot detect a version that reads it *during* construction and caches a transform,
because the cache is already built and the numbers do not move. The old technique could have
passed while `objective` became operative.

### Guards whose docstrings claimed more than they delivered

- **`objectives._observed_margin` passed one row** to a helper that asserts constancy by
  requiring exactly one distinct bit pattern — unconditionally true on a size-1 array. Its
  docstring called the constancy "asserted rather than assumed"; it was assumed. This is the
  export path for the intercept of every zero-tree model, and `_verify_against_source_margin` is
  tautological for such a model, so nothing downstream would have caught a row-dependent margin.
  Two rows now, with distinct values.
- **`validate._check_categorical_splits` searched a truncated array.** `1 in split_type` over an
  array shorter than the node count searches the wrong domain: on a real categorical model,
  clearing the other two signals and truncating `split_type` to `[0]` made this gate **pass**
  while `trees.extract_trees` refused the same document. `trees.py` carried the length check with
  a rationale and a test; the fix had been applied in one of the two modules that read the field.
  `validate_source_model`'s docstring invites use as a standalone gate, so a false pass there is a
  defect regardless of what catches it afterwards.
- **Three bare `int()` calls** leaked `ValueError` out of modules documenting that a caller
  receives "one of the structured exceptions in `xgboost_bridge.errors`". `num_feature = "1.0"`
  was enough. The identical defect at the identical idiom had already been found and fixed once,
  for `best_iteration`.
- **Two silent defaults in `validate.py`.** `attributes` present but not a dict made the D038
  early-stopping refusal `return None` — absence legitimately defaults, a wrong *type* is not
  absence. And emptiness of `feature_names` was checked only `and num_feature != 0`, so
  `feature_names: []` with `num_feature: "0"` passed the whole gate, including the length check
  (`0 == 0`), and export would emit an artifact our own reader and our own published schema both
  refuse. XGBoost declines to configure a 0-feature learner, which is why no test caught it.

### FORMAT.md §13 contradicted itself, and nothing recorded which horn won

"A reader **MUST NOT** raise on a node that is unreachable from the root" cannot hold alongside
the requirements two paragraphs above it that an out-of-range child, an out-of-range
`split_indices`, and a non-finite `node_values` entry **MUST** raise.

Resolved in favour of the narrow reading, which is what both readers already do: **every per-node
rule applies to every node, reachable or not**, and the only exemption is the cycle-and-tree
check, because an unreachable cycle cannot make the walk non-terminating. Measured on five
artifacts differing only at an unreachable node — canonical content and an unreachable cycle
load; `split_indices = 2147483647`, an out-of-range child, and `default_left = 7` each raise. A
conforming producer neutralizing per §8.3 writes individually-valid values, so copying XGBoost's
own dead marker through verbatim is not conforming. The wide claim appeared in three places and
is corrected in all of them.

### Published text that was wrong

Both package READMEs promised that "every error carries structured properties — `field`, `value`,
`expected`, `index`, `feature`". **No error carries all of them**, and
`FeatureKeyMismatchError` and `UnsupportedObjectiveError` carry none — so a caller following the
README read `err.field` as `undefined`, or as an `AttributeError`. The attribute names differ
between the two languages as well (`missing`/`extra` versus `missing_keys`/`extra_keys`), which I
also got wrong on the first attempt at the correction. Now enumerated per class, and pinned by a
test that checks the names against the classes in both directions — this text is frozen into the
wheel's `METADATA` and the npm tarball at publish.

`artifact.ts` also told a reader that `ErrorCode` "enumerates five codes" and that
`MALFORMED_ARTIFACT` and `NON_FINITE_FEATURE` reach a consumer's `default` branch through a type
assertion. D048 widened the union to seven; the comment was not updated, `grep "as ErrorCode"`
returns nothing, and `tsc` is clean. It was directing a reader to handle the two most common
failures in a fallback.

### Two guards with no test in either direction

`elementAt` and the walk's `default_left` re-check. Both fire correctly and both were reachable
only through the public constructor, so under D019 — "redundant safeguards are untested
safeguards" — reverting either turned zero tests red. `elementAt`'s own docstring names the
silent failure it prevents: an out-of-range typed-array read yields `undefined`,
`Math.fround(undefined)` is `NaN`, and `NaN` is this format's *missing value*, so the walk would
route down a legitimate branch and return a confident wrong number. Both now pinned, along with
every `ErrorCode` string a consumer can switch on — `NON_FINITE_FEATURE` was pinned by nothing.

### Not acted on, with reasons

**`export._self_check_rows` can miss a live leaf.** Its sufficiency argument — "a node whose
interval is empty is reachable by no input at all" — is **false for `NaN`**, which routes by
`default_left` and ignores thresholds entirely, so a node with an empty finite box can still be
reached. A constructed 7-node case reaches leaf 6 only via `[nan, 5.0]`, and corrupting that
leaf's value leaves every self-check margin identical. Latent rather than live: a search of 162
fitted models — `exact`/`hist`/`approx` × depth 4/8/12 × 0/30/70 % missing × 2/6 columns × 3
seeds — found **0** live nodes with an empty box. The argument is wrong and the exposure is
unreached; recorded here rather than papered over, and a fix that makes coverage an asserted
invariant rather than an argument is the right shape.

**`UNSUPPORTED_BOOSTER` is unreachable in the JavaScript package** — booster refusal is
export-side and lives in Python — while `types.ts` says the union is "every error this package can
raise". Left as-is: removing it from the union would be a breaking type change for a caller who
switches exhaustively, and the honest fix is a comment, which is what it now has.

**`PAIRED_TRANSFORM` is a frozen ordinary-prototype object** where its sibling
`OUTPUT_FUNCTIONS` uses `Object.create(null)` and documents that as load-bearing. Not
exploitable — `objective` is allow-list-checked first, and the subsequent comparison would throw
on a prototype hit — but the safety is an ordering property rather than a structural one.

**FORMAT.md §5.5's "each operation is a separate statement" is enforced by an AST scan on the
Python side and by nothing on the TypeScript side.** A fused `Math.fround(a * b + c)` would leave
every JavaScript ULP test green — it would score *better* against mpmath — and be caught only by
the parity harness, which lives in the Python suite.

Suites: Python 1041 → 1056, Node 138 → 148.

## D060 — The three sharpest open items, closed; npm moves to Trusted Publishing

*2026-08-06* — **completes D058's not-acted-on list except where noted; supersedes release.yml's npm token**

### FORMAT.md §5.5 is now structural on the TypeScript side

The rule — one arithmetic operation per statement, with a named intermediate — was
AST-enforced in Python since D046 and enforced by nothing in TypeScript. That asymmetry was the
sharpest thing left, because of *which direction* it fails in: a fused `Math.fround(a * b + c)`
contracts to a single rounding, which is **more accurate** than two rounded operations, so it
would score *better* against the mpmath reference and leave every JavaScript ULP test green.
Nothing in the JS suite would notice. The only check that would is the parity harness, which
lives in the Python suite — so anyone running `npm test` alone would see a passing, more
accurate, wrong implementation.

Wrong rather than merely different: float32 semantics are simulable only because each
`+ − × ÷` rounds separately (§5.1), so a contracted multiply-add breaks the exactness argument
the whole transform rests on.

Enforced with the real TypeScript AST via the compiler API — `typescript` is already a dev
dependency for `tsc`, so no dependency was added, and a `*` inside a comment or a string can
neither trip it nor hide from it. Three companion assertions keep it honest: the file must parse,
at least 20 arithmetic statements must be found (or the scan is passing vacuously — the same
failure mode the Python scan guards), and the scan must catch a fused multiply-add injected into
a copy of the source. That last one is the verify-by-reverting step, inline.

The first version of the scan reported `expF32` as a single 25-operator statement, because a
function *declaration* is itself a statement and the walk descended into its body. Corrected to
stop at nested blocks.

### `dist/index.cjs` is executed by the suite

It ships, the export map routes `require()` to it, and no test had ever run it — every other test
imports the ESM build. Not hypothetical: the audit had already found a real CJS defect, an export
map whose `require` condition pointed at a `types` entry that did not resolve, and it was found
by hand.

`tools/clean_install_js.sh` does predict through this entry point and runs in CI, so the path was
covered — from *outside* the repository, against an installed tarball. That is the right place
for a packaging check and the wrong place for the only execution of a shipped build output.

Five assertions: identical named exports to the ESM build, bit-identical margins and outputs
across all corpus rows, bit-exactness against XGBoost's recorded ground truth (not merely
agreement with the sibling build — two builds of one mistake agree perfectly), identical refusal
codes on three cases the readers were recently changed to refuse, and no `require` or `import` of
its own, since zero runtime dependencies has to hold for the file `require()` actually loads.

### The export self-check's coverage is asserted rather than argued

`_leaf_reaching_rows` justified its own sufficiency: a node whose feature-interval is empty "is
reachable in the graph but reachable by **no input at all**, so no row can exist for it and no
change to it can alter any prediction."

False for `NaN`, which routes by `default_left` and ignores thresholds entirely. A 7-node tree
where leaf 6's finite box is `col0 >= 20 ∧ col0 < 10` — empty — is reached by `[nan, 5.0]`: the
box tracking drops it, `_representative_row` returns `None`, the row is skipped **silently**, and
corrupting leaf 6's value leaves every self-check margin identical. The mandatory self-check of
§8.3 would pass over a wrong leaf.

`_assert_sample_reaches_every_live_leaf` now walks the sample and requires every live leaf to be
reached, naming the misses per tree. Confirmed to fire on exactly that constructed case, and to
be quiet on every fixture — the search of 162 fitted models that found 0 live nodes with an empty
box is what makes it safe to enforce rather than merely diagnostic.

**Ordering matters and I got it wrong first.** Asserting coverage inside `_self_check_rows`, before
the margin comparison, inverted the two checks: corrupted thresholds can make a leaf's box empty,
so the coverage complaint fired on an artifact whose actual defect was a wrong number, replacing
`MarginMismatchError` with the weaker finding. Coverage is a statement about the sufficiency of a
check that **passed**, so it runs after it.

### npm publishes via Trusted Publishing, with no token

`NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}` is gone. No npm credential exists in this repository,
which makes the npm side symmetric with PyPI's — that side never had one. `id-token: write` on the
publish job is now the whole of the credential.

npm Trusted Publishing requires npm ≥ 11.5.1 and Node 20 bundles 10.8.2, so the CLI is upgraded
in place rather than by moving to a newer Node: the build happened in `verify` under Node 20, and
this job only uploads a tarball it already holds. Upgrading a build toolchain to change a publish
mechanism would be the wrong trade.

`--tag rc` and `publishConfig.tag` both stay. The flag is what npm 10.8.2 needed and the manifest
field is what makes the behaviour independent of remembering the flag.

### Left open deliberately

`PAIRED_TRANSFORM`'s ordinary prototype, and the fixture generators' self-checks not running in
CI. Both judged not worth acting on now: the first is unexploitable because `objective` is
allow-list-checked before that table is consulted, and the second is a coverage gap in a
regeneration path whose output is already verified byte-identical.

Suites: Python 1056 → 1058, Node 148 → 157.

## D061 — npm sets `latest` on a package's first publish, and it cannot be removed

*2026-08-06* — **corrects a claim I made about `publishConfig.tag`; adds the post-publish dist-tag assertion**

Two npm behaviours, both measured on this package rather than reasoned about, and both
affecting what `npm install xgboost-predictor` resolves to.

### `publishConfig.tag` is not honoured, so `--tag` is load-bearing

I had said the manifest field made the `--tag rc` flag redundant. It does not. Probed on npm
12.0.1 with a throwaway manifest: `publishConfig.registry` **is** applied — a sentinel registry
appears in the publish target — while `publishConfig.tag: "rc"` still reports `with tag latest`.
The merge mechanism works; `tag` specifically is not picked up by that code path.

So the flag is the only thing keeping a release candidate off the default tag. Removing it as
redundant is a plausible future cleanup with a user-visible effect, so `release.yml` must carry
exactly one `npm publish` invocation and it must carry `--tag rc` — pinned by a test, along with
the converse: if the version ever stops being a prerelease, `--tag rc` becomes wrong rather than
protective and hides a final release behind a non-default tag.

That test's first version matched any line *containing* `npm publish`, which caught a prose
mention inside a YAML block scalar — trimming and testing for a leading `#` does not exclude a
block scalar. It matches invocations now.

### `latest` is assigned unconditionally on the first publish

**`--tag rc` worked and was not sufficient.** After a manual `npm publish --tag rc` of
`1.0.0-rc.1`, measured:

```
rc     -> 1.0.0-rc.1
latest -> 1.0.0-rc.1
```

npm sets `latest` on a package's first publish regardless of `--tag`, and then refuses to remove
it: `npm dist-tag rm` returns **E400**, because `latest` is protected and every package must have
one. So it cannot be corrected until a non-prerelease version exists, and until then
`npm i xgboost-predictor` installs a release candidate.

**First publish is exactly when the protection is weakest**, which is worth stating plainly: the
flag that governs the tag is honoured, the manifest field that would back it up is not, and the
one assignment that cannot be undone happens before either has anything to constrain. Anyone
bootstrapping a package by publishing a prerelease first — the normal way to create a package so
that Trusted Publishing can be configured, since npm has no pending-publisher equivalent to
PyPI's — hits this.

Not a wrong number, and not silent either once looked at. But it *is* a wrong-version-installed
path, and nothing verified the tags at all.

### The assertion that was missing

`release.yml` now checks, after every npm publish, what the dist-tags actually point at:

- `rc` must point at the version just published, with retries for registry propagation.
- For a **prerelease**, `latest` pointing at a prerelease is reported rather than failed — it is
  npm's bootstrap behaviour and cannot be undone.
- For a **final release**, `latest` **must** equal the just-published version and **must not** be
  a prerelease. That is the assertion the step exists for: it is what confirms `1.0.0` displaced
  the RC, and it fails loudly if an RC is still sitting there.

### A target selector, because the npm path could not otherwise be rehearsed

`publish-javascript` has `needs: [verify, publish-python]`, and `publish-python` targets real
PyPI, which has no project and no Trusted Publisher. Dispatching `release.yml` would therefore
fail at PyPI and **skip the npm job entirely** — the rehearsal would not happen. Caught before
spending a dispatch.

A `targets` input now selects `both` / `npm-only` / `pypi-only`. Two guards keep it from becoming
a way to half-publish a release: `verify` fails if a **non-prerelease** version is dispatched with
anything other than `both`, and `publish-javascript`'s condition requires `verify` to have
succeeded and `publish-python` to have either succeeded or been *deliberately skipped* — never
failed. A bare `needs` would have skipped the npm job whenever the PyPI job was skipped, which is
the opposite of what is wanted.

### Versions

`1.0.0-rc.1` is spent on npm and carries **no attestation** — a manual publish cannot produce
provenance, which requires CI with OIDC. `1.0.0-rc.2` is the rehearsal version, and npm is
therefore one RC ahead of PyPI from here on. Cosmetic, recorded so it is not mistaken for drift.

Fixture proof for the bump, as for D057: 0 of 299 margin patterns, 0 of 299 output patterns, 0 of
3258 node values moved; `exporter_version` changed in 23 of 23; content with that field removed
byte-identical in all 23.

## D062 — 1.0.0 released to both registries; the verification step failed, the release did not

*2026-08-07*

`xgboost-bridge` 1.0.0 on PyPI and `xgboost-predictor` 1.0.0 on npm, both through Trusted
Publishing with attestations and **no stored credential in the repository**. `release.yml`
executed with `targets=both` and the `release` environment's required reviewer.

### What the run reported, and what actually happened

The run is recorded as a failure. **Both packages published successfully.** `publish-python`
succeeded; `publish-javascript`'s publish step succeeded — `+ xgboost-predictor@1.0.0`, provenance
signed and logged to sigstore — and the step *after* it, my own dist-tag assertion, failed with:

```
FAIL: the rc tag never came to point at 1.0.0
```

The retry loop hardcoded `rc`, because every publish before this one was a release candidate. For
a final release the tag used is `latest`, so `rc` stayed at `1.0.0-rc.2` and the loop exhausted
six attempts waiting for something that was never going to happen. The assertion now waits on the
tag the publish step actually used, passed through the environment so the two cannot disagree.

**This is the same defect class as the seventeen findings before it** — a check whose reach did
not match its claim — and it is worth recording that it landed in the verification layer at the
one moment verification mattered most. A green publish and a red run is a confusing state to
inherit, and the fix is the boring one: never hardcode the thing you are about to compute.

It was also nearly a genuinely bad outcome. The instruction was to stop rather than re-dispatch if
a failure could leave one registry published and the other not, and `success / failure` across two
publish jobs is exactly what that looks like from the outside. Measuring the registries before
acting is what distinguished "the release is complete and a check is broken" from "half of 1.0.0
is out". Diagnose before remediating, especially when remediation is irreversible.

### The hardcoded tag was caught twice, once cheaply and once expensively

The publish step itself had `--tag rc` hardcoded, which would have put 1.0.0 on the `rc` tag and
left `latest` on `1.0.0-rc.1` permanently. That was caught **before any dispatch**, by this
repository's own test — *"the manifest declares a prerelease version while the rc tag is in
force"* — which went red the moment the version stopped being a prerelease. That test existed
precisely to fail here, and it did its job.

The identical mistake in the assertion's retry loop was not covered by any test, and cost a
confusing run. The lesson is not "write more tests" but a specific one: the *tag* was derived and
the *check on the tag* was not, so the two halves of one decision were expressed differently.

### Verified from outside, after publishing

Neither registry's result is taken from the workflow's own output:

| | measured |
|---|---|
| `uv pip install "xgboost-bridge==1.0.0"` into an empty venv | installs; `__version__` 1.0.0; **5/5 margins bit-exact** vs XGBoost ground truth |
| `npm install xgboost-predictor` into an empty project | resolves **1.0.0**; **5/5 margins bit-exact** |
| npm dist-tags | `latest -> 1.0.0`, `rc -> 1.0.0-rc.2` |
| npm attestation on 1.0.0 | present |
| PyPI files | wheel and sdist, both attested |

`npm install` with no version resolving to 1.0.0 is the observation that matters: `latest` moved
off the release candidates, which is what the non-prerelease branch of the assertion exists to
require and what npm's first-publish behaviour (D061) made impossible until a final version
existed.

### The split-release hazard, mapped rather than discovered

Before dispatching, the failure modes were enumerated: `publish-python` failing skips npm and
ships nothing (safe); `publish-python` succeeding and npm failing spends 1.0.0 on PyPI only, and a
re-dispatch would fail on the duplicate *before* reaching npm — so the obvious recovery,
`targets=npm-only`, was blocked by the non-prerelease guard. An `allow_partial` input now exists
for exactly that case, which means the recovery path was in place before it could be needed rather
than being written immediately after a partial release.

The fixture proof for the bump is the same as D057 and D061: 0 of 299 margin patterns, 0 of 299
output patterns, 0 of 3258 node values moved; `exporter_version` changed in 23 of 23; content with
that one field removed byte-identical in all 23.

Suites at release: Python **1058**, Node **160**, corpus parity `0.0`, 100,000-row parity `0.0`.

## D063 — Workflow logic moves into a tested module; the sweep found a fourth and fifth copy

*2026-08-07* — **completes D062; adds `RELEASING.md`**

### The assertion is now provable offline

`tools/check_dist_tags.mjs` holds the dist-tag decision as a module with a CLI attached, and
`packages/js/test/dist-tags.test.js` exercises it against **recorded registry responses** — the
actual output of `npm view xgboost-predictor dist-tags --json` at each stage of this project's
releases. No test touches a registry.

Both branches are covered, including the one that had never executed: a final release must own
`latest` and must not leave a prerelease there. And the **old rule is kept in the module** rather
than described in a comment, so a test can run it against the exact state that existed when the
1.0.0 run failed and require the two rules to disagree. A regression test that reimplements the bug
inside itself proves only that the author remembered it.

Writing those tests corrected my understanding twice, which is the argument for extraction better
than any I could make in prose:

- Two cases I expected to fail *passed*, because a mismatch on the expected tag is treated as
  **unsettled** — a propagation race, retried — rather than a verdict. That means the
  final-release branch is reachable only when a final version is published under a *non-default
  tag*, which is precisely the D062 publish-step defect. The verdict carries `settled` so a
  caller can tell "wrong" from "not yet"; a boolean could not.
- The CLI silently did nothing when run by hand. `import.meta.url` percent-encodes a path — this
  repository lives under directories with spaces — while `process.argv[1]` carries them
  literally, so `import.meta.url === \`file://${process.argv[1]}\`` never matched. It would have
  worked in CI, whose paths have no spaces. Fixed with `pathToFileURL`.

### The sweep: five hardcoded-for-prerelease assumptions, not three

Every one was written when every publish was a release candidate, and 1.0.0 was the first input
that distinguished them.

| | Where | Caught by |
|---|---|---|
| 1 | `--tag rc` in the publish step | This repository's own test, **before any dispatch** |
| 2 | `rc` in the assertion's retry loop | The 1.0.0 run — a green publish inside a red run |
| 3 | The `targets` guard requiring `both` | Reasoning about the split-release hazard before dispatching |
| 4 | A loose `/(a|b|rc|alpha|beta|dev)/i` in that guard | **This sweep** |
| 5 | A fourth inline `/-(?:rc|alpha|beta)\./` in the tag derivation | **This sweep** |

Four and five are the interesting ones. There were **four** definitions of "prerelease" in one
file, and they were not identical: the guard's matched any version *containing* `a` or `b`, and
the tag derivation's lacked a trailing `\d+$` anchor. All four now come from the module, which
holds one predicate per ecosystem — `isPrerelease` for npm's `1.0.0-rc.1`, `isPrereleasePep440`
for PyPI's `1.0.0rc1`. They cannot be one regex, because the grammars genuinely differ, but they
can live in one tested place, and a test asserts they agree about every version this project has
shipped. A test also forbids the workflow from carrying an inline copy again.

The guard's looseness mattered in the wrong direction: a *mis*-classification there would treat a
final release as a prerelease and skip the both-registries requirement, which is the half-published
release the guard exists to prevent.

### And a sixth, made while fixing the others

The first version of this work computed the dist-tag inside `publish-javascript` by importing the
module — in a job that deliberately has **no checkout**, because its whole design is "download an
artifact, call the publisher, do nothing else." It would have failed on a missing file.

The tag is derived in `verify` now and passed as a job output. That is better than a repair: the
credentialed job computes nothing at all, which is the shape the trust-domain split (D054) is for.

**The general rule this yields, and the reason it belongs next to the other two principles:
anything in a workflow that makes a decision belongs where a test can reach it.** Inline YAML is
not a testable place. Every defect above was in a decision expressed as shell, and the two that
were caught cheaply were caught because their *counterparts* had been extracted into files.

### `RELEASING.md`

What a maintainer runs, in what order, what each guard refuses and why, and what to do when it
fails — including the case that actually happened, where `success / failure` across two publish
jobs was indistinguishable from a split release and both packages had in fact published. The
knowledge was in these entries and in one head; the next release may be far enough out that
neither is available.

Suites: Python 1058, Node 160 → **174**.
