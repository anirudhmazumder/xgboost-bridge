# 1.0 readiness

State of the repository at the tip of `main`. Every number below was measured at that tip, not carried forward from an earlier report.

Nothing has been pushed and nothing has been published.

---

## Verification gates

| Check | Threshold | Result |
|---|---|---|
| Python suite | All pass; count never decreases | **960 passed**, 0 failed, 0 skipped, 0 xfailed |
| Node suite | All pass; count never decreases | **112 pass**, 0 fail, 0 skipped, 0 todo |
| **Margin parity, Python vs JS** | **Exactly `0.0`**, bit patterns | **`0.0`** across 299 corpus rows, and across **100,004** generated rows |
| **Output parity, Python vs JS** | **Exactly `0.0`**, bit patterns | **`0.0`** across 299 corpus rows, and across **100,004** generated rows |
| Python vs XGBoost, margin | Absolute ≤ `1e-6` | **`0.0` bit-exact**, 289/289 rows |
| Python vs XGBoost, output | Relative ≤ `1e-6` | max **`9.56e-08`**, 283/289 bit-exact |
| Bundled transform vs `mpmath` (50 dp) | Max ULP, per side | `exp` **1**, `sigmoid` **2**, both languages |
| `tsc --noEmit` | Clean, separate from build | clean |
| JS tests import `dist/` | Confirmed | confirmed, and a stale bundle is refused |
| Export determinism | Byte-identical across runs | byte-identical, in-process and cross-interpreter |
| JS runtime dependencies | **0** | **0** |
| Vocabulary scrub | Executable, self-testing, clean | clean |

The single warning in the Python suite is XGBoost's own `booster=gblinear` deprecation notice, emitted during the test that asserts gblinear is refused.

### Cross-platform reproducibility is asserted by construction and is not yet evidence

**CI has never executed.** Every figure in the table above was measured on a single machine: **darwin/arm64**, CPython 3.12.8, numpy 2.5.1, XGBoost 3.3.0, and Node 20.19.0 / 24.7.0 / 24.18.0. The wheel was additionally exercised at the declared floor, Python 3.10.20 with numpy 1.24.4 — on that same machine.

The claim that these results hold on another platform rests on **argument, not measurement**:

- Float32 arithmetic is simulated from `+ - * /` and exact power-of-two scaling only, all of which IEEE-754 mandates to be correctly rounded, so they cannot vary by platform.
- The transform calls no `libm`, which is the one component that demonstrably *does* vary — measured at up to 2 ULP between V8 and Apple's `libm`.
- The margin comparison against XGBoost was measured bit-stable across thread counts, batch sizes, and both prediction APIs: 1440 per-row observations, 0 divergences — **on this platform**. GPU was not measured at all.

That reasoning is sound, and it is still not evidence. Two things it cannot rule out: an x86-64 `libm` reaching a different result somewhere a platform function is still called that we believe is off the path, and XGBoost's own `expf` differing by architecture — which would move the six accepted output divergences rather than the margins.

**The first green CI run on Linux x86-64 is the point at which this claim becomes evidence.** Until then every number above is single-platform, single-architecture. The `python` and `javascript` jobs in `ci.yml` have been checked against what exists locally but have never run.

### Parity row counts: why 289 and 299 are both correct

The two figures answer different questions; neither is a subset error.

| Count | What it is |
|---|---|
| **289** | Corpus rows carrying XGBoost ground truth. The denominator for *margin vs XGBoost* |
| **299** | All corpus rows: the 289 above **plus** the 10 rows of `adversarial/non_finite_input_refusal`. The denominator for *parity* |

The 10 extra rows carry `+/-inf` feature values and **deliberately have no numeric ground truth**, because XGBoost itself has no single answer for them: it raises through `DMatrix` and treats them as ordinary comparable values through `inplace_predict`, giving two different predictions for one input. D022 refuses them for exactly that reason, so there is no XGBoost margin to compare against.

They remain part of the **parity** gate, because agreement on *refusing* is as much a cross-language property as agreement on a value — both sides must refuse the same rows with the same error type. Excluding them would make a row silently skipped on both sides indistinguishable from a row that passed, which is the failure the harness's own injection tests exist to catch.

### Parity at scale

299 rows is thin for the project's headline gate, so both measurement points were re-run at **100,004 rows** across all 23 artifacts, with the two languages fed **bit-identical float64 inputs** exchanged as bit patterns rather than decimals:

```
rows compared             100004
margin-point mismatches        0
output-point mismatches        0
refusal disagreements          0
fixtures with any mismatch  0 of 23
```

Composition is adversarial rather than uniform, because uniform rows cannot find the defect that matters — 0 of 20,000 random continuous rows detect the float32 comparison error. Of the 100,004: **5,828 land exactly on a `float32(threshold)`** drawn from the artifact under test, 112 carry `NaN` on the missing-value path, and 654 carry a subnormal or a value beyond `1e30`. Sensitivity was confirmed at this scale: injecting one ULP into a single row of 100,004 produces exactly one margin-point mismatch and zero output-point mismatches.

---

## Accuracy, per objective

Against XGBoost's own recorded `predict()` output, across all 23 fixtures:

| Objective | Rows | Output bit-exact | Max relative error |
|---|---|---|---|
| `reg:squarederror` | 174 | **174/174** | `0.0` |
| `binary:logistic` | 89 | 87/89 | `9.56e-08` |
| `survival:cox` | 26 | 22/26 | `7.03e-08` |

**Margins are bit-exact on all 289 rows for all three objectives.**

The six output divergences are `libm` differences inside the bundled `exp`, expected by construction: XGBoost's own `expf` is not correctly rounded, so bit-exactness with it at the output stage is unreachable by any implementation. Note the shape of the result — `reg:squarederror` uses the identity transform and diverges on **nothing**, while every divergence falls on the two objectives that call the bundled `exp`. That is the pattern a correct implementation produces.

The six are pinned as an exact set of `(fixture, row)` pairs plus the count, in both languages. Movement in either direction fails, **including an improvement**. That is what keeps the gate a tripwire rather than a band a future defect could hide inside.

---

## Corpus

| | |
|---|---|
| Fixtures | 23 — 15 ordinary, 8 adversarial |
| Rows | 299 — 289 value-bearing, 10 refusal |
| Regeneration | byte-identical, verified independently |
| Ground truth | XGBoost's own `predict()` and `predict(output_margin=True)`, stored as float32 bit patterns |

Ground truth is stored as uint32 hex bit patterns rather than JSON numbers, because JSON cannot represent `±inf` and `survival:cox` genuinely returns `+inf` above margin ≈ `88.72`. Bit patterns also handle `NaN` and `-0.0` with no special case, and they make the correct comparison the only convenient one.

Composition was treated as a correctness requirement rather than coverage bookkeeping. At `base_score = 0.5` **every** broken intercept variant scores 5000/5000, and Cox's estimated default gives an intercept of exactly `0.0` where placement stops mattering — so a corpus built on defaults certifies a wrong implementation. The corpus carries values far from `0.5` in both directions, inside and outside the logistic clamp, per objective.

The adversarial set exists to break a plausible-but-wrong implementation rather than to demonstrate a working one. Ordinary rows cannot do that: 0 of 20,000 random continuous rows detect the float32 comparison defect. The 83 disagreement rows place feature values exactly on `float32(threshold)` at the 85 of 147 internal nodes measured hazardous — selected by measuring the gap direction, not by hoping.

Six deliberately-broken walk variants, measured against the 184 value-producing adversarial rows: sample-side cast removed **96** wrong; `<=` instead of `<` **107**; leaf values un-narrowed **94**; intercept added last **115**; trees reversed **95**; threshold-side cast removed **1**. All non-zero, so no protection is uncovered.

---

## Versions actually tested

An untested version is an unrecognized input, so this is a list rather than a range.

| | |
|---|---|
| XGBoost, reference | **3.3.0** — everything is verified against this |
| XGBoost, drift probe | **3.4.0-dev**, commit `e787a447de12c15bdf06f65ddbf79b056743113d`, and released **3.4.0** |
| Python | 3.12.8 (published floor `>=3.10`; `export` extra needs `>=3.12`) |
| Node | 20.19.0, 24.7.0, 24.18.0 |
| Platform | darwin/arm64. **GPU not measured.** |

The version ceiling exists because drift here is real and silent: XGBoost 3.4.0-dev relocated `weight_drop`, and XGBoost 3.3.0 reads such an artifact returning **0/400 rows correct at max error 1.26, with zero warnings and exit code 0**, then drops the field on re-save. Unrecognized-*field* detection catches additions and structurally cannot catch relocations or removals.

---

## Refusal matrix

13 error classes plus a base, all subclassing `XGBoostBridgeError`, all carrying structured attributes rather than only a message.

| Refused | Why it is refused rather than approximated |
|---|---|
| `dart` (`weight_drop` at **either** JSON path) | Only one in-artifact signal exists, confirmed by exhaustive key census. The invariant that motivated supporting dart is the one that refuses it |
| `gblinear` | Deprecated upstream, and an entirely separate inference path with no trees |
| Categorical splits | They **invert** the child convention — in-set routes right — and `split_conditions` holds a subnormal, not a threshold |
| Multi-class, multi-target | An objective allow-list alone lets `reg:squarederror` with `num_target=2` through, producing `(N,2)` margins a scalar predictor accepts and gets wrong |
| Empty `feature_names` | A strict-key policy with no keys to check reads as enforced and is not |
| Ambiguous early-stopped tree count | The correct count is not a property of the model: the same file gives different answers loaded as a `Booster` versus through the sklearn estimator, diverging by 1.55, with no field distinguishing them |
| Untested XGBoost version | See above |
| `±inf` feature values | Upstream itself is inconsistent — raises through `DMatrix`, ordinary value through `inplace_predict` |
| Non-finite derived intercept | Cox at `base_score=0.0` derives `-inf`; at any negative value, `NaN`. Both accepted silently by XGBoost |
| Malformed artifact structure | Unequal array lengths, out-of-range indices, cycles, unrecognized keys at any level |
| Feature-key mismatch | A typo becomes a missing-value path, which is legitimate model structure, so the mistake compounds into a confident wrong number |

`NaN` is **not** refused: it is the missing value and routes by `default_left`.

---

## Open items, deferred past 1.0

None of these blocks the release. Each is recorded so a later reader does not mistake absence of evidence for evidence.

**Unmeasured, and honest about it**
- The **sklearn `XGBClassifier` route** in the arity probe. `scikit-learn` was not installed and installing it was out of scope for a read-only probe. It is the largest single hole in that probe.
- The **`save_best=True`** early-stopping callback. If it trims the model, the ambiguity predicate is satisfied and export proceeds — which is safe, because a satisfied predicate means both readings coincide. Inferred, not observed.
- **GPU**, and any platform other than darwin/arm64 for most probes.
- `learner.gradient_booster.model.cats` — empty in every model producible, including two with genuine categorical splits. Needs probing before any categorical support.

**Known and accepted**
- The logistic clamp's **exact form** — input clamp on the margin versus output floor — is observationally identical, and an upper clamp at `+88.7` is undetectable because both give exactly `1.0`. Either satisfies the spec. The constant is pinned; the mechanism is not.
- Two `format_version` refusals are **not expressible in JavaScript**: `1.0` is indistinguishable from `1` after `JSON.parse`, and `-0` passes `Number.isInteger` in an index array (with no numeric consequence, being `=== 0` as an index). Python rejects the first; JavaScript cannot.
- `SUPPORTED_OBJECTIVES` is defined in two modules, with a test pinning their agreement so they cannot drift silently. Collapsing them is a module-dependency question, deferred.
- `MalformedTreeError`'s name is used for some non-tree failures. Behaviour is right, wording is imprecise.
- `UnsupportedBoosterError` is unreachable from the JavaScript side — a v1 artifact carries no booster field, since boosters are refused at export. A live public class with no producer.
- The parity harness makes `pytest` depend on `node` and a current `packages/js/dist/`. Intended, but it introduces an ordering relationship between the two suites.

**Before publishing**
- The `release` environment must be created in repository settings with a reviewer rule. GitHub auto-creates a referenced environment **without** protection, so the gate in `release.yml` is only real once a human configures it.
- No PyPI Trusted Publisher entry and no npm token exist. Both are required, and neither should be created until the release is actually intended.

---

## AI-authorship disclosure

*Written for the 1.0 announcement, per the deferral recorded as D012.*

**This library was written by an AI system — Claude — working under human direction.** The human owner set the goals and the constraints, made the scope decisions, and reviewed the work; the design, the code, the probes, the tests and the documentation were produced by the model.

That is disclosed prominently rather than in a footnote, because a numerical library asks for a particular kind of trust and readers deserve to know what they are extending it to.

What was done to earn it is visible in the repository rather than asserted here:

- **Every empirical claim is measured, and the measurement is on disk.** Eleven probe reports, 10,664 lines, each recording the commands run and their real output. Where a claim could not be measured, it says so.
- **Four confidently-held beliefs were falsified by those probes** and are recorded with what replaced them, rather than quietly corrected: the logistic intercept is not `logit(base_score)`; dart cannot be detected by two independent signals; the output transform runs in float32, not float64; `num_class` can legitimately be `"1"`.
- **Three defects were found in shipped code by review and adversarial testing**, not by review comments: four of five float32 sites in the tree walk were pinned by no test at all; the validation gate was the more lenient of two modules checking the same thing; and a refusal that two documents specified was implemented nowhere.
- **Mistakes the model made are in the commit history**, including a spec example computed at the wrong precision, a statistic propagated with the wrong denominator, a verification that measured a deliberate clamp as 1.5 million ULP of error, and an edit reported as applied that had silently matched nothing.
- **The numerical core is validated against an external high-precision reference, per side independently.** Cross-language agreement is measured separately and is never treated as evidence of correctness — two identical implementations agreeing proves only that the code was written twice.

The project's own premise is that a plausible wrong number is worse than a crash. The same standard was applied to its authorship: where the model could not establish something, the repository says so.

---

## Decision coverage

Every `DECISIONS.md` entry asserting runtime behaviour is now mapped to a test that goes red when the behaviour is reverted. The map is `DECISION_COVERAGE.md`: **38 behavioural** entries, **13 process or metadata**.

It exists because D022 was specified in two documents, implemented nowhere, and found by accident four days later while every gate stayed green. Six behavioural decisions turned out to have **no pinning test** — including the leaf-detection rule and the rule that no predictor reads `provenance`, each of which could be reverted with **zero tests going red in either language**. All six are now pinned. Nothing else was found specified-but-unimplemented.

Five behaviours cannot be reverted in isolation because another site absorbs them. Each is named with its absorbing site rather than covered by a test that would prove nothing.

## Handoff

Complete and verified. The two things that were never mine remain undone: **pushing to a remote, and publishing to PyPI or npm.**

21 commits on `main`, working tree clean.

`origin` is configured, **private, and empty** — no branches, no commits, `size: 0`. GitHub did not auto-create a README or an initial commit, so the local history is a clean first push with nothing to reconcile. Verified read-only through the GitHub API; `git fetch` could not authenticate in this environment.
