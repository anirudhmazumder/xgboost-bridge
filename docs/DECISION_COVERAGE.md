# Decision coverage

Every entry in `DECISIONS.md`, classified, and — where it describes behaviour — mapped to the
test that goes red when the behaviour is reverted.

**Why this file exists.** D022 recorded "`±inf` at predict time raises" on 2026-08-01, `FORMAT.md`
§9.3 restated it, and nothing implemented it. `walk_margin` returned an ordinary-looking
`np.float32(13.74516)` for an infinite input for four days while every gate stayed green. It was
found by accident. **An unpinned behavioural decision is a defect, not a gap:** the record makes
it look handled, which is worse than no record.

## Method

A pinning claim is worth nothing unmeasured, so every claim below was measured the same way:

1. Revert the behaviour in the source — one site at a time, never two together (D019).
2. Run the **whole** suite, both languages, and record which tests go red.
3. Restore the file and verify it is byte-identical with `shasum -a 256`.

A test that stays green with the behaviour reverted does not pin it, and is recorded as such.
Where a behaviour cannot be reverted in isolation because another site absorbs it, the absorbing
site is named and its own pin is verified — see [Absorption](#absorption). Reverts that produce
zero red because the change is not observable on any admissible input are listed separately, in
[Reverts that are no-ops](#reverts-that-are-no-ops), so they are not mistaken for gaps.

Counts in the tables are red **test** counts from a full-suite run.

## Counts

| | |
|---|---|
| Entries | **51** (D001–D051) |
| **Behavioural** — asserts what the shipped code does or refuses at run, export or load time | **38** |
| **Process / metadata** — tooling, versions, documentation, layout, routing, or how work is verified | **13** |
| Behavioural entries with **no** pinning test before this audit | **6** |
| Tests added | **9** (7 Python, 2 Node) |
| Behaviours **specified but unimplemented** (the D022 class) | **0** |

Four entries are split: their behavioural half is classified behavioural and their process half is
noted in the same row (D004, D010, D019, D026). Three entries carry both a behavioural and a
packaging half and are counted behavioural (D009, D050, D051).

## The map — behavioural entries

| ID | Implementing site | Pinning test(s) | Revert performed | Red |
|---|---|---|---|---|
| **D003** | `validate.SUPPORTED_OBJECTIVES`; the seven-key envelope reserves nothing | `test_validate::test_rejects_multiclass_objectives`, `test_objectives::test_supported_objectives_agree_with_the_export_gate`; JS `artifact.test.js` "objective must be one of the three the format enumerates" | added `multi:softmax`/`multi:softprob` to the allow-list (both languages) | 3 py, 1 js |
| **D004** | parse-time float32 narrowing: `predict._read_node_values`, `artifact.ts readNodeValues`; both comparison-site casts in `trees.walk_margin` / `predict.ts walk` | `test_trees::test_the_walk_casts_the_sample_side_of_the_comparison_itself`, `…_the_threshold_side_…`, `test_predict::test_node_values_are_loaded_into_a_read_only_float32_array`, `…test_a_threshold_that_is_not_float32_exact_is_narrowed_at_parse_time`; JS "node_values is loaded into a Float32Array, not an array of doubles", "the sample side of the comparison is cast to float32" | dropped the sample-side cast; dropped the threshold-side cast; parsed `node_values` as float64 — each separately | 18 / 1 / 25 py; 3 / 0 / 5 js |
| **D005** | `predict._feature_row` key-set equality; `predict.ts featureRow` | `test_predict::test_a_missing_feature_key_raises`, `…test_an_extra_feature_key_raises`, `…test_a_typoed_feature_key_raises_and_is_reported_as_both`, `…test_a_key_no_split_reads_must_still_be_present`; JS "the input key set must equal feature_names exactly" | disabled the mismatch raise | 6 py, 2 js |
| **D006** | absence of any file-reading entry point in `packages/js/src` | JS "there is no fromFile, in any spelling (D006)"; **added** "the bundle exports no module-level file-reading entry point either (D006)" | added `Predictor.fromFile`; added a module-level `export function fromFile` | 1 js; **0 js before the added test**, 1 js after |
| **D007** | `predict._check_keys`, `_read_format_version`, `_read_objective`; `artifact.ts checkKeys` | `test_predict::test_an_eighth_top_level_key_raises`, `…test_an_unrecognized_key_is_reported_deterministically`, `…test_an_unrecognized_provenance_key_raises`, `…test_an_unrecognized_tree_key_raises`, `…test_a_format_version_other_than_the_integer_one_raises`; JS "an unrecognized key raises at every level of the document" | ignored unrecognized keys; accepted `format_version` through a coercing comparison | 4 / 6 py; 1 js |
| **D008** | `export.to_json` (`sort_keys=True`, compact separators, one trailing newline) | `test_export::test_serialized_keys_are_sorted_lexicographically_at_every_level`, `…test_export_is_byte_identical_across_two_calls_in_the_same_process`, `…_across_two_separate_interpreter_invocations` | `sort_keys=False` | 1 py |
| **D009** | `packages/js/package.json` `dependencies: {}`; the bundle imports nothing | JS "the package declares zero runtime dependencies", "the shipped bundle imports nothing at all" | declared one runtime dependency in the manifest | 1 js |
| **D010** *(split: packaging is process; "no xgboost needed to read and predict" is behaviour)* | deferred `import xgboost` inside `objectives._observed_margin`, `_observed_zero_round_margin`, `export._verify_against_source_margin` | **added** `test_scaffold::test_the_package_imports_and_predicts_with_the_export_extra_absent` | moved `import xgboost` to module scope in `objectives.py` | **0 py before the added test**, 1 py after |
| **D015** | `predict._read_intercept` (narrow, never transform); `provenance` read by no prediction path | `test_predict::test_intercept_is_narrowed_to_float32_at_parse_time`, `…test_negative_zero_intercept_is_not_normalized_on_read`; **added** `…test_no_prediction_path_function_reads_the_provenance_block` and `…test_predictions_are_unchanged_when_the_provenance_block_is_corrupted`; JS "the intercept is narrowed on read, never transformed, and keeps its sign"; **added** JS "no prediction method mentions `provenance` in its own source (D015)" | scaled the intercept on read; read `provenance` on the prediction path (plain, then obfuscated) | 55 py; **0 py / 0 js before the added tests**, then 1 py + 23 py and 2 js + 8 js |
| **D016** | `validate._check_booster` — allow-list, then `weight_drop` at both paths | `test_validate::test_rejects_gblinear`, `…test_rejects_dart_with_rate_drop_producing_weight_drop`, `…test_dart_weight_drop_detected_at_the_gradient_booster_path`, `…_at_the_relocated_model_path`, `test_export::test_export_refuses_dart_before_reaching_the_numeric_path` | accepted `gblinear`; ignored `weight_drop` at both paths; ignored it at the relocated path only | 1 / 4 / 1 py |
| **D017** | `validate._check_output_arity`, `num_target` | `test_validate::test_rejects_reg_squarederror_with_num_target_2`, `…_even_with_num_class_1`, `…test_num_target_is_compared_as_a_string_not_an_integer` | dropped the `num_target` check | 3 py |
| **D018** | `validate._check_version`; `export.DEFAULT_TESTED_VERSIONS` | `test_validate::test_rejects_when_the_producing_version_is_not_in_tested_versions`, `test_export::test_export_refuses_an_untested_xgboost_version_before_anything_numeric` | disabled the ceiling | 2 py |
| **D019** *(split: "keep both narrowing sites" is behaviour; "verify each in isolation" is the method of this file)* | the leaf-value cast at the add and the accumulator cast in `walk_margin` / `predict.ts walk`; the parse-time sites in `predict.py` / `artifact.ts` | `test_trees::test_the_walk_narrows_a_leaf_value_before_adding_it`, `…test_the_walk_narrows_leaf_values_read_from_a_float64_array`, `…test_the_walk_narrows_the_intercept_before_any_tree`, `…test_the_walk_narrows_a_float64_intercept_before_the_first_add`; JS "the accumulator is narrowed to float32 after every single addition", "leaf values are narrowed on read, and it changes the accumulated margin" | four separate single-site reverts (leaf cast; accumulator cast; walk intercept cast; parse-time array dtype) | 2 / 111 / 4 / 25 py; 0 / 6 / 0 / 5 js — see [Absorption](#absorption) |
| **D020** | `export._build_provenance` copies three named fields and nothing else; `learner.attributes` is never read into the artifact | `test_export::test_early_stopped_models_learner_attributes_do_not_leak_into_the_artifact`, `…test_provenance_records_xgboost_version_and_base_score_verbatim` | folded `learner.attributes` into `provenance` | 2 py |
| **D021** | `validate._check_feature_names`; `predict._read_feature_names` | `test_validate::test_rejects_model_fit_from_a_bare_array_with_no_feature_names`, `test_export::test_feature_names_override_is_required_for_a_bare_array_model`, `test_predict::test_feature_names_that_cannot_support_a_strict_key_policy_raise` | dropped the empty-names refusal at the gate; dropped it in the reader | 2 / 1 py |
| **D022** | superseded in implementation by **D045** — see that row | | | |
| **D025** | `trees.walk_margin`; `predict.ts walk` | 8 tests named in the revert column, plus the whole corpus and parity sets | intercept added last; trees reversed; float64 sum narrowed once at the end; non-strict `<=`; leaf test keyed on `right_children` | 113 / 113 / 111 / 67 / **0→1** py; 7 / 7 / 6 / 5 / 0 js |
| **D026** *(split: float64 output transform superseded by D032; float32 margin is behaviour; two measurement points is method)* | `Predictor.margin` returns `np.float32` untouched | `test_predict::test_margin_is_float32_and_never_widened`; `test_parity::test_margin_parity_is_exactly_zero`, `…test_output_parity_is_exactly_zero` | widened the margin to float64 | 1 py |
| **D027** | `trees.neutralize_dead_nodes`, `_neutralize_node`, `reachable_nodes`; `export._verify_against_source_margin` | `test_trees::test_neutralize_raises_when_a_marked_node_is_reachable`, `…_when_an_unreachable_node_is_unmarked`, `…test_neutralization_preserves_array_lengths_and_indices`, `…test_neutralized_nodes_carry_the_canonical_safe_values`, `…test_neutralizing_a_live_node_disagrees_with_xgboost`, `test_export::test_export_refuses_an_artifact_whose_live_node_was_cleared`, `…test_export_refuses_node_values_read_from_the_wrong_source_array` | dropped the marker/reachability agreement; compacted instead of neutralizing; wrote non-canonical values; removed the export self-check | 2 / 14 / 2 / 3 py |
| **D028** | absence of any `objective` read on either prediction path | `test_predict::test_no_prediction_path_function_reads_the_objective_field`, `…test_predictions_are_unchanged_when_the_objective_field_is_corrupted`, `test_parity::test_neither_predictor_branches_on_objective`; JS "no prediction method mentions `objective` in its own source", "nothing in the shipped bundle branches on `objective`" | inserted a no-op `if objective == …`; inserted an obfuscated `getattr`-based branch that changes the number | 1 / 22 py; 2 js. Both checks are necessary: the obfuscated revert leaves the source scan **green** |
| **D030** | `transform.exp_f32`, `sigmoid_f32` built from `+ − * /` only; `transform.ts` likewise | `test_transform::test_transform_calls_no_platform_transcendental`, `…test_transform_uses_only_the_four_correctly_rounded_operations`, `…test_exp_at_its_known_worst_input`; JS "no platform exponential and no exponentiation operator in src/", "…in the shipped bundle", "Math.fround is the only Math member the prediction path uses" | called the platform exponential | 8 py, 5 js |
| **D032** | float32 semantics per intermediate; `SIGMOID_MARGIN_FLOOR` applied before the transform | `test_transform::test_sigmoid_below_the_floor_is_that_one_value_and_never_zero`, `…test_sigmoid_output_of_exactly_zero_is_unreachable`, `…test_every_arithmetic_step_is_a_narrowed_named_intermediate`, `test_predict::test_logistic_clamp_floor_rows_match_xgboost_bit_for_bit`; JS "sigmoidF32 floors at the measured margin and is never 0", "below the clamp floor the check is a predicate…" | removed the sigmoid floor; made one intermediate float64 | 14 / 1 py; 4 js |
| **D034** | `objectives.verify_intercept` compares against XGBoost's observed zero-tree margin | `test_objectives::test_verify_intercept_fires_on_a_recipe_error`, `…test_verify_intercept_rejects_a_one_ulp_error`, `…test_verify_intercept_rejects_positive_zero_for_negative_zero`, `…test_verify_intercept_rejects_a_value_that_is_not_a_float32`, +5 more | made `verify_intercept` return without comparing | 9 py |
| **D035** | the clamp in `objectives._logistic_intercept` | `test_objectives::test_logistic_clamp_is_load_bearing`, `…test_logistic_clamp_keeps_the_logarithm_in_its_domain_at_one`, `…test_logistic_clamp_saturates_to_the_pinned_bounds`, `…test_logistic_intercept_is_bit_exact_against_xgboost` | removed the clamp | 6 py |
| **D036** | the zero-trees-and-`boost_from_average == "1"` cell in `objectives.derive_intercept` | `test_objectives::test_boost_from_average_selects_the_intercept_space`, `…test_raw_space_is_not_clamped_either`, `…test_verify_intercept_accepts_the_raw_space_of_a_zero_tree_default` | ignored `boost_from_average` | 5 py |
| **D037** | `validate._check_output_arity` — string comparisons, `num_class ∈ {"0","1"}`, per-tree `size_leaf_vector` | `test_validate::test_accepts_num_class_one_on_every_in_scope_objective`, `…test_num_class_is_compared_as_a_string_not_an_integer`, `…test_size_leaf_vector_is_compared_as_a_string_not_an_integer`, `…test_output_arity_gate_num_class_as_a_json_integer_still_rejects`, `…test_accepts_zero_boosting_round_model`; **added** `…test_size_leaf_vector_is_checked_on_every_tree_not_only_the_first` | compared each of the three fields against a bare `int`; required `num_class == "0"`; coerced `num_class` with `int()`; checked only `trees[0]` | 74 / 72 / 62 / 3 / 1 py; **0 py before the added test**, 1 py after |
| **D038** | `validate._check_early_stopping` | `test_validate::test_rejects_early_stopped_model_with_ambiguous_tree_count`, `…test_accepts_early_stopped_model_whose_best_iteration_is_unambiguous`, `…test_accepts_model_with_no_best_iteration_attribute_at_all`, `…test_rejects_best_iteration_that_indexes_past_iteration_indptr`, `…test_rejects_non_integer_best_iteration` | removed the ambiguity raise | 1 py |
| **D039** | `objectives._LOGISTIC_CLAMP_LOW` / `_HIGH` | `test_objectives::test_logistic_clamp_saturates_to_the_pinned_bounds`, `…test_logistic_clamp_is_load_bearing`, `…test_cox_is_not_clamped`, `…test_regression_is_not_clamped` | moved both bounds to `1e-7` | 6 py |
| **D040** | `np.log` of a float32 in `_cox_intercept` and `_logistic_intercept`; the float32 snap in `_read_base_score` | `test_objectives::test_cox_float32_log_route_matches_xgboost_where_float64_does_not`, `…test_logistic_float32_log_route_matches_xgboost_where_float64_does_not`, `…test_base_score_is_snapped_to_float32_before_the_transform`, `…test_textbook_logit_is_not_the_logistic_transform` | float64 logarithm for Cox; for logistic; the textbook ratio form; skipped the float32 snap | 2 / 1 / 10 / 10 py |
| **D041** | `UnsupportedModelShapeError` for arity, `MalformedTreeError` for structure | `test_trees::test_extract_trees_raises_on_vector_leaves`, `…test_a_vector_leaf_model_is_refused`, and the exact-class assertions throughout `test_validate` | raised `MalformedTreeError` where the arity class belongs | 2 py |
| **D043** | the two oracle shapes in `verify_intercept`; the finiteness refusal in `export.export_model` | `test_objectives::test_verify_intercept_accepts_the_raw_space_of_a_zero_tree_default`, `…test_verify_intercept_rejects_link_space_in_the_raw_cell`, `test_export::test_export_raises_on_a_non_finite_cox_intercept_at_zero_base_score`, `…at_negative_base_score` | made the oracle always refit; accepted a non-finite intercept | 4 / 2 py |
| **D044** | `fixtures/generate/reference.py` emission; bit patterns in every corpus file | `fixtures/tests/test_corpus::test_decimal_fields_agree_with_bit_patterns_where_finite`, `…test_rewalk_reproduces_expected_margin_bit_for_bit`, `test_predict::test_margin_is_bit_exact_on_every_row_of_the_whole_corpus`; JS "margin reproduces XGBoost's recorded margin bit-for-bit on every corpus row" | replaced one fixture's first `expected_margin` entry with a decimal; made one `margin_decimal` entry disagree with its bit pattern | 5 py + 1 js; 1 py |
| **D045** | the whole-row `±inf` check at the top of `walk_margin`; the same in `predict.ts featureRow` | `test_trees::test_an_infinite_feature_value_is_refused`, `…test_an_infinity_in_a_column_no_node_reads_is_still_refused`, `…test_every_row_of_the_refusal_fixture_is_refused`, `test_predict::test_infinite_value_raises_even_in_a_column_no_node_reads`, `test_parity::test_both_sides_refuse_the_same_rows_and_refuse_them_the_same_way`; JS "infinity in the input raises, and NaN in the same input does not", "the infinity check covers the whole row, not only the columns a node reads" | removed the guard; narrowed it to the first column only | 12 py, 3 js / 3 js |
| **D046** | the pinned constants in `transform.py` / `transform.ts`; the AST and token scans | `test_transform::test_every_constant_has_the_intended_bit_pattern`, `…test_no_constant_is_left_unpinned`, `…test_constants_equal_their_mathematical_definitions`, `…test_every_arithmetic_step_is_a_narrowed_named_intermediate`, `…test_transform_does_not_vectorize`; JS "every transform constant matches the integer bit pattern Python pins" | flipped one bit of `LN2_LO`, each language separately | 1 py, 1 js — and **every ULP test stayed green**, which is the whole argument for pinning constants as integers |
| **D047** | `predict.Predictor.from_json` structural narrowing, read-only arrays, §13 validation, the reachable-subgraph cycle check | `test_predict::test_node_values_are_loaded_into_a_read_only_float32_array` (22 fixtures), `…test_node_values_cannot_be_mutated_through_the_public_view`, `…test_a_cycle_reachable_from_the_root_raises_rather_than_hanging`, `…test_a_self_loop_raises`, `…test_output_bit_divergences_from_xgboost_are_exactly_the_recorded_set` | parsed `node_values` as float64; left the array writeable; removed the cycle check | 25 / 22 / 2 py — see [Absorption](#absorption) |
| **D048** | `packages/js/src/{transform,artifact,predict}.ts`; the null-prototype transform table and the constructor's own-property guard; the two new `ErrorCode` values | JS "the transform table has a null prototype, so inherited names do not resolve", "the public constructor refuses an outputTransform it does not implement", "a rejected transform yields no object at all, so nothing wrong can be called", "a threshold read back through the public view is the engine's float32" | used an ordinary object literal; removed the own-property guard; reused another error code | 1 / 2 / 1 js |
| **D050** *(schema is behavioural; licence and release configuration are process)* | `schema/xgboost-bridge-v1.schema.json` — `additionalProperties: false` everywhere, two required `description` contents | `test_schema::test_rejects_an_eighth_top_level_key`, `…test_objective_description_states_it_is_non_operative_metadata`, `…test_intercept_description_states_it_is_the_operative_value`, `…test_every_fixture_artifact_validates`; JS `schema.test.js` mirrors the key sets and descriptions | set top-level `additionalProperties: true`; rewrote the `objective` description | 1 py; 1 py + 1 js |
| **D051** *(three code defects are behavioural; the packaging defects are process, except the extra/ceiling coupling)* | ① `export._verify_against_source_margin`; ② `predict._feature_value`; ③ `transform.ts` null-prototype table + `predict.ts` guard; ④ the `export` extra in `packages/python/pyproject.toml` | ① see D027; ② `test_predict::test_a_feature_value_that_is_not_a_number_raises`, `…test_the_refusal_is_a_bridge_error_a_caller_can_catch`, `…test_a_real_nan_is_still_the_missing_value_and_still_routes_by_default_left`; ③ see D048; ④ **added** `test_export::test_the_export_extra_pins_exactly_the_enumerated_version_ceiling` | ① removed the self-check; ② coerced values with bare `float()`; ③ see D048; ④ widened the extra to `xgboost>=3.3,<4` | ① 3 py; ② 16 py; ③ 1+2 js; ④ **0 py before the added test**, 1 py after |
| **D052** *(the `$id` host and the doc/manifest agreement are pinnable; the workflow and metadata changes are process)* | ① `$id` in `schema/xgboost-bridge-v1.schema.json`; ② the export specifier stated in `COMPAT.md` / `README.md` / `packages/python/README.md`; ③ the camelCase hump forms in `UNAMBIGUOUS_RE` and `AMBIGUOUS_IDENTIFIER_RE` | ① **added** `test_schema::test_schema_id_is_hosted_where_the_project_actually_controls_the_namespace`; ② **added** `test_export::test_user_facing_docs_state_the_same_xgboost_specifier_as_the_manifest`; ③ `test_vocabulary_scrub::test_scrub_detects_what_it_claims_to_detect`, extended with six camelCase violations and two ordinary-English negatives | ① pointed `$id` back at the unowned domain; ② restored COMPAT.md's `>=3.3,<4` sentence; ③ dropped the hump alternation; separately, made the hump alternation case-insensitive | ① **0 py before the added test**, 1 py after; ② **0 py before the added test**, 1 py after; ③ 1 py on the dropped alternation, **2 py** on the IGNORECASE variant — the self-test *and* the corpus scan, because the false positives it creates are present in the repository's own prose |

## The map — process / metadata entries

No runtime behaviour, so no revert is possible. Where a check exists anyway it is named, because a
process decision with an executable check is strictly better off than one without.

| ID | Why it is not behavioural | Check, if any |
|---|---|---|
| **D001** | Which XGBoost version verification runs against, and that a second pass records its resolved version. A property of the verification matrix, not of shipped code. | The behavioural shadow — the enumerated ceiling — is D018. The dependency pin itself is now tied to that ceiling by the test added for D051 ④. |
| **D002** | The contents of `.gitignore`. | none |
| **D011** | Where the Node suite imports from, and that typecheck is a separate step. Test-harness topology. | JS "every test file imports the built bundle and never src/ (D011)" — **verified red** (1) by adding a temporary test file containing the forbidden import text, then deleting it |
| **D012** | When an authorship disclosure is published. | none, and none is possible |
| **D013** | Two `requires-python` floors across three manifests. | none |
| **D014** | That test files are `.js` rather than transpiled `.ts`. | implied by the D011 scan, which reads `.test.js` files |
| **D023** | A decision to *measure* rather than design. Resolved by D038, which is behavioural and pinned. | via D038 |
| **D024** | Workflow actions pinned to commit SHAs. | none |
| **D029** | That the vocabulary scrub is executable rather than a manual grep. | `test_vocabulary_scrub::test_scrub_detects_what_it_claims_to_detect` — the scrub's implementation lives in the test file, so its self-test *is* the pin by construction; `…test_scrub_covers_a_meaningful_number_of_files` guards against it silently scanning nothing |
| **D031** | `mpmath` in the dev dependency group. | the transform's mpmath tests would fail to import without it |
| **D033** | Which gate is relative and which absolute, and the inf/NaN/zero rules. Comparison policy, implemented in test and harness code. | `test_predict::test_output_meets_the_relative_gate_against_xgboost`, `…test_no_corpus_output_is_nan`, `…test_cox_overflow_rows_match_xgboost_infinity_as_bit_patterns` |
| **D042** | How concurrent agents schedule suite runs. | none, and none is possible |
| **D049** | The parity *measurement*: cross-language agreement is verification machinery, not behaviour of either predictor. | 24 tests in `test_parity.py`, nine of which exist to prove the harness can fail. **Verified red** (2) by neutering the harness's margin-point comparison: `test_an_injected_one_ulp_difference_fails_at_the_margin_point` and `test_an_injected_negative_zero_normalization_fails_at_the_margin_point` |

## Unpinned before this audit

Six behavioural claims had no test that went red when the behaviour was reverted. Each now has one,
and each added test was confirmed red under the revert and green after.

### 1. D025 — "Leaf iff `left_children[i] == -1`"

Reverting the walk's loop condition to `right_children[node] != LEAF_CHILD` turned **0** tests red
in **both** languages. The two tests coincide at every scalar leaf, so no fitted model can tell
them apart; they part company only on the vector-leaf shape, where a leaf's `right_children` slot
carries a block index.

**Added:** `test_trees::test_the_leaf_test_is_left_children_and_never_right_children` — hands
`walk_margin` a tree whose leaf carries `right_children = 5`, and requires the leaf's own value.
Red 1 under the revert. The reverted walk reaches a *different* leaf and returns `-8.0` where the
answer is `0.25`: an ordinary-looking number, not an error.

**Not added on the JavaScript side, and why.** `walk` is private and `fromJSON` refuses the only
artifact shape that distinguishes the two tests, so the rule is unobservable through the public
API there. That refusal is the absorbing site and is itself pinned — see
[Absorption](#absorption) item 5.

### 2. D037 — "`size_leaf_vector == "1"` for **every** tree"

Truncating the gate's loop to `trees[:1]` turned **0** tests red: every existing case edits
`trees[0]`, and every real model's trees agree.

**Added:** `test_validate::test_size_leaf_vector_is_checked_on_every_tree_not_only_the_first` —
edits the **last** tree of a three-round model. Red 1 under the revert.

### 3. D006 — "No `fromFile` in the JavaScript package"

The existing test checks `Predictor.fromFile` and `Predictor.prototype.fromFile`. It does not check
the spelling a contributor would actually reach for: a plain exported function beside `fromJSON`.
`index.ts` re-exports everything `predict.ts` declares, so adding `export function fromFile`
shipped a filesystem entry point with **0** tests red.

**Added:** JS "the bundle exports no module-level file-reading entry point either (D006)" — scans
the whole export surface. Red 1 under the revert. The scan asserts it would fire on `fromFile` and
`loadArtifactFromPath`, so it cannot pass for want of a pattern.

### 4. D010 — "Reading an artifact and predicting must not require XGBoost"

Moving `import xgboost` to module scope in `objectives.py` turned **0** tests red. Every environment
this repository tests in has XGBoost installed, and D051 already recorded that nothing is ever
tested as an installed package — so `import xgboost_bridge` would have failed on every base install
with the suite fully green.

**Added:** `test_scaffold::test_the_package_imports_and_predicts_with_the_export_extra_absent` —
a child process installs a `sys.meta_path` finder that refuses `xgboost`, then loads a corpus
fixture and predicts. Red 1 under the revert. The margin is checked against the fixture's recorded
XGBoost bit pattern; the output is checked against this process's own value, because XGBoost's
recorded output differs by one bit on a pinned set of rows by construction (D032, D047) and that row
is one of them.

### 5. D015 — "`provenance` is read by no predictor, in either language"

`test_provenance_is_exposed_and_read_by_no_prediction` asserts only that the block is exposed with
the right keys — which a predictor reading it passes. Reading `provenance["base_score"]` on the
prediction path turned **0** tests red in **both** languages. That field holds the value XGBoost
stored *unclamped and untransformed*, so anything derived from it is wrong by up to 13.8 in margin
space (D035).

**Added**, mirroring D028's pair exactly, because D047 established that one check is not enough:

- `test_predict::test_no_prediction_path_function_reads_the_provenance_block` — AST scan.
- `test_predict::test_predictions_are_unchanged_when_the_provenance_block_is_corrupted` — the block
  is replaced after loading and every bit pattern must be unchanged.
- JS "no prediction method mentions `provenance` in its own source (D015)" — method-source scan plus
  the behavioural half.

Red under a plain read: 1 py, 2 js. Red under an obfuscated read that changes the number: 38 py
(23 of them the new behavioural case) and 8 js. **The Python source scan stays green under the
obfuscated revert**, so neither half is redundant — the same finding D047 recorded for `objective`.

### 6. D051 ④ / D018 — the `export` extra must equal the enumerated ceiling

Widening the extra back to `xgboost>=3.3,<4` — the exact defect D051 fixed — turned **0** tests red,
because the workspace pins the version and every test runs against the source tree. A fresh
`pip install "xgboost-bridge[export]"` would resolve 3.4.0 and raise `UnsupportedVersionError` on
the first `export_model` call.

**Added:** `test_export::test_the_export_extra_pins_exactly_the_enumerated_version_ceiling` — reads
the manifest and requires the extra to *equal* `xgboost==` + `DEFAULT_TESTED_VERSIONS`. A range
spelling fails even when it resolves correctly today, which is the point: `3.3.1` would also be
untested and would also raise. Red 1 under the revert.

## Absorption

Cases where a site cannot be reverted in isolation because another site already produces the same
number. Each absorbing site is named, and each is itself pinned — a chain that ends in nothing
pinned would be no better than an unpinned site.

1. **Python parse-time float32 narrowing of `node_values`** (D004, D047). The revert turns 25 tests
   red and **zero prediction or parity tests** — every red is structural (`dtype`, read-only-ness,
   the non-finite refusals). **Absorbing site:** `trees.walk_margin`, which narrows both operands,
   and `np.float32(np.float64(x)) == np.float32(x)`, so narrowing at parse is idempotent with
   narrowing at comparison. The pin is therefore structural too, exactly as D047 disclosed. What
   differs is every *other* consumer of the array, which is §9.2's argument for the site.
2. **JavaScript walk-site intercept cast** `Math.fround(this.intercept)` — 0 red. **Absorbing site:**
   `artifact.ts narrow()`, which narrows the intercept at parse time; reverting *that* turns 2 red.
3. **JavaScript threshold-side cast** in `walk` — 0 red. **Absorbing site:** `artifact.ts
   readNodeValues`, whose `Float32Array` makes the value already float32; reverting *that* turns 5
   red.
4. **JavaScript leaf-value cast at the add** — 0 red. **Absorbing site:** the same `Float32Array`.
5. **The leaf test in JavaScript** (D025) — 0 red, and unpinnable through the public API.
   **Absorbing site:** the §13 leaf-shape refusal in `artifact.ts checkChildLinks` — a leaf whose
   right child is not `-1` is refused, so on every loadable artifact the two leaf tests are
   equivalent. Verified: reverting that refusal turns 1 JS test red ("a leaf whose right child is
   not -1 raises"), and the Python equivalent in `predict._check_child_links` turns 1 Python test
   red. The rule itself is pinned where it is observable, on `walk_margin`.

The absorption pattern is **mirrored** between the languages, as D048 recorded: Python's absorbed
site is the parse-time one, JavaScript's are the three comparison-site casts. Which site looks
redundant is a property of the host language's promotion rules, not of the algorithm. Every absorbed
cast stays, per D045's reasoning: it is the defence for artifacts this exporter did not produce.

## Reverts that are no-ops

Not gaps. Recorded so the zero is not read as one.

- **`SUPPORTED_BOOSTERS` widened to `("gbtree", "dart")`** — 0 red. dart serializes
  `gradient_booster.name` as `"gbtree"`, so the string `"dart"` never appears in that field and the
  allow-list entry is unreachable. The load-bearing member is the *absence* of `"gblinear"`, and
  that is pinned: admitting it turns `test_rejects_gblinear` red.
- **A `margin_decimal` entry perturbed in the 16th significant digit** — 0 red, because both
  decimals narrow to the same float32. Perturbing it to `-2.5` turns 1 red. The non-normative
  decimal fields are pinned exactly to float32 resolution, which is the resolution that matters.

## Specified but unimplemented

**None found.** No entry in `DECISIONS.md` describes a refusal or a value that the shipped code does
not produce. D022 itself is now implemented, by D045, and is pinned by 12 Python and 3 Node tests.

Three weaker observations, reported rather than acted on because each is a scope question rather
than a missing pin:

1. **D020's "explicit whitelist" is a rule, not a mechanism.** No whitelist structure exists in
   `export.py`; the behaviour D020 asserts — nothing under `learner.attributes` reaches the artifact —
   holds because `_build_provenance` copies three named fields and nothing else, and that *is*
   pinned (2 red). A future contributor looking for the whitelist to add an entry to will not find
   one.
2. **D030's "at least 1e6 sample points per objective" is not run by any automated gate.** The
   always-on ULP tests use 3000 deterministic points plus the known-worst inputs; the 1e6-point
   sweep is a manual invocation, which `test_transform.py`'s own module docstring states plainly and
   which is consistent with D051's recorded-not-fixed note about generator self-checks. The max-ULP
   *bound* is separately pinned at the inputs that produce it, so the number is not unguarded — the
   sample size is.
3. **`COMPAT.md`'s tested-version list is not compared to `DEFAULT_TESTED_VERSIONS` by any test.**
   Same shape as the `export` extra defect, but a divergence here produces documentation drift
   rather than a wrong number or a broken install, so no test was added.

## Gate numbers at the end of this audit

| Check | Result |
|---|---|
| Python suite | **962 passed**, 0 failed (was 960; +2 for D052) |
| Node suite | **112 pass**, 0 fail, 0 skipped, 0 todo (was 110; +2) |
| `tsc --noEmit` | clean, run as a step separate from the build |
| Margin parity, Python vs JavaScript | **0** mismatches over 299 rows |
| Output parity, Python vs JavaScript | **0** mismatches |
| Refusal disagreements / input-bit disagreements | **0** / **0** |
| Source files altered | **none** — every temporary revert restored and verified byte-identical with `shasum -a 256` |
