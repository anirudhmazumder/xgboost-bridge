"""Verification of the adversarial fixture corpus written by
``fixtures/generate/adversarial.py``.

Two jobs, kept separate because they have different oracles:

1. **Composition checks** -- does each fixture actually exhibit the specific
   property its case (A1-A7) requires? These check the fixture's own
   recorded metadata and, where a real independent signal exists (A5's raw
   deletion marker), cross-check it against that signal rather than against
   a re-derivation of the same computation.

2. **The broken-variant table** -- six deliberately wrong copies of the
   margin walk, each reverting exactly one protection FORMAT.md section 10
   specifies, run against every row of every value-producing fixture in
   this corpus. The oracle for "wrong" is ``expected_margin``, recorded once
   from XGBoost's own ``predict(output_margin=True)`` at generation time --
   never a re-derivation of the walk. This file never calls
   ``booster.predict`` itself; XGBoost need not be installed to run it,
   which is the entire point of storing ground truth as bit patterns
   (D044).

Per CLAUDE.md and D019: **narrowing after every accumulator addition
partially absorbs leaf-value narrowing**, so a broken variant that reverts
both at once pins neither -- each of the six protections below is reverted
**alone**. Per FORMAT.md section 10.1's own correction, a broken variant fed
*weak* Python floats is invisible for four of the five float32 sites, because
NEP 50 quietly promotes a weak scalar to match whatever strong-typed operand
it meets. Every row below is therefore fed to every variant as a genuine
``np.float64`` array (a *strong* scalar container, exactly the shape
``matrix[i]`` produces for a real caller) -- never a Python list of floats.

One further point this file's own docstring in ``adversarial.py`` explains at
length and is only summarized here: two of the six broken variants (reading
a threshold or a leaf value without its float32 narrowing) are provably
no-ops if fed this fixture corpus's own **exported artifact** values,
because D044's emission rule already makes those values recover their exact
float32 bit-for-bit at any width. Only XGBoost's own *raw*, short-decimal
``split_conditions`` token -- read directly from
``booster.save_raw(raw_format="json")`` before export ever narrows anything,
and carried for this purpose alone in ``meta.raw_node_values_per_tree`` --
carries the genuine imprecision those two variants need to be observable at
all. This mirrors ``packages/python/tests/test_trees.py``'s own
``_un_narrowed_trees`` helper, reimplemented here rather than imported,
since this suite's job is to distrust the corpus independently rather than
to share a helper with the thing it is checking.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_FIXTURES_ROOT = Path(__file__).resolve().parents[1]
if str(_FIXTURES_ROOT) not in sys.path:
    sys.path.insert(0, str(_FIXTURES_ROOT))

from generate.adversarial import (  # noqa: E402
    ADVERSARIAL_DIR,
    FIXTURE_NAMES,
    REFUSAL_ONLY_FIXTURE,
)

from xgboost_bridge.trees import reachable_nodes, walk_margin  # noqa: E402


def _load_corpus() -> dict[str, dict[str, Any]]:
    if not ADVERSARIAL_DIR.is_dir():
        return {}
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(ADVERSARIAL_DIR.glob("*.json"))
    }


#: Loaded once at collection time, exactly as `fixtures/tests/test_corpus.py` does.
CORPUS: dict[str, dict[str, Any]] = _load_corpus()

#: Every fixture except the refusal-only one -- i.e. every fixture that
#: carries a real XGBoost margin/output in every row.
VALUE_PRODUCING_NAMES: tuple[str, ...] = tuple(
    name for name in FIXTURE_NAMES if name != REFUSAL_ONLY_FIXTURE
)


def _bits_to_float32(bits: str) -> np.float32:
    assert bits.startswith("0x") and len(bits) == 10, f"not a uint32 hex bit pattern: {bits!r}"
    return np.uint32(int(bits, 16)).view(np.float32)[()]


def _bits_of(value: float) -> str:
    return f"0x{int(np.float32(value).view(np.uint32)):08x}"


def _row_to_features(row: list[float | str | None]) -> np.ndarray:
    """The wire encoding back to a strong ``np.float64`` row.

    ``null`` -> NaN (D044's existing convention). This corpus's
    value-producing fixtures never emit the ``"inf"``/``"-inf"`` string
    sentinel -- that encoding is unique to the refusal-only fixture, which
    this helper is never called on.
    """
    values: list[float] = []
    for value in row:
        if value is None:
            values.append(float("nan"))
        else:
            assert isinstance(value, (int, float)), f"unexpected non-finite sentinel in value row: {value!r}"
            values.append(float(value))
    return np.asarray(values, dtype=np.float64)


# ---------------------------------------------------------------------------
# Basic presence and shape
# ---------------------------------------------------------------------------


def test_adversarial_corpus_is_non_empty() -> None:
    assert CORPUS, f"adversarial fixture corpus at {ADVERSARIAL_DIR} is empty"


def test_every_required_adversarial_fixture_is_present() -> None:
    missing = set(FIXTURE_NAMES) - set(CORPUS)
    assert not missing, f"required adversarial fixtures missing: {sorted(missing)}"
    extra = set(CORPUS) - set(FIXTURE_NAMES)
    assert not extra, (
        f"adversarial corpus contains fixtures not declared in FIXTURE_NAMES: {sorted(extra)} -- "
        "add them to fixtures/generate/adversarial.py's builder tuple"
    )


def test_adversarial_corpus_lives_in_its_own_directory_not_the_ordinary_corpus() -> None:
    """The ordinary corpus's own exact-fixture-list test globs
    ``fixtures/corpus/*.json`` non-recursively; this fixture set must never
    land there or it silently breaks that check's completeness guarantee."""
    ordinary_dir = ADVERSARIAL_DIR.parent
    assert ADVERSARIAL_DIR != ordinary_dir
    assert ADVERSARIAL_DIR.parent == ordinary_dir
    for name in FIXTURE_NAMES:
        assert not (ordinary_dir / f"{name}.json").exists(), (
            f"{name}.json must not exist directly under {ordinary_dir}; "
            "it belongs only under fixtures/corpus/adversarial/"
        )


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_ground_truth_arrays_match_row_count(name: str) -> None:
    fixture = CORPUS[name]
    row_count = len(fixture["rows"])
    for key in ("expected_margin", "expected_output", "margin_decimal", "output_decimal"):
        assert len(fixture[key]) == row_count, f"{name}: {key} length mismatch"
    assert fixture["meta"]["row_count"] == row_count


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_meta_records_ground_truth_provenance(name: str) -> None:
    fixture = CORPUS[name]
    for key in ("xgboost_version", "numpy_version", "objective", "seed", "row_count", "description"):
        assert key in fixture["meta"], f"{name}: meta missing {key!r}"
    assert fixture["meta"]["xgboost_version"] != "latest"


@pytest.mark.parametrize("name", sorted(n for n in CORPUS if n != REFUSAL_ONLY_FIXTURE))
def test_value_producing_fixtures_carry_a_real_bit_pattern_on_every_row(name: str) -> None:
    """Every row of a value-producing fixture has real ground truth --
    the invariant this whole corpus rests on, except for the one fixture
    explicitly exempted by design (A6's refusal rows)."""
    fixture = CORPUS[name]
    for index, (margin, output) in enumerate(
        zip(fixture["expected_margin"], fixture["expected_output"])
    ):
        assert isinstance(margin, str) and margin.startswith("0x"), (
            f"{name}: expected_margin[{index}] is not a bit-pattern string: {margin!r}"
        )
        assert isinstance(output, str) and output.startswith("0x"), (
            f"{name}: expected_output[{index}] is not a bit-pattern string: {output!r}"
        )


@pytest.mark.parametrize("name", sorted(n for n in CORPUS if n != REFUSAL_ONLY_FIXTURE))
def test_rewalk_reproduces_expected_margin_bit_for_bit(name: str) -> None:
    """``walk_margin`` over the recorded artifact must match XGBoost's own
    margin, row for row -- the same independent-oracle check
    ``fixtures/tests/test_corpus.py`` runs for the ordinary corpus, applied
    here to rows deliberately chosen to be difficult rather than realistic."""
    fixture = CORPUS[name]
    trees = fixture["artifact"]["trees"]
    intercept = fixture["artifact"]["intercept"]
    rows = fixture["rows"]
    expected = fixture["expected_margin"]

    mismatches: list[tuple[int, str, str]] = []
    for index, (row, expected_bits) in enumerate(zip(rows, expected)):
        computed = walk_margin(trees, intercept, _row_to_features(row))
        computed_bits = _bits_of(computed)
        if computed_bits != expected_bits:
            mismatches.append((index, expected_bits, computed_bits))

    assert not mismatches, (
        f"{name}: walk_margin disagreed with expected_margin on "
        f"{len(mismatches)}/{len(rows)} rows: {mismatches}"
    )


# ---------------------------------------------------------------------------
# A1 -- float32/float64 threshold disagreement
# ---------------------------------------------------------------------------


def test_a1_hazardous_thresholds_were_measured_not_assumed() -> None:
    meta = CORPUS["float32_threshold_disagreement"]["meta"]
    hazardous = meta["hazardous_node_count"]
    total = meta["total_internal_node_count"]
    resolved = meta["resolved_row_count"]
    print(
        f"\nA1: {hazardous}/{total} internal nodes measured hazardous "
        f"(raw float64 parse > float32 narrowing); {resolved} resolved to a row"
    )
    assert 0 < hazardous <= total
    assert 0 < resolved <= hazardous


def test_a1_every_row_sits_exactly_on_its_own_float32_threshold() -> None:
    """Each row's targeted feature value must be bit-identical to the
    threshold it targets -- otherwise the row is not testing what its own
    meta claims it tests."""
    fixture = CORPUS["float32_threshold_disagreement"]
    trees = fixture["artifact"]["trees"]
    targets = fixture["meta"]["targets"]
    for row, target in zip(fixture["rows"], targets):
        tree = trees[target["tree_index"]]
        threshold = tree["node_values"][target["node_index"]]
        feature = tree["split_indices"][target["node_index"]]
        assert _bits_of(row[feature]) == _bits_of(threshold)


# ---------------------------------------------------------------------------
# A2 -- equality-boundary routing
# ---------------------------------------------------------------------------


def test_a2_below_exact_above_are_three_distinct_float32_values_per_node() -> None:
    fixture = CORPUS["equality_boundary_routing"]
    for record in fixture["meta"]["node_records"]:
        bits = record["bits"]
        assert len({bits["below"], bits["exact"], bits["above"]}) == 3, (
            f"node {record}: below/exact/above are not three distinct float32 values"
        )


def test_a2_below_and_above_route_to_a_different_margin_than_exact() -> None:
    """The equality boundary is only pinned if the three rows in a triple
    actually produce different margins -- otherwise the triple is
    decorative, per CLAUDE.md's own warning about boundary fixtures a buggy
    build could still pass."""
    fixture = CORPUS["equality_boundary_routing"]
    labels = fixture["meta"]["row_labels"]
    margins = fixture["expected_margin"]
    by_node: dict[str, dict[str, str]] = {}
    for label, margin in zip(labels, margins):
        tree_index, node_index, which = label.split(":")
        by_node.setdefault(f"{tree_index}:{node_index}", {})[which] = margin

    at_least_one_distinguishing_triple = False
    for node_key, triple in by_node.items():
        assert set(triple) == {"below", "exact", "above"}, f"{node_key}: incomplete triple {triple}"
        if triple["below"] != triple["exact"] or triple["exact"] != triple["above"]:
            at_least_one_distinguishing_triple = True
    assert at_least_one_distinguishing_triple, (
        "equality_boundary_routing: no triple's margin differs across below/exact/above -- "
        "the fixture would not detect a broken equality direction"
    )


# ---------------------------------------------------------------------------
# A3 -- extreme and denormal features
# ---------------------------------------------------------------------------


def test_a3_rows_contain_the_required_extreme_values() -> None:
    fixture = CORPUS["extreme_and_denormal_features"]
    flat = [value for row in fixture["rows"] for value in row]
    smallest_subnormal = float(np.float32(1.4e-45))
    negative_subnormal = float(np.float32(-5.6e-44))
    near_max = float(np.float32(3.4e38))

    assert smallest_subnormal in flat
    assert negative_subnormal in flat
    assert near_max in flat
    assert -near_max in flat
    assert 0.0 in flat
    assert any(_bits_of(v) == _bits_of(-0.0) for v in flat), "no exact -0.0 in any row"
    assert 16_777_217.0 in flat, "missing the first integer float32 cannot represent exactly"
    assert -16_777_217.0 in flat


def test_a3_the_lossy_integer_actually_loses_precision_in_float32() -> None:
    """Confirms the adversarial value chosen for A3 is adversarial: 2**24 + 1
    narrows to a *different* float32 than itself -- otherwise the row would
    not be exercising the "integers large enough to lose float32 precision"
    requirement at all."""
    value = 16_777_217.0
    assert float(np.float32(value)) != value
    assert float(np.float32(value)) == 16_777_216.0


# ---------------------------------------------------------------------------
# A4 -- the logistic clamp floor, at the output level
# ---------------------------------------------------------------------------


#: XGBoost's own measured logistic clamp floor output (FORMAT.md section
#: 5.2, D032): the exact float32 value `predict()` returns for any margin at
#: or below f32(-88.7), and never 0.0. This is an independently-recorded
#: empirical constant, not a re-derivation of anything this corpus computes
#: -- the oracle for the check below is that separately-measured number.
_LOGISTIC_CLAMP_FLOOR_OUTPUT_BITS = _bits_of(3.006635794144578e-39)


def test_a4_at_least_one_row_is_below_the_logistic_clamp_floor() -> None:
    meta = CORPUS["logistic_clamp_floor_output"]["meta"]
    below = meta["below_clamp_row_count"]
    lowest = meta["lowest_margin_achieved_overall"]
    print(f"\nA4: {below} rows below the logistic clamp floor; lowest margin achieved {lowest}")
    assert below > 0, (
        f"logistic_clamp_floor_output: 0 rows below the clamp floor; lowest margin "
        f"achieved was {lowest} -- could not push a fitted model's margins below f32(-88.7)"
    )


def test_a4_below_clamp_rows_have_the_exact_measured_floor_output_bits() -> None:
    """Independent-oracle check: XGBoost's own recorded output for each
    below-clamp row must equal the *separately measured* floor constant
    (D032), not merely be small. A float64 sigmoid would instead give a
    relative error of 1.0 here (CLAUDE.md) -- this check is what would catch
    that, since it compares against a hardcoded, independently-measured
    number rather than against this fixture's own generation."""
    fixture = CORPUS["logistic_clamp_floor_output"]
    floor = np.float32(-88.7)
    below_rows_checked = 0
    for margin_bits, output_bits in zip(fixture["expected_margin"], fixture["expected_output"]):
        margin = _bits_to_float32(margin_bits)
        if margin < floor:
            assert output_bits == _LOGISTIC_CLAMP_FLOOR_OUTPUT_BITS, (
                f"row at margin {margin!r} (below clamp) has output bits {output_bits}, "
                f"expected the measured floor bits {_LOGISTIC_CLAMP_FLOOR_OUTPUT_BITS}"
            )
            below_rows_checked += 1
    assert below_rows_checked == fixture["meta"]["below_clamp_row_count"]


# ---------------------------------------------------------------------------
# A5 -- neutralization detection
# ---------------------------------------------------------------------------


def test_a5_dead_indices_agree_with_the_raw_deletion_marker() -> None:
    """The independent-oracle check: reachability from the root (computed on
    the *exported*, already-neutralized artifact) must agree with the raw
    ``split_indices == 2147483647`` marker, read directly from the *source*
    document before export touches anything. These are two different
    signals computed two different ways; agreement is evidence, not an
    assumption (D027, CLAUDE.md)."""
    meta = CORPUS["gamma_pruned_neutralization"]["meta"]
    dead = meta["dead_node_indices_per_tree"]
    raw = meta["raw_deleted_marker_indices_per_tree"]
    assert len(dead) == len(raw)
    for tree_index, (dead_here, raw_here) in enumerate(zip(dead, raw)):
        assert dead_here == raw_here, (
            f"gamma_pruned_neutralization: tree {tree_index} disagreement between "
            f"reachability-based dead set {dead_here} and raw deletion marker {raw_here}"
        )


def test_a5_at_least_one_tree_has_genuinely_dead_nodes() -> None:
    meta = CORPUS["gamma_pruned_neutralization"]["meta"]
    total_dead = sum(len(indices) for indices in meta["dead_node_indices_per_tree"])
    print(f"\nA5: dead node indices per tree = {meta['dead_node_indices_per_tree']}")
    assert total_dead > 0


def test_a5_no_dead_index_is_reachable_from_the_root() -> None:
    fixture = CORPUS["gamma_pruned_neutralization"]
    dead = fixture["meta"]["dead_node_indices_per_tree"]
    for tree, dead_here in zip(fixture["artifact"]["trees"], dead):
        live = reachable_nodes(tree)
        overlap = live & set(dead_here)
        assert not overlap, f"dead indices {overlap} are reachable from the root"


# ---------------------------------------------------------------------------
# A6a -- missing values, value-producing
# ---------------------------------------------------------------------------


def test_a6a_exercises_both_default_left_directions() -> None:
    fixture = CORPUS["missing_value_adversarial"]
    meta = fixture["meta"]
    left_target = meta["default_left_1_target"]
    right_target = meta["default_left_0_target"]

    tree = fixture["artifact"]["trees"][left_target["tree_index"]]
    assert tree["default_left"][left_target["node_index"]] == 1
    tree = fixture["artifact"]["trees"][right_target["tree_index"]]
    assert tree["default_left"][right_target["node_index"]] == 0

    assert any(value is None for value in fixture["rows"][0])
    assert any(value is None for value in fixture["rows"][1])


def test_a6a_has_a_row_with_every_feature_missing() -> None:
    fixture = CORPUS["missing_value_adversarial"]
    assert any(all(value is None for value in row) for row in fixture["rows"]), (
        "missing_value_adversarial: no row has every feature missing"
    )


# ---------------------------------------------------------------------------
# A6b -- non-finite input, refusal only
# ---------------------------------------------------------------------------


def test_a6b_carries_no_ground_truth() -> None:
    fixture = CORPUS[REFUSAL_ONLY_FIXTURE]
    assert all(value is None for value in fixture["expected_margin"])
    assert all(value is None for value in fixture["expected_output"])
    assert all(value is None for value in fixture["margin_decimal"])
    assert all(value is None for value in fixture["output_decimal"])
    assert fixture["meta"]["expected_behavior"] == "raise"


def test_a6b_every_row_contains_a_non_finite_sentinel() -> None:
    fixture = CORPUS[REFUSAL_ONLY_FIXTURE]
    for row in fixture["rows"]:
        assert any(value in ("inf", "-inf") for value in row), (
            f"non_finite_input_refusal: row {row!r} carries no +/-inf sentinel"
        )


def test_a6b_is_a_separate_fixture_from_every_value_producing_case() -> None:
    """A consumer reading fixture names alone must not be able to confuse a
    refusal-only fixture with a value-producing one."""
    assert REFUSAL_ONLY_FIXTURE not in (
        set(FIXTURE_NAMES) - {REFUSAL_ONLY_FIXTURE}
    )
    for name in FIXTURE_NAMES:
        if name == REFUSAL_ONLY_FIXTURE:
            continue
        rows = CORPUS[name]["rows"]
        for row in rows:
            for value in row:
                assert value not in ("inf", "-inf"), (
                    f"{name}: value-producing fixture unexpectedly carries a non-finite "
                    "sentinel that belongs only in the refusal-only fixture"
                )


# ---------------------------------------------------------------------------
# A7 -- zero-leaf intercept isolation
# ---------------------------------------------------------------------------


def test_a7_every_tree_is_a_single_leaf_node() -> None:
    fixture = CORPUS["zero_leaf_intercept_isolation"]
    for tree in fixture["artifact"]["trees"]:
        assert tree["left_children"] == [-1], f"not a single-node (root-is-leaf) tree: {tree}"


def test_a7_every_leaf_is_exactly_zero() -> None:
    fixture = CORPUS["zero_leaf_intercept_isolation"]
    for tree in fixture["artifact"]["trees"]:
        assert tree["node_values"][0] == 0.0, f"leaf is {tree['node_values'][0]!r}, not exactly 0.0"


def test_a7_every_row_margin_equals_the_intercept_exactly() -> None:
    fixture = CORPUS["zero_leaf_intercept_isolation"]
    intercept_bits = _bits_of(fixture["artifact"]["intercept"])
    for margin_bits in fixture["expected_margin"]:
        assert margin_bits == intercept_bits, (
            f"row margin {margin_bits} != intercept {intercept_bits} -- a tree diluted the margin"
        )


def test_a7_intercept_is_not_at_a_degenerate_value() -> None:
    """D025's own trap: at base_score=0.5 (logistic) or 1.0 (Cox) the
    intercept collapses to a value at which every broken accumulation
    variant also scores full marks. A7's whole point is to isolate the
    intercept, so it must not accidentally sit at the one place that traps
    every other fixture."""
    base_score = CORPUS["zero_leaf_intercept_isolation"]["meta"]["base_score"]
    assert base_score is not None
    assert abs(base_score - 0.5) > 0.1


# ---------------------------------------------------------------------------
# The broken-variant table (VERIFICATION section)
# ---------------------------------------------------------------------------
#
# Six deliberately wrong copies of the margin walk, each reverting exactly
# one FORMAT.md section 10 protection, in isolation (never two at once, per
# D019). Every row of every value-producing fixture in this corpus is fed to
# every variant as a strong np.float64 array. The oracle is
# `expected_margin`, recorded once from XGBoost's own predict() at
# generation time -- these functions never call XGBoost.


def _strong_clean_trees(trees: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The artifact's own trees, with ``node_values`` as a strong,
    explicitly float32-typed array -- never a Python list of weak floats."""
    out = []
    for tree in trees:
        out.append(
            {
                "default_left": tree["default_left"],
                "left_children": tree["left_children"],
                "right_children": tree["right_children"],
                "split_indices": tree["split_indices"],
                "node_values": np.asarray([np.float32(v) for v in tree["node_values"]], dtype=np.float32),
            }
        )
    return out


def _strong_raw_trees(
    trees: list[dict[str, Any]], raw_per_tree: list[list[float]], role: str
) -> list[dict[str, Any]]:
    """Like ``_strong_clean_trees``, but nodes matching ``role`` get the raw,
    un-narrowed float64 parse of their own token instead of the exported
    artifact's (already narrowed-and-widened-back) value.

    ``role="internal"``: thresholds carry genuine extra precision, leaves
    stay clean -- this is what makes "cast only the sample value, not the
    threshold" observable at all (see this module's own docstring).
    ``role="leaf"``: the reverse, for "leaf values un-narrowed".
    """
    assert role in ("internal", "leaf")
    out = []
    for tree, raw_values in zip(trees, raw_per_tree, strict=True):
        left_children = tree["left_children"]
        values = []
        for index in range(len(left_children)):
            is_internal = left_children[index] != -1
            wants_raw = (role == "internal" and is_internal) or (role == "leaf" and not is_internal)
            values.append(raw_values[index] if wants_raw else float(np.float32(tree["node_values"][index])))
        out.append(
            {
                "default_left": tree["default_left"],
                "left_children": tree["left_children"],
                "right_children": tree["right_children"],
                "split_indices": tree["split_indices"],
                # A strong np.float64 array -- the only shape in which an
                # un-narrowed value's dtype survives contact with the
                # accumulator's np.float32 (FORMAT.md section 10.1).
                "node_values": np.asarray(values, dtype=np.float64),
            }
        )
    return out


def _leaf_lookup(tree: dict[str, Any], values: np.ndarray) -> tuple[int, Any]:
    """Walk to the leaf using the always-correct routing rule (both sides
    cast to float32, strict '<', NaN by default_left) and return
    ``(leaf_index, node_values[leaf_index])``.

    Shared by every broken variant below whose deviation is *not* in
    routing -- i.e. everything except the '<=' variant, which inlines its
    own routing. Routing is identical to FORMAT.md section 10; the
    deviation each variant tests lives entirely in what happens with the
    leaf value once it is found, or in what array the routing itself reads
    from (``trees`` is the caller's choice, not this function's).
    """
    node = 0
    left = tree["left_children"]
    right = tree["right_children"]
    split_indices = tree["split_indices"]
    node_values = tree["node_values"]
    default_left = tree["default_left"]
    while left[node] != -1:
        value = values[split_indices[node]]
        if value != value:
            node = left[node] if default_left[node] == 1 else right[node]
        elif np.float32(value) < np.float32(node_values[node]):
            node = left[node]
        else:
            node = right[node]
    return node, node_values[node]


def _walk_cast_only_sample(trees: list[dict[str, Any]], intercept: float, values: np.ndarray) -> np.float32:
    """Revert the threshold-side cast: the sample is cast, the threshold
    (read from ``trees``, which the caller must build with role="internal"
    raw values) is not."""
    accumulator = np.float32(intercept)
    for tree in trees:
        node = 0
        left = tree["left_children"]
        right = tree["right_children"]
        split_indices = tree["split_indices"]
        node_values = tree["node_values"]
        default_left = tree["default_left"]
        while left[node] != -1:
            value = values[split_indices[node]]
            if value != value:
                node = left[node] if default_left[node] == 1 else right[node]
            elif np.float32(value) < node_values[node]:  # threshold NOT cast
                node = left[node]
            else:
                node = right[node]
        accumulator = np.float32(accumulator + np.float32(node_values[node]))
    return accumulator


def _walk_cast_only_threshold(trees: list[dict[str, Any]], intercept: float, values: np.ndarray) -> np.float32:
    """Revert the sample-side cast: the threshold is cast, the sample is not."""
    accumulator = np.float32(intercept)
    for tree in trees:
        node = 0
        left = tree["left_children"]
        right = tree["right_children"]
        split_indices = tree["split_indices"]
        node_values = tree["node_values"]
        default_left = tree["default_left"]
        while left[node] != -1:
            value = values[split_indices[node]]
            if value != value:
                node = left[node] if default_left[node] == 1 else right[node]
            elif value < np.float32(node_values[node]):  # sample NOT cast
                node = left[node]
            else:
                node = right[node]
        accumulator = np.float32(accumulator + np.float32(node_values[node]))
    return accumulator


def _walk_leq(trees: list[dict[str, Any]], intercept: float, values: np.ndarray) -> np.float32:
    """Revert strict '<' to '<=': equality now routes LEFT instead of RIGHT."""
    accumulator = np.float32(intercept)
    for tree in trees:
        node = 0
        left = tree["left_children"]
        right = tree["right_children"]
        split_indices = tree["split_indices"]
        node_values = tree["node_values"]
        default_left = tree["default_left"]
        while left[node] != -1:
            value = values[split_indices[node]]
            if value != value:
                node = left[node] if default_left[node] == 1 else right[node]
            elif np.float32(value) <= np.float32(node_values[node]):  # '<=' instead of '<'
                node = left[node]
            else:
                node = right[node]
        accumulator = np.float32(accumulator + np.float32(node_values[node]))
    return accumulator


def _walk_leaf_unnarrowed(trees: list[dict[str, Any]], intercept: float, values: np.ndarray) -> np.float32:
    """Correct routing throughout; the leaf value found is added without its
    own float32 narrowing (``trees`` must be built with role="leaf" raw
    values). The accumulator is still narrowed after the add, per spec --
    this variant isolates the leaf-read site alone (D019)."""
    accumulator = np.float32(intercept)
    for tree in trees:
        _, leaf_value = _leaf_lookup(tree, values)
        accumulator = np.float32(accumulator + leaf_value)  # leaf value NOT cast
    return accumulator


def _walk_intercept_last(trees: list[dict[str, Any]], intercept: float, values: np.ndarray) -> np.float32:
    """Correct routing and leaf narrowing throughout; the intercept is added
    once at the end instead of seeding the accumulator."""
    accumulator = np.float32(0.0)
    for tree in trees:
        _, leaf_value = _leaf_lookup(tree, values)
        accumulator = np.float32(accumulator + np.float32(leaf_value))
    return np.float32(accumulator + np.float32(intercept))


def _walk_reverse_order(trees: list[dict[str, Any]], intercept: float, values: np.ndarray) -> np.float32:
    """Correct in every other respect; trees are walked in reverse array order."""
    accumulator = np.float32(intercept)
    for tree in reversed(trees):
        _, leaf_value = _leaf_lookup(tree, values)
        accumulator = np.float32(accumulator + np.float32(leaf_value))
    return accumulator


def _wrong_row_count(name: str) -> dict[str, int]:
    """Run all six broken variants against one fixture's rows; return the
    count of rows on which each variant disagrees with `expected_margin`."""
    fixture = CORPUS[name]
    trees = fixture["artifact"]["trees"]
    intercept = fixture["artifact"]["intercept"]
    raw_per_tree = fixture["meta"]["raw_node_values_per_tree"]
    rows = fixture["rows"]
    expected = fixture["expected_margin"]

    trees_raw_internal = _strong_raw_trees(trees, raw_per_tree, "internal")
    trees_raw_leaf = _strong_raw_trees(trees, raw_per_tree, "leaf")
    trees_clean = _strong_clean_trees(trees)

    counts = {
        "cast_only_sample": 0,
        "cast_only_threshold": 0,
        "leq_instead_of_lt": 0,
        "leaf_unnarrowed": 0,
        "intercept_added_last": 0,
        "reverse_tree_order": 0,
    }
    for row, expected_bits in zip(rows, expected):
        values = _row_to_features(row)
        assert values.dtype == np.float64  # the strong container the walk needs to be tested at all

        if _bits_of(_walk_cast_only_sample(trees_raw_internal, intercept, values)) != expected_bits:
            counts["cast_only_sample"] += 1
        if _bits_of(_walk_cast_only_threshold(trees_clean, intercept, values)) != expected_bits:
            counts["cast_only_threshold"] += 1
        if _bits_of(_walk_leq(trees_clean, intercept, values)) != expected_bits:
            counts["leq_instead_of_lt"] += 1
        if _bits_of(_walk_leaf_unnarrowed(trees_raw_leaf, intercept, values)) != expected_bits:
            counts["leaf_unnarrowed"] += 1
        if _bits_of(_walk_intercept_last(trees_clean, intercept, values)) != expected_bits:
            counts["intercept_added_last"] += 1
        if _bits_of(_walk_reverse_order(trees_clean, intercept, values)) != expected_bits:
            counts["reverse_tree_order"] += 1

    return counts


def _total_wrong_row_counts() -> tuple[dict[str, int], int]:
    totals = {
        "cast_only_sample": 0,
        "cast_only_threshold": 0,
        "leq_instead_of_lt": 0,
        "leaf_unnarrowed": 0,
        "intercept_added_last": 0,
        "reverse_tree_order": 0,
    }
    total_rows = 0
    for name in VALUE_PRODUCING_NAMES:
        counts = _wrong_row_count(name)
        total_rows += len(CORPUS[name]["rows"])
        for key, value in counts.items():
            totals[key] += value
    return totals, total_rows


def test_broken_variant_table_every_protection_shows_a_measured_wrong_count() -> None:
    """The verification section's own requirement, stated as a single test:
    every one of the six broken variants must produce a nonzero wrong-row
    count somewhere in this corpus, or the gap is real and must be reported
    rather than hidden. This assertion is a regression guard -- if it ever
    goes red, some fixture's coverage of one of these six protections has
    been lost, which is exactly the D019 failure mode ("a redundant
    safeguard is an untested safeguard") applied to this corpus rather than
    to the tree walk itself.
    """
    totals, total_rows = _total_wrong_row_counts()
    print(f"\nbroken-variant table over {total_rows} rows across {len(VALUE_PRODUCING_NAMES)} fixtures:")
    for key, value in totals.items():
        print(f"  {key}: {value}/{total_rows} rows wrong")

    uncovered = [key for key, value in totals.items() if value == 0]
    assert not uncovered, (
        f"the following broken variants produced 0 wrong rows across the entire adversarial "
        f"corpus, meaning the fixtures do not cover them: {uncovered}"
    )


@pytest.mark.parametrize(
    "variant_key",
    [
        "cast_only_sample",
        "cast_only_threshold",
        "leq_instead_of_lt",
        "leaf_unnarrowed",
        "intercept_added_last",
        "reverse_tree_order",
    ],
)
def test_broken_variant_disagrees_with_xgboost_somewhere(variant_key: str) -> None:
    """One test per protection, so a reader sees which specific protection
    regressed rather than a single combined failure."""
    totals, total_rows = _total_wrong_row_counts()
    assert totals[variant_key] > 0, (
        f"{variant_key}: 0/{total_rows} rows wrong across the adversarial corpus -- "
        "this protection is not covered by any fixture here"
    )


def test_the_spec_compliant_walk_disagrees_with_no_broken_variant_by_coincidence() -> None:
    """Sanity check on the harness itself, using the real, shipped
    ``walk_margin`` (not a local reimplementation) as the one non-broken
    baseline: it must reproduce ``expected_margin`` everywhere the broken
    variants are tested against, or the wrong-row counts above would not
    mean what they claim to mean."""
    for name in VALUE_PRODUCING_NAMES:
        fixture = CORPUS[name]
        trees = fixture["artifact"]["trees"]
        intercept = fixture["artifact"]["intercept"]
        for row, expected_bits in zip(fixture["rows"], fixture["expected_margin"]):
            computed = walk_margin(trees, intercept, _row_to_features(row))
            assert _bits_of(computed) == expected_bits, f"{name}: real walk_margin disagreed with XGBoost"
