"""Tests for tree extraction, dead-node neutralization, and the margin walk.

XGBoost is the oracle. Hand-built dicts appear here only for the shapes
XGBoost will not produce on demand -- a marker that disagrees with
reachability, a vector-leaf leaf, a non-finite threshold -- and for the worked
example in FORMAT.md section 16. Every numeric claim about the walk is settled
against ``predict(output_margin=True)`` on a real fitted model, compared on
float32 **bit patterns** rather than with ``==``, because ``-0.0 == 0.0`` is
true and they are different artifacts.

Several tests deliberately implement a *wrong* variant of the walk and require
it to disagree with XGBoost. Those are not redundant: a boundary fixture that
a buggy build still passes is decorative, so the rows are shown to be
sensitive to the specific rule they exist to pin.

**Weak scalars, and why some tests here insist on float64 inputs.** Under
NEP 50 a Python ``float`` is a *weak* scalar: in ``python_float < np.float32(t)``
the Python side is narrowed and the comparison happens in float32 whether or
not the walk cast it, and ``np.float32(acc) + python_float`` stays float32 the
same way. A suite that only ever hands the walk Python floats therefore passes
with **any one** of its float32 casts deleted -- correct for the wrong reason,
and blind to the one call a caller actually writes,
``walk_margin(trees, intercept, matrix[i])`` on a float64 feature matrix. An
``np.float64``, and an element of a ``dtype=np.float64`` array, are *strong*:
they pull the operation into float64 and leave the cast as the only thing
holding the invariant up. The tests below that pin an individual cast use those
forms deliberately.

That is not a hypothetical about this file. An adversarial review of its
previous revision measured that removing any one of the five float32 sites in
``walk_margin`` left all 73 tests passing, and that reducing the walk to a
plain float64 running sum -- raw intercept, both add-site casts dropped -- left
the four 2000-row bit-exactness tests it then had green, with a single type
assertion red. Against this revision that same reduction turns 6 tests red. All
eight 2000-row bit-exactness tests in section 1 still pass under it, and that is
expected rather than a gap: a real extracted tree holds float32-exact Python
floats, so weak promotion genuinely does absorb per-add narrowing there. The
casts are load-bearing for every *other* shape the walk can be handed, which is
what the tests below feed it.

Each protection is verified by reverting it in the source, one at a time, and
confirming which tests go red. Four of the five float32 sites in ``walk_margin``
turn tests red on their own, and the damage each one prevents is a measured row
count rather than an argument:

=================================  =========================  =========  =========
site reverted alone                role                       tests red  rows wrong
=================================  =========================  =========  =========
``np.float32(feature_value)``      sample side of ``<``               2    126/629
``np.float32(threshold)``          threshold side of ``<``            1     96/464
``np.float32(node_values[node])``  leaf, before the add               2   980/2000
``np.float32(intercept)``          accumulator's first value          2    35/2000
``np.float32(accumulator + ...)``  wrap around the add                0         --
=================================  =========================  =========  =========

The last one could **not** be made to fail on its own, and is recorded here
rather than covered by a test that proves nothing. It is a provable no-op, not
merely an unmeasured one: given the two casts above it, both operands of that
addition are ``np.float32``, ``np.float32 + np.float32`` is ``np.float32``
computed in float32, and wrapping an ``np.float32`` in ``np.float32(...)``
changes no bit. What is load-bearing, and what the ``float64 sum narrowed once
at the end`` variant below pins at 473/2000, is that the accumulator stays
float32 across every addition -- not the wrap that documents it. The wrap stays
because the moment either operand widens, it is the only thing standing between
a float64 running sum and a green suite.
"""

from __future__ import annotations

import functools
import json
import pathlib
import math
from collections.abc import Sequence
from typing import Any, Callable, NamedTuple

import numpy as np
import pytest

# Imported unconditionally rather than through importorskip: XGBoost is the
# oracle for every numeric claim in this file, and a skip would report a green
# suite that measured nothing.
import xgboost as xgb

from xgboost_bridge.errors import (
    CategoricalSplitError,
    MalformedTreeError,
    NonFiniteFeatureError,
    UnsupportedModelShapeError,
)
from xgboost_bridge.trees import (
    DELETED_NODE_MARKER,
    LEAF_CHILD,
    TREE_KEYS,
    _neutralize_node,
    extract_trees,
    neutralize_dead_nodes,
    reachable_nodes,
    walk_margin,
)

SEED = 20260804
COLUMN_COUNT = 8
COLUMN_NAMES = [f"f{index}" for index in range(COLUMN_COUNT)]
TRAIN_ROWS = 1500
PREDICT_ROWS = 2000

# base_score values chosen away from the two degenerate points where every wrong
# accumulation variant also scores full marks: binary:logistic at 0.5 gives an
# intercept of exactly -0.0, and survival:cox at 1.0 gives exactly 0.0
# (probes/accumulation.md section 8). 0.987654 is included because a large
# intercept is what makes intercept placement observable.
MARGIN_CASES = [
    ("reg:squarederror", 0.3),
    ("binary:logistic", 0.7),
    ("binary:logistic", 0.987654),
    ("survival:cox", 0.7),
]


# --------------------------------------------------------------------------
# fitted models
# --------------------------------------------------------------------------


def _matrix(rows: int, seed: int, missing_fraction: float = 0.0) -> np.ndarray:
    generator = np.random.default_rng(seed)
    values = generator.normal(size=(rows, COLUMN_COUNT))
    if missing_fraction:
        values[generator.random(values.shape) < missing_fraction] = np.nan
    return values


def _labels(objective: str, values: np.ndarray, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed + 1)
    dense = np.nan_to_num(values)
    signal = 1.5 * dense[:, 0] - dense[:, 1] ** 2 + 0.5 * dense[:, 2] * dense[:, 3]
    if objective == "reg:squarederror":
        return signal + generator.normal(scale=0.25, size=values.shape[0])
    if objective == "binary:logistic":
        return (signal + generator.normal(scale=0.5, size=values.shape[0]) > 0).astype(float)
    if objective == "survival:cox":
        # Sign convention: positive = event, negative = right-censored.
        duration = np.exp(0.4 * signal + generator.normal(scale=0.3, size=values.shape[0])) + 0.1
        censored = generator.random(values.shape[0]) < 0.3
        return np.where(censored, -duration, duration)
    raise AssertionError(f"no label recipe for {objective}")


def _intercept(objective: str, model: dict[str, Any]) -> np.float32:
    """The margin-space intercept, per FORMAT.md section 6.1.

    Derived here rather than imported: this module owns the walk, not the
    intercept, and a test that shared an implementation with the thing it
    checks would not be able to fail on a recipe error.
    """
    stored = model["learner"]["learner_model_param"]["base_score"]
    assert isinstance(stored, str) and stored.startswith("[") and stored.endswith("]")
    value = np.float32(float(stored[1:-1]))
    if objective == "reg:squarederror":
        return np.float32(value)
    if objective == "survival:cox":
        return np.float32(math.log(value))
    if objective == "binary:logistic":
        clamped = min(max(value, np.float32(1e-6)), np.float32(np.float32(1.0) - np.float32(1e-6)))
        odds = np.float32(np.float32(np.float32(1.0) / clamped) - np.float32(1.0))
        return np.float32(-math.log(float(odds)))
    raise AssertionError(f"no intercept recipe for {objective}")


class Case(NamedTuple):
    booster: Any
    model: dict[str, Any]
    source_trees: list[dict[str, Any]]
    trees: list[dict[str, Any]]
    intercept: np.float32
    reference: np.ndarray


@functools.lru_cache(maxsize=None)
def _predict_matrix() -> np.ndarray:
    return _matrix(PREDICT_ROWS, SEED + 99, missing_fraction=0.12)


@functools.lru_cache(maxsize=None)
def _predict_rows() -> list[list[float]]:
    return [[float(value) for value in row] for row in _predict_matrix()]


@functools.lru_cache(maxsize=None)
def _case(
    objective: str,
    base_score: float,
    rounds: int = 60,
    gamma: float = 0.0,
    max_depth: int = 4,
) -> Case:
    train = _matrix(TRAIN_ROWS, SEED, missing_fraction=0.1)
    label = _labels(objective, train, SEED)
    dtrain = xgb.DMatrix(train, label=label, feature_names=COLUMN_NAMES)
    booster = xgb.train(
        {
            "objective": objective,
            "booster": "gbtree",
            "tree_method": "exact",
            "max_depth": max_depth,
            "eta": 0.3,
            "gamma": gamma,
            # base_score is passed explicitly: on a model left at the default,
            # boost_from_average is "1" and a zero-tree margin is the raw
            # base_score rather than the link transform (FORMAT.md 6.1).
            "base_score": base_score,
            "seed": SEED,
            "nthread": 1,
        },
        dtrain,
        num_boost_round=rounds,
    )
    model = json.loads(booster.save_raw(raw_format="json").decode("utf-8"))
    reference = booster.predict(
        xgb.DMatrix(_predict_matrix(), feature_names=COLUMN_NAMES), output_margin=True
    )
    return Case(
        booster=booster,
        model=model,
        source_trees=model["learner"]["gradient_booster"]["model"]["trees"],
        trees=extract_trees(model),
        intercept=_intercept(objective, model),
        reference=reference,
    )


def _pruned_case() -> Case:
    """A gamma-pruned model whose dead nodes are interleaved with live ones."""
    return _case("reg:squarederror", 0.3, rounds=12, gamma=5.0, max_depth=6)


def _leaf_only_case() -> Case:
    """gamma large enough that every split is blocked: leaf-only roots."""
    return _case("reg:squarederror", 0.3, rounds=12, gamma=1e9, max_depth=4)


# --------------------------------------------------------------------------
# comparison helpers
# --------------------------------------------------------------------------


def _bits(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).view(np.int32)


def _walk_all(
    trees: list[dict[str, Any]],
    intercept: float | np.floating[Any],
    rows: Sequence[Sequence[float] | np.ndarray],
) -> np.ndarray:
    return np.array([walk_margin(trees, intercept, row) for row in rows], dtype=np.float32)


def _strong_float64_rows(rows: list[list[float]]) -> list[np.ndarray]:
    """The same values, as strong ``np.float64`` scalars rather than Python floats.

    This is the shape a caller has: a row of a float64 feature matrix. It is
    also the only shape in which the sample-side narrowing can be observed at
    all, for the weak-scalar reason in this module's docstring.
    """
    return [np.asarray(row, dtype=np.float64) for row in rows]


def _bit_exact_count(walked: np.ndarray, reference: np.ndarray) -> int:
    return int(np.sum(_bits(walked) == _bits(reference)))


def _max_abs_error(walked: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.max(np.abs(walked.astype(np.float64) - reference.astype(np.float64)))
    )


def _leaf_value(tree: dict[str, Any], values: list[float]) -> np.float32:
    """The correct leaf lookup, shared by the deliberately-wrong variants below."""
    node = 0
    while tree["left_children"][node] != -1:
        value = values[tree["split_indices"][node]]
        if value != value:
            node = (
                tree["left_children"][node]
                if tree["default_left"][node] == 1
                else tree["right_children"][node]
            )
        elif np.float32(value) < np.float32(tree["node_values"][node]):
            node = tree["left_children"][node]
        else:
            node = tree["right_children"][node]
    return np.float32(tree["node_values"][node])


def _variant_walk(
    trees: list[dict[str, Any]],
    intercept: np.float32,
    values: list[float],
    compare: Callable[[float, float], bool],
) -> np.float32:
    """The walk with one rule swapped out, for the deliberate red tests."""
    accumulator = np.float32(intercept)
    for tree in trees:
        node = 0
        while tree["left_children"][node] != -1:
            value = values[tree["split_indices"][node]]
            if value != value:
                node = (
                    tree["left_children"][node]
                    if tree["default_left"][node] == 1
                    else tree["right_children"][node]
                )
            else:
                node = (
                    tree["left_children"][node]
                    if compare(value, tree["node_values"][node])
                    else tree["right_children"][node]
                )
        accumulator = np.float32(accumulator + np.float32(tree["node_values"][node]))
    return accumulator


def _adversarial_rows(case: Case) -> tuple[list[list[float]], np.ndarray]:
    """Rows whose every column sits exactly on one of XGBoost's own thresholds.

    The values are the *float64 parses* of the serialized tokens, which is the
    hazardous reading: on 104/104 measured thresholds that float64 is a
    different number from the engine's float32
    (``probes/float32_thresholds.md`` section 8b).
    """
    thresholds = sorted(
        {
            float(tree["split_conditions"][index])
            for tree in case.source_trees
            for index, left in enumerate(tree["left_children"])
            if left != -1
        }
    )
    rows = [[value] * COLUMN_COUNT for value in thresholds]
    reference = case.booster.predict(
        xgb.DMatrix(np.array(rows, dtype=np.float64), feature_names=COLUMN_NAMES),
        output_margin=True,
    )
    return rows, reference


def _un_narrowed_trees(case: Case, which: str = "all") -> list[dict[str, Any]]:
    """The extracted trees with ``node_values`` left as the float64 parse.

    Only the live nodes take a source value; a neutralized slot keeps its
    canonical ``0.0``, so this variant differs from the real extraction in
    exactly one respect -- the missing narrowing.

    ``which`` selects which role loses its narrowing, because ``node_values``
    overloads two: ``"internal"`` leaves the thresholds un-narrowed and the leaf
    outputs float32-exact, ``"leaf"`` does the reverse, and ``"all"`` un-narrows
    both. Splitting the roles is what makes the two casts on those values
    separately attributable -- a fixture that un-narrows both goes red when
    either cast is reverted and cannot say which.

    ``node_values`` is a ``dtype=np.float64`` array rather than a list of Python
    floats, and that is the point of the helper rather than a detail of it. A
    Python float is a weak scalar: ``np.float32(v) < python_float`` narrows the
    Python side back down and hides a missing threshold-side cast, and
    ``np.float32(acc) + python_float`` hides a missing leaf-side cast the same
    way. An element of this array is a strong ``np.float64`` and hides neither.
    """
    assert which in ("all", "internal", "leaf")
    variants = []
    for source, tree in zip(case.source_trees, case.trees, strict=True):
        live = reachable_nodes(tree)
        values: list[float] = []
        for index in range(len(tree["left_children"])):
            if index not in live:
                values.append(0.0)
            elif which in ("all", "internal" if tree["left_children"][index] != -1 else "leaf"):
                values.append(float(source["split_conditions"][index]))
            else:
                values.append(tree["node_values"][index])
        variants.append(
            {
                "default_left": tree["default_left"],
                "left_children": tree["left_children"],
                "right_children": tree["right_children"],
                "split_indices": tree["split_indices"],
                "node_values": np.asarray(values, dtype=np.float64),
            }
        )
    return variants


# --------------------------------------------------------------------------
# hand-built source models, for shapes XGBoost will not produce
# --------------------------------------------------------------------------


def _source_tree(
    *,
    left_children: list[int],
    right_children: list[int],
    split_indices: list[int],
    split_conditions: list[float],
    default_left: list[int],
    column_count: int = 2,
    **overrides: Any,
) -> dict[str, Any]:
    node_count = len(left_children)
    tree: dict[str, Any] = {
        "base_weights": [0.0] * node_count,
        "categories": [],
        "categories_nodes": [],
        "categories_segments": [],
        "categories_sizes": [],
        "default_left": default_left,
        "id": 0,
        "left_children": left_children,
        "loss_changes": [0.0] * node_count,
        "parents": [DELETED_NODE_MARKER] + [0] * (node_count - 1),
        "right_children": right_children,
        "split_conditions": split_conditions,
        "split_indices": split_indices,
        "split_type": [0] * node_count,
        "sum_hessian": [1.0] * node_count,
        "tree_param": {
            "num_deleted": "0",
            "num_feature": str(column_count),
            "num_nodes": str(node_count),
            "size_leaf_vector": "1",
        },
    }
    tree.update(overrides)
    return tree


def _three_node_source() -> dict[str, Any]:
    return _source_tree(
        left_children=[1, -1, -1],
        right_children=[2, -1, -1],
        split_indices=[0, 0, 0],
        split_conditions=[0.5, -0.25, 0.75],
        default_left=[1, 0, 0],
    )


def _source_model(
    trees: list[dict[str, Any]],
    column_count: int = 2,
    feature_types: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "learner": {
            "attributes": {},
            "feature_names": [f"f{index}" for index in range(column_count)],
            "feature_types": [] if feature_types is None else feature_types,
            "gradient_booster": {
                "model": {
                    "cats": {"enc": [], "feature_segments": [], "sorted_idx": []},
                    "gbtree_model_param": {
                        "num_parallel_tree": "1",
                        "num_trees": str(len(trees)),
                    },
                    "iteration_indptr": list(range(len(trees) + 1)),
                    "tree_info": [0] * len(trees),
                    "trees": trees,
                },
                "name": "gbtree",
            },
            "learner_model_param": {
                "base_score": "[5E-1]",
                "boost_from_average": "1",
                "num_class": "0",
                "num_feature": str(column_count),
                "num_target": "1",
            },
            "objective": {
                "name": "reg:squarederror",
                "reg_loss_param": {"scale_pos_weight": "1"},
            },
        },
        "version": [3, 3, 0],
    }


# ==========================================================================
# 1. the walk against XGBoost, per objective
# ==========================================================================


@pytest.mark.parametrize(("objective", "base_score"), MARGIN_CASES)
def test_walk_reproduces_xgboost_margin_bit_for_bit(objective: str, base_score: float) -> None:
    case = _case(objective, base_score)
    rows = _predict_rows()
    walked = _walk_all(case.trees, case.intercept, rows)

    exact = _bit_exact_count(walked, case.reference)
    error = _max_abs_error(walked, case.reference)
    print(
        f"\n{objective} base_score={base_score} trees={len(case.trees)} "
        f"intercept={float(case.intercept)!r} "
        f"bit-exact {exact}/{len(rows)} max abs err {error!r}"
    )

    assert len(rows) >= 2000
    assert exact == len(rows), f"{objective}: {exact}/{len(rows)} bit-exact"
    assert error == 0.0


@pytest.mark.parametrize(("objective", "base_score"), MARGIN_CASES)
def test_walk_reproduces_xgboost_margin_on_float64_numpy_rows(
    objective: str, base_score: float
) -> None:
    """The same claim through the call a caller writes: ``matrix[i]``.

    The test above hands the walk lists of Python floats, which are weak
    scalars; this one hands it rows of the float64 matrix that was fed to
    ``predict``, whose elements are strong ``np.float64``. Same rows, same
    reference margins, and the only difference is that nothing here narrows on
    the walk's behalf.
    """
    case = _case(objective, base_score)
    rows = list(_predict_matrix())
    assert all(row.dtype == np.float64 for row in rows)
    walked = _walk_all(case.trees, case.intercept, rows)

    exact = _bit_exact_count(walked, case.reference)
    error = _max_abs_error(walked, case.reference)
    print(
        f"\n{objective} base_score={base_score} float64 rows: "
        f"bit-exact {exact}/{len(rows)} max abs err {error!r}"
    )

    assert len(rows) >= 2000
    assert exact == len(rows), f"{objective}: {exact}/{len(rows)} bit-exact"
    assert error == 0.0


def test_zero_round_model_has_no_trees_and_the_margin_is_the_intercept() -> None:
    case = _case("binary:logistic", 0.7, rounds=0)
    assert case.model["learner"]["gradient_booster"]["model"]["trees"] == []
    assert case.trees == []
    # A zero-tree model at boost_from_average "1" reports the raw base_score
    # rather than the link transform, so XGBoost is not the oracle here --
    # only that the walk returns the intercept it was handed, untouched.
    margin = walk_margin(case.trees, case.intercept, _predict_rows()[0])
    assert _bits(margin) == _bits(case.intercept)


def test_leaf_only_trees_are_extracted_and_walked() -> None:
    case = _leaf_only_case()
    assert all(tree["left_children"][0] == -1 for tree in case.trees)
    walked = _walk_all(case.trees, case.intercept, _predict_rows())
    assert _bit_exact_count(walked, case.reference) == PREDICT_ROWS
    assert _max_abs_error(walked, case.reference) == 0.0


# ==========================================================================
# 2. neutralization
# ==========================================================================


def test_pruned_model_really_has_dead_nodes() -> None:
    case = _pruned_case()
    deleted = [int(tree["tree_param"]["num_deleted"]) for tree in case.source_trees]
    print(f"\nnum_deleted per tree: {deleted}")
    assert sum(deleted) > 0, "the pruned model has no dead nodes to neutralize"


def test_dead_set_is_interleaved_with_live_nodes_not_a_trailing_suffix() -> None:
    case = _pruned_case()
    interleaved = []
    for index, tree in enumerate(case.trees):
        live = reachable_nodes(tree)
        dead = [n for n in range(len(tree["left_children"])) if n not in live]
        if dead and max(live) > min(dead):
            interleaved.append((index, dead, sorted(live)[-1]))
    if interleaved:
        index, dead, last_live = interleaved[0]
        print(
            f"\ntree {index}: {len(case.trees[index]['left_children'])} nodes, "
            f"dead={dead}, highest live index={last_live}"
        )
    assert interleaved, "no tree has a dead node followed by a live one"


def test_neutralization_reports_exactly_the_unreachable_nodes() -> None:
    case = _pruned_case()
    for index, (source, tree) in enumerate(zip(case.source_trees, case.trees, strict=True)):
        marked = {
            node
            for node, column in enumerate(source["split_indices"])
            if column == DELETED_NODE_MARKER
        }
        live = reachable_nodes(tree)
        dead = {n for n in range(len(tree["left_children"])) if n not in live}
        assert dead == marked, f"tree {index}: reachability and marker disagree"
        assert len(marked) == int(source["tree_param"]["num_deleted"])


def test_neutralization_leaves_no_out_of_range_split_index() -> None:
    case = _pruned_case()
    column_count = int(case.model["learner"]["learner_model_param"]["num_feature"])
    for tree in case.trees:
        assert DELETED_NODE_MARKER not in tree["split_indices"]
        for column in tree["split_indices"]:
            assert 0 <= column < column_count


def test_neutralization_preserves_array_lengths_and_indices() -> None:
    case = _pruned_case()
    for source, tree in zip(case.source_trees, case.trees, strict=True):
        node_count = int(source["tree_param"]["num_nodes"])
        assert len(source["left_children"]) == node_count
        for key in TREE_KEYS:
            assert len(tree[key]) == node_count
        # No renumbering: every live child reference is unchanged.
        live = reachable_nodes(tree)
        for node in sorted(live):
            assert tree["left_children"][node] == source["left_children"][node]
            assert tree["right_children"][node] == source["right_children"][node]


def test_neutralized_nodes_carry_the_canonical_safe_values() -> None:
    case = _pruned_case()
    positive_zero_bits = _bits(np.float32(0.0))
    seen = 0
    for tree in case.trees:
        live = reachable_nodes(tree)
        for node in range(len(tree["left_children"])):
            if node in live:
                continue
            seen += 1
            assert tree["split_indices"][node] == 0
            assert tree["left_children"][node] == -1
            assert tree["right_children"][node] == -1
            assert tree["default_left"][node] == 0
            assert _bits(np.float32(tree["node_values"][node])) == positive_zero_bits
    assert seen > 0


def test_walk_reproduces_xgboost_margin_after_neutralization() -> None:
    case = _pruned_case()
    walked = _walk_all(case.trees, case.intercept, _predict_rows())
    exact = _bit_exact_count(walked, case.reference)
    error = _max_abs_error(walked, case.reference)
    print(f"\npruned model: bit-exact {exact}/{PREDICT_ROWS} max abs err {error!r}")
    assert exact == PREDICT_ROWS
    assert error == 0.0


def test_neutralizing_a_live_node_disagrees_with_xgboost() -> None:
    """The red test for section 8.3: clearing a live node must be detectable.

    A neutralization that clears a live node produces a plausible wrong
    number, not an error, so the check that guards against it has to be shown
    to fire. This corrupts the reachability result by hand -- neutralizing the
    root of the first tree, which every row visits -- and requires the
    walk-versus-XGBoost comparison to go red.
    """
    case = _pruned_case()
    rows = _predict_rows()
    assert _bit_exact_count(_walk_all(case.trees, case.intercept, rows), case.reference) == len(rows)

    corrupted = [
        {key: list(tree[key]) for key in TREE_KEYS} for tree in case.trees
    ]
    assert corrupted[0]["left_children"][0] != -1, "tree 0's root must be a split"
    _neutralize_node(corrupted[0], 0)

    walked = _walk_all(corrupted, case.intercept, rows)
    disagreeing = int(np.sum(_bits(walked) != _bits(case.reference)))
    print(f"\nlive root neutralized: {disagreeing}/{len(rows)} rows now disagree")
    assert disagreeing > 0, (
        "neutralizing a live node changed nothing, so this fixture cannot "
        "detect the bug it exists to detect"
    )


def test_neutralize_raises_when_a_marked_node_is_reachable() -> None:
    tree = {
        "default_left": [1, 0, 0],
        "left_children": [1, -1, -1],
        "node_values": [0.5, -0.25, 0.75],
        "right_children": [2, -1, -1],
        "split_indices": [0, DELETED_NODE_MARKER, 0],
    }
    with pytest.raises(MalformedTreeError) as caught:
        neutralize_dead_nodes(tree, "trees[0]")
    assert caught.value.field == "split_indices"
    assert caught.value.location == "trees[0]"


def test_neutralize_raises_when_an_unreachable_node_is_unmarked() -> None:
    tree = {
        "default_left": [0, 0, 0],
        "left_children": [-1, -1, -1],
        "node_values": [0.5, -0.25, 0.75],
        "right_children": [-1, -1, -1],
        "split_indices": [0, 0, 0],
    }
    with pytest.raises(MalformedTreeError):
        neutralize_dead_nodes(tree, "trees[0]")


def test_neutralize_returns_the_dead_indices_and_mutates_in_place() -> None:
    tree = {
        "default_left": [1, 0, 0, 1, 1],
        "left_children": [1, -1, -1, -1, -1],
        "node_values": [0.5, -0.25, 0.75, 1.5, 2.5],
        "right_children": [2, -1, -1, -1, -1],
        "split_indices": [0, 0, 0, DELETED_NODE_MARKER, DELETED_NODE_MARKER],
    }
    assert neutralize_dead_nodes(tree) == (3, 4)
    assert tree["split_indices"] == [0, 0, 0, 0, 0]
    assert tree["node_values"] == [0.5, -0.25, 0.75, 0.0, 0.0]
    assert tree["default_left"] == [1, 0, 0, 0, 0]
    assert len(tree["left_children"]) == 5


def test_reachable_nodes_finds_only_what_the_child_arrays_reach() -> None:
    tree = {
        "left_children": [1, 3, -1, -1, -1],
        "right_children": [2, 4, -1, -1, -1],
    }
    assert reachable_nodes(tree) == frozenset({0, 1, 2, 3, 4})
    assert reachable_nodes({"left_children": [-1, -1], "right_children": [-1, -1]}) == frozenset({0})


# ==========================================================================
# 3. float32 discipline, with the wrong variants required to go red
# ==========================================================================


def test_walk_is_bit_exact_on_rows_sitting_exactly_on_thresholds() -> None:
    case = _case("reg:squarederror", 0.3)
    rows, reference = _adversarial_rows(case)
    walked = _walk_all(case.trees, case.intercept, rows)
    exact = _bit_exact_count(walked, reference)
    print(f"\nadversarial rows: bit-exact {exact}/{len(rows)}")
    assert len(rows) > 100
    assert exact == len(rows)
    assert _max_abs_error(walked, reference) == 0.0


def test_leaving_the_sample_side_uncast_disagrees_with_xgboost() -> None:
    case = _case("reg:squarederror", 0.3)
    rows, reference = _adversarial_rows(case)
    walked = np.array(
        [
            _variant_walk(case.trees, case.intercept, row, lambda v, t: float(v) < float(t))
            for row in rows
        ],
        dtype=np.float32,
    )
    disagreeing = int(np.sum(_bits(walked) != _bits(reference)))
    print(f"\nsample side uncast: {disagreeing}/{len(rows)} rows disagree")
    assert disagreeing > 0


def test_leaving_the_threshold_side_uncast_disagrees_with_xgboost() -> None:
    """The canonical one-sided cast: sample narrowed, threshold not.

    The threshold has to come from the float64 parse for this to be
    reachable -- in an extracted tree it is already float32-exact, so a
    one-sided cast on *that* array is harmless. This is why the narrowing
    belongs at parse time and not at the comparison site.
    """
    case = _case("binary:logistic", 0.7)
    rows, reference = _adversarial_rows(case)
    un_narrowed = _un_narrowed_trees(case)
    walked = np.array(
        [
            _variant_walk(
                un_narrowed, case.intercept, row, lambda v, t: float(np.float32(v)) < float(t)
            )
            for row in rows
        ],
        dtype=np.float32,
    )
    disagreeing = int(np.sum(_bits(walked) != _bits(reference)))
    print(f"\nthreshold side uncast: {disagreeing}/{len(rows)} rows disagree")
    assert disagreeing > 0


def test_the_walk_casts_the_sample_side_of_the_comparison_itself() -> None:
    """The sample-side cast inside ``walk_margin``, pinned in isolation.

    The rows are the float64 parses of XGBoost's own threshold tokens, handed
    over as strong ``np.float64`` values -- the shape ``matrix[i]`` produces.
    On 104/104 measured thresholds that float64 is a different number from the
    engine's float32 (``probes/float32_thresholds.md`` section 8b), so an
    uncast sample side compares a genuinely-below value against the threshold
    and routes left where the engine, comparing two equal float32s, routes
    right.

    Measured by reverting ``np.float32(feature_value)`` in the source: 126 of
    these 629 rows come back with a different margin, bit-exact 503/629. The
    same rows as Python floats detect nothing, because the weak-scalar rule
    narrows them anyway.
    """
    case = _case("reg:squarederror", 0.3)
    rows, reference = _adversarial_rows(case)
    strong = _strong_float64_rows(rows)
    assert any(
        float(np.float32(value)) != float(value) for row in strong for value in row
    ), "no un-narrowable row value in this model, so the test would prove nothing"
    walked = _walk_all(case.trees, case.intercept, strong)
    exact = _bit_exact_count(walked, reference)
    print(f"\nfloat64 rows on thresholds: bit-exact {exact}/{len(rows)}")
    assert len(rows) > 100
    assert exact == len(rows)
    assert _max_abs_error(walked, reference) == 0.0


def test_the_walk_casts_the_threshold_side_of_the_comparison_itself() -> None:
    """The threshold-side cast inside ``walk_margin``, pinned in isolation.

    An extracted tree already holds float32-exact thresholds, so removing the
    threshold-side cast changes nothing there -- verified by reverting it, and
    the reason this test hands the real walk an array that was *not* narrowed
    at parse time. The walk then has to recover the engine's routing on its
    own, which it can, because narrowing the float64 parse is exact: 0/341
    divergence between ``float32(float64(text))`` and ``float32(text)``
    (``probes/float32_thresholds.md`` section 8a). The hazard is *using* the
    un-narrowed value, not parsing it.

    The un-narrowed array is ``dtype=np.float64`` rather than a list of Python
    floats, and that is what makes the cast observable: a Python float would be
    narrowed back down by the weak-scalar rule and the reverted build would
    still pass. Measured by reverting ``np.float32(threshold)`` in the source:
    96 of these 464 rows come back with a different margin, bit-exact 368/464.
    """
    case = _case("binary:logistic", 0.7)
    rows, reference = _adversarial_rows(case)
    un_narrowed = _un_narrowed_trees(case, "internal")
    assert all(
        tree["node_values"].dtype == np.float64 for tree in un_narrowed
    ), "the thresholds must be strong float64 or this test proves nothing"
    assert any(
        float(np.float32(value)) != float(value)
        for tree, source in zip(un_narrowed, case.trees, strict=True)
        for index, value in enumerate(tree["node_values"])
        if source["left_children"][index] != -1
    ), "no un-narrowed threshold in this model, so the test would prove nothing"
    walked = _walk_all(un_narrowed, case.intercept, rows)
    exact = _bit_exact_count(walked, reference)
    print(f"\nun-narrowed thresholds: bit-exact {exact}/{len(rows)}")
    assert exact == len(rows)
    assert _max_abs_error(walked, reference) == 0.0


def test_a_non_strict_operator_disagrees_with_xgboost() -> None:
    case = _case("reg:squarederror", 0.3)
    rows, reference = _adversarial_rows(case)
    walked = np.array(
        [
            _variant_walk(
                case.trees,
                case.intercept,
                row,
                lambda v, t: np.float32(v) <= np.float32(t),
            )
            for row in rows
        ],
        dtype=np.float32,
    )
    disagreeing = int(np.sum(_bits(walked) != _bits(reference)))
    print(f"\n'<=' instead of '<': {disagreeing}/{len(rows)} rows disagree")
    assert disagreeing > 0


# ==========================================================================
# 4. accumulation rules, with the wrong variants required to go red
# ==========================================================================


def _accumulation_variants(case: Case) -> dict[str, Callable[[list[float]], np.float32]]:
    def intercept_last(values: list[float]) -> np.float32:
        accumulator = np.float32(0.0)
        for tree in case.trees:
            accumulator = np.float32(accumulator + _leaf_value(tree, values))
        return np.float32(accumulator + np.float32(case.intercept))

    def reversed_order(values: list[float]) -> np.float32:
        accumulator = np.float32(case.intercept)
        for tree in reversed(case.trees):
            accumulator = np.float32(accumulator + _leaf_value(tree, values))
        return accumulator

    def widened_running_sum(values: list[float]) -> np.float32:
        accumulator = float(case.intercept)
        for tree in case.trees:
            accumulator = accumulator + float(_leaf_value(tree, values))
        return np.float32(accumulator)

    return {
        "intercept last": intercept_last,
        "reversed tree order": reversed_order,
        "float64 sum narrowed once at the end": widened_running_sum,
    }


@pytest.mark.parametrize(
    "variant",
    ["intercept last", "reversed tree order", "float64 sum narrowed once at the end"],
)
def test_deviating_from_the_accumulation_recipe_disagrees_with_xgboost(variant: str) -> None:
    case = _case("binary:logistic", 0.7)
    rows = _predict_rows()
    walk = _accumulation_variants(case)[variant]
    walked = np.array([walk(row) for row in rows], dtype=np.float32)
    exact = _bit_exact_count(walked, case.reference)
    print(f"\n{variant}: bit-exact {exact}/{len(rows)}")
    assert exact < len(rows)


def test_the_walk_narrows_a_leaf_value_before_adding_it() -> None:
    """The leaf-side cast on the add line, pinned as one rounding step.

    A float64 leaf must reach the accumulator through a single rounding -- narrow
    the leaf, then add in float32 -- not through a float64 add that is narrowed
    afterwards. The two disagree wherever the float32 add would land on a tie
    that the extra float64 precision breaks, which is what this leaf is built to
    hit: ``float32`` of it is exactly ``2**-24``, ``1 + 2**-24`` is halfway
    between two float32s and rounds to even, and the float64 add clears the
    halfway point and rounds up instead.
    """
    leaf = 2.0**-24 + 2.0**-50
    tree = {
        "default_left": [0],
        "left_children": [-1],
        "node_values": np.asarray([leaf], dtype=np.float64),
        "right_children": [-1],
        "split_indices": [0],
    }
    stored = tree["node_values"][0]
    assert isinstance(stored, np.float64)
    assert np.float32(stored) == np.float32(2.0**-24)
    assert np.float32(np.float32(1.0) + np.float32(stored)) == np.float32(1.0)
    assert np.float32(np.float32(1.0) + stored) != np.float32(1.0)

    margin = walk_margin([tree], np.float32(1.0), [0.0])
    assert _bits(margin) == _bits(np.float32(1.0))


def test_the_walk_narrows_leaf_values_read_from_a_float64_array() -> None:
    """The same cast at model scale, against XGBoost.

    The thresholds here are float32-exact and only the leaf outputs are left as
    the float64 parse, so this fixture is sensitive to the leaf-side cast alone.
    Measured by reverting ``np.float32(node_values[node])`` in the source: 980
    of these 2000 rows come back with a different margin, bit-exact 1020/2000.
    """
    case = _case("binary:logistic", 0.7)
    un_narrowed = _un_narrowed_trees(case, "leaf")
    assert any(
        float(np.float32(value)) != float(value)
        for tree, source in zip(un_narrowed, case.trees, strict=True)
        for index, value in enumerate(tree["node_values"])
        if source["left_children"][index] == -1 and index in reachable_nodes(source)
    ), "no un-narrowed leaf in this model, so the test would prove nothing"
    walked = _walk_all(un_narrowed, case.intercept, _predict_rows())
    exact = _bit_exact_count(walked, case.reference)
    print(f"\nun-narrowed leaf values: bit-exact {exact}/{PREDICT_ROWS}")
    assert exact == PREDICT_ROWS
    assert _max_abs_error(walked, case.reference) == 0.0


def test_the_walk_narrows_the_intercept_before_any_tree() -> None:
    """The intercept-side cast, pinned on a model with no trees at all.

    With no trees the margin *is* the intercept, so nothing downstream can
    launder a float64 one: the returned value is either the float32 the format
    specifies or the float64 it was handed. ``np.float64`` is used rather than a
    Python float because every other intercept in this file is already an
    ``np.float32`` from the derivation helper, which is exactly what left this
    cast unpinned.
    """
    widened = np.float64(0.1)
    margin = walk_margin([], widened, [])
    assert isinstance(margin, np.float32)
    assert float(margin) == float(np.float32(0.1))
    assert float(margin) != float(widened)


def test_the_walk_narrows_a_float64_intercept_before_the_first_add() -> None:
    """The same cast at model scale, against XGBoost.

    The intercept handed over is one float64 ULP above the engine's float32
    intercept, so it narrows back to exactly that float32 and the correct walk
    stays bit-exact -- while an accumulator that starts at the un-narrowed value
    carries the extra precision into the first add. Measured by reverting
    ``np.float32(intercept)`` in the source: 35 of these 2000 rows come back
    with a different margin, bit-exact 1965/2000.
    """
    case = _case("binary:logistic", 0.7)
    widened = np.nextafter(np.float64(case.intercept), np.float64(np.inf))
    assert isinstance(widened, np.float64)
    assert _bits(np.float32(widened)) == _bits(case.intercept)
    assert float(widened) != float(case.intercept)

    walked = _walk_all(case.trees, widened, _predict_rows())
    exact = _bit_exact_count(walked, case.reference)
    print(f"\nfloat64 intercept: bit-exact {exact}/{PREDICT_ROWS}")
    assert exact == PREDICT_ROWS
    assert _max_abs_error(walked, case.reference) == 0.0


# ==========================================================================
# 5. emission: values are float32-exact and never tidied
# ==========================================================================


def test_extracted_values_are_float32_fixed_points() -> None:
    case = _case("reg:squarederror", 0.3)
    total = 0
    for tree in case.trees:
        for value in tree["node_values"]:
            total += 1
            assert isinstance(value, float)
            assert float(np.float32(value)) == value
    assert total > 0


def test_extracted_values_match_the_source_thresholds_at_float32() -> None:
    case = _case("reg:squarederror", 0.3)
    for source, tree in zip(case.source_trees, case.trees, strict=True):
        live = reachable_nodes(tree)
        for node in sorted(live):
            assert _bits(np.float32(tree["node_values"][node])) == _bits(
                np.float32(source["split_conditions"][node])
            )


def test_extracted_values_survive_a_json_round_trip_bit_for_bit() -> None:
    case = _case("reg:squarederror", 0.3)
    for tree in case.trees:
        restored = json.loads(json.dumps(tree["node_values"]))
        assert np.array_equal(
            _bits(np.asarray(restored, dtype=np.float32)),
            _bits(np.asarray(tree["node_values"], dtype=np.float32)),
        )


def test_rounding_the_values_would_land_on_a_different_float32() -> None:
    """Why no rounding, formatting, or tidying step is permitted anywhere.

    8 significant digits already corrupts 2/341 measured thresholds
    (``probes/float32_thresholds.md`` section 8c). If this test ever finds
    nothing to corrupt, the emission tests above are proving less than they
    look like they prove.
    """
    case = _case("reg:squarederror", 0.3)
    values = [value for tree in case.trees for value in tree["node_values"]]
    corrupted = sum(
        1
        for value in values
        if _bits(np.float32(float(f"{value:.8g}"))) != _bits(np.float32(value))
    )
    print(f"\n8-significant-digit rounding corrupts {corrupted}/{len(values)} values")
    assert corrupted > 0


def test_signed_zero_is_not_normalized() -> None:
    model = _source_model(
        [
            _source_tree(
                left_children=[-1],
                right_children=[-1],
                split_indices=[0],
                split_conditions=[-0.0],
                default_left=[0],
            )
        ]
    )
    tree = extract_trees(model)[0]
    assert _bits(np.float32(tree["node_values"][0])) == _bits(np.float32(-0.0))
    assert json.dumps(tree["node_values"]) == "[-0.0]"
    assert _bits(walk_margin([tree], np.float32(-0.0), [0.0])) == _bits(np.float32(-0.0))


# ==========================================================================
# 6. the walk's node semantics, on hand-built trees
# ==========================================================================


WORKED_EXAMPLE = [
    {
        "default_left": [1, 0, 0],
        "left_children": [1, -1, -1],
        "node_values": [0.5, -0.25, 0.75],
        "right_children": [2, -1, -1],
        "split_indices": [0, 0, 0],
    },
    {
        "default_left": [0],
        "left_children": [-1],
        "node_values": [0.125],
        "right_children": [-1],
        "split_indices": [0],
    },
]


def test_worked_example_from_the_format_specification() -> None:
    margin = walk_margin(WORKED_EXAMPLE, 0.40546515583992004, [0.25, 9.0])
    assert _bits(margin) == _bits(np.float32(0.28046516))


def _single_split(threshold: float, default_left: int = 0) -> list[dict[str, Any]]:
    return [
        {
            "default_left": [default_left, 0, 0],
            "left_children": [1, -1, -1],
            "node_values": [threshold, -1.0, 1.0],
            "right_children": [2, -1, -1],
            "split_indices": [0, 0, 0],
        }
    ]


def test_a_value_equal_to_the_threshold_routes_right() -> None:
    threshold = float(np.float32(0.1))
    assert walk_margin(_single_split(threshold), 0.0, [threshold]) == np.float32(1.0)


def test_a_value_one_float32_ulp_below_the_threshold_routes_left() -> None:
    threshold = np.float32(0.1)
    below = float(np.nextafter(threshold, np.float32(-np.inf)))
    assert walk_margin(_single_split(float(threshold)), 0.0, [below]) == np.float32(-1.0)


def test_a_value_below_in_float64_but_equal_in_float32_routes_right() -> None:
    """The parse hazard, as a single node.

    ``0.1`` as a float64 is strictly below the float32 nearest ``0.1``. A walk
    comparing in float64 sends this row left; the engine sends it right,
    because both sides are float32 there and the values are then equal.

    The value is handed over twice, and only the second form reaches the hazard.
    The two comparisons asserted first say why: a Python float is a weak scalar
    and is narrowed to float32 before the comparison happens, so it routes right
    with or without the walk's sample-side cast, while an ``np.float64`` -- what
    indexing a float64 feature matrix yields -- keeps the comparison in float64
    and routes left unless the walk narrows it.
    """
    threshold = float(np.float32(0.1))
    assert 0.1 < threshold
    assert not (0.1 < np.float32(threshold))
    assert np.float64(0.1) < np.float32(threshold)

    assert walk_margin(_single_split(threshold), 0.0, [0.1]) == np.float32(1.0)
    assert walk_margin(_single_split(threshold), 0.0, [np.float64(0.1)]) == np.float32(1.0)
    assert walk_margin(
        _single_split(threshold), 0.0, np.asarray([0.1], dtype=np.float64)
    ) == np.float32(1.0)


def test_missing_value_routes_left_when_default_left_is_one() -> None:
    assert walk_margin(_single_split(0.5, default_left=1), 0.0, [float("nan")]) == np.float32(-1.0)


def test_missing_value_routes_right_when_default_left_is_zero() -> None:
    assert walk_margin(_single_split(0.5, default_left=0), 0.0, [float("nan")]) == np.float32(1.0)


def test_the_walk_raises_on_a_default_left_outside_zero_and_one() -> None:
    """An out-of-range ``default_left`` is refused, not read as "not left".

    ``extract_trees`` already rejects anything but 0 or 1, so the export path
    cannot produce this tree. The walk is reachable independently -- it is the
    normative algorithm the JavaScript predictor and the parity harness are
    checked against -- and a ``2`` here used to route right, which is a legal
    direction and therefore a plausible wrong number rather than an error.

    The second half records the scope of the check deliberately: the walk reads
    ``default_left`` only on the missing-value path, so a row that has a value
    at that column is routed by the comparison alone and is unaffected. Widening
    the check to every node on every row would put it in the hot loop to catch a
    tree that ``extract_trees`` has already refused.
    """
    trees = _single_split(0.5, default_left=2)
    with pytest.raises(MalformedTreeError) as caught:
        walk_margin(trees, 0.0, [float("nan")])
    assert caught.value.field == "default_left"
    assert caught.value.value == 2

    assert walk_margin(trees, 0.0, [9.0]) == np.float32(1.0)
    assert walk_margin(trees, 0.0, [-9.0]) == np.float32(-1.0)


def test_no_trees_returns_the_intercept_untouched() -> None:
    for intercept in (np.float32(-0.0), np.float32(0.0), np.float32(0.40546516)):
        assert _bits(walk_margin([], intercept, [])) == _bits(intercept)


def test_a_leaf_only_tree_contributes_its_value() -> None:
    tree = {
        "default_left": [0],
        "left_children": [-1],
        "node_values": [0.125],
        "right_children": [-1],
        "split_indices": [0],
    }
    assert _bits(walk_margin([tree], np.float32(0.5), [0.0])) == _bits(np.float32(0.625))


def test_the_leaf_test_is_left_children_and_never_right_children() -> None:
    """Step 3 of the normative recipe: a node is a leaf iff ``left_children`` is ``-1``.

    ``right_children[i] == -1`` coincides at a scalar leaf, so on every tree
    this library accepts the two tests are interchangeable and no prediction
    can tell them apart. They part company on the vector-leaf shape, where a
    leaf's ``right_children`` slot carries a block index into ``leaf_weights``
    instead of ``-1`` (``probes/tree_structure.md`` section 7g). ``extract_trees``
    and the reader both refuse that shape, so the rule is observable only
    through ``walk_margin`` itself -- which is public, is normative, and is what
    the JavaScript port and the parity harness are measured against.

    Node 1 below is a leaf whose ``right_children`` entry is ``5``. Keyed on
    ``left_children`` the walk adds that leaf's value. Keyed on
    ``right_children`` it reads node 1 as internal and descends into slots that
    are not children, which yields an ordinary-looking number rather than an
    error -- this project's failure signature, and the reason the rule is
    stated as an iff rather than as a convenience.
    """
    tree = {
        "default_left": [0, 0, 0],
        "left_children": [1, LEAF_CHILD, LEAF_CHILD],
        "node_values": [0.5, 0.25, -8.0],
        "right_children": [2, 5, LEAF_CHILD],
        "split_indices": [0, 0, 0],
    }
    assert tree["left_children"][1] == LEAF_CHILD
    assert tree["right_children"][1] != LEAF_CHILD

    margin = walk_margin([tree], np.float32(0.0), [0.0])
    assert _bits(margin) == _bits(np.float32(0.25))

    # The same tree read the other way reaches a different leaf, so the two
    # tests are genuinely distinguishable here and the assertion above is not
    # passing for want of a difference to find.
    assert _bits(np.float32(tree["node_values"][2])) != _bits(np.float32(0.25))


def test_the_walk_returns_a_float32() -> None:
    assert isinstance(walk_margin(WORKED_EXAMPLE, 0.5, [0.25, 9.0]), np.float32)


# ==========================================================================
# 7. extraction shape and strictness
# ==========================================================================


def test_extract_trees_returns_exactly_the_five_specified_keys() -> None:
    case = _case("reg:squarederror", 0.3)
    assert len(case.trees) == len(case.source_trees)
    assert len(case.trees) > 0
    for tree in case.trees:
        assert tuple(sorted(tree)) == TREE_KEYS
        assert set(tree) == set(TREE_KEYS)


def test_extract_trees_keeps_the_serialized_tree_order() -> None:
    case = _case("reg:squarederror", 0.3)
    for source, tree in zip(case.source_trees, case.trees, strict=True):
        assert tree["left_children"] == source["left_children"]


def test_extract_trees_types_are_plain_ints_and_floats() -> None:
    case = _case("reg:squarederror", 0.3)
    tree = case.trees[0]
    for key in ("left_children", "right_children", "split_indices", "default_left"):
        assert all(type(value) is int for value in tree[key])
    assert all(type(value) is float for value in tree["node_values"])
    assert set(tree["default_left"]) <= {0, 1}


def test_extract_trees_accepts_a_hand_built_model() -> None:
    trees = extract_trees(_source_model([_three_node_source()]))
    assert trees == [
        {
            "default_left": [1, 0, 0],
            "left_children": [1, -1, -1],
            "node_values": [0.5, -0.25, 0.75],
            "right_children": [2, -1, -1],
            "split_indices": [0, 0, 0],
        }
    ]


def _mutated(mutate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    tree = _three_node_source()
    mutate(tree)
    return _source_model([tree])


def _drop_learner_key(key: str) -> dict[str, Any]:
    model = _source_model([_three_node_source()])
    del model["learner"][key]
    return model


def _with_trees_field(value: Any) -> dict[str, Any]:
    model = _source_model([_three_node_source()])
    model["learner"]["gradient_booster"]["model"]["trees"] = value
    return model


def _with_integer_column_count() -> dict[str, Any]:
    model = _source_model([_three_node_source()])
    model["learner"]["learner_model_param"]["num_feature"] = 2
    return model


SHAPE_REJECTIONS: list[tuple[str, dict[str, Any]]] = [
    ("no learner", {"version": [3, 3, 0]}),
    ("no gradient_booster", _drop_learner_key("gradient_booster")),
    ("no feature_types", _drop_learner_key("feature_types")),
    ("trees is not an array", _with_trees_field({})),
    ("num_feature is an int", _with_integer_column_count()),
    ("array lengths disagree", _mutated(lambda t: t.__setitem__("default_left", [1, 0]))),
    ("default_left is not 0 or 1", _mutated(lambda t: t.__setitem__("default_left", [1, 2, 0]))),
    ("a threshold is not finite", _mutated(lambda t: t.__setitem__("split_conditions", [float("inf"), -0.25, 0.75]))),
    ("a threshold is NaN", _mutated(lambda t: t.__setitem__("split_conditions", [float("nan"), -0.25, 0.75]))),
    ("a threshold is a string", _mutated(lambda t: t.__setitem__("split_conditions", ["5E-1", -0.25, 0.75]))),
    ("a child index is a float", _mutated(lambda t: t.__setitem__("left_children", [1.0, -1, -1]))),
    ("tree_param is absent", _mutated(lambda t: t.pop("tree_param"))),
    ("split_indices is absent", _mutated(lambda t: t.pop("split_indices"))),
    ("a child index is out of range", _mutated(lambda t: t.__setitem__("right_children", [9, -1, -1]))),
    ("a child index points backwards", _mutated(lambda t: t.__setitem__("left_children", [1, 0, -1]))),
    ("a child index points at the node itself", _mutated(lambda t: t.__setitem__("left_children", [1, 1, -1]))),
    ("both children are the same node", _mutated(lambda t: t.__setitem__("right_children", [1, -1, -1]))),
    ("a leaf's right child is a block index", _mutated(lambda t: t.__setitem__("right_children", [2, 0, -1]))),
    ("split_indices names a column that does not exist", _mutated(lambda t: t.__setitem__("split_indices", [7, 0, 0]))),
    ("split_type carries an unknown value", _mutated(lambda t: t.__setitem__("split_type", [2, 0, 0]))),
    ("split_type is shorter than the node count", _mutated(lambda t: t.__setitem__("split_type", [0]))),
    ("split_type is longer than the node count", _mutated(lambda t: t.__setitem__("split_type", [0, 0, 0, 0]))),
    ("the tree has no nodes", _mutated(lambda t: t.update({
        "left_children": [], "right_children": [], "split_indices": [],
        "split_conditions": [], "default_left": [], "split_type": [],
    }))),
    ("a marked node is reachable", _mutated(lambda t: t.__setitem__("split_indices", [0, DELETED_NODE_MARKER, 0]))),
]


@pytest.mark.parametrize(
    ("description", "model"), SHAPE_REJECTIONS, ids=[case[0] for case in SHAPE_REJECTIONS]
)
def test_extract_trees_raises_on_an_unrecognized_shape(
    description: str, model: dict[str, Any]
) -> None:
    with pytest.raises(MalformedTreeError):
        extract_trees(model)


ARITY_REJECTIONS: list[tuple[str, dict[str, Any]]] = [
    ("size_leaf_vector is 2", _mutated(lambda t: t["tree_param"].__setitem__("size_leaf_vector", "2"))),
    ("size_leaf_vector is the integer 1", _mutated(lambda t: t["tree_param"].__setitem__("size_leaf_vector", 1))),
]


@pytest.mark.parametrize(
    ("description", "model"), ARITY_REJECTIONS, ids=[case[0] for case in ARITY_REJECTIONS]
)
def test_extract_trees_raises_on_vector_leaves(
    description: str, model: dict[str, Any]
) -> None:
    """Vector leaves are an arity fact, so they raise the arity error.

    The integer-``1`` case is not redundant: the field is a JSON string, and an
    integer comparison would silently never fire.
    """
    with pytest.raises(UnsupportedModelShapeError) as caught:
        extract_trees(model)
    assert caught.value.field == "size_leaf_vector"


CATEGORICAL_REJECTIONS: list[tuple[str, dict[str, Any]]] = [
    ("split_type contains 1", _mutated(lambda t: t.__setitem__("split_type", [1, 0, 0]))),
    ("categories_nodes is non-empty", _mutated(lambda t: t.__setitem__("categories_nodes", [0]))),
    (
        "feature_types contains 'c'",
        _source_model([_three_node_source()], feature_types=["c", "float"]),
    ),
]


@pytest.mark.parametrize(
    ("description", "model"),
    CATEGORICAL_REJECTIONS,
    ids=[case[0] for case in CATEGORICAL_REJECTIONS],
)
def test_extract_trees_refuses_a_categorical_split_on_each_signal_alone(
    description: str, model: dict[str, Any]
) -> None:
    """Each of the three signals is checked in isolation.

    All three fired together on every categorical model measured, so a suite
    that only used a real categorical model would pin none of them
    individually.
    """
    with pytest.raises(CategoricalSplitError) as caught:
        extract_trees(model)
    assert len(caught.value.signals) == 1


@functools.lru_cache(maxsize=None)
def _categorical_model_json() -> str:
    """A really-fitted categorical model, serialized so callers can mutate a copy."""
    generator = np.random.default_rng(SEED)
    codes = generator.integers(0, 6, size=800).astype(float)
    other = generator.normal(size=800)
    values = np.column_stack([codes, other])
    label = np.where(np.isin(codes, [0, 2, 5]), -5.0, 5.0) + 0.01 * other
    dtrain = xgb.DMatrix(
        values,
        label=label,
        feature_names=["a", "b"],
        feature_types=["c", "float"],
        enable_categorical=True,
    )
    booster = xgb.train(
        {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "max_depth": 2,
            "base_score": 0.0,
            "seed": SEED,
            "nthread": 1,
        },
        dtrain,
        num_boost_round=1,
    )
    return booster.save_raw(raw_format="json").decode("utf-8")


def test_a_real_categorical_model_is_refused() -> None:
    model = json.loads(_categorical_model_json())
    with pytest.raises(CategoricalSplitError) as caught:
        extract_trees(model)
    print(f"\ncategorical signals: {caught.value.signals}")
    assert len(caught.value.signals) >= 1


def test_a_truncated_split_type_cannot_hide_a_categorical_node() -> None:
    """``split_type`` is per-node, so its length is checked like the other five.

    Measured on the three-node source tree before this check existed:
    ``split_type == [0, 1, 0]`` was refused as categorical, while a truncated
    ``split_type == [0]`` was **accepted** and exported as a numeric tree. The
    truncation dropped the only entry the signal exists to examine, and
    ``_extract_one``'s length checks did not cover it because it reads five
    arrays and this is a sixth.
    """
    categorical = _mutated(lambda t: t.__setitem__("split_type", [0, 1, 0]))
    with pytest.raises(CategoricalSplitError):
        extract_trees(categorical)

    truncated = _mutated(lambda t: t.__setitem__("split_type", [0]))
    with pytest.raises(MalformedTreeError) as caught:
        extract_trees(truncated)
    assert caught.value.field == "split_type"
    assert caught.value.value == 1


def test_an_emptied_categories_nodes_is_still_refused_on_a_real_model() -> None:
    """``categories_nodes`` gets no length check, and why that is not the same hole.

    Its length is the number of categorical nodes, not the node count, and its
    entries are node indices (``probes/tree_structure.md`` section 3), so there
    is nothing to length-check it against; any non-empty value refuses whatever
    its length. The only way to lose the signal is to empty the array, which no
    length check can detect. What closes that hole is the redundancy of the other
    two signals, measured here on a really-fitted categorical model with this
    signal and the ``feature_types`` signal both removed by hand.
    """
    model = json.loads(_categorical_model_json())
    for tree in model["learner"]["gradient_booster"]["model"]["trees"]:
        assert tree["categories_nodes"], "this model has no categorical node to hide"
        tree["categories_nodes"] = []
    model["learner"]["feature_types"] = ["float", "float"]

    with pytest.raises(CategoricalSplitError) as caught:
        extract_trees(model)
    print(f"\nsurviving signals: {caught.value.signals}")
    assert all("split_type contains 1" in signal for signal in caught.value.signals)


def test_a_vector_leaf_model_is_refused() -> None:
    generator = np.random.default_rng(SEED)
    values = generator.normal(size=(600, 3))
    label = np.column_stack([values[:, 0], values[:, 1]])
    dtrain = xgb.DMatrix(values, label=label, feature_names=["a", "b", "c"])
    booster = xgb.train(
        {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "multi_strategy": "multi_output_tree",
            "max_depth": 2,
            "seed": SEED,
            "nthread": 1,
        },
        dtrain,
        num_boost_round=2,
    )
    model = json.loads(booster.save_raw(raw_format="json").decode("utf-8"))
    with pytest.raises(UnsupportedModelShapeError) as caught:
        extract_trees(model)
    assert caught.value.field == "size_leaf_vector"


# --------------------------------------------------------------------------
# D022: infinite feature values are refused; NaN is the missing value.
#
# These pin a guard that was specified in DECISIONS.md and FORMAT.md but was
# not implemented until an adversarial fixture pass noticed the gap. Before
# the guard, walk_margin returned an ordinary float for an infinite input --
# a plausible wrong number, which is the failure this project exists to
# prevent, reachable in shipped code.
# --------------------------------------------------------------------------


def _two_feature_split() -> list[dict[str, Any]]:
    """One split on column 0; column 1 is never read by any node."""
    return [
        {
            "default_left": [1, 0, 0],
            "left_children": [1, -1, -1],
            "node_values": [0.5, -1.0, 1.0],
            "right_children": [2, -1, -1],
            "split_indices": [0, 0, 0],
        }
    ]


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_an_infinite_feature_value_is_refused(value: float) -> None:
    with pytest.raises(NonFiniteFeatureError) as caught:
        walk_margin(_two_feature_split(), 0.0, [value, 0.0])
    assert caught.value.index == 0
    assert caught.value.value == value


def test_nan_is_the_missing_value_and_is_not_refused() -> None:
    """NaN must NOT raise: it routes by default_left, which is model structure.

    Refusing it would reject every model fitted on data with gaps.
    """
    margin = walk_margin(_two_feature_split(), 0.0, [float("nan"), 0.0])
    assert margin == np.float32(-1.0)


def test_an_infinity_in_a_column_no_node_reads_is_still_refused() -> None:
    """The row is checked up front, not lazily at visited nodes.

    Column 1 is never read by this tree. A lazy check would accept this input,
    which would make the same invalid row raise or not depending on which
    branches the tree happens to take -- an outcome that is a property of the
    model rather than of the input. Reverting the guard to a per-node check
    turns this test red and leaves the two above green.
    """
    with pytest.raises(NonFiniteFeatureError) as caught:
        walk_margin(_two_feature_split(), 0.0, [0.25, float("inf")])
    assert caught.value.index == 1


def test_every_row_of_the_refusal_fixture_is_refused() -> None:
    """The adversarial corpus records these rows as expected-to-raise.

    The fixture deliberately carries no numeric ground truth for them, so this
    is the only check that can confirm the recorded expectation is met.
    """
    path = (
        pathlib.Path(__file__).resolve().parents[3]
        / "fixtures"
        / "corpus"
        / "adversarial"
        / "non_finite_input_refusal.json"
    )
    fixture = json.loads(path.read_text(encoding="utf-8"))
    assert fixture["meta"]["expected_behavior"] == "raise"
    artifact = fixture["artifact"]
    assert fixture["rows"], "refusal fixture must carry rows"
    for row in fixture["rows"]:
        values = [
            float("nan") if v is None else float(v) if not isinstance(v, str) else float(v)
            for v in row
        ]
        with pytest.raises(NonFiniteFeatureError):
            walk_margin(artifact["trees"], artifact["intercept"], values)
