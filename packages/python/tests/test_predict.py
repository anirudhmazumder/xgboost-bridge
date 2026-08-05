"""Tests for the reference predictor (predict.py).

**What the oracles are, and why they cannot share the defect being looked
for.** Three independent ones are used here, and no test in this file compares
one of this library's values against another of its own:

1. **The fixture corpus.** Every ``expected_margin`` and ``expected_output``
   in ``fixtures/corpus/`` was recorded from XGBoost's own ``predict()`` at
   generation time (D044) and is stored as a uint32 bit pattern so the
   comparison cannot degrade into ``==``, under which ``-0.0 == 0.0``. This
   file re-reads those recorded patterns; it never recomputes them, and it
   never consults the generator's own self-check.
2. **A live XGBoost model.** One test fits three real models, exports them,
   loads the artifacts through :class:`Predictor` and compares against
   ``booster.predict()`` in the same process. That catches a defect a frozen
   corpus cannot: one introduced by a *change in the export path* that the
   corpus files predate.
3. **FORMAT.md's own worked example** (section 16), whose margin is stated in
   the specification independently of any code here.

The margin gate is exact: bit-for-bit, 289/289. The output gate against
XGBoost is **relative** and is a different gate (FORMAT.md section 5.7) --
XGBoost's own ``expf`` is not correctly rounded, so bit-exactness at the
output is unreachable and is explicitly not a goal (section 5.2). The rows
where the bundled transform and XGBoost's ``libm`` differ are enumerated
below and pinned exactly, so the figure is a tripwire rather than a tolerance:
any change in that set fails, in either direction.

Cross-language parity is a separate gate and is not measured here.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xgboost as xgb

from xgboost_bridge import errors, export
from xgboost_bridge.objectives import OUTPUT_TRANSFORMS
from xgboost_bridge.predict import (
    ENVELOPE_KEYS,
    PROVENANCE_KEYS,
    READABLE_FORMAT_VERSION,
    TREE_KEYS,
    Predictor,
)
from xgboost_bridge.transform import OUTPUT_FUNCTIONS, SIGMOID_FLOOR_OUTPUT
from xgboost_bridge.trees import TREE_KEYS as SOURCE_TREE_KEYS

REPO_ROOT = Path(__file__).resolve().parents[3]
CORPUS_DIR = REPO_ROOT / "fixtures" / "corpus"
ADVERSARIAL_DIR = CORPUS_DIR / "adversarial"

PREDICT_SOURCE_PATH = (
    REPO_ROOT / "packages" / "python" / "src" / "xgboost_bridge" / "predict.py"
)

POSITIVE_INFINITY_BITS = "0x7f800000"


# ---------------------------------------------------------------------------
# Corpus loading. Nothing here regenerates a fixture; a missing corpus fails
# loudly rather than making every parametrized test collect zero cases.
# ---------------------------------------------------------------------------


def _load_directory(directory: Path) -> dict[str, dict[str, Any]]:
    if not directory.is_dir():
        return {}
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.json"))
    }


CORPUS = _load_directory(CORPUS_DIR)
ADVERSARIAL = _load_directory(ADVERSARIAL_DIR)
FIXTURES: dict[str, dict[str, Any]] = {**CORPUS, **ADVERSARIAL}

#: Fixtures that record no numeric ground truth because refusal *is* the
#: specified behaviour (D022, D045).
REFUSAL_FIXTURES = tuple(
    sorted(
        name
        for name, fixture in FIXTURES.items()
        if fixture["meta"].get("ground_truth") == "none"
    )
)

#: Fixtures carrying XGBoost's recorded margin and output for every row.
GROUND_TRUTH_FIXTURES = tuple(
    sorted(name for name in FIXTURES if name not in REFUSAL_FIXTURES)
)

#: The rows where the bundled float32 transform and XGBoost's ``libm``
#: disagree at the last bit. Recorded exactly rather than bounded by a count,
#: so a change in either direction fails: growing means the transform drifted,
#: shrinking means it changed to chase a value FORMAT.md section 5.2 says is
#: unreachable. All six are ordinary 1-ULP differences; the rows that matter
#: most -- Cox's ``+inf`` and the logistic clamp floor -- are bit-exact and are
#: checked separately below.
KNOWN_LIBM_DIVERGENCES: frozenset[tuple[str, int]] = frozenset(
    {
        ("binary_logistic_base_score_low_inside_clamp", 2),
        ("single_feature_model", 0),
        ("survival_cox_base_score_low", 1),
        ("survival_cox_base_score_low", 2),
        ("survival_cox_overflow_to_infinity", 0),
        ("survival_cox_overflow_to_infinity", 8),
    }
)

#: The Python-vs-XGBoost output gate (FORMAT.md section 5.7). Not a parity
#: gate: cross-language parity is exact and no tolerance touches it.
OUTPUT_RELATIVE_GATE = 1e-6


def _bits(value: object) -> str:
    """The uint32 bit pattern of a value as float32, in the fixtures' spelling."""
    return f"0x{int(np.float32(value).view(np.uint32)):08x}"


def _from_bits(pattern: str) -> np.float32:
    assert pattern.startswith("0x") and len(pattern) == 10, f"not a bit pattern: {pattern!r}"
    return np.uint32(int(pattern, 16)).view(np.float32)[()]


def _row_mapping(fixture: dict[str, Any], row: list[Any]) -> dict[str, float]:
    """Turn a fixture row into the mapping a predictor takes.

    Two wire conventions, both recorded in each fixture's ``meta`` (D044):
    ``null`` is a missing value and becomes ``NaN``; the strings ``"inf"`` and
    ``"-inf"`` appear only in the refusal fixture, because JSON has no
    infinity literal.
    """
    names = fixture["artifact"]["feature_names"]
    assert len(names) == len(row), "fixture row width disagrees with feature_names"
    values: dict[str, float] = {}
    for name, value in zip(names, row):
        if value is None:
            values[name] = float("nan")
        else:
            values[name] = float(value)
    return values


def _predictor(name: str) -> Predictor:
    return Predictor.from_json(FIXTURES[name]["artifact"])


# ---------------------------------------------------------------------------
# The corpus is actually there.
# ---------------------------------------------------------------------------


def test_both_corpus_directories_are_present_and_populated() -> None:
    """A suite that silently loaded no fixture would pass every check below."""
    assert len(CORPUS) >= 15, f"only {len(CORPUS)} fixtures found at {CORPUS_DIR}"
    assert len(ADVERSARIAL) >= 8, (
        f"only {len(ADVERSARIAL)} fixtures found at {ADVERSARIAL_DIR}"
    )
    assert set(CORPUS).isdisjoint(ADVERSARIAL), (
        "a fixture name appears in both directories, so one shadows the other"
    )


def test_the_fixtures_that_matter_most_are_present() -> None:
    """Named individually: these are the cases a regression would hide in."""
    required = (
        "binary_logistic_signed_zero",
        "equality_boundary_routing",
        "float32_threshold_disagreement",
        "gamma_pruned_dead_nodes",
        "gamma_pruned_neutralization",
        "logistic_clamp_floor_output",
        "missing_value_both_directions",
        "non_finite_input_refusal",
        "reg_squarederror_zero_tree",
        "survival_cox_overflow_to_infinity",
        "zero_leaf_intercept_isolation",
    )
    missing = sorted(set(required) - set(FIXTURES))
    assert not missing, f"required fixtures absent from the corpus: {missing}"


def test_the_refusal_fixture_is_the_only_one_without_ground_truth() -> None:
    assert REFUSAL_FIXTURES == ("non_finite_input_refusal",)


# ---------------------------------------------------------------------------
# Margin: bit-for-bit against XGBoost's recorded output.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", GROUND_TRUTH_FIXTURES)
def test_margin_reproduces_the_recorded_xgboost_margin_bit_for_bit(name: str) -> None:
    fixture = FIXTURES[name]
    predictor = Predictor.from_json(fixture["artifact"])

    mismatches: list[tuple[int, str, str]] = []
    for index, (row, expected) in enumerate(
        zip(fixture["rows"], fixture["expected_margin"])
    ):
        computed = _bits(predictor.margin(_row_mapping(fixture, row)))
        if computed != expected:
            mismatches.append((index, expected, computed))

    assert not mismatches, (
        f"{name}: margin disagreed with XGBoost on {len(mismatches)}/"
        f"{len(fixture['rows'])} rows (row, expected, computed): {mismatches}"
    )


def test_margin_is_bit_exact_on_every_row_of_the_whole_corpus() -> None:
    """The aggregate figure, pinned: the gate is 289/289, not "mostly"."""
    total = 0
    exact = 0
    for name in GROUND_TRUTH_FIXTURES:
        fixture = FIXTURES[name]
        predictor = Predictor.from_json(fixture["artifact"])
        for row, expected in zip(fixture["rows"], fixture["expected_margin"]):
            total += 1
            exact += _bits(predictor.margin(_row_mapping(fixture, row))) == expected

    assert total == 289, (
        f"the corpus supplied {total} ground-truth rows, not the 289 this gate "
        "was measured on; a fixture was added or lost"
    )
    assert exact == total, f"margin bit-exact on only {exact}/{total} corpus rows"


def test_margin_is_float32_and_never_widened() -> None:
    for name in GROUND_TRUTH_FIXTURES:
        fixture = FIXTURES[name]
        predictor = Predictor.from_json(fixture["artifact"])
        value = predictor.margin(_row_mapping(fixture, fixture["rows"][0]))
        assert isinstance(value, np.float32), f"{name}: margin returned {type(value)}"


# ---------------------------------------------------------------------------
# Output: the relative gate against XGBoost, plus the exact-bit tripwire.
# ---------------------------------------------------------------------------


def _relative_error(ours: np.float32, theirs: np.float32) -> float:
    """FORMAT.md section 5.7's instrument, with its stated special cases.

    ``NaN`` is always a failure on either side, infinities are compared as bit
    patterns and never divided, and where XGBoost's value is ``0.0`` or
    ``-0.0`` exact bit equality is required instead of a ratio.
    """
    if math.isnan(float(ours)) or math.isnan(float(theirs)):
        return math.inf
    if math.isinf(float(theirs)) or math.isinf(float(ours)):
        return 0.0 if _bits(ours) == _bits(theirs) else math.inf
    if float(theirs) == 0.0:
        return 0.0 if _bits(ours) == _bits(theirs) else math.inf
    return abs(float(ours) - float(theirs)) / abs(float(theirs))


@pytest.mark.parametrize("name", GROUND_TRUTH_FIXTURES)
def test_output_meets_the_relative_gate_against_xgboost(name: str) -> None:
    fixture = FIXTURES[name]
    predictor = Predictor.from_json(fixture["artifact"])

    worst = 0.0
    worst_row: tuple[int, float, float] | None = None
    for index, (row, expected) in enumerate(
        zip(fixture["rows"], fixture["expected_output"])
    ):
        ours = predictor.output(_row_mapping(fixture, row))
        theirs = _from_bits(expected)
        error = _relative_error(ours, theirs)
        if error > worst:
            worst = error
            worst_row = (index, float(ours), float(theirs))

    # Max, never mean: a mean hides one catastrophic row in a large corpus.
    assert worst <= OUTPUT_RELATIVE_GATE, (
        f"{name}: max relative output error {worst:.3e} exceeds "
        f"{OUTPUT_RELATIVE_GATE:.0e} at row {worst_row}"
    )


def test_output_bit_divergences_from_xgboost_are_exactly_the_recorded_set() -> None:
    """Pin the ``libm`` divergence exactly, in both directions.

    Bit-exactness with XGBoost at the output is unreachable by construction --
    its own ``expf`` is not correctly rounded -- so this is not a gate on
    correctness. It is a tripwire on *change*: the bundled transform is built
    from ``+ - * /`` in float32 and the fixtures are frozen, so this set is
    deterministic, and any movement in it means the transform moved.
    """
    observed: set[tuple[str, int]] = set()
    total = 0
    for name in GROUND_TRUTH_FIXTURES:
        fixture = FIXTURES[name]
        predictor = Predictor.from_json(fixture["artifact"])
        for index, (row, expected) in enumerate(
            zip(fixture["rows"], fixture["expected_output"])
        ):
            total += 1
            if _bits(predictor.output(_row_mapping(fixture, row))) != expected:
                observed.add((name, index))

    assert observed == KNOWN_LIBM_DIVERGENCES, (
        "output bit-divergence set moved: newly divergent "
        f"{sorted(observed - KNOWN_LIBM_DIVERGENCES)}, no longer divergent "
        f"{sorted(KNOWN_LIBM_DIVERGENCES - observed)}"
    )
    assert total - len(observed) == 283, (
        f"output bit-exact on {total - len(observed)}/{total} rows, expected 283/289"
    )


def test_no_corpus_output_is_nan() -> None:
    """NaN is always a failure, on either side, with no exception.

    Stated separately because NaN compares unequal to everything including
    itself, so a harness that only checks a ratio silently *skips* exactly
    these rows.
    """
    for name in GROUND_TRUTH_FIXTURES:
        fixture = FIXTURES[name]
        predictor = Predictor.from_json(fixture["artifact"])
        for index, (row, expected) in enumerate(
            zip(fixture["rows"], fixture["expected_output"])
        ):
            assert not math.isnan(float(_from_bits(expected))), (
                f"{name} row {index}: recorded ground truth is NaN"
            )
            ours = predictor.output(_row_mapping(fixture, row))
            assert not math.isnan(float(ours)), f"{name} row {index}: output is NaN"


def test_cox_overflow_rows_match_xgboost_infinity_as_bit_patterns() -> None:
    """The ``+inf`` rows, called out on their own.

    ``survival:cox`` has no clamp and genuinely returns ``+inf`` above margin
    approximately 88.72. An infinity must never drop out of a comparison, and
    a finite-versus-``inf`` disagreement is the qualitative failure a float64
    transform produces here (D032).
    """
    name = "survival_cox_overflow_to_infinity"
    fixture = FIXTURES[name]
    predictor = Predictor.from_json(fixture["artifact"])

    infinite_rows = 0
    matched = 0
    for row, expected in zip(fixture["rows"], fixture["expected_output"]):
        if expected != POSITIVE_INFINITY_BITS:
            continue
        infinite_rows += 1
        matched += _bits(predictor.output(_row_mapping(fixture, row))) == expected

    assert infinite_rows == 2, (
        f"{name}: {infinite_rows} rows record +inf, expected 2 -- the fixture changed"
    )
    assert matched == infinite_rows, f"{name}: only {matched}/{infinite_rows} +inf rows match"


def test_logistic_clamp_floor_rows_match_xgboost_bit_for_bit() -> None:
    """The clamp-floor rows, called out on their own.

    XGBoost's ``binary:logistic`` floors at margin ``f32(-88.7)`` and returns
    exactly ``3.006635794144578e-39`` -- never ``0.0``. A float64 transform is
    not off by a ULP here, it is relatively 100% wrong, and an absolute gate
    cannot see it.
    """
    name = "logistic_clamp_floor_output"
    fixture = FIXTURES[name]
    predictor = Predictor.from_json(fixture["artifact"])
    floor_bits = _bits(SIGMOID_FLOOR_OUTPUT)

    floor_rows = 0
    matched = 0
    for row, expected in zip(fixture["rows"], fixture["expected_output"]):
        if expected != floor_bits:
            continue
        floor_rows += 1
        ours = predictor.output(_row_mapping(fixture, row))
        matched += _bits(ours) == expected
        assert float(ours) != 0.0, "the floor must not return 0.0"

    assert floor_rows == 21, (
        f"{name}: {floor_rows} rows sit at the clamp floor, expected 21 -- "
        "the fixture changed"
    )
    assert matched == floor_rows, f"{name}: only {matched}/{floor_rows} floor rows match"


def test_signed_zero_margin_survives_as_negative_zero() -> None:
    """``-0.0`` is reachable through an ordinary default and is not normalized."""
    fixture = FIXTURES["binary_logistic_signed_zero"]
    predictor = Predictor.from_json(fixture["artifact"])
    assert _bits(predictor.intercept) == "0x80000000"
    for index, row in enumerate(fixture["rows"]):
        margin = predictor.margin(_row_mapping(fixture, row))
        assert _bits(margin) == "0x80000000", (
            f"row {index}: margin bits {_bits(margin)}, expected 0x80000000; "
            "-0.0 == 0.0 is True, so only the bit pattern can see this"
        )


# ---------------------------------------------------------------------------
# The refusal fixture.
# ---------------------------------------------------------------------------


def test_non_finite_input_fixture_raises_on_every_row() -> None:
    name = "non_finite_input_refusal"
    fixture = FIXTURES[name]
    predictor = Predictor.from_json(fixture["artifact"])

    refused = 0
    for row in fixture["rows"]:
        mapping = _row_mapping(fixture, row)
        with pytest.raises(errors.NonFiniteFeatureError) as margin_failure:
            predictor.margin(mapping)
        assert math.isinf(margin_failure.value.value)
        assert 0 <= margin_failure.value.index < len(fixture["artifact"]["feature_names"])
        with pytest.raises(errors.NonFiniteFeatureError):
            predictor.output(mapping)
        refused += 1

    assert refused == len(fixture["rows"]) == 10, (
        f"{name}: refused {refused} of {len(fixture['rows'])} rows"
    )


def test_infinite_value_raises_even_in_a_column_no_node_reads() -> None:
    """D045: the whole row is checked before the walk, deliberately.

    A lazy check makes the same invalid input raise or not depending on which
    branches this particular model takes -- the outcome becomes a property of
    the model instead of the input.
    """
    artifact = _worked_example()
    predictor = Predictor.from_json(artifact)
    # feature_b is read by no split in the worked example.
    assert all(
        index == 0
        for tree in artifact["trees"]
        for index in tree["split_indices"]
    )
    with pytest.raises(errors.NonFiniteFeatureError) as failure:
        predictor.margin({"feature_a": 0.25, "feature_b": float("inf")})
    assert failure.value.index == 1


def test_the_infinity_refusal_is_not_reimplemented_in_the_reader() -> None:
    """One implementation, not two that could disagree.

    The refusal belongs to ``walk_margin`` (D045). A second copy here would be
    a second thing to keep in agreement with it, and the failure mode of a
    disagreement is that one path accepts what the other refuses.
    """
    source = PREDICT_SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    raised = {
        node.exc.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and isinstance(node.exc.func, ast.Name)
    }
    assert "NonFiniteFeatureError" not in raised, (
        "predict.py raises NonFiniteFeatureError itself; the walk already does"
    )
    assert "NonFiniteFeatureError" not in {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }


def test_nan_is_accepted_and_routes_by_default_left() -> None:
    """``NaN`` is the missing value, not an error, and both directions work."""
    artifact = _artifact(
        trees=[
            {
                "default_left": [1, 0, 0],
                "left_children": [1, -1, -1],
                "node_values": [0.5, -0.25, 0.75],
                "right_children": [2, -1, -1],
                "split_indices": [0, 0, 0],
            }
        ],
        intercept=0.0,
    )
    left = Predictor.from_json(artifact)
    assert _bits(left.margin({"feature_a": float("nan"), "feature_b": 1.0})) == _bits(-0.25)

    artifact["trees"][0]["default_left"] = [0, 0, 0]
    right = Predictor.from_json(artifact)
    assert _bits(right.margin({"feature_a": float("nan"), "feature_b": 1.0})) == _bits(0.75)


# ---------------------------------------------------------------------------
# FORMAT.md section 9.2: the narrowing is structural, not a habit.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_node_values_are_loaded_into_a_read_only_float32_array(name: str) -> None:
    """The invariant is a property of the data structure (FORMAT.md 9.2).

    This is the test that goes red if ``node_values`` is held as a list of
    Python floats. It has to be a structural assertion rather than a
    prediction one: ``walk_margin`` narrows both operands itself and narrowing
    is idempotent, so no *prediction* over this corpus can distinguish a
    float32 array from a float64 list. What differs is every other consumer of
    the array -- a re-serializer, an inspection utility, an arithmetic
    transform -- which is precisely the argument section 9.2 makes.
    """
    predictor = _predictor(name)
    assert len(predictor.trees) == len(FIXTURES[name]["artifact"]["trees"])
    for index, tree in enumerate(predictor.trees):
        values = tree["node_values"]
        assert isinstance(values, np.ndarray), (
            f"{name} tree {index}: node_values is {type(values)}, not an ndarray"
        )
        assert values.dtype == np.float32, (
            f"{name} tree {index}: node_values dtype is {values.dtype}, not float32"
        )
        assert not values.flags.writeable, (
            f"{name} tree {index}: node_values is writeable; a caller could "
            "mutate the loaded model's thresholds"
        )


@pytest.mark.parametrize("name", GROUND_TRUTH_FIXTURES)
def test_every_loaded_node_value_is_the_float32_of_the_serialized_number(name: str) -> None:
    fixture = FIXTURES[name]
    predictor = Predictor.from_json(fixture["artifact"])
    for index, (raw, loaded) in enumerate(
        zip(fixture["artifact"]["trees"], predictor.trees)
    ):
        for position, serialized in enumerate(raw["node_values"]):
            assert _bits(loaded["node_values"][position]) == _bits(serialized), (
                f"{name} tree {index} node {position}: loaded value is not the "
                "float32 of the serialized number"
            )


def test_a_threshold_that_is_not_float32_exact_is_narrowed_at_parse_time() -> None:
    """A hand-edited artifact, which the reader's contract must still handle.

    Our own exporter emits ``float(np.float32(x))``, so every threshold it
    writes already recovers its float32 at any width -- which means our own
    corpus cannot exercise this. ``0.1`` as a float64 is a *different number*
    from the float32 the engine compares against, and after loading, the
    artifact's value must be the float32 one.
    """
    artifact = _artifact()
    artifact["trees"] = [
        {
            "default_left": [1, 0, 0],
            "left_children": [1, -1, -1],
            # 0.1 and 0.3 are not float32-exact as written.
            "node_values": [0.1, 0.3, -0.3],
            "right_children": [2, -1, -1],
            "split_indices": [0, 0, 0],
        }
    ]
    loaded = Predictor.from_json(artifact).trees[0]["node_values"]

    for position, serialized in enumerate([0.1, 0.3, -0.3]):
        assert float(loaded[position]) != serialized, (
            f"position {position}: {serialized!r} is float32-exact after all, so "
            "this test is not exercising narrowing"
        )
        assert _bits(loaded[position]) == _bits(np.float32(serialized))
        assert float(loaded[position]) == float(np.float32(serialized))


def test_intercept_is_narrowed_to_float32_at_parse_time() -> None:
    artifact = _artifact(intercept=0.1, trees=[])
    predictor = Predictor.from_json(artifact)
    assert isinstance(predictor.intercept, np.float32)
    assert float(predictor.intercept) != 0.1
    assert _bits(predictor.intercept) == _bits(np.float32(0.1))
    # With no trees the margin is the intercept alone, untransformed.
    assert _bits(predictor.margin({"feature_a": 0.0, "feature_b": 0.0})) == _bits(
        np.float32(0.1)
    )


def test_negative_zero_intercept_is_not_normalized_on_read() -> None:
    predictor = Predictor.from_json(_artifact(intercept=-0.0, trees=[]))
    assert _bits(predictor.intercept) == "0x80000000"
    assert _bits(predictor.margin({"feature_a": 1.0, "feature_b": 2.0})) == "0x80000000"


def test_node_values_cannot_be_mutated_through_the_public_view() -> None:
    predictor = Predictor.from_json(_worked_example())
    values = predictor.trees[0]["node_values"]
    with pytest.raises(ValueError):
        values[0] = 999.0
    with pytest.raises(TypeError):
        predictor.trees[0]["node_values"] = np.zeros(3, dtype=np.float32)


def test_accumulation_follows_the_normative_recipe_on_the_worked_example() -> None:
    """FORMAT.md section 16's worked example: the spec states the numbers.

    An oracle that is neither this code nor XGBoost. The margin
    ``0.28046515583992004`` is written in the specification; the printed
    sigmoid there is a float64 rendering whose float32 value is what section
    5.1 requires, so the output is compared as float32 bits.
    """
    predictor = Predictor.from_json(_worked_example())
    row = {"feature_a": 0.25, "feature_b": 9.0}
    assert _bits(predictor.margin(row)) == _bits(0.28046515583992004)
    assert _bits(predictor.output(row)) == _bits(np.float32(0.5696602593994496))


# ---------------------------------------------------------------------------
# Envelope validation (FORMAT.md section 13).
# ---------------------------------------------------------------------------


def _worked_example() -> dict[str, Any]:
    """FORMAT.md section 16, verbatim."""
    return {
        "feature_names": ["feature_a", "feature_b"],
        "format_version": 1,
        "intercept": 0.40546515583992004,
        "objective": "binary:logistic",
        "output_transform": "sigmoid",
        "provenance": {
            "base_score": "[6E-1]",
            "exporter_version": "0.1.0.dev0",
            "xgboost_version": "3.3.0",
        },
        "trees": [
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
        ],
    }


def _artifact(**overrides: Any) -> dict[str, Any]:
    """A fresh, deep, valid artifact with ``overrides`` applied."""
    artifact = _worked_example()
    artifact.update(overrides)
    return artifact


def test_a_valid_hand_built_artifact_loads() -> None:
    predictor = Predictor.from_json(_worked_example())
    assert predictor.format_version == READABLE_FORMAT_VERSION
    assert predictor.feature_names == ("feature_a", "feature_b")
    assert predictor.output_transform == "sigmoid"
    assert len(predictor.trees) == 2


@pytest.mark.parametrize("version", [0, 2, 1.0, "1", True, None, [1], 1.5, -1])
def test_a_format_version_other_than_the_integer_one_raises(version: Any) -> None:
    with pytest.raises(errors.UnsupportedVersionError) as failure:
        Predictor.from_json(_artifact(format_version=version))
    assert failure.value.supported == (1,)


def test_format_version_two_raises_and_names_the_value() -> None:
    with pytest.raises(errors.UnsupportedVersionError) as failure:
        Predictor.from_json(_artifact(format_version=2))
    assert failure.value.version == 2


def test_an_eighth_top_level_key_raises() -> None:
    artifact = _worked_example()
    artifact["tree_info"] = [0, 0]
    with pytest.raises(errors.UnrecognizedFieldError) as failure:
        Predictor.from_json(artifact)
    assert failure.value.field == "tree_info"
    assert failure.value.location is None


def test_an_unrecognized_key_is_reported_deterministically() -> None:
    artifact = _worked_example()
    artifact["zzz_last"] = 1
    artifact["aaa_first"] = 1
    with pytest.raises(errors.UnrecognizedFieldError) as failure:
        Predictor.from_json(artifact)
    assert failure.value.field == "aaa_first"


@pytest.mark.parametrize("key", sorted(ENVELOPE_KEYS))
def test_an_absent_required_envelope_key_raises(key: str) -> None:
    artifact = _worked_example()
    del artifact[key]
    with pytest.raises(errors.MalformedTreeError) as failure:
        Predictor.from_json(artifact)
    assert failure.value.field == key
    assert failure.value.value == "<absent>"


@pytest.mark.parametrize("artifact", [None, [], "an artifact", 1, np.float32(1.0)])
def test_an_artifact_that_is_not_a_json_object_raises(artifact: Any) -> None:
    with pytest.raises(errors.MalformedTreeError):
        Predictor.from_json(artifact)


@pytest.mark.parametrize(
    "objective",
    ["multi:softprob", "reg:logistic", "REG:SQUAREDERROR", "reg:squarederror ", "", None, 1],
)
def test_an_objective_outside_the_enumerated_set_raises(objective: Any) -> None:
    with pytest.raises(errors.UnsupportedObjectiveError) as failure:
        Predictor.from_json(_artifact(objective=objective))
    assert failure.value.objective == objective


@pytest.mark.parametrize("transform", ["logit", "relu", "softplus", "SIGMOID", "", None, 1])
def test_an_output_transform_outside_the_enumerated_set_raises(transform: Any) -> None:
    with pytest.raises(errors.MalformedTreeError) as failure:
        Predictor.from_json(_artifact(output_transform=transform))
    assert failure.value.field == "output_transform"


@pytest.mark.parametrize(
    ("objective", "transform"),
    [
        ("binary:logistic", "identity"),
        ("binary:logistic", "exp"),
        ("reg:squarederror", "sigmoid"),
        ("reg:squarederror", "exp"),
        ("survival:cox", "sigmoid"),
        ("survival:cox", "identity"),
    ],
)
def test_an_objective_transform_pairing_that_does_not_match_raises(
    objective: str, transform: str
) -> None:
    with pytest.raises(errors.MalformedTreeError) as failure:
        Predictor.from_json(_artifact(objective=objective, output_transform=transform))
    assert failure.value.field == "output_transform"
    assert OUTPUT_TRANSFORMS[objective] in failure.value.expected


@pytest.mark.parametrize(
    "objective", sorted(OUTPUT_TRANSFORMS)
)
def test_every_paired_objective_and_transform_loads(objective: str) -> None:
    artifact = _artifact(
        objective=objective, output_transform=OUTPUT_TRANSFORMS[objective]
    )
    assert Predictor.from_json(artifact).output_transform == OUTPUT_TRANSFORMS[objective]


@pytest.mark.parametrize(
    "intercept",
    ["0.5", None, [0.5], True, float("inf"), float("-inf"), float("nan"), 1e40, -1e40],
)
def test_an_intercept_that_is_not_a_finite_json_number_raises(intercept: Any) -> None:
    with pytest.raises(errors.MalformedTreeError) as failure:
        Predictor.from_json(_artifact(intercept=intercept))
    assert failure.value.field == "intercept"


def test_a_non_finite_intercept_spelled_as_a_json_token_raises() -> None:
    """``json.loads`` accepts ``Infinity`` and ``NaN``; the reader must not."""
    parsed = json.loads('{"a": Infinity, "b": NaN}')
    assert math.isinf(parsed["a"]) and math.isnan(parsed["b"])

    for token in ("a", "b"):
        artifact = _artifact(intercept=parsed[token])
        with pytest.raises(errors.MalformedTreeError) as failure:
            Predictor.from_json(artifact)
        assert failure.value.field == "intercept"


def test_an_intercept_that_is_a_json_integer_loads() -> None:
    """A JSON number with no fractional part is still a JSON number."""
    predictor = Predictor.from_json(_artifact(intercept=2, trees=[]))
    assert _bits(predictor.intercept) == _bits(2.0)


@pytest.mark.parametrize(
    "feature_names",
    [
        [],
        ["feature_a", "feature_a"],
        ["feature_a", 1],
        ["feature_a", None],
        "feature_a",
        {"feature_a": 0},
        None,
    ],
)
def test_feature_names_that_cannot_support_a_strict_key_policy_raise(
    feature_names: Any,
) -> None:
    with pytest.raises(errors.MalformedTreeError) as failure:
        Predictor.from_json(_artifact(feature_names=feature_names))
    assert failure.value.field == "feature_names"


def test_a_split_index_outside_the_feature_name_range_raises() -> None:
    artifact = _worked_example()
    artifact["trees"][0]["split_indices"] = [2, 0, 0]
    with pytest.raises(errors.MalformedTreeError) as failure:
        Predictor.from_json(artifact)
    assert failure.value.field == "split_indices"
    assert failure.value.location == "trees[0]"


def test_a_negative_split_index_raises() -> None:
    artifact = _worked_example()
    artifact["trees"][0]["split_indices"] = [0, -1, 0]
    with pytest.raises(errors.MalformedTreeError) as failure:
        Predictor.from_json(artifact)
    assert failure.value.field == "split_indices"


def test_a_split_index_is_checked_at_a_leaf_too() -> None:
    """The range check is total, which is what neutralization buys (section 8.3).

    A leaf's ``split_indices`` entry is never read by the walk, so a reader
    could skip it -- but then the check would need an exception for pruned
    models, and an exception that applies to every artifact is not a check.
    """
    artifact = _worked_example()
    artifact["trees"][1]["split_indices"] = [7]
    with pytest.raises(errors.MalformedTreeError) as failure:
        Predictor.from_json(artifact)
    assert failure.value.location == "trees[1]"


# ---------------------------------------------------------------------------
# provenance: non-operative, validated anyway.
# ---------------------------------------------------------------------------


def test_provenance_is_exposed_and_read_by_no_prediction() -> None:
    predictor = Predictor.from_json(_worked_example())
    assert predictor.provenance["base_score"] == "[6E-1]"
    assert set(predictor.provenance) == PROVENANCE_KEYS


def test_an_unrecognized_provenance_key_raises() -> None:
    artifact = _worked_example()
    artifact["provenance"]["boost_from_average"] = "1"
    with pytest.raises(errors.UnrecognizedFieldError) as failure:
        Predictor.from_json(artifact)
    assert failure.value.field == "boost_from_average"
    assert failure.value.location == "provenance"


@pytest.mark.parametrize("key", sorted(PROVENANCE_KEYS))
def test_an_absent_provenance_key_raises(key: str) -> None:
    artifact = _worked_example()
    del artifact["provenance"][key]
    with pytest.raises(errors.MalformedTreeError) as failure:
        Predictor.from_json(artifact)
    assert failure.value.field == key
    assert failure.value.location == "provenance"


@pytest.mark.parametrize("key", sorted(PROVENANCE_KEYS))
def test_a_provenance_value_of_the_wrong_json_type_raises(key: str) -> None:
    """A numeric field carried as a number where the format says string.

    ``provenance.base_score`` is the raw bracketed string XGBoost stored, e.g.
    ``"[6E-1]"``. A parsed number there would be a *derived* value in a block
    whose entire purpose is fidelity.
    """
    artifact = _worked_example()
    artifact["provenance"][key] = 0.6
    with pytest.raises(errors.MalformedTreeError) as failure:
        Predictor.from_json(artifact)
    assert failure.value.field == key


@pytest.mark.parametrize("provenance", [None, [], "3.3.0", 1])
def test_a_provenance_block_that_is_not_a_json_object_raises(provenance: Any) -> None:
    with pytest.raises(errors.MalformedTreeError) as failure:
        Predictor.from_json(_artifact(provenance=provenance))
    assert failure.value.field == "provenance"


# ---------------------------------------------------------------------------
# Tree validation (FORMAT.md section 8).
# ---------------------------------------------------------------------------


def test_an_unrecognized_tree_key_raises() -> None:
    artifact = _worked_example()
    artifact["trees"][0]["base_weights"] = [0.0, 0.0, 0.0]
    with pytest.raises(errors.UnrecognizedFieldError) as failure:
        Predictor.from_json(artifact)
    assert failure.value.field == "base_weights"
    assert failure.value.location == "trees[0]"


@pytest.mark.parametrize("key", sorted(TREE_KEYS))
def test_an_absent_tree_key_raises(key: str) -> None:
    artifact = _worked_example()
    del artifact["trees"][0][key]
    with pytest.raises(errors.MalformedTreeError) as failure:
        Predictor.from_json(artifact)
    assert failure.value.field == key
    assert failure.value.location == "trees[0]"


@pytest.mark.parametrize("key", sorted(TREE_KEYS - {"left_children"}))
def test_tree_arrays_of_unequal_length_raise(key: str) -> None:
    artifact = _worked_example()
    artifact["trees"][0][key] = artifact["trees"][0][key][:-1]
    with pytest.raises(errors.MalformedTreeError) as failure:
        Predictor.from_json(artifact)
    assert failure.value.field == key


def test_a_short_left_children_array_raises_through_its_own_length() -> None:
    """``left_children`` defines the node count, so shortening it moves the count."""
    artifact = _worked_example()
    artifact["trees"][0]["left_children"] = [1, -1]
    with pytest.raises(errors.MalformedTreeError):
        Predictor.from_json(artifact)


def test_a_tree_with_no_nodes_raises() -> None:
    artifact = _worked_example()
    artifact["trees"][0] = {
        "default_left": [],
        "left_children": [],
        "node_values": [],
        "right_children": [],
        "split_indices": [],
    }
    with pytest.raises(errors.MalformedTreeError) as failure:
        Predictor.from_json(artifact)
    assert failure.value.field == "left_children"


@pytest.mark.parametrize("child", [3, 99, -2, -5])
def test_a_child_index_out_of_range_raises(child: int) -> None:
    artifact = _worked_example()
    artifact["trees"][0]["left_children"] = [child, -1, -1]
    with pytest.raises(errors.MalformedTreeError) as failure:
        Predictor.from_json(artifact)
    assert failure.value.field == "left_children"


def test_a_right_child_index_out_of_range_raises() -> None:
    artifact = _worked_example()
    artifact["trees"][0]["right_children"] = [17, -1, -1]
    with pytest.raises(errors.MalformedTreeError) as failure:
        Predictor.from_json(artifact)
    assert failure.value.field == "right_children"


def test_a_leaf_whose_right_child_is_not_minus_one_raises() -> None:
    """The vector-leaf signature: that slot carries a block index, not a child."""
    artifact = _worked_example()
    artifact["trees"][0]["right_children"] = [2, 5, -1]
    with pytest.raises(errors.MalformedTreeError) as failure:
        Predictor.from_json(artifact)
    assert failure.value.field == "right_children"


@pytest.mark.parametrize("value", [2, -1, 7, True, False, 1.0, "1", None])
def test_a_default_left_entry_that_is_not_zero_or_one_raises(value: Any) -> None:
    artifact = _worked_example()
    artifact["trees"][0]["default_left"] = [value, 0, 0]
    with pytest.raises(errors.MalformedTreeError) as failure:
        Predictor.from_json(artifact)
    assert failure.value.field == "default_left"


@pytest.mark.parametrize("value", [1.5, "1", None, True, [1]])
def test_a_child_index_that_is_not_an_integer_raises(value: Any) -> None:
    artifact = _worked_example()
    artifact["trees"][0]["left_children"] = [1, value, -1]
    with pytest.raises(errors.MalformedTreeError) as failure:
        Predictor.from_json(artifact)
    assert failure.value.field == "left_children"


@pytest.mark.parametrize("value", ["0.5", None, True, [0.5], {"a": 1}])
def test_a_node_value_that_is_not_a_json_number_raises(value: Any) -> None:
    artifact = _worked_example()
    artifact["trees"][0]["node_values"] = [0.5, value, 0.75]
    with pytest.raises(errors.MalformedTreeError) as failure:
        Predictor.from_json(artifact)
    assert failure.value.field == "node_values"


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan"), 1e40, -1e40])
def test_a_non_finite_node_value_raises(value: float) -> None:
    artifact = _worked_example()
    artifact["trees"][0]["node_values"] = [0.5, value, 0.75]
    with pytest.raises(errors.MalformedTreeError) as failure:
        Predictor.from_json(artifact)
    assert failure.value.field == "node_values"
    assert failure.value.location == "trees[0]"


def test_a_node_value_that_overflows_float32_raises_rather_than_becoming_inf() -> None:
    """``1e40`` is a finite float64 and an infinite float32.

    Narrowing happens at parse time, so the refusal has to happen there too --
    a reader that checked finiteness *before* narrowing would accept this and
    walk an infinite threshold.
    """
    artifact = _worked_example()
    artifact["trees"][0]["node_values"] = [1e40, -0.25, 0.75]
    assert math.isfinite(1e40)
    with pytest.raises(errors.MalformedTreeError):
        Predictor.from_json(artifact)


@pytest.mark.parametrize("trees", [None, {}, "trees", 1])
def test_a_trees_field_that_is_not_an_array_raises(trees: Any) -> None:
    with pytest.raises(errors.MalformedTreeError) as failure:
        Predictor.from_json(_artifact(trees=trees))
    assert failure.value.field == "trees"


@pytest.mark.parametrize("tree", [None, [], "tree", 1])
def test_a_tree_that_is_not_a_json_object_raises(tree: Any) -> None:
    with pytest.raises(errors.MalformedTreeError):
        Predictor.from_json(_artifact(trees=[tree]))


def test_an_empty_tree_list_loads_and_the_margin_is_the_intercept() -> None:
    """A zero-boosting-round model serializes ``"trees": []`` and is valid."""
    predictor = Predictor.from_json(_artifact(trees=[], intercept=0.40546515583992004))
    assert predictor.trees == ()
    assert _bits(predictor.margin({"feature_a": 1.0, "feature_b": 2.0})) == _bits(
        np.float32(0.40546515583992004)
    )


def test_a_cycle_reachable_from_the_root_raises_rather_than_hanging() -> None:
    """A refusal against a hang rather than against a wrong number.

    ``walk_margin`` follows children until it meets a leaf, so a cycle never
    terminates and the caller has nothing to catch. Every index here is in
    range, so the range checks cannot see it.
    """
    artifact = _worked_example()
    artifact["trees"][0] = {
        "default_left": [1, 0, 0],
        "left_children": [1, 0, -1],
        "node_values": [0.5, 0.25, 0.75],
        "right_children": [2, 2, -1],
        "split_indices": [0, 0, 0],
    }
    with pytest.raises(errors.MalformedTreeError):
        Predictor.from_json(artifact)


def test_a_self_loop_raises() -> None:
    artifact = _worked_example()
    artifact["trees"][0] = {
        "default_left": [1, 0, 0],
        "left_children": [1, 1, -1],
        "node_values": [0.5, 0.25, 0.75],
        "right_children": [2, 2, -1],
        "split_indices": [0, 0, 0],
    }
    with pytest.raises(errors.MalformedTreeError):
        Predictor.from_json(artifact)


def test_a_shared_subtree_is_not_refused() -> None:
    """Two parents, one child, no cycle: it terminates, so it is not refused."""
    artifact = _worked_example()
    artifact["trees"][0] = {
        "default_left": [1, 1, 0, 0],
        "left_children": [1, 3, 3, -1],
        "node_values": [0.5, 0.25, 0.75, -0.5],
        "right_children": [2, 3, 3, -1],
        "split_indices": [0, 0, 0, 0],
    }
    predictor = Predictor.from_json(artifact)
    margin = predictor.margin({"feature_a": 0.1, "feature_b": 0.0})
    assert _bits(margin) == _bits(
        np.float32(np.float32(np.float32(0.40546515583992004) + np.float32(-0.5)) + np.float32(0.125))
    )


def test_a_node_unreachable_from_the_root_does_not_raise() -> None:
    """FORMAT.md section 13: a reader MUST NOT raise on an unreachable node.

    Neutralized dead slots are legitimate content, indistinguishable from a
    leaf carrying ``0.0``, and never visited. A reader that rejected them would
    reject every pruned model.
    """
    artifact = _worked_example()
    artifact["trees"][0] = {
        # Nodes 3 and 4 are neutralized dead slots: unreachable from node 0.
        "default_left": [1, 0, 0, 0, 0],
        "left_children": [1, -1, -1, -1, -1],
        "node_values": [0.5, -0.25, 0.75, 0.0, 0.0],
        "right_children": [2, -1, -1, -1, -1],
        "split_indices": [0, 0, 0, 0, 0],
    }
    predictor = Predictor.from_json(artifact)
    assert len(predictor.trees[0]["node_values"]) == 5
    assert _bits(predictor.margin({"feature_a": 0.25, "feature_b": 9.0})) == _bits(
        0.28046515583992004
    )


def test_an_unreachable_internal_node_does_not_raise_either() -> None:
    """Not only neutralized leaves: any unreachable node is left alone.

    A dead set is not in general a trailing suffix and a stale link can point
    into it, so a reader must tolerate unreachable *internal* nodes too, as
    long as every index it carries is in range.
    """
    artifact = _worked_example()
    artifact["trees"][0] = {
        "default_left": [1, 0, 0, 1, 0, 0],
        "left_children": [1, -1, -1, 4, -1, -1],
        "node_values": [0.5, -0.25, 0.75, 1.5, 100.0, 200.0],
        "right_children": [2, -1, -1, 5, -1, -1],
        "split_indices": [0, 0, 0, 1, 0, 0],
    }
    predictor = Predictor.from_json(artifact)
    assert _bits(predictor.margin({"feature_a": 0.25, "feature_b": 9.0})) == _bits(
        0.28046515583992004
    )


def test_the_pruned_fixtures_carry_nodes_unreachable_from_the_root() -> None:
    """The MUST-NOT-raise rule needs a fixture where it actually applies."""
    for name in ("gamma_pruned_dead_nodes", "gamma_pruned_neutralization"):
        fixture = FIXTURES[name]
        dead = fixture["meta"].get("dead_node_indices_per_tree")
        assert dead is not None, f"{name}: meta records no dead node indices"
        assert sum(len(indices) for indices in dead) > 0, (
            f"{name}: no tree has an unreachable node, so it does not exercise the rule"
        )
        Predictor.from_json(fixture["artifact"])


# ---------------------------------------------------------------------------
# Strict feature keys (D005).
# ---------------------------------------------------------------------------


def test_a_missing_feature_key_raises() -> None:
    predictor = Predictor.from_json(_worked_example())
    with pytest.raises(errors.FeatureKeyMismatchError) as failure:
        predictor.margin({"feature_a": 0.25})
    assert failure.value.missing_keys == frozenset({"feature_b"})
    assert failure.value.extra_keys == frozenset()


def test_an_extra_feature_key_raises() -> None:
    predictor = Predictor.from_json(_worked_example())
    with pytest.raises(errors.FeatureKeyMismatchError) as failure:
        predictor.margin({"feature_a": 0.25, "feature_b": 9.0, "feature_c": 1.0})
    assert failure.value.missing_keys == frozenset()
    assert failure.value.extra_keys == frozenset({"feature_c"})


def test_a_typoed_feature_key_raises_and_is_reported_as_both() -> None:
    """A typo is one missing key and one extra key, and is diagnosed as such.

    This is the case D005 exists for: lenient handling would route
    ``feature_b`` down the missing-value branch -- legitimate model structure
    -- and return a confident wrong number instead of an error.
    """
    predictor = Predictor.from_json(_worked_example())
    with pytest.raises(errors.FeatureKeyMismatchError) as failure:
        predictor.margin({"feature_a": 0.25, "featureb": 9.0})
    assert failure.value.missing_keys == frozenset({"feature_b"})
    assert failure.value.extra_keys == frozenset({"featureb"})


def test_a_key_no_split_reads_must_still_be_present() -> None:
    """FORMAT.md section 16: ``feature_b`` is read by no split and is required."""
    predictor = Predictor.from_json(_worked_example())
    with pytest.raises(errors.FeatureKeyMismatchError):
        predictor.margin({"feature_a": 0.25})
    assert _bits(predictor.margin({"feature_a": 0.25, "feature_b": 9.0})) == _bits(
        0.28046515583992004
    )


def test_output_applies_the_same_strict_key_policy_as_margin() -> None:
    predictor = Predictor.from_json(_worked_example())
    with pytest.raises(errors.FeatureKeyMismatchError):
        predictor.output({"feature_a": 0.25})


@pytest.mark.parametrize("row", [None, [0.25, 9.0], "row", 1, (0.25, 9.0)])
def test_a_row_that_is_not_a_mapping_raises(row: Any) -> None:
    predictor = Predictor.from_json(_worked_example())
    with pytest.raises(errors.MalformedTreeError):
        predictor.margin(row)


# ---------------------------------------------------------------------------
# Strict feature *values*: D005 one level down. Nothing is coerced.
# ---------------------------------------------------------------------------

#: Values a prediction input may not carry, with the type each one reports.
#: ``"nan"`` heads the list because it is the dangerous one: ``float("nan")``
#: of that string is the **missing value**, so a lenient constructor turns a
#: quoted number into a legitimate model branch and returns a confident wrong
#: prediction with no error at all.
_REFUSED_FEATURE_VALUES: tuple[tuple[str, Any, str], ...] = (
    ("the string nan", "nan", "str"),
    ("a quoted number", "0.5", "str"),
    ("a quoted integer", "1", "str"),
    ("the string inf", "inf", "str"),
    ("an empty string", "", "str"),
    ("True", True, "bool"),
    ("False", False, "bool"),
    ("None", None, "NoneType"),
    ("a list", [0.5], "list"),
    ("a tuple", (0.5,), "tuple"),
    ("a dict", {"value": 0.5}, "dict"),
    ("bytes", b"0.5", "bytes"),
    ("an int too large for a float64", 10**400, "int"),
    # numpy 2 spells this type's `__name__` "bool", like the builtin, while
    # being a different type -- it is not an `np.integer`, so it is refused by
    # the type test rather than by the `bool` branch.
    ("a numpy bool", np.bool_(True), "bool"),
    ("a complex number", complex(0.5, 0.0), "complex"),
)


@pytest.mark.parametrize(
    ("label", "value", "type_name"),
    _REFUSED_FEATURE_VALUES,
    ids=[label for label, _value, _type in _REFUSED_FEATURE_VALUES],
)
def test_a_feature_value_that_is_not_a_number_raises(
    label: str, value: Any, type_name: str
) -> None:
    """Structured, per FORMAT.md section 13: the name, the value, and its type.

    ``float()`` accepts every string in this list. ``"nan"`` and ``"inf"``
    would become the missing value and an infinity respectively -- one a
    silent wrong number, the other a refusal raised for the wrong reason --
    and ``True`` would become ``1.0``, which is a coercion rather than a
    feature. ``10 ** 400`` is refused here rather than being allowed to
    overflow to ``inf``, because ``float()`` raises ``OverflowError`` there,
    which carries no structured attributes.
    """
    predictor = Predictor.from_json(_worked_example())
    with pytest.raises(errors.InvalidFeatureValueError) as failure:
        predictor.margin({"feature_a": value, "feature_b": 9.0})
    assert failure.value.feature == "feature_a"
    assert failure.value.value_type == type_name
    assert failure.value.expected
    # The value is carried verbatim, so a caller can log what arrived.
    assert type(failure.value.value) is type(value)


@pytest.mark.parametrize(
    ("label", "value", "type_name"),
    _REFUSED_FEATURE_VALUES,
    ids=[label for label, _value, _type in _REFUSED_FEATURE_VALUES],
)
def test_output_refuses_the_same_values_as_margin(
    label: str, value: Any, type_name: str
) -> None:
    predictor = Predictor.from_json(_worked_example())
    with pytest.raises(errors.InvalidFeatureValueError):
        predictor.output({"feature_a": value, "feature_b": 9.0})


def test_a_refused_value_in_any_column_raises() -> None:
    """Not only the first column: every value is checked, in order."""
    predictor = Predictor.from_json(_worked_example())
    with pytest.raises(errors.InvalidFeatureValueError) as failure:
        predictor.margin({"feature_a": 0.25, "feature_b": "9.0"})
    assert failure.value.feature == "feature_b"


def test_the_refusal_is_a_bridge_error_a_caller_can_catch() -> None:
    """Not a bare ``TypeError`` or ``OverflowError`` escaping the library."""
    predictor = Predictor.from_json(_worked_example())
    for value in (None, [0.5], 10**400):
        with pytest.raises(errors.XGBoostBridgeError):
            predictor.margin({"feature_a": value, "feature_b": 9.0})


_ACCEPTED_FEATURE_VALUES: tuple[tuple[str, Any], ...] = (
    ("a Python float", 0.25),
    ("a Python int", 1),
    ("a negative int", -3),
    ("a float32", np.float32(0.25)),
    ("a float64", np.float64(0.25)),
    ("an int64", np.int64(1)),
    ("an int32", np.int32(-3)),
    ("a float16", np.float16(0.25)),
    ("negative zero", -0.0),
)


@pytest.mark.parametrize(
    ("label", "value"),
    _ACCEPTED_FEATURE_VALUES,
    ids=[label for label, _value in _ACCEPTED_FEATURE_VALUES],
)
def test_real_numbers_are_still_accepted(label: str, value: Any) -> None:
    """The refusal must not narrow what a caller can legitimately pass.

    ``numpy`` scalars are included deliberately: iterating a matrix row --
    which is how a caller ordinarily builds a mapping -- hands out
    ``np.float64`` values, and ``np.float32`` arrives from anything already
    narrowed.
    """
    predictor = Predictor.from_json(_worked_example())
    got = predictor.margin({"feature_a": value, "feature_b": 9.0})
    expected = predictor.margin({"feature_a": float(value), "feature_b": 9.0})
    assert _bits(got) == _bits(expected)


def test_a_real_nan_is_still_the_missing_value_and_still_routes_by_default_left() -> None:
    """The value a caller passes when they mean "missing", both directions.

    The point of the refusal above is that this remains the *only* way to say
    it. If the string ``"nan"`` also worked, a quoted-number producer would
    reach this branch by accident.
    """
    artifact = _artifact(
        trees=[
            {
                "default_left": [1, 0, 0],
                "left_children": [1, -1, -1],
                "node_values": [0.5, -0.25, 0.75],
                "right_children": [2, -1, -1],
                "split_indices": [0, 0, 0],
            }
        ],
        intercept=0.0,
    )
    left = Predictor.from_json(artifact)
    assert _bits(left.margin({"feature_a": float("nan"), "feature_b": 1.0})) == _bits(-0.25)
    assert _bits(left.margin({"feature_a": np.float32("nan"), "feature_b": 1.0})) == _bits(
        -0.25
    )
    with pytest.raises(errors.InvalidFeatureValueError):
        left.margin({"feature_a": "nan", "feature_b": 1.0})

    artifact["trees"][0]["default_left"] = [0, 0, 0]
    right = Predictor.from_json(artifact)
    assert _bits(right.margin({"feature_a": float("nan"), "feature_b": 1.0})) == _bits(0.75)
    with pytest.raises(errors.InvalidFeatureValueError):
        right.margin({"feature_a": "nan", "feature_b": 1.0})


def test_an_infinite_value_still_raises_the_refusal_that_owns_it() -> None:
    """``±inf`` is a number, so it passes the type check and reaches D045's
    refusal in the walk. The two refusals must not be confused: one is "not a
    number", the other is "a number this library declines"."""
    predictor = Predictor.from_json(_worked_example())
    for value in (float("inf"), float("-inf"), np.float32("inf")):
        with pytest.raises(errors.NonFiniteFeatureError):
            predictor.margin({"feature_a": value, "feature_b": 9.0})


def test_a_batch_helper_refuses_a_non_numeric_value_too() -> None:
    predictor = Predictor.from_json(_worked_example())
    rows = [
        {"feature_a": 0.25, "feature_b": 9.0},
        {"feature_a": "0.25", "feature_b": 9.0},
    ]
    with pytest.raises(errors.InvalidFeatureValueError):
        predictor.margins(rows)


def test_key_order_in_the_input_does_not_matter() -> None:
    predictor = Predictor.from_json(_worked_example())
    forward = predictor.margin({"feature_a": 0.25, "feature_b": 9.0})
    backward = predictor.margin({"feature_b": 9.0, "feature_a": 0.25})
    assert _bits(forward) == _bits(backward)


# ---------------------------------------------------------------------------
# `objective` is non-operative metadata (D028).
# ---------------------------------------------------------------------------

#: The functions a prediction actually passes through.
PREDICTION_PATH_FUNCTIONS = frozenset(
    {
        "margin",
        "output",
        "margins",
        "outputs",
        "_collect",
        "_feature_row",
        "_feature_value",
    }
)


def _function_definitions(source: str) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef)
    }


def test_no_prediction_path_function_reads_the_objective_field() -> None:
    """A source-level check, mirrored on the JavaScript side (D028).

    Without it a future contributor adds ``if objective == ...`` to a
    prediction path and the field quietly becomes a second source of truth
    about behaviour ``output_transform`` already determines.
    """
    source = PREDICT_SOURCE_PATH.read_text(encoding="utf-8")
    definitions = _function_definitions(source)
    missing = sorted(PREDICTION_PATH_FUNCTIONS - set(definitions))
    assert not missing, (
        f"prediction-path functions not found in predict.py: {missing} -- this "
        "check silently covers nothing if the names drift"
    )

    forbidden_constants = {"objective", *OUTPUT_TRANSFORMS}
    findings: list[str] = []
    for name in sorted(PREDICTION_PATH_FUNCTIONS):
        definition = definitions[name]
        docstring_nodes = {
            id(statement.value)
            for statement in definition.body
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)
        }
        for node in ast.walk(definition):
            if isinstance(node, ast.Attribute) and "objective" in node.attr:
                findings.append(f"{name}: attribute {node.attr!r}")
            elif isinstance(node, ast.Name) and "objective" in node.id:
                findings.append(f"{name}: name {node.id!r}")
            elif (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value in forbidden_constants
                and id(node) not in docstring_nodes
            ):
                findings.append(f"{name}: literal {node.value!r}")

    assert not findings, "the prediction path reads `objective`:\n" + "\n".join(findings)


@pytest.mark.parametrize("name", GROUND_TRUTH_FIXTURES)
def test_predictions_are_unchanged_when_the_objective_field_is_corrupted(name: str) -> None:
    """The behavioural half of D028, since a source check can be evaded.

    The field is overwritten *after* loading, so the load-time pairing check
    (which is the field's entire job) still ran on the real value.
    """
    fixture = FIXTURES[name]
    rows = [_row_mapping(fixture, row) for row in fixture["rows"]]

    honest = Predictor.from_json(fixture["artifact"])
    before = [(_bits(honest.margin(row)), _bits(honest.output(row))) for row in rows]

    corrupted = Predictor.from_json(fixture["artifact"])
    corrupted._objective = "not:an:objective"  # noqa: SLF001 -- the point of the test
    after = [(_bits(corrupted.margin(row)), _bits(corrupted.output(row))) for row in rows]

    assert before == after, f"{name}: predictions changed with the objective field"


# ---------------------------------------------------------------------------
# The reader's constants must not drift from the writer's.
# ---------------------------------------------------------------------------


def test_the_readers_key_sets_agree_with_the_writers() -> None:
    """Stated independently in both modules, compared here.

    A reader that imported the writer's key set could not detect a writer that
    emits the wrong one. Two statements and one comparison can.
    """
    assert ENVELOPE_KEYS == export.ENVELOPE_KEYS
    assert PROVENANCE_KEYS == export.PROVENANCE_KEYS
    assert READABLE_FORMAT_VERSION == export.FORMAT_VERSION
    assert TREE_KEYS == frozenset(SOURCE_TREE_KEYS)
    assert len(ENVELOPE_KEYS) == 7
    assert len(PROVENANCE_KEYS) == 3
    assert len(TREE_KEYS) == 5


def test_every_objective_has_a_transform_and_every_transform_is_implemented() -> None:
    assert set(OUTPUT_TRANSFORMS.values()) <= set(OUTPUT_FUNCTIONS)
    assert set(OUTPUT_FUNCTIONS) == {"identity", "sigmoid", "exp"}


# ---------------------------------------------------------------------------
# Batch helpers: a loop over the single-row path, and nothing more.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", GROUND_TRUTH_FIXTURES)
def test_batch_helpers_agree_with_the_single_row_path_bit_for_bit(name: str) -> None:
    fixture = FIXTURES[name]
    predictor = Predictor.from_json(fixture["artifact"])
    rows = [_row_mapping(fixture, row) for row in fixture["rows"]]

    margins = predictor.margins(rows)
    outputs = predictor.outputs(rows)
    assert margins.dtype == np.float32
    assert outputs.dtype == np.float32
    assert margins.shape == (len(rows),)

    for index, row in enumerate(rows):
        assert _bits(margins[index]) == _bits(predictor.margin(row))
        assert _bits(outputs[index]) == _bits(predictor.output(row))
        assert _bits(margins[index]) == fixture["expected_margin"][index]


def test_batch_helpers_accept_an_empty_sequence() -> None:
    predictor = Predictor.from_json(_worked_example())
    assert predictor.margins([]).shape == (0,)
    assert predictor.outputs([]).dtype == np.float32


def test_a_batch_helper_raises_on_the_first_bad_row() -> None:
    predictor = Predictor.from_json(_worked_example())
    rows = [{"feature_a": 0.25, "feature_b": 9.0}, {"feature_a": 0.25}]
    with pytest.raises(errors.FeatureKeyMismatchError):
        predictor.margins(rows)


# ---------------------------------------------------------------------------
# The live oracle: a real model, exported and read back in this process.
# ---------------------------------------------------------------------------

_LIVE_BASE_PARAMS = {"tree_method": "exact", "max_depth": 3, "eta": 0.3, "nthread": 1}

#: Deliberately far from 0.5 in both directions. At ``base_score = 0.5`` the
#: logistic intercept is exactly ``-0.0`` and every wrong intercept-placement
#: variant scores 5000/5000, so a model built there validates a broken
#: implementation (FORMAT.md section 10).
_LIVE_CASES = (
    ("reg:squarederror", 7.5),
    ("binary:logistic", 0.987654),
    ("survival:cox", 0.13),
)


def _fit(objective: str, base_score: float, seed: int = 20260805):
    rng = np.random.default_rng(seed)
    columns = 5
    rows = 400
    matrix_values = rng.normal(size=(rows, columns))

    label_rng = np.random.default_rng(seed + 1)
    if objective == "binary:logistic":
        label = (label_rng.random(rows) > 0.5).astype(np.float64)
    elif objective == "survival:cox":
        # Sign convention: positive is an event, negative is right-censored.
        magnitude = label_rng.exponential(scale=2.0, size=rows) + 0.05
        sign = np.where(label_rng.random(rows) > 0.3, 1.0, -1.0)
        label = magnitude * sign
    else:
        label = label_rng.normal(size=rows)

    names = [f"column_{index}" for index in range(columns)]
    train = xgb.DMatrix(matrix_values, label=label, feature_names=names)
    params: dict[str, Any] = dict(_LIVE_BASE_PARAMS)
    params["objective"] = objective
    params["seed"] = seed
    params["base_score"] = float(base_score)
    booster = xgb.train(params, train, num_boost_round=8, verbose_eval=False)
    return booster, matrix_values, names


@pytest.mark.parametrize(("objective", "base_score"), _LIVE_CASES)
def test_a_freshly_exported_model_reproduces_xgboosts_margin_bit_for_bit(
    objective: str, base_score: float
) -> None:
    """The oracle a frozen corpus cannot be: XGBoost, in this process, now.

    The corpus files predate any later change to the export path. This test
    fits, exports, loads and compares end to end, so a defect introduced
    between the corpus's generation and today has somewhere to show up.
    """
    booster, matrix_values, names = _fit(objective, base_score)
    artifact = export.export_model(booster)
    predictor = Predictor.from_json(json.loads(export.to_json(artifact)))

    expected_margin = np.asarray(
        booster.predict(
            xgb.DMatrix(matrix_values, feature_names=names), output_margin=True
        ),
        dtype=np.float32,
    )
    expected_output = np.asarray(
        booster.predict(xgb.DMatrix(matrix_values, feature_names=names)),
        dtype=np.float32,
    )

    rows = [dict(zip(names, row)) for row in matrix_values]
    margins = predictor.margins(rows)
    outputs = predictor.outputs(rows)

    margin_mismatches = [
        (index, _bits(expected_margin[index]), _bits(margins[index]))
        for index in range(len(rows))
        if _bits(expected_margin[index]) != _bits(margins[index])
    ]
    assert not margin_mismatches, (
        f"{objective}: margin disagreed with XGBoost on "
        f"{len(margin_mismatches)}/{len(rows)} rows: {margin_mismatches[:5]}"
    )

    worst = 0.0
    worst_row = None
    for index in range(len(rows)):
        error = _relative_error(outputs[index], expected_output[index])
        if error > worst:
            worst = error
            worst_row = (index, float(outputs[index]), float(expected_output[index]))
    assert worst <= OUTPUT_RELATIVE_GATE, (
        f"{objective}: max relative output error {worst:.3e} at row {worst_row}"
    )


def test_a_freshly_exported_zero_tree_model_loads_and_returns_its_intercept() -> None:
    """The configuration where the intercept *is* the entire output."""
    rng = np.random.default_rng(4321)
    matrix_values = rng.normal(size=(3, 2))
    names = ["column_0", "column_1"]
    train = xgb.DMatrix(
        matrix_values, label=np.asarray([0.0, 1.0, 0.0]), feature_names=names
    )
    booster = xgb.train(
        {
            "objective": "binary:logistic",
            "base_score": 0.5,
            "nthread": 1,
            "tree_method": "exact",
        },
        train,
        num_boost_round=0,
    )
    predictor = Predictor.from_json(export.export_model(booster))
    assert predictor.trees == ()

    expected = np.asarray(
        booster.predict(xgb.DMatrix(matrix_values, feature_names=names), output_margin=True),
        dtype=np.float32,
    )
    for index, row in enumerate(matrix_values):
        assert _bits(predictor.margin(dict(zip(names, row)))) == _bits(expected[index])
    # `base_score=0.5` passed explicitly puts the intercept in link space,
    # where it is exactly -0.0 (FORMAT.md section 6.3).
    assert _bits(predictor.intercept) == "0x80000000"
