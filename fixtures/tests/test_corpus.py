"""Verification of the fixture corpus written by `fixtures/generate/corpus.py`.

Every check here has an oracle independent of what it checks: ground truth
in each fixture file is XGBoost's own `predict()` output, recorded once at
generation time and never recomputed here. This suite re-walks the recorded
artifact with this repository's own `walk_margin` and asks whether it
reproduces that already-recorded XGBoost output bit-for-bit -- it does not
re-derive the fixture, and it does not trust `fixtures/generate/corpus.py`'s
own generation-time self-check, which runs before a fixture is written and
could in principle be skipped or defeated without this suite noticing
otherwise.

Nothing here regenerates the corpus. Regeneration and its byte-identical
determinism is verified separately (by running the generator twice and
diffing file hashes), because refitting real models on every test run would
make this suite slow and would reintroduce exactly the "needs XGBoost
installed" dependency the corpus exists to remove from the JavaScript side.
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

from generate.corpus import CORPUS_DIR, FIXTURE_NAMES  # noqa: E402

from xgboost_bridge.trees import walk_margin  # noqa: E402

#: FORMAT.md section 3 -- the seven required top-level artifact keys.
ENVELOPE_KEYS = frozenset(
    {
        "format_version",
        "objective",
        "output_transform",
        "intercept",
        "feature_names",
        "trees",
        "provenance",
    }
)


def _load_corpus() -> dict[str, dict[str, Any]]:
    if not CORPUS_DIR.is_dir():
        return {}
    return {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in sorted(CORPUS_DIR.glob("*.json"))}


#: Loaded once at collection time. A corpus that silently lost a fixture
#: between generation and this test run must fail rather than pass quietly
#: on an empty dict, hence `test_corpus_is_non_empty` below rather than
#: skipping when `CORPUS` happens to be empty.
CORPUS: dict[str, dict[str, Any]] = _load_corpus()


def _bits_to_float32(bits: str) -> np.float32:
    """Parse a `"0xXXXXXXXX"` ground-truth token back to the float32 it names."""
    assert bits.startswith("0x") and len(bits) == 10, f"not a uint32 hex bit pattern: {bits!r}"
    return np.uint32(int(bits, 16)).view(np.float32)[()]


def _bits_of(value: float) -> str:
    return f"0x{int(np.float32(value).view(np.uint32)):08x}"


def _row_to_features(row: list[float | None]) -> list[float]:
    """The wire encoding's `null` becomes `NaN`, exactly as a real predictor must do."""
    return [float("nan") if value is None else float(value) for value in row]


def test_corpus_is_non_empty() -> None:
    assert CORPUS, f"fixture corpus at {CORPUS_DIR} is empty"


def test_every_required_fixture_is_present() -> None:
    """A corpus that silently lost a required fixture must fail, not pass quietly."""
    missing = set(FIXTURE_NAMES) - set(CORPUS)
    assert not missing, f"required fixtures missing from the corpus: {sorted(missing)}"
    extra = set(CORPUS) - set(FIXTURE_NAMES)
    assert not extra, (
        f"corpus contains fixtures not declared in FIXTURE_NAMES: {sorted(extra)} -- "
        "add them to fixtures/generate/corpus.py's builder tuple"
    )


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_envelope_has_exactly_seven_keys(name: str) -> None:
    artifact = CORPUS[name]["artifact"]
    present = frozenset(artifact)
    assert present == ENVELOPE_KEYS, (
        f"{name}: artifact keys {sorted(present)} do not match the required "
        f"envelope {sorted(ENVELOPE_KEYS)}"
    )


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_format_version_is_one(name: str) -> None:
    assert CORPUS[name]["artifact"]["format_version"] == 1


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_ground_truth_arrays_match_row_count(name: str) -> None:
    fixture = CORPUS[name]
    row_count = len(fixture["rows"])
    assert len(fixture["expected_margin"]) == row_count, f"{name}: expected_margin length mismatch"
    assert len(fixture["expected_output"]) == row_count, f"{name}: expected_output length mismatch"
    assert len(fixture["margin_decimal"]) == row_count, f"{name}: margin_decimal length mismatch"
    assert len(fixture["output_decimal"]) == row_count, f"{name}: output_decimal length mismatch"
    assert fixture["meta"]["row_count"] == row_count, f"{name}: meta.row_count disagrees with rows"


def _assert_decimal_agrees_with_bits(bits: str, decimal: float | str, *, where: str) -> None:
    """The decimal field is non-normative, but it must still agree with the bit pattern.

    This is the check that catches a hand-edit of either field (CLAUDE.md's
    "the decimal fields agree with the bit patterns wherever finite"
    requirement) -- comparison is on bits, never on `==`, so `-0.0` and
    `0.0` are not conflated with each other.
    """
    value = _bits_to_float32(bits)
    if isinstance(decimal, str):
        # Non-finite ground truth is rendered as its Python repr ("inf",
        # "-inf", or "nan") rather than a bare JSON token (D044 discussion
        # in fixtures/generate/corpus.py).
        if math.isnan(float(value)):
            assert decimal == "nan", f"{where}: bits are NaN but decimal is {decimal!r}"
        else:
            assert not math.isfinite(float(value)), f"{where}: decimal {decimal!r} claims non-finite but bits are finite"
            assert decimal == repr(float(value)), f"{where}: decimal {decimal!r} does not match bits-derived {repr(float(value))!r}"
        return
    assert math.isfinite(float(value)), f"{where}: decimal {decimal!r} is finite but bits {bits} are not"
    bits_from_decimal = _bits_of(decimal)
    assert bits_from_decimal == bits, (
        f"{where}: decimal {decimal!r} narrows to bit pattern {bits_from_decimal}, "
        f"disagreeing with the recorded {bits}"
    )


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_decimal_fields_agree_with_bit_patterns_where_finite(name: str) -> None:
    fixture = CORPUS[name]
    for index, (bits, decimal) in enumerate(zip(fixture["expected_margin"], fixture["margin_decimal"])):
        _assert_decimal_agrees_with_bits(bits, decimal, where=f"{name}.margin_decimal[{index}]")
    for index, (bits, decimal) in enumerate(zip(fixture["expected_output"], fixture["output_decimal"])):
        _assert_decimal_agrees_with_bits(bits, decimal, where=f"{name}.output_decimal[{index}]")


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_rewalk_reproduces_expected_margin_bit_for_bit(name: str) -> None:
    """`walk_margin` over the recorded artifact must match XGBoost's own margin, row for row.

    The oracle is `expected_margin`, which was recorded from
    `booster.predict(output_margin=True)` at generation time -- this test
    never recomputes it, and never trusts the generator's own self-check in
    its place.
    """
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
# Composition requirements: the specific properties CLAUDE.md and FORMAT.md
# require the corpus to exhibit, not merely a fixture count.
# ---------------------------------------------------------------------------


def test_gamma_pruned_fixture_has_genuinely_dead_nodes() -> None:
    fixture = CORPUS["gamma_pruned_dead_nodes"]
    num_deleted = fixture["meta"]["num_deleted_per_tree"]
    assert num_deleted, "gamma_pruned_dead_nodes: no tree_param.num_deleted recorded"
    assert sum(num_deleted) > 0, f"gamma_pruned_dead_nodes: num_deleted_per_tree={num_deleted}, expected > 0 somewhere"

    dead_indices = fixture["meta"]["dead_node_indices_per_tree"]
    for tree_index, (deleted_count, indices) in enumerate(zip(num_deleted, dead_indices)):
        assert len(indices) == deleted_count, (
            f"gamma_pruned_dead_nodes: tree {tree_index} has tree_param.num_deleted="
            f"{deleted_count} but {len(indices)} nodes unreachable from the root"
        )


def test_missing_value_fixture_exercises_both_default_left_directions() -> None:
    fixture = CORPUS["missing_value_both_directions"]
    meta = fixture["meta"]
    left_target = meta["default_left_1_target"]
    right_target = meta["default_left_0_target"]

    tree = fixture["artifact"]["trees"][left_target["tree_index"]]
    assert tree["default_left"][left_target["node_index"]] == 1

    tree = fixture["artifact"]["trees"][right_target["tree_index"]]
    assert tree["default_left"][right_target["node_index"]] == 0

    # Every constructed row carries at least one missing value, or the
    # fixture would not be exercising the missing-value path at all.
    assert any(value is None for value in fixture["rows"][0])
    assert any(value is None for value in fixture["rows"][1])


def test_survival_cox_overflow_fixture_contains_at_least_one_infinity() -> None:
    fixture = CORPUS["survival_cox_overflow_to_infinity"]
    positive_infinity_bits = _bits_of(float("inf"))
    inf_count = sum(1 for bits in fixture["expected_output"] if bits == positive_infinity_bits)
    assert inf_count > 0, "survival_cox_overflow_to_infinity: expected_output contains no +inf"
    assert inf_count < len(fixture["expected_output"]), (
        "survival_cox_overflow_to_infinity: every row is +inf; the boundary is not visible"
    )


def test_binary_logistic_signed_zero_fixture_has_the_required_intercept() -> None:
    fixture = CORPUS["binary_logistic_signed_zero"]
    intercept_bits = _bits_of(fixture["artifact"]["intercept"])
    assert intercept_bits == "0x80000000", (
        f"binary_logistic_signed_zero: intercept bit pattern is {intercept_bits}, "
        "expected 0x80000000 (negative zero)"
    )
    assert all(bits == "0x80000000" for bits in fixture["expected_margin"]), (
        "binary_logistic_signed_zero: not every row's expected_margin is negative zero"
    )


#: `f32(1e-6)` and `f32(1 - 1e-6)` -- the logistic clamp bounds (D035, D039).
#: Duplicated from `xgboost_bridge.objectives` deliberately: this test is
#: checking that the *fixture composition* honors the clamp boundary, not
#: re-deriving the clamp itself, so it does not import the constant it is
#: testing against.
_LOGISTIC_CLAMP_LOW = float(np.float32(1e-6))
_LOGISTIC_CLAMP_HIGH = float(np.float32(1.0) - np.float32(1e-6))

#: name -> (must be below 0.5, must be at least this far from 0.5). The
#: corpus-composition trap this guards against (CLAUDE.md): at
#: base_score=0.5 every broken intercept-placement variant scores 5000/5000,
#: so "some base_score was recorded" is not enough -- it must be far enough
#: from 0.5 that placement could actually be observed to matter.
_DIRECTIONAL_BASE_SCORE_FIXTURES: dict[str, tuple[bool, float]] = {
    "reg_squarederror_base_score_low": (True, 5.0),
    "reg_squarederror_base_score_high": (False, 5.0),
    "binary_logistic_base_score_low_inside_clamp": (True, 0.1),
    "binary_logistic_base_score_high_inside_clamp": (False, 0.1),
    "survival_cox_base_score_low": (True, 0.1),
    "survival_cox_base_score_high": (False, 0.1),
}


def test_base_score_values_are_recorded_far_from_one_half_in_both_directions() -> None:
    """The corpus-composition trap: a base_score of 0.5 tests nothing (CLAUDE.md)."""
    for name, (below, min_distance) in _DIRECTIONAL_BASE_SCORE_FIXTURES.items():
        base_score = CORPUS[name]["meta"]["base_score"]
        assert base_score is not None, f"{name}: meta.base_score is None"
        distance = abs(base_score - 0.5)
        assert distance >= min_distance, (
            f"{name}: base_score={base_score} is only {distance} from 0.5, "
            f"expected at least {min_distance}"
        )
        if below:
            assert base_score < 0.5, f"{name}: base_score={base_score} is not below 0.5"
        else:
            assert base_score > 0.5, f"{name}: base_score={base_score} is not above 0.5"


def test_logistic_clamp_fixtures_sit_on_the_correct_side_of_the_clamp_boundary() -> None:
    """`FORMAT.md` section 6.1: the corpus needs at least one base_score inside and
    one outside `[f32(1e-6), f32(1-1e-6)]` on each side, so the clamp itself is exercised."""
    inside = (
        "binary_logistic_base_score_low_inside_clamp",
        "binary_logistic_base_score_high_inside_clamp",
    )
    for name in inside:
        base_score = CORPUS[name]["meta"]["base_score"]
        assert _LOGISTIC_CLAMP_LOW < base_score < _LOGISTIC_CLAMP_HIGH, (
            f"{name}: base_score={base_score} is not inside the logistic clamp domain"
        )

    below = CORPUS["binary_logistic_base_score_below_clamp"]["meta"]["base_score"]
    assert below < _LOGISTIC_CLAMP_LOW, f"base_score={below} does not sit below the clamp floor"

    above = CORPUS["binary_logistic_base_score_above_clamp"]["meta"]["base_score"]
    assert above > _LOGISTIC_CLAMP_HIGH, f"base_score={above} does not sit above the clamp ceiling"


def test_single_feature_and_single_row_fixtures_have_the_minimal_shape() -> None:
    single_feature = CORPUS["single_feature_model"]
    assert len(single_feature["artifact"]["feature_names"]) == 1

    single_row = CORPUS["single_row_model"]
    assert len(single_row["rows"]) == 1


def test_meta_records_ground_truth_provenance() -> None:
    """Every fixture must record enough to know what produced its ground truth."""
    required_meta_keys = {
        "xgboost_version",
        "numpy_version",
        "objective",
        "base_score",
        "seed",
        "row_count",
        "description",
    }
    for name, fixture in CORPUS.items():
        missing = required_meta_keys - set(fixture["meta"])
        assert not missing, f"{name}: meta is missing {sorted(missing)}"
        assert fixture["meta"]["xgboost_version"] != "latest"
        assert isinstance(fixture["meta"]["description"], str) and fixture["meta"]["description"]


# ---------------------------------------------------------------------------
# Fixture provenance must name the exporter that actually built them
#
# Nothing checked this. The whole suite passed with every fixture stamped
# `0.1.0.dev0` while the installed package reported `1.0.0rc1`, which is the
# stale-provenance case D052 recorded as a risk of the two unlinked version
# literals -- observed here rather than reasoned about. A fixture whose
# provenance names a different exporter than the one under test is either stale
# or was built by something else, and both are worth a failure: the corpus is
# the record of what this exporter produces, so a mislabelled record is a record
# of nothing.
#
# This is also what makes the version bump auditable. It fails until the corpus
# is regenerated, so a bump cannot land with 23 fixtures still claiming the old
# version.
# ---------------------------------------------------------------------------


def test_every_fixture_names_the_current_exporter_version() -> None:
    from xgboost_bridge import __version__

    stale = {
        name: fixture["artifact"]["provenance"]["exporter_version"]
        for name, fixture in CORPUS.items()
        if fixture["artifact"]["provenance"]["exporter_version"] != __version__
    }
    assert not stale, (
        f"these fixtures were built by a different exporter than the installed "
        f"{__version__}; regenerate the corpus: {stale}"
    )


def test_the_exporter_version_is_a_single_value_across_the_whole_corpus() -> None:
    """A split corpus means a partial regeneration, which is how half the
    fixtures end up describing one exporter and half another."""
    versions = {
        fixture["artifact"]["provenance"]["exporter_version"]
        for fixture in CORPUS.values()
    }
    assert len(versions) == 1, f"corpus spans multiple exporter versions: {sorted(versions)}"
