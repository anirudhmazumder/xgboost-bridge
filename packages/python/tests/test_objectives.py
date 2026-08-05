"""Tests for per-objective intercept derivation.

Every numerical assertion here has **XGBoost's own observed margin** as its
oracle. Nothing is compared against a second derivation of the recipe under
test: a check whose oracle shares the defect it looks for cannot fire, and
that exact blindness let a real ``base_score`` clamping defect through an
earlier version of this project (D034).

The instrument is a zero-boosting-round fit with ``base_score`` passed
**explicitly**. With no trees the margin *is* the intercept, so no leaf value
enters the chain. Explicit is mandatory: left at its default,
``boost_from_average`` stays ``"1"`` and XGBoost emits the raw
``base_score``, which would make every measurement here meaningless (D036).
Each fit asserts the configuration it got back -- zero trees,
``boost_from_average == "0"``, and a ``base_score`` that round-tripped to the
same float32 -- so a silently mis-configured instrument fails rather than
reports.

Comparison is on float32 **bit patterns** throughout. ``-0.0 == 0.0`` is
``True`` in Python and the two are different artifacts; a NaN margin also
compares unequal to itself, so a value comparison would silently skip exactly
the rows that should fail.

Sweep values deliberately target the inputs where candidate implementations
diverge, not a comfortable middle. A sample that does not target the
disagreeing inputs cannot distinguish two implementations, and its silence is
not evidence of equivalence (D040).
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest
import xgboost as xgb

from xgboost_bridge import objectives, validate
from xgboost_bridge.errors import (
    InterceptMismatchError,
    MalformedTreeError,
    UnsupportedObjectiveError,
)

# --------------------------------------------------------------------------
# Instrument
# --------------------------------------------------------------------------

#: Labels legal for each objective. A zero-round model has no trees, so these
#: cannot influence any margin -- but each objective validates its labels at
#: configure time. ``survival:cox`` uses the sign convention: positive is an
#: event, negative is right-censored.
LABELS: dict[str, list[float]] = {
    "reg:squarederror": [0.5, 1.5, -0.5, 2.0],
    "binary:logistic": [0.0, 1.0, 1.0, 0.0],
    "survival:cox": [1.0, -2.0, 3.0, -4.0],
}

FEATURE_NAMES = ["c0", "c1"]


def _features() -> np.ndarray:
    return np.asarray(
        [[0.25, -1.5], [1.75, 0.5], [-0.75, 2.25], [0.0, -0.125]], dtype=np.float32
    )


def _matrix(objective: str) -> xgb.DMatrix:
    return xgb.DMatrix(
        _features(),
        label=np.asarray(LABELS[objective], dtype=np.float32),
        feature_names=FEATURE_NAMES,
        nthread=1,
    )


def fit(
    objective: str, base_score: float | None = None, rounds: int = 0
) -> tuple[xgb.Booster, dict[str, Any]]:
    """Fit a model and return it with its parsed serialized form."""
    params: dict[str, Any] = {
        "objective": objective,
        "nthread": 1,
        "seed": 20260804,
        "max_depth": 2,
        "tree_method": "exact",
    }
    if base_score is not None:
        params["base_score"] = float(base_score)
    booster = xgb.train(params, _matrix(objective), num_boost_round=rounds)
    return booster, json.loads(booster.save_raw(raw_format="json"))


def observed_margin(booster: xgb.Booster, objective: str) -> np.float32:
    """XGBoost's own margin, as one float32, checked for constancy on bits."""
    margin = np.asarray(
        booster.predict(_matrix(objective), output_margin=True), dtype=np.float32
    )
    assert margin.ndim == 1 and margin.size == len(LABELS[objective])
    patterns = set(margin.view(np.uint32).tolist())
    assert len(patterns) == 1, f"zero-tree margin is not constant: {sorted(patterns)}"
    return np.float32(margin[0])


def zero_tree_oracle(
    objective: str, base_score: float
) -> tuple[xgb.Booster, dict[str, Any], np.float32]:
    """Fit the zero-round instrument and assert the configuration it returned."""
    booster, document = fit(objective, base_score=base_score, rounds=0)
    model_param = document["learner"]["learner_model_param"]
    assert document["learner"]["gradient_booster"]["model"]["trees"] == []
    assert model_param["boost_from_average"] == "0", model_param
    stored = np.float32(float(model_param["base_score"].strip("[]")))
    assert bits32(stored) == bits32(np.float32(base_score)), (
        f"base_score did not round-trip: passed {base_score!r}, "
        f"stored {model_param['base_score']!r}"
    )
    return booster, document, observed_margin(booster, objective)


def bits32(value: Any) -> int:
    return int(np.float32(value).view(np.uint32))


# --------------------------------------------------------------------------
# Sweeps. Each spans the region where a candidate recipe would diverge.
# --------------------------------------------------------------------------

# Both sides of the clamp, the pinned transition pairs themselves (D039),
# the values where the textbook logit is wrong (0.48, 0.987654, 0.99), and
# 0.5 where the intercept is negative zero.
LOGISTIC_SWEEP: tuple[float, ...] = (
    0.0,
    1.401298464324817e-45,
    1e-38,
    1e-30,
    1e-20,
    1e-12,
    1e-8,
    1e-7,
    9.9e-7,
    9.99999883788405e-07,
    9.999999974752427e-07,
    1.0000001111620804e-06,
    1.0000002248489182e-06,
    1.1e-6,
    1e-5,
    1e-3,
    0.01,
    0.05,
    0.1,
    0.2,
    0.25,
    0.3,
    0.3333333,
    0.4,
    0.45,
    0.48,
    0.49,
    0.499,
    0.5,
    0.501,
    0.51,
    0.52,
    0.55,
    0.6,
    0.6666667,
    0.7,
    0.75,
    0.8,
    0.9,
    0.95,
    0.987654,
    0.99,
    0.999,
    0.9999,
    0.99999,
    0.9999988675117493,
    0.999998927116394,
    0.999999,
    0.9999999,
    1.0,
)

# Several orders of magnitude, from the smallest positive float32 subnormal
# to the largest finite float32, plus the [0.5, 1.1] window where the two
# logarithm routes diverge.
COX_SWEEP: tuple[float, ...] = (
    1.401298464324817e-45,
    1e-44,
    1e-42,
    1e-40,
    1e-38,
    1e-30,
    1e-20,
    1e-12,
    1e-8,
    1e-7,
    9.9e-7,
    1e-6,
    1.1e-6,
    1e-5,
    1e-4,
    1e-3,
    0.01,
    0.1,
    0.25,
    0.5,
    0.7,
    0.8,
    0.9,
    0.99,
    0.9975585341453552,
    0.999,
    0.9999998807907104,
    1.0,
    1.0000007152557373,
    1.00210702419281,
    1.0038461685180664,
    1.01,
    1.1,
    1.5,
    2.0,
    3.1415927,
    7.5,
    100.0,
    1000.0,
    1e6,
    1e12,
    1e20,
    1e30,
    1e38,
    3.4028234663852886e38,
)

# Negative, zero, subnormal and huge: the identity link constrains nothing,
# so the sweep must not be confined to (0, 1).
REGRESSION_SWEEP: tuple[float, ...] = (
    -3.4028234663852886e38,
    -1e38,
    -1e30,
    -1e6,
    -1234.5678,
    -100.0,
    -7.5,
    -1.0,
    -0.5,
    -0.3,
    -1e-6,
    -1e-38,
    -1.401298464324817e-45,
    -0.0,
    0.0,
    1.401298464324817e-45,
    1e-38,
    1e-20,
    1e-7,
    1e-6,
    1e-3,
    0.1,
    0.25,
    0.3,
    0.48,
    0.5,
    0.75,
    0.987654,
    1.0,
    1.0000001192092896,
    1.5,
    2.0,
    7.5,
    39.882263,
    100.0,
    1234.5678,
    1e6,
    1e12,
    1e20,
    1e30,
    1e38,
    3.4028234663852886e38,
)

# float32 `base_score` values where a float64 logarithm narrowed to float32
# and a float32 logarithm land on different bit patterns. Found by scanning
# consecutive float32 values in the region where |log| is small -- 5765 of
# 2516584 for Cox in [0.9, 1.1], 3621 of 2516584 for logistic in
# [0.45, 0.55]. Ordinary values cannot distinguish the two routes.
COX_LOG_ROUTE_DISAGREEMENTS: tuple[float, ...] = (
    0.9975585341453552,
    0.9981271028518677,
    0.9994231462478638,
    0.991168200969696,
    0.9999998807907104,
    1.0000007152557373,
    1.00210702419281,
    1.0103390216827393,
)

LOGISTIC_LOG_ROUTE_DISAGREEMENTS: tuple[float, ...] = (
    0.45000141859054565,
    0.45004647970199585,
    0.4500989615917206,
    0.4502125084400177,
    0.451135516166687,
    0.4882951080799103,
    0.49932295083999634,
    0.49952593445777893,
)


# --------------------------------------------------------------------------
# The interface export.py is written against
# --------------------------------------------------------------------------


def test_supported_objectives_are_the_three_in_scope() -> None:
    assert objectives.SUPPORTED_OBJECTIVES == (
        "reg:squarederror",
        "binary:logistic",
        "survival:cox",
    )


def test_output_transform_pairing_is_the_measured_one() -> None:
    """The pairing is measured, not assumed: 107500/107500 bit-exact rows
    across 43 fitted models, eleven candidate transforms scored per
    objective (``probes/output_transform.md``)."""
    assert dict(objectives.OUTPUT_TRANSFORMS) == {
        "reg:squarederror": "identity",
        "binary:logistic": "sigmoid",
        "survival:cox": "exp",
    }


def test_output_transforms_covers_exactly_the_supported_objectives() -> None:
    assert tuple(objectives.OUTPUT_TRANSFORMS) == objectives.SUPPORTED_OBJECTIVES


def test_output_transforms_cannot_be_mutated_by_a_caller() -> None:
    with pytest.raises(TypeError):
        objectives.OUTPUT_TRANSFORMS["reg:squarederror"] = "sigmoid"  # type: ignore[index]


def test_supported_objectives_agree_with_the_export_gate() -> None:
    """Two modules name the same three objectives. If either list changes
    without the other, this fails rather than letting an objective be
    exportable with an unverified intercept space, or vice versa."""
    assert objectives.SUPPORTED_OBJECTIVES == validate.SUPPORTED_OBJECTIVES


# --------------------------------------------------------------------------
# Bit-exactness against XGBoost's own zero-tree margin
# --------------------------------------------------------------------------


def _sweep_report(objective: str, values: tuple[float, ...]) -> tuple[int, list[str]]:
    hits = 0
    failures: list[str] = []
    for base_score in values:
        _booster, document, observed = zero_tree_oracle(objective, base_score)
        derived = objectives.derive_intercept(document)
        if bits32(derived) == bits32(observed):
            hits += 1
        else:
            failures.append(
                f"base_score={base_score!r}: derived {derived!r} "
                f"bits=0x{bits32(derived):08X} vs XGBoost {float(observed)!r} "
                f"bits=0x{bits32(observed):08X}"
            )
    return hits, failures


def test_logistic_intercept_is_bit_exact_against_xgboost() -> None:
    total = len(LOGISTIC_SWEEP)
    assert total >= 40
    hits, failures = _sweep_report("binary:logistic", LOGISTIC_SWEEP)
    assert hits == total, f"binary:logistic {hits}/{total}\n" + "\n".join(failures)


def test_cox_intercept_is_bit_exact_against_xgboost() -> None:
    total = len(COX_SWEEP)
    assert total >= 40
    hits, failures = _sweep_report("survival:cox", COX_SWEEP)
    assert hits == total, f"survival:cox {hits}/{total}\n" + "\n".join(failures)


def test_regression_intercept_is_bit_exact_against_xgboost() -> None:
    total = len(REGRESSION_SWEEP)
    assert total >= 40
    hits, failures = _sweep_report("reg:squarederror", REGRESSION_SWEEP)
    assert hits == total, f"reg:squarederror {hits}/{total}\n" + "\n".join(failures)


def test_intercept_is_exactly_representable_as_float32() -> None:
    """FORMAT.md section 6 requires it, and section 9.1's emission rule
    depends on it: a float64 that is not a widened float32 does not recover
    that float32 through a shortest round-trip repr."""
    for objective, values in (
        ("binary:logistic", LOGISTIC_SWEEP),
        ("survival:cox", COX_SWEEP),
        ("reg:squarederror", REGRESSION_SWEEP),
    ):
        for base_score in values[:8]:
            _booster, document, _observed = zero_tree_oracle(objective, base_score)
            derived = objectives.derive_intercept(document)
            widened = float(np.float32(derived))
            assert np.float64(widened).view(np.uint64) == np.float64(derived).view(
                np.uint64
            ), f"{objective} at {base_score!r} derived a non-float32 value {derived!r}"


# --------------------------------------------------------------------------
# The clamp is load-bearing
# --------------------------------------------------------------------------


def _unclamped_logistic(base_score: float) -> np.float32:
    """The same recipe with the clamp removed, and nothing else changed."""
    probability = np.float32(base_score)
    reciprocal = np.float32(np.float32(1.0) / probability)
    odds = np.float32(reciprocal - np.float32(1.0))
    return np.float32(-np.log(odds))


@pytest.mark.parametrize(
    ("base_score", "expected_unclamped"),
    [
        (1e-12, -27.63102149963379),
        (1e-7, -16.11809539794922),
        (0.9999999, 15.942384719848633),
    ],
)
def test_logistic_clamp_is_load_bearing(
    base_score: float, expected_unclamped: float
) -> None:
    """Skipping the clamp gives a plausible wrong intercept, by up to 13.8
    in margin space. XGBoost clamps before deriving and stores the value
    unclamped, so the clamp cannot be recovered from the artifact (D035)."""
    _booster, document, observed = zero_tree_oracle("binary:logistic", base_score)
    derived = objectives.derive_intercept(document)
    unclamped = _unclamped_logistic(base_score)

    assert bits32(derived) == bits32(observed)
    assert bits32(unclamped) == bits32(expected_unclamped), (
        f"the unclamped recipe moved: {float(unclamped)!r}"
    )
    assert bits32(unclamped) != bits32(observed), (
        "the unclamped recipe agrees with XGBoost here, so this value does "
        "not pin the clamp"
    )
    error = abs(float(unclamped) - float(observed))
    assert error > 2.0, f"margin-space error only {error!r}"


def test_logistic_clamp_saturates_to_the_pinned_bounds() -> None:
    """The saturated intercepts are pinned float32 values, and every
    ``base_score`` past the transition pair reaches exactly one of them
    (D039)."""
    saturated_low = -13.815509796142578
    saturated_high = 13.745160102844238

    for base_score in (0.0, 1.401298464324817e-45, 1e-38, 1e-12, 1e-7):
        _booster, document, observed = zero_tree_oracle("binary:logistic", base_score)
        assert bits32(objectives.derive_intercept(document)) == bits32(saturated_low)
        assert bits32(observed) == bits32(saturated_low)

    for base_score in (0.999998927116394, 0.9999999, 1.0):
        _booster, document, observed = zero_tree_oracle("binary:logistic", base_score)
        assert bits32(objectives.derive_intercept(document)) == bits32(saturated_high)
        assert bits32(observed) == bits32(saturated_high)


def test_logistic_clamp_keeps_the_logarithm_in_its_domain_at_one() -> None:
    """``base_score = 1.0`` stores as ``[1E0]`` and XGBoost accepts it. The
    unclamped argument there is exactly ``0.0``, where a float64 logarithm
    raises ``math domain error`` outright rather than producing a wrong
    number -- so this input separates the clamp from the transform."""
    _booster, document, observed = zero_tree_oracle("binary:logistic", 1.0)
    assert document["learner"]["learner_model_param"]["base_score"] == "[1E0]"

    unclamped_argument = np.float32(np.float32(np.float32(1.0) / np.float32(1.0)) - 1.0)
    assert bits32(unclamped_argument) == bits32(0.0)
    with pytest.raises(ValueError, match="math domain error"):
        math.log(float(unclamped_argument))

    assert bits32(objectives.derive_intercept(document)) == bits32(observed)
    assert bits32(observed) == bits32(13.745160102844238)


def _clamped_to_logistic_bounds(base_score: float) -> np.float32:
    """The logistic clamp applied where it does not belong."""
    probability = np.float32(base_score)
    low = np.float32(1e-6)
    high = np.float32(1.0) - np.float32(1e-6)
    if probability < low:
        return low
    if probability > high:
        return high
    return probability


@pytest.mark.parametrize("base_score", [1e-38, 1e-12, 1.0, 1e6, 1e30])
def test_cox_is_not_clamped(base_score: float) -> None:
    """Reusing the logistic bounds here would be inference by analogy, and it
    is wrong on 27 of 34 measured Cox values. Every value below is outside
    the logistic window, so the two hypotheses must disagree."""
    _booster, document, observed = zero_tree_oracle("survival:cox", base_score)
    derived = objectives.derive_intercept(document)
    wrongly_clamped = np.float32(np.log(_clamped_to_logistic_bounds(base_score)))

    assert bits32(derived) == bits32(observed)
    assert bits32(wrongly_clamped) != bits32(observed)


@pytest.mark.parametrize("base_score", [-1e38, -1.0, 1e-38, 1.5, 1e30])
def test_regression_is_not_clamped(base_score: float) -> None:
    """Wrong on 22 of 25 measured regression values. Negative and huge
    ``base_score`` values are accepted and stored verbatim, which is what
    confirms the space is not constrained to ``(0, 1)``."""
    _booster, document, observed = zero_tree_oracle("reg:squarederror", base_score)
    derived = objectives.derive_intercept(document)
    wrongly_clamped = _clamped_to_logistic_bounds(base_score)

    assert bits32(derived) == bits32(observed)
    assert bits32(derived) == bits32(np.float32(base_score))
    assert bits32(wrongly_clamped) != bits32(observed)


# --------------------------------------------------------------------------
# The logarithm is a float32 logarithm
# --------------------------------------------------------------------------


def test_cox_float32_log_route_matches_xgboost_where_float64_does_not() -> None:
    """On the inputs where the two routes disagree, only the float32 route
    matches XGBoost. An ordinary sweep cannot see this -- the routes agree
    on 99.945% of float32 inputs (D040)."""
    total = len(COX_LOG_ROUTE_DISAGREEMENTS)
    float32_route_hits = 0
    float64_route_hits = 0
    rows: list[str] = []

    for base_score in COX_LOG_ROUTE_DISAGREEMENTS:
        snapped = np.float32(base_score)
        route_float32 = np.float32(np.log(snapped))
        route_float64 = np.float32(math.log(float(snapped)))
        assert bits32(route_float32) != bits32(route_float64), (
            f"{base_score!r} no longer discriminates the two log routes"
        )

        _booster, document, observed = zero_tree_oracle("survival:cox", base_score)
        derived = objectives.derive_intercept(document)
        assert bits32(derived) == bits32(route_float32), (
            "derive_intercept is not taking the float32 route"
        )
        float32_route_hits += bits32(route_float32) == bits32(observed)
        float64_route_hits += bits32(route_float64) == bits32(observed)
        rows.append(
            f"{base_score!r}: xgb=0x{bits32(observed):08X} "
            f"f32=0x{bits32(route_float32):08X} f64=0x{bits32(route_float64):08X}"
        )

    assert float32_route_hits == total, "\n".join(rows)
    assert float64_route_hits == 0, "\n".join(rows)


def test_logistic_float32_log_route_matches_xgboost_where_float64_does_not() -> None:
    """Same discrimination for the logistic transform. The disagreeing
    inputs cluster near ``base_score = 0.5``, where the intercept's
    magnitude is small; a sweep of the clamp window that misses that
    neighbourhood reports "no difference" and proves nothing."""
    total = len(LOGISTIC_LOG_ROUTE_DISAGREEMENTS)
    float32_route_hits = 0
    float64_route_hits = 0
    rows: list[str] = []

    for base_score in LOGISTIC_LOG_ROUTE_DISAGREEMENTS:
        snapped = np.float32(base_score)
        odds = np.float32(np.float32(np.float32(1.0) / snapped) - np.float32(1.0))
        route_float32 = np.float32(-np.log(odds))
        route_float64 = np.float32(-math.log(float(odds)))
        assert bits32(route_float32) != bits32(route_float64), (
            f"{base_score!r} no longer discriminates the two log routes"
        )

        _booster, document, observed = zero_tree_oracle("binary:logistic", base_score)
        derived = objectives.derive_intercept(document)
        assert bits32(derived) == bits32(route_float32), (
            "derive_intercept is not taking the float32 route"
        )
        float32_route_hits += bits32(route_float32) == bits32(observed)
        float64_route_hits += bits32(route_float64) == bits32(observed)
        rows.append(
            f"{base_score!r}: xgb=0x{bits32(observed):08X} "
            f"f32=0x{bits32(route_float32):08X} f64=0x{bits32(route_float64):08X}"
        )

    assert float32_route_hits == total, "\n".join(rows)
    assert float64_route_hits == 0, "\n".join(rows)


def test_textbook_logit_is_not_the_logistic_transform() -> None:
    """``log(p / (1 - p))`` is the formula a reasonable person would write.

    Measured here: bit-wrong on 9 of these 10 values, right on ``0.7``, and
    breaching the ``1e-6`` margin gate at ``0.99`` and ``0.987654``. Right on
    some values and wrong on others with no error raised is this project's
    exact failure signature, which is why the value it gets right is named
    rather than left to a count.
    """
    sweep = (0.45, 0.48, 0.49, 0.6, 0.7, 0.75, 0.9, 0.95, 0.99, 0.987654)
    right: list[float] = []
    worst = 0.0
    for base_score in sweep:
        _booster, document, observed = zero_tree_oracle("binary:logistic", base_score)
        derived = objectives.derive_intercept(document)
        assert bits32(derived) == bits32(observed)

        probability = np.float32(base_score)
        textbook = np.float32(
            math.log(float(probability) / (1.0 - float(probability)))
        )
        if bits32(textbook) == bits32(observed):
            right.append(base_score)
        else:
            worst = max(worst, abs(float(textbook) - float(observed)))

    assert right == [0.7], f"the textbook formula was right on {right}"
    assert worst > 1e-6, f"worst textbook error {worst!r} does not breach the gate"


def test_base_score_is_snapped_to_float32_before_the_transform() -> None:
    """``float("[4.8E-1]"[1:-1])`` is ``0.48``; the value XGBoost holds is
    ``0.47999998927116394``. Deriving from the unsnapped float64 is 1 ULP
    wrong for Cox and 20 ULP wrong for logistic."""
    _booster, document, observed = zero_tree_oracle("survival:cox", 0.7)
    assert document["learner"]["learner_model_param"]["base_score"] == "[7E-1]"
    derived = objectives.derive_intercept(document)
    unsnapped = np.float32(np.log(np.float64(0.7)))

    assert bits32(derived) == bits32(observed)
    assert bits32(derived) == bits32(-0.3566749691963196)
    assert bits32(unsnapped) == bits32(-0.3566749393939972)
    assert bits32(unsnapped) != bits32(observed)


def test_logistic_derives_from_the_snapped_value_too() -> None:
    """``base_score = 0.48`` is where the formulations spread out furthest.

    Deriving from the unsnapped ``0.48`` rather than from the float32 XGBoost
    holds, ``0.47999998927116394``, lands 29 ULP away -- an absolute error of
    ``2.1606683731079102e-07``, which no ``1e-6`` gate would catch and which
    would make cross-language parity nonzero instead.
    """
    _booster, document, observed = zero_tree_oracle("binary:logistic", 0.48)
    assert document["learner"]["learner_model_param"]["base_score"] == "[4.8E-1]"
    derived = objectives.derive_intercept(document)

    unsnapped_probability = np.float64(0.48)
    unsnapped_odds = np.float32(
        np.float32(np.float64(1.0) / unsnapped_probability) - np.float32(1.0)
    )
    unsnapped = np.float32(-np.log(unsnapped_odds))

    assert bits32(derived) == bits32(observed)
    assert bits32(derived) == bits32(-0.08004285395145416)
    assert bits32(unsnapped) != bits32(observed)
    assert bits32(unsnapped) - bits32(observed) == -29
    assert abs(float(unsnapped) - float(observed)) == 2.1606683731079102e-07


# --------------------------------------------------------------------------
# boost_from_average selects the space
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("objective", "raw", "linked"),
    [
        ("binary:logistic", 0.5, -0.0),
        ("survival:cox", 0.5, -0.6931471824645996),
        ("reg:squarederror", 0.5, 0.5),
    ],
)
def test_boost_from_average_selects_the_intercept_space(
    objective: str, raw: float, linked: float
) -> None:
    """Zero trees plus ``boost_from_average == "1"`` means the raw
    ``base_score`` is the margin. The same model with ``base_score`` passed
    explicitly flips the field to ``"0"`` and the link applies. Both sides
    are asserted against XGBoost's own margin (D036)."""
    default_booster, default_document = fit(objective, base_score=None, rounds=0)
    default_param = default_document["learner"]["learner_model_param"]
    assert default_param["boost_from_average"] == "1"
    assert default_param["base_score"] == "[5E-1]"
    assert default_document["learner"]["gradient_booster"]["model"]["trees"] == []

    default_observed = observed_margin(default_booster, objective)
    default_derived = objectives.derive_intercept(default_document)
    assert bits32(default_observed) == bits32(raw)
    assert bits32(default_derived) == bits32(raw)

    _explicit_booster, explicit_document, explicit_observed = zero_tree_oracle(
        objective, 0.5
    )
    explicit_derived = objectives.derive_intercept(explicit_document)
    assert bits32(explicit_observed) == bits32(linked)
    assert bits32(explicit_derived) == bits32(linked)

    if objective != "reg:squarederror":
        assert bits32(default_derived) != bits32(explicit_derived), (
            "the two spaces coincide here, so this objective does not pin "
            "boost_from_average"
        )


def test_raw_space_is_not_clamped_either() -> None:
    """In the raw cell the stored value is emitted verbatim, so the logistic
    clamp must not run there: ``base_score = 1.0`` gives margin ``1.0``, not
    the saturated ``13.745160102844238``."""
    booster, document = fit("binary:logistic", base_score=None, rounds=0)
    document["learner"]["learner_model_param"]["base_score"] = "[1E0]"
    assert document["learner"]["learner_model_param"]["boost_from_average"] == "1"
    assert bits32(objectives.derive_intercept(document)) == bits32(1.0)
    assert bits32(observed_margin(booster, "binary:logistic")) == bits32(0.5)


@pytest.mark.parametrize(
    "objective", ["reg:squarederror", "binary:logistic", "survival:cox"]
)
def test_a_model_with_trees_always_applies_the_transform(objective: str) -> None:
    """``boost_from_average`` is ``"1"`` on an ordinary fitted model too. The
    raw cell is zero trees **and** ``"1"``, not ``"1"`` alone -- keying on
    the string alone puts every trained model in the wrong space."""
    booster, document = fit(objective, base_score=None, rounds=4)
    model_param = document["learner"]["learner_model_param"]
    assert model_param["boost_from_average"] == "1"
    assert len(document["learner"]["gradient_booster"]["model"]["trees"]) == 4

    stored = np.float32(float(model_param["base_score"].strip("[]")))
    derived = objectives.derive_intercept(document)
    _oracle_booster, _oracle_document, observed = zero_tree_oracle(
        objective, float(stored)
    )
    assert bits32(derived) == bits32(observed)
    objectives.verify_intercept(booster, derived)


# --------------------------------------------------------------------------
# Signed zero
# --------------------------------------------------------------------------


def test_logistic_at_one_half_derives_negative_zero() -> None:
    """``-log(f32(1/0.5 - 1)) = -log(1) = -0.0``, bit pattern
    ``0x80000000``. Reachable through an ordinary default, and never
    normalized. ``base_score`` must be passed explicitly or the model is in
    the raw space and this tests nothing (FORMAT.md section 6.3)."""
    _booster, document, observed = zero_tree_oracle("binary:logistic", 0.5)
    derived = objectives.derive_intercept(document)

    assert bits32(derived) == 0x80000000
    assert bits32(observed) == 0x80000000
    assert math.copysign(1.0, derived) == -1.0
    # The comparison that cannot see the difference, stated so the bit
    # comparison above is not mistaken for redundancy.
    assert derived == 0.0
    assert bits32(0.0) == 0x00000000


def test_cox_at_one_derives_positive_zero() -> None:
    """The two objectives differ in the sign of zero: ``ln(1.0)`` is ``+0.0``
    where the logistic form gives ``-0.0``."""
    _booster, document, observed = zero_tree_oracle("survival:cox", 1.0)
    derived = objectives.derive_intercept(document)
    assert bits32(derived) == 0x00000000
    assert bits32(observed) == 0x00000000


def test_regression_preserves_a_stored_negative_zero() -> None:
    _booster, document, observed = zero_tree_oracle("reg:squarederror", -0.0)
    assert document["learner"]["learner_model_param"]["base_score"] == "[-0E0]"
    assert bits32(objectives.derive_intercept(document)) == 0x80000000
    assert bits32(observed) == 0x80000000


# --------------------------------------------------------------------------
# Non-finite intercepts XGBoost itself produces
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("base_score", "expected_bits"),
    [(0.0, 0xFF800000), (-0.5, 0x7FC00000), (-1e-06, 0x7FC00000)],
)
def test_cox_reproduces_xgboost_non_finite_intercepts(
    base_score: float, expected_bits: int
) -> None:
    """``survival:cox`` accepts ``base_score <= 0`` with no error and no
    warning, giving ``-inf`` at zero and ``NaN`` below it. The derivation
    reproduces XGBoost rather than second-guessing it; whether such an
    intercept may be *emitted* is a format question (FORMAT.md sections 6,
    9.3) and is not decided here."""
    booster, document = fit("survival:cox", base_score=base_score, rounds=0)
    model_param = document["learner"]["learner_model_param"]
    assert model_param["boost_from_average"] == "0"

    margin = np.asarray(
        booster.predict(_matrix("survival:cox"), output_margin=True), dtype=np.float32
    )
    patterns = set(margin.view(np.uint32).tolist())
    assert patterns == {expected_bits}, f"XGBoost margin bits {sorted(patterns)}"

    derived = objectives.derive_intercept(document)
    assert bits32(derived) == expected_bits


def test_deriving_a_non_finite_intercept_emits_no_warning() -> None:
    """The value carries the information; a warning that depends on ambient
    numpy error state would make the derivation's behaviour caller-dependent
    instead of deterministic."""
    _booster, document = fit("survival:cox", base_score=0.0, rounds=0)
    with np.errstate(all="raise"):
        derived = objectives.derive_intercept(document)
    assert bits32(derived) == 0xFF800000


# --------------------------------------------------------------------------
# Parsing, and failing loudly
# --------------------------------------------------------------------------


def _document(
    objective: str = "binary:logistic",
    base_score: Any = "[5E-1]",
    boost_from_average: Any = "0",
    trees: Any = None,
) -> dict[str, Any]:
    """A minimal parsed model, shaped exactly like a serialized one."""
    return {
        "learner": {
            "objective": {"name": objective},
            "learner_model_param": {
                "base_score": base_score,
                "boost_from_average": boost_from_average,
                "num_class": "0",
                "num_feature": "2",
                "num_target": "1",
            },
            "feature_names": list(FEATURE_NAMES),
            "gradient_booster": {"model": {"trees": [] if trees is None else trees}},
        },
        "version": [3, 3, 0],
    }


def test_minimal_document_shape_agrees_with_a_real_one() -> None:
    """The hand-built document above is only useful if it has the shape the
    derivation reads out of a real artifact, so that is asserted rather than
    assumed."""
    _booster, real = zero_tree_oracle("binary:logistic", 0.5)[:2]
    hand_built = _document(base_score="[5E-1]", boost_from_average="0")
    assert bits32(objectives.derive_intercept(real)) == bits32(
        objectives.derive_intercept(hand_built)
    )
    real_param = real["learner"]["learner_model_param"]
    hand_param = hand_built["learner"]["learner_model_param"]
    assert set(hand_param) <= set(real_param)
    assert real_param["base_score"] == hand_param["base_score"]


def test_base_score_is_read_as_a_bracketed_string() -> None:
    assert bits32(objectives.derive_intercept(_document(base_score="[4.8E-1]"))) == (
        bits32(objectives.derive_intercept(_document(base_score="[4.7999999E-1]")))
    )


@pytest.mark.parametrize("stored", [0.5, ["[5E-1]"], None, 1, {"0": "5E-1"}])
def test_non_string_base_score_raises(stored: Any) -> None:
    with pytest.raises(MalformedTreeError) as caught:
        objectives.derive_intercept(_document(base_score=stored))
    assert caught.value.field == "base_score"
    assert caught.value.location == "learner.learner_model_param.base_score"


@pytest.mark.parametrize("stored", ["5E-1", "[5E-1", "5E-1]", "", "[", "[]"])
def test_base_score_without_a_single_bracketed_element_raises(stored: str) -> None:
    with pytest.raises(MalformedTreeError) as caught:
        objectives.derive_intercept(_document(base_score=stored))
    assert caught.value.field == "base_score"


def test_multi_element_base_score_raises() -> None:
    """A two-element ``base_score`` is what a multi-output model serializes.
    Its meaning is not established, so it raises rather than having its first
    element read."""
    with pytest.raises(MalformedTreeError) as caught:
        objectives.derive_intercept(_document(base_score="[5E-1,6E-1]"))
    assert "one element" in str(caught.value)


@pytest.mark.parametrize("stored", ["[abc]", "[5E-1 6E-1]", "[--1]", "[0x1]"])
def test_unparseable_base_score_raises(stored: str) -> None:
    with pytest.raises(MalformedTreeError):
        objectives.derive_intercept(_document(base_score=stored))


@pytest.mark.parametrize("stored", ["[nan]", "[inf]", "[-inf]", "[1E40]"])
def test_non_finite_base_score_raises(stored: str) -> None:
    """XGBoost's own parser refuses ``nan`` and ``inf`` in this field, so a
    non-finite value here is a hand-edited artifact, not a model. ``1E40``
    overflows float32 and is refused for the same reason."""
    with pytest.raises(MalformedTreeError) as caught:
        objectives.derive_intercept(_document(base_score=stored))
    assert "finite" in str(caught.value)


@pytest.mark.parametrize("stored", [1, 0, True, "", "2", "yes", None, "1 "])
def test_unrecognized_boost_from_average_raises(stored: Any) -> None:
    """The field is a JSON string. An integer comparison would silently
    never fire, disabling the space selection rather than tripping it."""
    with pytest.raises(MalformedTreeError) as caught:
        objectives.derive_intercept(_document(boost_from_average=stored))
    assert caught.value.field == "boost_from_average"


@pytest.mark.parametrize(
    "objective",
    ["reg:logistic", "count:poisson", "survival:aft", "multi:softprob", "", "cox"],
)
def test_unsupported_objective_raises(objective: str) -> None:
    with pytest.raises(UnsupportedObjectiveError) as caught:
        objectives.derive_intercept(_document(objective=objective))
    assert caught.value.objective == objective
    assert caught.value.supported == objectives.SUPPORTED_OBJECTIVES


@pytest.mark.parametrize(
    ("remove", "location"),
    [
        (("learner",), None),
        (("learner", "objective"), "learner"),
        (("learner", "learner_model_param"), "learner"),
        (("learner", "gradient_booster"), "learner"),
    ],
)
def test_absent_field_raises(remove: tuple[str, ...], location: str | None) -> None:
    document = _document()
    container: Any = document
    for key in remove[:-1]:
        container = container[key]
    del container[remove[-1]]

    with pytest.raises(MalformedTreeError) as caught:
        objectives.derive_intercept(document)
    assert caught.value.field == remove[-1]
    assert caught.value.location == location


@pytest.mark.parametrize("field", ["base_score", "boost_from_average"])
def test_absent_model_param_field_raises(field: str) -> None:
    document = _document()
    del document["learner"]["learner_model_param"][field]
    with pytest.raises(MalformedTreeError) as caught:
        objectives.derive_intercept(document)
    assert caught.value.field == field


def test_absent_trees_key_raises() -> None:
    """A zero-round model serializes ``"trees": []`` -- present and empty,
    on all three objectives. An absent key is a shape no probe measured."""
    document = _document()
    del document["learner"]["gradient_booster"]["model"]["trees"]
    with pytest.raises(MalformedTreeError) as caught:
        objectives.derive_intercept(document)
    assert caught.value.field == "trees"


def test_non_list_trees_raises() -> None:
    with pytest.raises(MalformedTreeError) as caught:
        objectives.derive_intercept(_document(trees={"0": {}}))
    assert caught.value.field == "trees"


# --------------------------------------------------------------------------
# verify_intercept: the independent oracle
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "objective", ["reg:squarederror", "binary:logistic", "survival:cox"]
)
@pytest.mark.parametrize("rounds", [0, 1, 6])
def test_verify_intercept_accepts_the_derived_intercept(
    objective: str, rounds: int
) -> None:
    booster, document = fit(objective, base_score=0.4, rounds=rounds)
    objectives.verify_intercept(booster, objectives.derive_intercept(document))


@pytest.mark.parametrize(
    "objective", ["reg:squarederror", "binary:logistic", "survival:cox"]
)
def test_verify_intercept_accepts_the_raw_space_of_a_zero_tree_default(
    objective: str,
) -> None:
    """The oracle for a zero-tree model is that model's own margin. Refitting
    with ``base_score`` passed explicitly would land in link space and
    reject a correct intercept."""
    booster, document = fit(objective, base_score=None, rounds=0)
    assert document["learner"]["learner_model_param"]["boost_from_average"] == "1"
    objectives.verify_intercept(booster, objectives.derive_intercept(document))


@pytest.mark.parametrize(
    ("objective", "link_space_value"),
    [("binary:logistic", -0.0), ("survival:cox", -0.6931471824645996)],
)
def test_verify_intercept_rejects_link_space_in_the_raw_cell(
    objective: str, link_space_value: float
) -> None:
    """The oracle for a zero-tree model must be that model's own margin. If
    it were a re-fit with ``base_score`` passed explicitly it would come back
    in link space and accept this wrong value."""
    booster, document = fit(objective, base_score=None, rounds=0)
    assert document["learner"]["learner_model_param"]["boost_from_average"] == "1"
    with pytest.raises(InterceptMismatchError) as caught:
        objectives.verify_intercept(booster, link_space_value)
    assert bits32(caught.value.observed) == bits32(0.5)


def test_verify_intercept_handles_a_model_with_no_feature_names() -> None:
    """A model fit from a bare array serializes ``feature_names`` as ``[]``.
    Export refuses such a model later (D021), but the oracle must still be
    able to read its margin rather than failing on the way to the refusal."""
    matrix = xgb.DMatrix(
        _features(), label=np.asarray(LABELS["survival:cox"], dtype=np.float32), nthread=1
    )
    booster = xgb.train(
        {"objective": "survival:cox", "nthread": 1, "base_score": 0.8},
        matrix,
        num_boost_round=0,
    )
    document = json.loads(booster.save_raw(raw_format="json"))
    assert document["learner"]["feature_names"] == []
    objectives.verify_intercept(booster, objectives.derive_intercept(document))


def test_verify_intercept_rejects_positive_zero_for_negative_zero() -> None:
    """The whole reason the comparison is on bit patterns: ``-0.0 == 0.0`` is
    ``True``, so an equality check passes here and the artifact ships with a
    sign flipped."""
    booster, document = fit("binary:logistic", base_score=0.5, rounds=0)
    derived = objectives.derive_intercept(document)
    assert bits32(derived) == 0x80000000

    objectives.verify_intercept(booster, derived)
    with pytest.raises(InterceptMismatchError) as caught:
        objectives.verify_intercept(booster, 0.0)
    assert caught.value.objective == "binary:logistic"
    assert bits32(caught.value.observed) == 0x80000000


def test_verify_intercept_rejects_a_one_ulp_error() -> None:
    booster, document = fit("survival:cox", base_score=0.7, rounds=3)
    derived = objectives.derive_intercept(document)
    objectives.verify_intercept(booster, derived)

    off_by_one_ulp = float(np.nextafter(np.float32(derived), np.float32(np.inf)))
    with pytest.raises(InterceptMismatchError) as caught:
        objectives.verify_intercept(booster, off_by_one_ulp)
    assert caught.value.derived == off_by_one_ulp


def test_verify_intercept_rejects_a_value_that_is_not_a_float32() -> None:
    """A float64 that no float32 recovers cannot be an XGBoost margin, and
    narrowing it silently would let the check pass on a value the exporter
    would never emit."""
    booster, document = fit("reg:squarederror", base_score=0.1, rounds=0)
    derived = objectives.derive_intercept(document)
    objectives.verify_intercept(booster, derived)

    with pytest.raises(InterceptMismatchError):
        objectives.verify_intercept(booster, 0.1)


def test_verify_intercept_fires_on_a_recipe_error() -> None:
    """The property that makes the oracle independent, tested directly:
    break the recipe and the check must still fire. The superseded version
    of this check re-derived the intercept from ``base_score``, so it could
    not fire on a recipe error -- and it passed the clamp defect of D035
    (D034)."""
    booster, document = fit("binary:logistic", base_score=0.3, rounds=2)
    honest = objectives.derive_intercept(document)
    objectives.verify_intercept(booster, honest)

    original = objectives._logistic_intercept
    try:
        objectives._logistic_intercept = lambda base_score: np.float32(  # type: ignore[assignment]
            float(base_score) + 1.0
        )
        broken = objectives.derive_intercept(document)
        assert bits32(broken) != bits32(honest)
        with pytest.raises(InterceptMismatchError):
            objectives.verify_intercept(booster, broken)
    finally:
        objectives._logistic_intercept = original  # type: ignore[assignment]

    assert bits32(objectives.derive_intercept(document)) == bits32(honest)


def test_verify_intercept_oracle_is_in_link_space_for_a_model_with_trees() -> None:
    """The oracle fit must pass ``base_score`` explicitly. If it did not,
    ``boost_from_average`` would stay ``"1"`` on the zero-round oracle and
    the raw value would be returned -- which would accept a raw-space
    intercept for a model with trees."""
    booster, document = fit("binary:logistic", base_score=0.75, rounds=5)
    derived = objectives.derive_intercept(document)
    objectives.verify_intercept(booster, derived)

    raw = float(np.float32(0.75))
    assert bits32(derived) != bits32(raw)
    with pytest.raises(InterceptMismatchError):
        objectives.verify_intercept(booster, raw)


def test_verify_intercept_raises_on_an_unsupported_objective() -> None:
    booster = xgb.train(
        {"objective": "reg:logistic", "nthread": 1, "base_score": 0.4},
        _matrix("binary:logistic"),
        num_boost_round=0,
    )
    with pytest.raises(UnsupportedObjectiveError):
        objectives.verify_intercept(booster, 0.0)


def test_verify_intercept_error_carries_structured_attributes() -> None:
    booster, _document = fit("survival:cox", base_score=0.25, rounds=0)
    with pytest.raises(InterceptMismatchError) as caught:
        objectives.verify_intercept(booster, 1.5)
    assert caught.value.derived == 1.5
    assert bits32(caught.value.observed) == bits32(-1.3862943649291992)
    assert caught.value.objective == "survival:cox"
    assert "does not match" in str(caught.value)
