"""Tests for artifact assembly and deterministic serialization (export.py).

This module is the mechanical half of export: it computes no numeric value
of its own, and every test below drives it through real fitted models
rather than hand-built shortcuts, per the brief this suite was built from.

**Ambiguity, resolved and flagged rather than guessed at:** the brief
requires ``provenance.base_score`` to carry "the ORIGINAL stored value".
FORMAT.md section 16's worked example renders that field as a float32-snapped
JSON number (``0.6000000238418579``), which is a *derived* value -- reaching
it would mean parsing the bracketed ``base_score`` string and snapping it to
float32 in ``export.py``, i.e. writing threshold/``base_score``-handling
code this module's scope explicitly excludes. This suite instead pins
``provenance.base_score`` to the *raw* string exactly as
``learner.learner_model_param.base_score`` holds it (e.g. ``"[4E-1]"``),
with no parsing performed in ``export.py`` at all -- the literal reading of
"ORIGINAL stored value". See ``test_provenance_records_xgboost_version_and_base_score_verbatim``.
This is called out again in the final report for review.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import warnings
from typing import Any

import numpy as np
import pytest
import xgboost as xgb

from xgboost_bridge import errors, export
from xgboost_bridge.trees import walk_margin

OBJECTIVES = ("reg:squarederror", "binary:logistic", "survival:cox")

_BASE_PARAMS = {"tree_method": "exact", "max_depth": 3, "eta": 0.3, "nthread": 1}


def _fit(
    objective: str,
    *,
    rows: int = 600,
    cols: int = 5,
    num_boost_round: int = 6,
    base_score: float | None = None,
    seed: int = 20260804,
) -> tuple[xgb.Booster, np.ndarray, list[str]]:
    """Fit a small, real, deterministic model and return it with its inputs.

    ``tree_method="exact"`` and ``nthread=1`` are what make training itself
    reproducible across processes -- required for the separate-interpreter
    determinism check below, not merely a style choice here.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, cols))

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

    names = [f"f{i}" for i in range(cols)]
    dtrain = xgb.DMatrix(x, label=label, feature_names=names)

    params: dict[str, Any] = dict(_BASE_PARAMS)
    params["objective"] = objective
    params["seed"] = seed
    if base_score is not None:
        params["base_score"] = float(base_score)

    booster = xgb.train(params, dtrain, num_boost_round=num_boost_round, verbose_eval=False)
    return booster, x, names


# ---------------------------------------------------------------------------
# 5. The envelope is exactly the seven keys of FORMAT.md section 3.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_export_produces_exactly_the_seven_envelope_keys(objective: str) -> None:
    booster, _x, names = _fit(objective)
    artifact = export.export_model(booster, feature_names=names)
    assert set(artifact) == export.ENVELOPE_KEYS
    assert set(artifact) == {
        "format_version",
        "objective",
        "output_transform",
        "intercept",
        "feature_names",
        "trees",
        "provenance",
    }


def test_envelope_check_rejects_an_eighth_key() -> None:
    booster, _x, names = _fit("reg:squarederror")
    artifact = export.export_model(booster, feature_names=names)
    artifact["unexpected_extra_key"] = 1
    with pytest.raises(errors.MalformedTreeError) as excinfo:
        export._assert_envelope_keys(artifact)
    assert excinfo.value.field == "<envelope>"


def test_envelope_check_rejects_a_missing_key() -> None:
    booster, _x, names = _fit("reg:squarederror")
    artifact = export.export_model(booster, feature_names=names)
    del artifact["intercept"]
    with pytest.raises(errors.MalformedTreeError) as excinfo:
        export._assert_envelope_keys(artifact)
    assert excinfo.value.field == "<envelope>"


def test_provenance_has_exactly_its_three_keys() -> None:
    booster, _x, names = _fit("binary:logistic")
    artifact = export.export_model(booster, feature_names=names)
    assert set(artifact["provenance"]) == export.PROVENANCE_KEYS
    assert set(artifact["provenance"]) == {
        "xgboost_version",
        "base_score",
        "exporter_version",
    }


def test_provenance_check_rejects_an_extra_key() -> None:
    bad_provenance = {
        "xgboost_version": "3.3.0",
        "base_score": "[5E-1]",
        "exporter_version": "0.1.0",
        "unexpected": 1,
    }
    with pytest.raises(errors.MalformedTreeError) as excinfo:
        export._assert_provenance_keys(bad_provenance)
    assert excinfo.value.field == "<provenance>"


def test_format_version_is_the_integer_one_not_a_string_or_float() -> None:
    booster, _x, names = _fit("reg:squarederror")
    artifact = export.export_model(booster, feature_names=names)
    assert artifact["format_version"] == 1
    assert isinstance(artifact["format_version"], int)
    assert not isinstance(artifact["format_version"], bool)
    assert not isinstance(artifact["format_version"], str)


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_objective_and_output_transform_pairing(objective: str) -> None:
    booster, _x, names = _fit(objective)
    artifact = export.export_model(booster, feature_names=names)
    assert artifact["objective"] == objective
    expected_transform = {
        "reg:squarederror": "identity",
        "binary:logistic": "sigmoid",
        "survival:cox": "exp",
    }[objective]
    assert artifact["output_transform"] == expected_transform


def test_provenance_records_xgboost_version_and_base_score_verbatim() -> None:
    """See this module's docstring: ``base_score`` is carried through with
    no parsing, per the "ORIGINAL stored value" reading of the brief."""
    booster, _x, names = _fit("reg:squarederror", base_score=0.25)
    document = json.loads(booster.save_raw(raw_format="json"))
    artifact = export.export_model(booster, feature_names=names)

    expected_version = ".".join(str(component) for component in document["version"])
    assert artifact["provenance"]["xgboost_version"] == expected_version
    stored_base_score = document["learner"]["learner_model_param"]["base_score"]
    assert isinstance(stored_base_score, str)
    assert artifact["provenance"]["base_score"] == stored_base_score
    assert artifact["provenance"]["exporter_version"] == export.__version__


# ---------------------------------------------------------------------------
# feature_names resolution
# ---------------------------------------------------------------------------


def test_feature_names_default_to_the_models_own() -> None:
    booster, _x, names = _fit("reg:squarederror")
    artifact = export.export_model(booster)
    assert artifact["feature_names"] == names


def test_feature_names_override_is_used_when_given() -> None:
    booster, _x, names = _fit("reg:squarederror")
    override = [f"custom_{i}" for i in range(len(names))]
    artifact = export.export_model(booster, feature_names=override)
    assert artifact["feature_names"] == override


def test_feature_names_override_is_validated_for_duplicates() -> None:
    booster, _x, names = _fit("reg:squarederror")
    duplicated = ["dup"] * len(names)
    with pytest.raises(errors.UnsupportedModelShapeError) as excinfo:
        export.export_model(booster, feature_names=duplicated)
    assert excinfo.value.field == "feature_names"


def test_feature_names_override_is_validated_for_length() -> None:
    booster, _x, names = _fit("reg:squarederror")
    with pytest.raises(errors.UnsupportedModelShapeError) as excinfo:
        export.export_model(booster, feature_names=names[:-1])
    assert excinfo.value.field == "feature_names"


def test_feature_names_override_rejects_a_non_string_entry() -> None:
    booster, _x, names = _fit("reg:squarederror")
    bad_names: list[Any] = list(names)
    bad_names[0] = 3
    with pytest.raises(errors.UnsupportedModelShapeError) as excinfo:
        export.export_model(booster, feature_names=bad_names)
    assert excinfo.value.field == "feature_names"


def test_feature_names_override_is_required_for_a_bare_array_model() -> None:
    """D021: a model fit from a bare array has no feature names of its own;
    supplying names here is the only way such a model can be exported."""
    rng = np.random.default_rng(20260900)
    x = rng.normal(size=(300, 3))
    y = rng.normal(size=300)
    dtrain = xgb.DMatrix(x, label=y)  # no feature_names -> serializes as []
    booster = xgb.train(
        {**_BASE_PARAMS, "objective": "reg:squarederror", "seed": 1},
        dtrain,
        num_boost_round=3,
        verbose_eval=False,
    )
    document = json.loads(booster.save_raw(raw_format="json"))
    assert document["learner"]["feature_names"] == []

    with pytest.raises(errors.MissingFeatureNamesError):
        export.export_model(booster)

    artifact = export.export_model(booster, feature_names=["a", "b", "c"])
    assert artifact["feature_names"] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Non-finite intercepts (D043, FORMAT.md section 9.3)
# ---------------------------------------------------------------------------


def _zero_round_cox(base_score: float) -> xgb.Booster:
    x = np.zeros((4, 2))
    y = np.array([1.0, -2.0, 3.0, -4.0])
    dtrain = xgb.DMatrix(x, label=y, feature_names=["a", "b"])
    return xgb.train(
        {"objective": "survival:cox", "nthread": 1, "base_score": float(base_score)},
        dtrain,
        num_boost_round=0,
    )


def test_export_raises_on_a_non_finite_cox_intercept_at_zero_base_score() -> None:
    """``survival:cox`` at ``base_score = 0.0`` derives ``-inf``; XGBoost
    accepts this with no error and no warning (D043)."""
    booster = _zero_round_cox(0.0)
    with pytest.raises(errors.NonFiniteInterceptError) as excinfo:
        export.export_model(booster)
    assert not math.isfinite(excinfo.value.intercept)
    assert math.isinf(excinfo.value.intercept)


def test_export_raises_on_a_non_finite_cox_intercept_at_negative_base_score() -> None:
    """``survival:cox`` with any negative ``base_score`` derives ``NaN``;
    XGBoost accepts this with no error and no warning (D043)."""
    booster = _zero_round_cox(-0.5)
    with pytest.raises(errors.NonFiniteInterceptError) as excinfo:
        export.export_model(booster)
    assert not math.isfinite(excinfo.value.intercept)
    assert math.isnan(excinfo.value.intercept)


# ---------------------------------------------------------------------------
# Validation runs first (order of operations)
# ---------------------------------------------------------------------------


def test_export_refuses_an_untested_xgboost_version_before_anything_numeric() -> None:
    booster, _x, names = _fit("reg:squarederror")
    with pytest.raises(errors.UnsupportedVersionError):
        export.export_model(
            booster, feature_names=names, tested_versions=frozenset({"0.0.1"})
        )


def test_export_refuses_dart_before_reaching_the_numeric_path() -> None:
    rng = np.random.default_rng(20261001)
    x = rng.normal(size=(200, 4))
    label_rng = np.random.default_rng(20261002)
    y = (label_rng.random(200) > 0.5).astype(np.float64)
    names = ["f0", "f1", "f2", "f3"]
    dtrain = xgb.DMatrix(x, label=y, feature_names=names)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # dart is deprecated upstream
        booster = xgb.train(
            {
                "objective": "binary:logistic",
                "booster": "dart",
                "rate_drop": 0.3,
                "skip_drop": 0.1,
                "nthread": 1,
                "seed": 1,
            },
            dtrain,
            num_boost_round=5,
            verbose_eval=False,
        )

    with pytest.raises(errors.UnsupportedBoosterError) as excinfo:
        export.export_model(booster, feature_names=names)
    assert excinfo.value.booster == "dart"


# ---------------------------------------------------------------------------
# 2. Determinism
# ---------------------------------------------------------------------------


def test_export_is_byte_identical_across_two_calls_in_the_same_process() -> None:
    booster, _x, names = _fit("binary:logistic", base_score=0.6)
    first = export.to_json(export.export_model(booster, feature_names=names))
    second = export.to_json(export.export_model(booster, feature_names=names))
    assert first == second


_SUBPROCESS_TRAINING_SCRIPT = """
import hashlib
import numpy as np
import xgboost as xgb
from xgboost_bridge import export

rng = np.random.default_rng(20260804)
x = rng.normal(size=(300, 4))
label_rng = np.random.default_rng(20260805)
y = (label_rng.random(300) > 0.5).astype(np.float64)
names = ["f0", "f1", "f2", "f3"]
dtrain = xgb.DMatrix(x, label=y, feature_names=names)
booster = xgb.train(
    {
        "tree_method": "exact",
        "max_depth": 3,
        "eta": 0.3,
        "nthread": 1,
        "objective": "binary:logistic",
        "seed": 20260804,
        "base_score": 0.6,
    },
    dtrain,
    num_boost_round=6,
    verbose_eval=False,
)
artifact = export.export_model(booster, feature_names=names)
text = export.to_json(artifact)
print(hashlib.sha256(text.encode("utf-8")).hexdigest())
"""


def _run_subprocess_export() -> str:
    result = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_TRAINING_SCRIPT],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_export_is_byte_identical_across_two_separate_interpreter_invocations() -> None:
    first = _run_subprocess_export()
    second = _run_subprocess_export()
    assert len(first) == 64  # a sha256 hex digest, so the comparison is meaningful
    assert first == second


def test_serialized_keys_are_sorted_lexicographically_at_every_level() -> None:
    """FORMAT.md section 12 / D008. Checked at every JSON object encountered
    while parsing, at any nesting depth -- the envelope, ``provenance``, and
    every tree dict."""

    def _assert_sorted_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        keys = [key for key, _ in pairs]
        assert keys == sorted(keys), f"keys not sorted at this level: {keys}"
        return dict(pairs)

    booster, _x, names = _fit("survival:cox", base_score=0.4)
    text = export.to_json(export.export_model(booster, feature_names=names))
    json.loads(text, object_pairs_hook=_assert_sorted_hook)


def test_serialization_has_no_insignificant_whitespace() -> None:
    booster, _x, names = _fit("reg:squarederror")
    text = export.to_json(export.export_model(booster, feature_names=names))
    body = text[:-1]  # the one trailing newline is checked separately below
    assert "\n" not in body
    assert ", " not in body
    assert ": " not in body


def test_serialization_ends_with_exactly_one_trailing_newline() -> None:
    booster, _x, names = _fit("reg:squarederror")
    text = export.to_json(export.export_model(booster, feature_names=names))
    assert text.endswith("\n")
    assert not text.endswith("\n\n")


def test_signed_zero_intercept_is_preserved_through_serialization() -> None:
    """``binary:logistic`` at ``base_score = 0.5`` derives exactly ``-0.0``
    (FORMAT.md section 6.3). ``json.dumps`` already preserves the sign; this
    pins that ``to_json`` does not accidentally normalize it away."""
    x = np.zeros((20, 2))
    y = np.array([0.0, 1.0] * 10)
    dtrain = xgb.DMatrix(x, label=y, feature_names=["a", "b"])
    booster = xgb.train(
        {"objective": "binary:logistic", "nthread": 1, "base_score": 0.5},
        dtrain,
        num_boost_round=0,
    )
    artifact = export.export_model(booster)
    assert math.copysign(1.0, artifact["intercept"]) == -1.0
    assert artifact["intercept"] == 0.0

    text = export.to_json(artifact)
    assert '"intercept":-0.0' in text
    reloaded = json.loads(text)
    assert math.copysign(1.0, reloaded["intercept"]) == -1.0


# ---------------------------------------------------------------------------
# 6. learner.attributes must never leak into the artifact (D020, D008).
# ---------------------------------------------------------------------------


def test_early_stopped_models_learner_attributes_do_not_leak_into_the_artifact() -> None:
    """``learner.attributes`` is the only nondeterministic surface measured
    in a source model; the v1 whitelist is empty (D020), so nothing from it
    may reach the artifact.

    Searched by exact substring match of every value ``learner.attributes``
    holds against the full serialized JSON text -- this does not depend on
    guessing which artifact field a leak would land in.
    """
    rng = np.random.default_rng(20260901)
    x = rng.normal(size=(300, 4))
    y = rng.normal(size=300)
    names = ["f0", "f1", "f2", "f3"]
    dtrain = xgb.DMatrix(x, label=y, feature_names=names)

    booster = xgb.train(
        {**_BASE_PARAMS, "objective": "reg:squarederror", "seed": 7},
        dtrain,
        num_boost_round=6,
        evals=[(dtrain, "train")],
        early_stopping_rounds=10,
        verbose_eval=False,
    )
    document = json.loads(booster.save_raw(raw_format="json"))
    attributes = document["learner"]["attributes"]
    assert "best_iteration" in attributes
    assert "best_score" in attributes
    best_score = attributes["best_score"]
    # The distinctive marker D020 names explicitly: a short value like
    # best_iteration's "5" is too short to search for as a substring without
    # false positives, but a full-precision float string is not.
    assert len(best_score) >= 8, f"best_score {best_score!r} is not full precision"

    iteration_indptr = document["learner"]["gradient_booster"]["model"]["iteration_indptr"]
    total_trees = len(document["learner"]["gradient_booster"]["model"]["trees"])
    best_iteration = int(attributes["best_iteration"])
    # Unambiguous (D038): both tree-count readings already agree, so the
    # gate accepts this model and export can proceed.
    assert iteration_indptr[best_iteration + 1] == total_trees

    artifact = export.export_model(booster, feature_names=names)
    text = export.to_json(artifact)

    assert "attributes" not in text
    assert best_score not in text
    for value in attributes.values():
        # Only values long enough to be a meaningful substring search are
        # checked here -- a short value like best_iteration's "5" is a
        # false-positive trap (it is a substring of countless legitimate
        # numbers in the artifact), not evidence of a leak.
        if len(value) < 8:
            continue
        assert value not in text, f"learner.attributes value {value!r} leaked into the artifact"


# ---------------------------------------------------------------------------
# 3. Round-trip through the walk: the artifact's own trees and intercept,
# fed to trees.walk_margin, must reproduce predict(output_margin=True)
# bit-for-bit.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_round_trip_through_the_walk_matches_predict_bit_for_bit(objective: str) -> None:
    booster, x, names = _fit(objective, rows=800, num_boost_round=8, base_score=0.55)
    artifact = export.export_model(booster, feature_names=names)

    intercept = artifact["intercept"]
    trees = artifact["trees"]

    dtrain_full = xgb.DMatrix(x, feature_names=names)
    expected = np.asarray(booster.predict(dtrain_full, output_margin=True), dtype=np.float32)

    total = len(expected)
    assert total >= 500

    exact = 0
    first_mismatch: tuple[int, float, float] | None = None
    for row_index in range(total):
        row = x[row_index]  # a float64 NumPy row, the documented input shape
        got = walk_margin(trees, intercept, row)
        if int(got.view(np.uint32)) == int(expected[row_index].view(np.uint32)):
            exact += 1
        elif first_mismatch is None:
            first_mismatch = (row_index, float(got), float(expected[row_index]))

    assert exact == total, f"{objective}: {exact}/{total} bit-exact; first mismatch {first_mismatch}"
    print(f"{objective}: round-trip {exact}/{total} bit-exact")


# ---------------------------------------------------------------------------
# 4. Emission fidelity: every emitted number round-trips through json.loads
# to a value whose float32 narrowing is unchanged.
# ---------------------------------------------------------------------------

_EMISSION_FIDELITY_BASE_SCORES: dict[str, tuple[float, ...]] = {
    "reg:squarederror": (-3.0, -0.3, 0.0, 0.6),
    "binary:logistic": (0.001, 0.1, 0.48, 0.9),
    "survival:cox": (0.01, 0.5, 1.0, 10.0),
}


def test_every_emitted_number_round_trips_to_an_unchanged_float32() -> None:
    """FORMAT.md section 9.1: the emission rule is
    ``float(np.float32(x))``, serialized by Python's ordinary shortest
    round-trip ``float`` repr. This confirms the *text* recovers exactly the
    same float32 for every ``intercept`` and every ``node_values`` entry
    across a real spread of fitted models.
    """
    checked = 0
    for objective, base_scores in _EMISSION_FIDELITY_BASE_SCORES.items():
        for base_score in base_scores:
            booster, _x, names = _fit(
                objective, base_score=base_score, num_boost_round=10, cols=6
            )
            artifact = export.export_model(booster, feature_names=names)
            text = export.to_json(artifact)
            reloaded = json.loads(text)

            original_numbers = [artifact["intercept"]]
            for tree in artifact["trees"]:
                original_numbers.extend(tree["node_values"])

            reloaded_numbers = [reloaded["intercept"]]
            for tree in reloaded["trees"]:
                reloaded_numbers.extend(tree["node_values"])

            assert len(original_numbers) == len(reloaded_numbers)
            for original_value, reloaded_value in zip(original_numbers, reloaded_numbers):
                original_bits = np.float32(original_value).view(np.uint32)
                reloaded_bits = np.float32(reloaded_value).view(np.uint32)
                assert original_bits == reloaded_bits, (
                    f"{objective} base_score={base_score!r}: "
                    f"{original_value!r} -> {reloaded_value!r} changed float32 bits"
                )
                checked += 1

    assert checked >= 500, f"only checked {checked} numbers; expected at least 500"
    print(f"emission fidelity: checked {checked} numbers")
