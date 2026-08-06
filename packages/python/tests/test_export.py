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
import importlib.util
import json
import math
import re
import subprocess
import sys
import tomllib
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xgboost as xgb

from xgboost_bridge import errors, export
from xgboost_bridge.trees import extract_trees, reachable_nodes, walk_margin

REPO_ROOT = Path(__file__).resolve().parents[3]

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
    gamma: float | None = None,
    max_depth: int | None = None,
) -> tuple[xgb.Booster, np.ndarray, list[str]]:
    """Fit a small, real, deterministic model and return it with its inputs.

    ``tree_method="exact"`` and ``nthread=1`` are what make training itself
    reproducible across processes -- required for the separate-interpreter
    determinism check below, not merely a style choice here.

    ``gamma`` and ``max_depth`` are omitted from ``params`` unless given, so
    every test written before they existed fits exactly the model it always
    did. A ``gamma`` above zero is what produces a **pruned** tree, i.e. one
    with dead nodes -- and it only works under ``tree_method="exact"``:
    ``hist`` declines to grow a losing split rather than growing and then
    pruning one, so it yields ``num_deleted == 0`` at every ``gamma``.
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
    if gamma is not None:
        params["gamma"] = float(gamma)
    if max_depth is not None:
        params["max_depth"] = int(max_depth)

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


def test_the_export_extra_pins_exactly_the_enumerated_version_ceiling() -> None:
    """The declared extra and the code's ceiling are one list, compared here.

    ``DEFAULT_TESTED_VERSIONS`` is the enumerated ceiling of D018: export raises
    for a producing version outside it. The published ``export`` extra decides
    which xgboost a fresh ``pip install`` actually resolves. When those two
    disagree the *installed* package raises ``UnsupportedVersionError`` on the
    first call -- which is what shipped once, because the extra said
    ``xgboost>=3.3,<4`` while 3.4.0 was on the index (D051).

    Every test in this repository runs against the source tree, where the
    workspace pins the version, so no prediction can see this. A comparison of
    the two declarations can. A range spelling fails here even when it happens
    to resolve correctly today: ``3.3.1`` would also be untested and would also
    raise, so the dependency specifier has to equal the tested list rather than
    contain it.
    """
    manifest = tomllib.loads(
        (REPO_ROOT / "packages" / "python" / "pyproject.toml").read_text(encoding="utf-8")
    )
    declared = manifest["project"]["optional-dependencies"]["export"]

    expected = sorted(f"xgboost=={version}" for version in export.DEFAULT_TESTED_VERSIONS)
    assert sorted(declared) == expected, (
        "the export extra and the enumerated version ceiling disagree: "
        f"{sorted(declared)} against {expected}"
    )


def _load_policy():
    """The shared historical-record exemption. See ``_policy`` for the rule.

    Loaded by path because pytest runs under ``--import-mode=importlib``. The
    vocabulary scrub loads the same module the same way, so the exemption has one
    definition and the two prose checks cannot drift apart -- which they did,
    within a single commit, before this module existed.
    """
    path = Path(__file__).resolve().with_name("_policy.py")
    spec = importlib.util.spec_from_file_location("_xgboost_bridge_test_policy", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_POLICY = _load_policy()

#: User-facing documents. A reader decides what to install from these, so a version
#: specifier stated here is a claim about the manifest and is pinned to it below.
#: The historical record is exempt -- a decision entry and a probe transcript must
#: stay free to quote a specifier that was withdrawn. That exemption is not
#: restated here; it comes from ``_policy`` and is asserted disjoint below.
USER_FACING_DOCS = (
    Path("COMPAT.md"),
    Path("README.md"),
    Path("packages") / "python" / "README.md",
)


def test_the_pinned_documents_are_not_part_of_the_historical_record() -> None:
    """The two prose checks share one exemption, and this is where they meet.

    Without this, ``USER_FACING_DOCS`` is an independent allow-list that happens
    to agree with the scrub's exemption today. Adding ``docs/DECISIONS.md`` to it
    would make the specifier check demand that a historical entry be edited to
    match the current manifest -- the exact inconsistency ``_policy`` was written
    to settle -- and nothing would have objected.
    """
    overlap = [str(doc) for doc in USER_FACING_DOCS if _POLICY.is_historical_record(doc)]
    assert not overlap, (
        "a document cannot be both pinned to the current manifest and exempt as a "
        f"historical record: {overlap}"
    )

#: ``xgboost`` followed by a PEP 440 operator and a version, in prose or a fenced
#: block: ``xgboost==3.3.0``, ``xgboost>=3.3,<4``.
_DOC_SPECIFIER_RE = re.compile(
    r"xgboost\s*(?:==|>=|<=|~=|!=|>|<)\s*[0-9][0-9A-Za-z.*+!-]*"
    r"(?:\s*,\s*(?:==|>=|<=|~=|!=|>|<)\s*[0-9][0-9A-Za-z.*+!-]*)*",
    re.IGNORECASE,
)


def test_user_facing_docs_state_the_same_xgboost_specifier_as_the_manifest() -> None:
    """Prose that contradicts the manifest is the defect this pins.

    ``COMPAT.md`` told a reader the export extra allowed ``xgboost>=3.3,<4`` and
    pointed at ``packages/python/pyproject.toml`` to confirm it, for the whole
    period after D051 pinned that manifest to ``xgboost==3.3.0``. The range it
    named is the exact specifier D051 records as a shipped defect. Every
    executable check passed the entire time: the test above pins the manifest,
    and nothing pinned the sentence describing it.

    So the same comparison is made against the prose. A specifier written in a
    user-facing document must equal the one a user would actually resolve.

    **What this does not catch, stated rather than implied:** a bare version
    named without an operator ("XGBoost 3.4 is supported") is prose this regex
    does not read, and a document that omits the specifier entirely cannot
    disagree with anything. This closes the contradiction case, not every way a
    document can mislead.
    """
    manifest = tomllib.loads(
        (REPO_ROOT / "packages" / "python" / "pyproject.toml").read_text(encoding="utf-8")
    )
    declared = manifest["project"]["optional-dependencies"]["export"]
    assert len(declared) == 1, f"expected a single export requirement, got {declared}"
    canonical = declared[0].replace(" ", "")

    findings: list[str] = []
    scanned = 0
    for relative in USER_FACING_DOCS:
        path = REPO_ROOT / relative
        assert path.is_file(), f"user-facing document is missing: {relative}"
        scanned += 1
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for match in _DOC_SPECIFIER_RE.finditer(line):
                stated = match.group(0).replace(" ", "").replace("`", "")
                if stated.lower() != canonical.lower():
                    findings.append(f"{relative}:{lineno}: states {stated!r}")

    assert scanned == len(USER_FACING_DOCS)
    assert not findings, (
        f"user-facing documentation contradicts the manifest ({canonical!r}):\n"
        + "\n".join(findings)
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
import importlib.util
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
# 7. The neutralization self-check (FORMAT.md section 8.3, D027).
#
# Every check upstream of it validates deadness *detection* -- that the
# `split_indices == 2147483647` marker agrees with reachability. None of them
# validates the arrays neutralization produced, and a neutralization that
# cleared a live node is silent wrongness. The oracle here is XGBoost's own
# `predict(output_margin=True)`, which shares nothing with this library's
# extraction, neutralization, or emission path.
#
# Note what the round-trip test above cannot cover: `_fit` set no `gamma`
# until now, so no export-level test had ever seen a tree with a dead node.
# The tests below fit a pruned model and assert `num_deleted > 0`, so they
# cannot silently stop covering the case.
# ---------------------------------------------------------------------------

_PRUNED_MODEL = {
    "objective": "binary:logistic",
    "rows": 600,
    "cols": 5,
    "num_boost_round": 4,
    "gamma": 5.0,
    "max_depth": 6,
}


def _fit_pruned() -> tuple[xgb.Booster, np.ndarray, list[str]]:
    """A real, deterministic, ``gamma``-pruned model: dead nodes and all."""
    return _fit(**_PRUNED_MODEL)  # type: ignore[arg-type]


def _dead_node_counts(booster: xgb.Booster) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Per-tree dead-node counts, read two independent ways.

    ``tree_param.num_deleted`` is XGBoost's own count, as a JSON string.
    The second count is the number of ``split_indices`` entries carrying the
    ``2147483647`` marker. Both are read from the *source* model, before this
    library touches anything, and a test that compared only one of them
    against itself would pin nothing.
    """
    document = json.loads(booster.save_raw(raw_format="json"))
    source_trees = document["learner"]["gradient_booster"]["model"]["trees"]
    reported = tuple(int(tree["tree_param"]["num_deleted"]) for tree in source_trees)
    marked = tuple(
        sum(1 for value in tree["split_indices"] if value == 2147483647)
        for tree in source_trees
    )
    return reported, marked


def test_the_self_check_model_really_carries_dead_nodes() -> None:
    """The precondition every test in this section depends on.

    ``tree_method="hist"`` produces no dead nodes at any ``gamma``; ``exact``
    is required. Asserted rather than assumed, so this section cannot quietly
    become a test of unpruned trees.
    """
    booster, _x, names = _fit_pruned()
    reported, marked = _dead_node_counts(booster)
    assert reported == marked, (
        f"XGBoost's own num_deleted {reported} disagrees with the "
        f"2147483647 marker count {marked}"
    )
    assert sum(reported) > 0, "the model carries no dead nodes at all"
    assert all(count > 0 for count in reported), (
        f"some tree carries no dead node: {reported}"
    )
    print(f"pruned self-check model: num_deleted per tree {reported}, total {sum(reported)}")

    # And it exports, which is the other half of the claim: a pruned model is
    # ordinary artifact content, not a refusal (FORMAT.md section 8.3).
    artifact = export.export_model(booster, feature_names=names)
    assert len(artifact["trees"]) == len(reported)


def test_the_self_check_sample_reaches_every_live_node() -> None:
    """The sufficiency claim, made executable against an independent oracle.

    ``_self_check_rows`` claims to reach every node the walk can visit. The
    oracle is XGBoost's ``pred_leaf``, which reports the node index of the
    leaf **it** landed on, per row and per tree -- not this library's walk, so
    a defect in the walk cannot make the coverage look complete. Every live
    leaf must be reported, and every live node is then covered because it is
    an ancestor of a reported leaf.
    """
    booster, _x, names = _fit_pruned()
    artifact = export.export_model(booster, feature_names=names)
    trees = artifact["trees"]
    rows = export._self_check_rows(trees, len(names))

    reached = np.asarray(
        booster.predict(
            xgb.DMatrix(rows, feature_names=booster.feature_names, nthread=1),
            pred_leaf=True,
        )
    ).reshape(len(rows), -1)
    assert reached.shape == (len(rows), len(trees))

    total_live = 0
    for index, tree in enumerate(trees):
        live = reachable_nodes(tree)
        total_live += len(live)
        live_leaves = {node for node in live if tree["left_children"][node] == -1}
        reported = {int(value) for value in reached[:, index]}
        assert reported <= live_leaves, (
            f"tree {index}: XGBoost reported leaves outside the live set: "
            f"{sorted(reported - live_leaves)}"
        )
        assert reported == live_leaves, (
            f"tree {index}: {len(live_leaves) - len(reported)} live leaves are "
            f"reached by no sampled row: {sorted(live_leaves - reported)}"
        )

        parent = {}
        for node in live:
            if tree["left_children"][node] != -1:
                parent[tree["left_children"][node]] = node
                parent[tree["right_children"][node]] = node
        covered: set[int] = set()
        for leaf in reported:
            node = leaf
            while True:
                covered.add(node)
                if node not in parent:
                    break
                node = parent[node]
        assert covered == live, (
            f"tree {index}: live nodes reached by no sampled row: "
            f"{sorted(live - covered)}"
        )

    assert total_live >= 30, f"only {total_live} live nodes; the check is near-vacuous"
    print(
        f"self-check sample: {len(rows)} rows cover {total_live} live nodes "
        f"across {len(trees)} trees"
    )


def test_the_self_check_sample_is_deterministic() -> None:
    """Two builds, byte-identical -- including the ``NaN`` block's bit patterns.

    Compared on ``tobytes()`` rather than with ``==``, since ``NaN != NaN``
    and ``-0.0 == 0.0``.
    """
    booster, _x, names = _fit_pruned()
    trees = export.export_model(booster, feature_names=names)["trees"]
    first = export._self_check_rows(trees, len(names))
    second = export._self_check_rows(trees, len(names))
    assert first.shape == second.shape
    assert first.tobytes() == second.tobytes()


def _clear_one_live_node(document: dict[str, Any]) -> list[dict[str, Any]]:
    """``extract_trees``, then neutralize a node the walk actually visits.

    The corruption lives here, in a copy of the export flow, and never in
    ``trees.py``: the point is to demonstrate that the self-check fires on a
    defect no other check can see, which requires the real extraction to run
    first. It is imported from ``trees`` rather than reached through
    ``export.extract_trees``, which is the name being replaced.
    """
    trees = extract_trees(document)
    victim = sorted(reachable_nodes(trees[0]))[1]
    trees[0]["split_indices"][victim] = 0
    trees[0]["node_values"][victim] = 0.0
    trees[0]["left_children"][victim] = -1
    trees[0]["right_children"][victim] = -1
    trees[0]["default_left"][victim] = 0
    return trees


def test_export_refuses_an_artifact_whose_live_node_was_cleared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure FORMAT.md section 8.3 names: a cleared *live* node.

    Nothing structural detects it. The cleared node is a well-formed leaf
    carrying ``0.0``, which is legitimate artifact content, its
    ``split_indices`` entry is in range, and the reachability marker still
    agrees with itself. Only the comparison against XGBoost's own margin sees
    it.
    """
    booster, _x, names = _fit_pruned()
    monkeypatch.setattr(export, "extract_trees", _clear_one_live_node)

    with pytest.raises(errors.MarginMismatchError) as failure:
        export.export_model(booster, feature_names=names)

    error = failure.value
    assert error.mismatches >= 1
    assert error.rows_compared >= error.mismatches
    assert 0 <= error.row_index < error.rows_compared
    assert np.float32(error.derived).view(np.uint32) != np.float32(
        error.observed
    ).view(np.uint32)
    print(
        f"cleared live node: refused on {error.mismatches}/{error.rows_compared} "
        f"rows, first margin error {abs(error.derived - error.observed):.6f}"
    )


def _read_base_weights_instead(document: dict[str, Any]) -> list[dict[str, Any]]:
    """``extract_trees``, then take ``node_values`` from the wrong source array.

    ``base_weights`` is the sibling array a plausible misreading picks up: it
    is per-node, the same length, and finite everywhere, so every structural
    check passes. FORMAT.md section 15 records it as off by ``5.10`` in margin
    space.
    """
    trees = extract_trees(document)
    source_trees = document["learner"]["gradient_booster"]["model"]["trees"]
    for tree, source in zip(trees, source_trees):
        tree["node_values"] = [float(np.float32(value)) for value in source["base_weights"]]
    return trees


def test_export_refuses_node_values_read_from_the_wrong_source_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``base_weights`` instead of ``split_conditions``: same shape, wrong numbers."""
    booster, _x, names = _fit_pruned()
    monkeypatch.setattr(export, "extract_trees", _read_base_weights_instead)

    with pytest.raises(errors.MarginMismatchError) as failure:
        export.export_model(booster, feature_names=names)
    error = failure.value
    assert error.mismatches >= 1
    print(
        f"base_weights substitution: refused on "
        f"{error.mismatches}/{error.rows_compared} rows, first margin error "
        f"{abs(error.derived - error.observed):.6f}"
    )


class _DoubledMarginBooster:
    """A booster whose ``predict`` returns two margins per row.

    The arity gate (D017) refuses multi-output models, so this shape cannot
    arrive through a validated model -- which is exactly why it is driven
    directly rather than left as an untested branch. Without the shape check
    the self-check would compare a scalar against a row of values.
    """

    def __init__(self, booster: xgb.Booster) -> None:
        self._booster = booster

    def __getattr__(self, name: str) -> Any:
        return getattr(self._booster, name)

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        margin = np.asarray(self._booster.predict(*args, **kwargs))
        return np.column_stack([margin, margin])


def test_the_self_check_refuses_a_margin_that_is_not_one_value_per_row() -> None:
    booster, _x, names = _fit_pruned()
    with pytest.raises(errors.MalformedTreeError) as failure:
        export.export_model(_DoubledMarginBooster(booster), feature_names=names)
    assert failure.value.field == "<observed margin>"


def test_the_self_check_runs_on_every_objective_and_agrees() -> None:
    """The clean side of the check: it passes, and on rows that sit ON the
    thresholds rather than near them.

    Every row of the sample is a boundary value -- the threshold itself, or
    the float32 immediately below it -- which is where a one-sided float32
    cast or an equality routed the wrong way changes the branch. That the
    walk agrees with XGBoost on all of them is a stronger statement than
    agreement on ordinary rows.
    """
    for objective in OBJECTIVES:
        booster, _x, names = _fit(objective, gamma=2.0, max_depth=6, num_boost_round=5)
        artifact = export.export_model(booster, feature_names=names)
        rows = export._self_check_rows(artifact["trees"], len(names))
        observed = np.asarray(
            booster.predict(
                xgb.DMatrix(rows, feature_names=booster.feature_names, nthread=1),
                output_margin=True,
            ),
            dtype=np.float32,
        )
        exact = sum(
            1
            for index in range(len(rows))
            if int(walk_margin(artifact["trees"], artifact["intercept"], rows[index]).view(np.uint32))
            == int(observed[index].view(np.uint32))
        )
        assert exact == len(rows), f"{objective}: {exact}/{len(rows)} bit-exact"
        print(f"{objective}: self-check sample {exact}/{len(rows)} bit-exact")


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


# ---------------------------------------------------------------------------
# The intercept comes from the engine, and the oracle that guards it fires
#
# D053: XGBoost derives this intercept with the platform's `logf`, which is not
# correctly rounded, and its own answer differs between darwin/arm64 and
# linux/x86_64 by 1 ULP on 29 of 58 discriminating inputs. Deriving the value
# instead of reading it out therefore guarantees a spurious refusal on some
# platform -- which is what CI's first Linux run reported.
#
# Taking the value from the oracle removes the intercept comparison, so the
# check that now stands behind it needs its own teeth proven. It already had
# them for *tree* defects -- neutering it turns the cleared-live-node and
# wrong-source-array tests red -- but nothing exercised it against an intercept
# error, which is precisely the failure it is now the last line against.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("direction", [np.inf, -np.inf])
def test_the_source_margin_check_fires_on_a_one_ulp_intercept(
    direction: float,
) -> None:
    """The end-to-end oracle detects a single-bit intercept error, in either
    direction. This is the protection that replaced the derived-versus-observed
    intercept comparison, and it is strictly stronger: it covers the trees and
    the accumulation as well, and its oracle is XGBoost's own ``predict`` on
    rows chosen to reach specific leaves rather than a second reading of the
    same zero-tree configuration."""
    booster, _x, names = _fit("binary:logistic", base_score=0.6)
    artifact = export.export_model(booster, feature_names=names)

    # The baseline must pass, or the negative result below proves nothing.
    export._verify_against_source_margin(booster, artifact)

    perturbed = dict(artifact)
    perturbed["intercept"] = float(
        np.nextafter(np.float32(artifact["intercept"]), np.float32(direction))
    )
    assert perturbed["intercept"] != artifact["intercept"], "nextafter did not move"

    with pytest.raises(errors.MarginMismatchError) as caught:
        export._verify_against_source_margin(booster, perturbed)
    assert caught.value.mismatches >= 1


@pytest.mark.parametrize("objective", ["binary:logistic", "survival:cox"])
@pytest.mark.parametrize("base_score", [0.9478001, 0.99, 0.999])
def test_export_succeeds_at_a_base_score_where_a_recipe_would_miss(
    objective: str, base_score: float
) -> None:
    """The regression test for the defect itself.

    These are values whose intercept passes through a logarithm, near 1 where
    the result is most sensitive to it. Before D053 the exporter derived the
    intercept and refused the model unless the derivation matched XGBoost
    bit-for-bit, so on a platform whose ``logf`` differs from numpy's this
    raised ``InterceptMismatchError`` on a perfectly ordinary model. Export must
    succeed, and the artifact must reproduce XGBoost's margin on every real row
    rather than merely on the leaf-reaching sample the exporter checks itself
    with.
    """
    booster, x, names = _fit(objective, base_score=base_score)
    artifact = export.export_model(booster, feature_names=names)

    matrix = xgb.DMatrix(x, feature_names=names)
    observed = np.asarray(booster.predict(matrix, output_margin=True), dtype=np.float32)
    trees = artifact["trees"]
    intercept = artifact["intercept"]

    mismatches = 0
    first: tuple[int, str, str] | None = None
    for index in range(len(x)):
        walked = walk_margin(trees, intercept, x[index])
        if int(walked.view(np.uint32)) != int(observed[index].view(np.uint32)):
            mismatches += 1
            if first is None:
                first = (
                    index,
                    f"0x{int(walked.view(np.uint32)):08X}",
                    f"0x{int(observed[index].view(np.uint32)):08X}",
                )
    assert mismatches == 0, (
        f"{objective} at base_score={base_score!r}: {mismatches}/{len(x)} rows "
        f"differ from XGBoost; first {first!r}"
    )
