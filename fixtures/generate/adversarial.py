"""The adversarial fixture generator.

Companion to ``fixtures/generate/corpus.py``, and deliberately built on top of
it rather than beside it: the fixture-level schema (D044 -- a uint32 hex bit
pattern of the float32 ground truth, a non-normative decimal alongside it, a
JSON ``null`` for a missing feature value) is reused unchanged, by importing
the small, application-agnostic helpers ``corpus.py`` already defines rather
than writing a second copy of them. Every value-producing fixture here still
carries XGBoost's own ``predict()`` output as ground truth -- nothing in this
module invents an expected value any way other than asking a real, freshly
fitted booster, with exactly one deliberate exception (the ``±inf`` refusal
rows of the non-finite-input fixture, which record no ground truth by
design, per CLAUDE.md).

Where this module differs from the ordinary corpus is purpose, not format:
these fixtures exist to break a plausible-but-wrong implementation, not to
demonstrate a working one. Ordinary random rows are close to useless for
that -- 0 of 20000 random continuous rows detect the float32 comparison
defect, while rows placed deliberately on a threshold detect it on 26 of 104
nodes (``probes/float32_thresholds.md`` section 7). Every builder below
exists because some specific invariant cannot be observed to fail without
it.

Written to ``fixtures/corpus/adversarial/``, a **separate directory** from
the ordinary corpus, precisely so this module never has to touch
``fixtures/generate/corpus.py`` or perturb ``fixtures/tests/test_corpus.py``'s
exact-fixture-list check -- that test globs ``fixtures/corpus/*.json``
non-recursively and never sees this directory.

One schema extension, made explicit here because it is not otherwise
documented: several fixtures below also carry ``meta.raw_node_values_per_tree``
-- the *raw*, un-narrowed float64 parse of each tree's ``split_conditions``
token, read directly from ``booster.save_raw(raw_format="json")`` before
export ever narrows anything, positionally aligned with the exported
artifact's own ``node_values``. This is not needed by any predictor and is
not part of FORMAT.md; it exists only so `fixtures/tests/test_adversarial.py`
can revert the float32 narrowing sites in isolation without silently
re-deriving a value the export path has already made exact. Using the
*exported* artifact's own (already narrowed-and-widened-back) value for this
purpose would prove nothing: D044's emission rule (``float(np.float32(x))``,
re-emitted at full float64 precision) makes that value's own float64 parse
recover the identical float32 bit-for-bit, by construction -- there is no
gap left to exploit. The gap exists only in XGBoost's own *short* decimal
token, which is why the raw token, not the artifact, is what the broken
variants need.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import xgboost as xgb

from generate.corpus import (
    _bits32,
    _decimal,
    _feature_names,
    _fit,
    _is_nan,
    _json_row,
    _logistic_labels,
    _parent_map,
    _path_to,
    _regression_labels,
    _resolved_xgboost_version,
    _sample_rows,
)
from xgboost_bridge.export import export_model
from xgboost_bridge.trees import reachable_nodes, walk_margin

ADVERSARIAL_DIR = Path(__file__).resolve().parents[1] / "corpus" / "adversarial"

__all__ = ["ADVERSARIAL_DIR", "build_all", "main"]


# ---------------------------------------------------------------------------
# Raw (pre-export, un-narrowed) access -- for verification only, never for a
# fixture's own ground truth.
# ---------------------------------------------------------------------------


def _raw_document(booster: xgb.Booster) -> dict[str, Any]:
    """The parsed ``booster.save_raw(raw_format="json")`` document, unmodified."""
    return json.loads(booster.save_raw(raw_format="json"))


def _raw_source_trees(document: dict[str, Any]) -> list[dict[str, Any]]:
    return document["learner"]["gradient_booster"]["model"]["trees"]


def _raw_node_values_per_tree(document: dict[str, Any]) -> list[list[float]]:
    """The float64 parse of ``split_conditions``, exactly as XGBoost wrote it.

    Positionally aligned with the exported artifact's ``node_values``:
    export changes no array length and renumbers no index (FORMAT.md section
    8.3), so index ``i`` here is index ``i`` there.
    """
    return [
        [float(value) for value in tree["split_conditions"]]
        for tree in _raw_source_trees(document)
    ]


def _hazardous_internal_nodes(
    trees: list[dict[str, Any]], raw_values_per_tree: list[list[float]]
) -> list[tuple[int, int]]:
    """``(tree_index, node_index)`` of every internal node where the raw

    float64 parse of the threshold token is strictly greater than its float32
    narrowing -- the direction ``probes/float32_thresholds.md`` section 7
    measured as hazardous for a one-sided cast that narrows the sample but
    not the threshold: ``float32(v) < RAW_float64`` is true (routes LEFT) at
    exactly ``v = float32(threshold)``, where the correct two-sided
    comparison is false (routes RIGHT).
    """
    hazardous: list[tuple[int, int]] = []
    for tree_index, (tree, raw_values) in enumerate(zip(trees, raw_values_per_tree, strict=True)):
        left_children = tree["left_children"]
        for node_index, left in enumerate(left_children):
            if left == -1:
                continue
            raw = raw_values[node_index]
            narrowed = float(np.float32(raw))
            if raw > narrowed:
                hazardous.append((tree_index, node_index))
    return hazardous


def _resolve_row_to_value(
    tree: dict[str, Any], target: int, num_feature: int, value: float
) -> list[float] | None:
    """A feature row that reaches ``target`` with ``value`` at its own feature.

    Generalizes ``corpus.py``'s ``_resolve_row`` (which always places ``NaN``
    at the target, for the missing-value fixture) to an arbitrary value --
    the exact threshold, or one of its float32 neighbours. Each ancestor on
    the root-to-target path gets a value driving the walk in the needed
    direction; if an ancestor shares the target's own split feature, the
    fixed ``value`` must itself already take that direction, or the row is
    infeasible and this returns ``None``.
    """
    path = _path_to(tree, target)
    target_feature = tree["split_indices"][target]
    row = [0.0] * num_feature
    assigned: dict[int, float] = {}

    for ancestor, direction in path:
        feature = tree["split_indices"][ancestor]
        threshold = tree["node_values"][ancestor]

        if feature == target_feature:
            takes_left = np.float32(value) < np.float32(threshold)
            needed = "left" if takes_left else "right"
            if needed != direction:
                return None
            continue

        candidate = threshold if direction == "right" else threshold - abs(threshold) - 1.0
        if feature in assigned:
            existing = assigned[feature]
            takes_left = np.float32(existing) < np.float32(threshold)
            if takes_left != (direction == "left"):
                return None
        else:
            assigned[feature] = candidate
            row[feature] = candidate

    row[target_feature] = float(value)
    return row


def _internal_nodes(trees: list[dict[str, Any]]) -> list[tuple[int, int]]:
    return [
        (tree_index, node_index)
        for tree_index, tree in enumerate(trees)
        for node_index, left in enumerate(tree["left_children"])
        if left != -1
    ]


# ---------------------------------------------------------------------------
# Fixture writers. Two shapes: value-producing (ground truth from XGBoost's
# own predict()) and refusal-only (no ground truth, by design -- A6).
# ---------------------------------------------------------------------------


def _ground_truth(
    booster: xgb.Booster, rows: list[list[float]], feature_names: list[str]
) -> tuple[list[str], list[str], list[float | str], list[float | str]]:
    """Identical in spirit to ``corpus.py``'s helper of the same name: ask
    XGBoost, and only XGBoost, what the margin and output are."""
    matrix = xgb.DMatrix(np.asarray(rows, dtype=np.float64), feature_names=feature_names, nthread=1)
    margin = np.asarray(booster.predict(matrix, output_margin=True), dtype=np.float32)
    output = np.asarray(booster.predict(matrix), dtype=np.float32)
    margin_bits = [_bits32(value) for value in margin]
    output_bits = [_bits32(value) for value in output]
    margin_decimal = [_decimal(value) for value in margin]
    output_decimal = [_decimal(value) for value in output]
    return margin_bits, output_bits, margin_decimal, output_decimal


def _write_adversarial_fixture(
    name: str,
    *,
    booster: xgb.Booster,
    rows: list[list[float]],
    feature_names: list[str],
    description: str,
    seed: int,
    base_score: float | None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one value-producing fixture, self-check it, and write it.

    Mirrors ``corpus.py``'s ``_write_fixture`` (same generation-time
    self-check: this repository's own ``walk_margin`` must reproduce
    XGBoost's observed margin bit-for-bit before anything is written), with
    one addition: ``meta.raw_node_values_per_tree`` records the un-narrowed
    float64 parse of every ``split_conditions`` token, for
    ``test_adversarial.py``'s broken-variant verification. See this module's
    own docstring for why the *raw* token, not the exported artifact's own
    value, is what that verification needs.
    """
    artifact = export_model(booster)
    margin_bits, output_bits, margin_decimal, output_decimal = _ground_truth(
        booster, rows, feature_names
    )

    intercept = artifact["intercept"]
    trees = artifact["trees"]
    for row, expected_bits in zip(rows, margin_bits):
        computed_bits = _bits32(walk_margin(trees, intercept, row))
        if computed_bits != expected_bits:
            raise AssertionError(
                f"adversarial fixture {name!r}: walk_margin disagrees with XGBoost's own "
                f"margin at generation time (row={row!r}, xgboost={expected_bits}, "
                f"walk_margin={computed_bits}); this is a defect to report, not to fix "
                "by adjusting the expected value"
            )

    document = _raw_document(booster)
    raw_node_values_per_tree = _raw_node_values_per_tree(document)

    meta: dict[str, Any] = {
        "name": name,
        "description": description,
        "objective": artifact["objective"],
        "base_score": base_score,
        "seed": seed,
        "row_count": len(rows),
        "xgboost_version": _resolved_xgboost_version(booster),
        "numpy_version": np.__version__,
        "decimal_fields_are_non_normative": True,
        "nan_encoding": (
            "A JSON null in `rows` denotes a missing (NaN) feature value. A "
            "non-finite margin_decimal/output_decimal entry is the string "
            "'inf', '-inf', or 'nan' rather than a bare JSON token; the bit "
            "pattern in expected_margin/expected_output is normative in "
            "every case."
        ),
        "raw_node_values_per_tree": raw_node_values_per_tree,
        "raw_node_values_note": (
            "The float64 parse of learner.gradient_booster.model.trees[i]."
            "split_conditions, read directly from booster.save_raw() before "
            "export narrows anything, aligned positionally with this "
            "fixture's own artifact.trees[i].node_values. Not part of "
            "FORMAT.md and not consumed by any predictor -- present only so "
            "a verification harness can revert a float32 narrowing site "
            "without silently re-deriving a value export has already made "
            "exact. See fixtures/generate/adversarial.py."
        ),
    }
    if extra_meta:
        meta.update(extra_meta)

    fixture = {
        "artifact": artifact,
        "rows": [_json_row(row) for row in rows],
        "expected_margin": margin_bits,
        "expected_output": output_bits,
        "margin_decimal": margin_decimal,
        "output_decimal": output_decimal,
        "meta": meta,
    }

    ADVERSARIAL_DIR.mkdir(parents=True, exist_ok=True)
    path = ADVERSARIAL_DIR / f"{name}.json"
    path.write_text(
        json.dumps(fixture, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return fixture


def _json_row_with_non_finite(row: Sequence[float]) -> list[float | str | None]:
    """Like ``corpus.py``'s ``_json_row``, extended for a row that may carry
    ``+inf``/``-inf`` as well as ``NaN``.

    ``NaN`` -> JSON ``null`` (D044's existing convention: the missing value).
    ``+inf``/``-inf`` -> the strings ``"inf"``/``"-inf"``, by direct analogy
    with D044's own non-finite *decimal* convention -- a bare ``Infinity``
    token is non-standard JSON and Python's own ``json`` module would emit
    exactly that if asked to serialize a raw ``float('inf')``. This encoding
    is used **only** by the refusal-only fixture below, is recorded
    explicitly in that fixture's own ``meta``, and is never mixed with the
    ordinary value-producing row encoding.
    """
    encoded: list[float | str | None] = []
    for value in row:
        if _is_nan(value):
            encoded.append(None)
        elif math.isinf(value):
            encoded.append("inf" if value > 0 else "-inf")
        else:
            encoded.append(float(value))
    return encoded


def _write_refusal_fixture(
    name: str,
    *,
    booster: xgb.Booster,
    rows: list[list[float]],
    feature_names: list[str],
    description: str,
    seed: int,
    base_score: float | None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the ``±inf`` refusal fixture. No ground truth is recorded.

    Per D022, ``±inf`` in prediction input is a refusal case for the
    predictor -- not a value to be computed. This writer therefore never
    asks XGBoost's own ``predict()`` for these rows (upstream's own handling
    of ``±inf`` is itself inconsistent between call paths, per D022, so it is
    not an oracle worth asking here) and records ``null`` in every
    ground-truth slot rather than a bit pattern, keeping the array-length
    invariant (``len(expected_margin) == len(rows)``) intact without
    pretending a value exists.
    """
    artifact = export_model(booster)

    meta: dict[str, Any] = {
        "name": name,
        "description": description,
        "objective": artifact["objective"],
        "base_score": base_score,
        "seed": seed,
        "row_count": len(rows),
        "xgboost_version": _resolved_xgboost_version(booster),
        "numpy_version": np.__version__,
        "decimal_fields_are_non_normative": True,
        "nan_encoding": (
            "A JSON null in `rows` denotes a missing (NaN) feature value; "
            "not exercised in this fixture (see non_finite_row_encoding "
            "below for what is)."
        ),
        "non_finite_row_encoding": (
            "This fixture's rows use the JSON strings 'inf' and '-inf' to "
            "denote +inf and -inf feature values -- an input-side extension "
            "of D044's existing non-finite *decimal* convention, needed "
            "because standard JSON has no Infinity literal. This encoding "
            "is unique to this refusal-only fixture; ordinary value-"
            "producing fixtures never use it."
        ),
        "ground_truth": "none",
        "expected_behavior": "raise",
        "expected_behavior_reason": (
            "D022: non-finite (+/-inf) feature values are a refusal case "
            "for the predictor, in both languages. Upstream XGBoost is "
            "itself inconsistent here (raises through DMatrix, does not "
            "raise through inplace_predict), so XGBoost's own predict() is "
            "not an independent oracle for this case and is deliberately "
            "not consulted. No numeric ground truth is recorded, by design "
            "(CLAUDE.md); expected_margin and expected_output are null for "
            "every row."
        ),
    }
    if extra_meta:
        meta.update(extra_meta)

    fixture = {
        "artifact": artifact,
        "rows": [_json_row_with_non_finite(row) for row in rows],
        "expected_margin": [None] * len(rows),
        "expected_output": [None] * len(rows),
        "margin_decimal": [None] * len(rows),
        "output_decimal": [None] * len(rows),
        "meta": meta,
    }

    ADVERSARIAL_DIR.mkdir(parents=True, exist_ok=True)
    path = ADVERSARIAL_DIR / f"{name}.json"
    path.write_text(
        json.dumps(fixture, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return fixture


# ---------------------------------------------------------------------------
# A1 / A2 share one fitted model: a real, moderately deep ensemble gives
# enough internal nodes for both the hazardous-threshold search (A1) and the
# equality-boundary spot checks (A2).
# ---------------------------------------------------------------------------


def _threshold_stress_model(seed: int) -> tuple[xgb.Booster, list[str], int]:
    num_feature = 6
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(600, num_feature))
    labels = _regression_labels(rng, features)
    feature_names = _feature_names(num_feature)
    booster = _fit(
        "reg:squarederror", features, labels, feature_names,
        seed=seed, num_boost_round=10, max_depth=4, eta=0.3, tree_method="hist",
    )
    return booster, feature_names, num_feature


def _build_float32_threshold_disagreement() -> dict[str, Any]:
    """A1: a row at ``float32(threshold)`` for every measured-hazardous node.

    Hazardous means the raw float64 parse of the threshold token is strictly
    greater than its float32 narrowing -- the direction
    ``probes/float32_thresholds.md`` section 7 measured as the one that a
    one-sided cast (sample narrowed, threshold not) gets wrong, routing LEFT
    where the correct two-sided comparison routes RIGHT. Selection is by
    measuring that gap directly on this fixture's own fitted model, never by
    assuming a fraction from the probe.
    """
    seed = 901
    booster, feature_names, num_feature = _threshold_stress_model(seed)
    artifact = export_model(booster)
    trees = artifact["trees"]

    document = _raw_document(booster)
    raw_values_per_tree = _raw_node_values_per_tree(document)
    hazardous = _hazardous_internal_nodes(trees, raw_values_per_tree)

    rows: list[list[float]] = []
    targets: list[dict[str, int]] = []
    for tree_index, node_index in hazardous:
        threshold = trees[tree_index]["node_values"][node_index]
        row = _resolve_row_to_value(trees[tree_index], node_index, num_feature, threshold)
        if row is None:
            continue
        rows.append(row)
        targets.append({"tree_index": tree_index, "node_index": node_index})

    total_internal = len(_internal_nodes(trees))
    if not rows:
        raise AssertionError(
            "float32_threshold_disagreement: no hazardous internal node resolved to a "
            f"row (found {len(hazardous)} hazardous of {total_internal} internal nodes); "
            "widen the underlying model"
        )

    return _write_adversarial_fixture(
        "float32_threshold_disagreement",
        booster=booster, rows=rows, feature_names=feature_names,
        description=(
            "reg:squarederror; each row's feature value is exactly float32(threshold) "
            "at an internal node measured (via the raw, un-narrowed split_conditions "
            "token) to be hazardous for a one-sided cast -- the raw float64 parse of "
            "the threshold is strictly greater than its float32 narrowing. A correct "
            "two-sided float32 comparison routes every one of these rows RIGHT; "
            "narrowing only the sample value routes it LEFT."
        ),
        seed=seed, base_score=None,
        extra_meta={
            "hazardous_node_count": len(hazardous),
            "total_internal_node_count": total_internal,
            "resolved_row_count": len(rows),
            "targets": targets,
        },
    )


def _build_equality_boundary_routing() -> dict[str, Any]:
    """A2: exact-threshold rows plus their float32 neighbours, at several nodes.

    For each of several internal nodes: the exact float32(threshold) (must
    route RIGHT), the float32 value immediately below it via
    ``np.nextafter`` (must route LEFT), and the float32 value immediately
    above it (must route RIGHT) -- pinning the strict-`<`-with-equality-
    right rule across the boundary rather than at one point
    (``probes/float32_thresholds.md`` section 4).
    """
    seed = 901  # same fitted model as A1; deliberately reused, not re-derived
    booster, feature_names, num_feature = _threshold_stress_model(seed)
    artifact = export_model(booster)
    trees = artifact["trees"]

    candidates = _internal_nodes(trees)
    # A deterministic, spread-out subset -- every 5th internal node, capped --
    # rather than the first few, so the sample is not concentrated in one tree.
    chosen = candidates[::5][:10]

    rows: list[list[float]] = []
    row_labels: list[str] = []
    node_records: list[dict[str, Any]] = []
    for tree_index, node_index in chosen:
        tree = trees[tree_index]
        threshold32 = np.float32(tree["node_values"][node_index])
        below = np.nextafter(threshold32, np.float32(-np.inf))
        above = np.nextafter(threshold32, np.float32(np.inf))

        triple: dict[str, str] = {}
        ok = True
        for label, value in (("below", below), ("exact", threshold32), ("above", above)):
            row = _resolve_row_to_value(tree, node_index, num_feature, float(value))
            if row is None:
                ok = False
                break
            rows.append(row)
            row_labels.append(f"{tree_index}:{node_index}:{label}")
            triple[label] = _bits32(float(value))
        if not ok:
            # Roll back the partial triple; an infeasible node contributes nothing
            # rather than a mismatched row/label count.
            while row_labels and row_labels[-1].startswith(f"{tree_index}:{node_index}:"):
                row_labels.pop()
                rows.pop()
            continue
        node_records.append(
            {"tree_index": tree_index, "node_index": node_index, "bits": triple}
        )

    if not node_records:
        raise AssertionError("equality_boundary_routing: no candidate node resolved to a row")

    return _write_adversarial_fixture(
        "equality_boundary_routing",
        booster=booster, rows=rows, feature_names=feature_names,
        description=(
            "reg:squarederror (same fitted model as float32_threshold_disagreement). "
            "For each of several internal nodes, three consecutive rows: the float32 "
            "value immediately below the threshold (must route LEFT), the threshold "
            "itself (must route RIGHT -- equality routes right), and the float32 value "
            "immediately above (must route RIGHT). row_labels in meta identify which "
            "row is which."
        ),
        seed=seed, base_score=None,
        extra_meta={"row_labels": row_labels, "node_records": node_records},
    )


# ---------------------------------------------------------------------------
# A3: denormal and extreme feature values.
# ---------------------------------------------------------------------------


def _build_extreme_and_denormal_features() -> dict[str, Any]:
    """A3: subnormals, near-max-float32 magnitudes, signed zero, and integers
    large enough to lose float32 precision -- fit on data whose own columns
    span those scales, so the model's thresholds are themselves extreme."""
    seed = 902
    num_feature = 5
    rng = np.random.default_rng(seed)
    col_large = rng.uniform(-1e30, 1e30, size=600)
    col_small = rng.uniform(-1e-30, 1e-30, size=600)
    col_tiny = rng.uniform(-1e-40, 1e-40, size=600)
    col_integer = rng.integers(-100_000_000, 100_000_000, size=600).astype(np.float64)
    col_ordinary = rng.uniform(-5.0, 5.0, size=600)
    features = np.column_stack([col_large, col_small, col_tiny, col_integer, col_ordinary])
    feature_names = _feature_names(num_feature)

    # Coefficients rescaled so every column contributes an O(1) signal despite
    # spanning wildly different magnitudes -- otherwise the extreme-scale
    # columns either swamp the label or never influence a split at all.
    labels = (
        1e-30 * col_large
        + 1e30 * col_small
        + 1e40 * col_tiny
        + 1e-8 * col_integer
        + col_ordinary
        + rng.normal(scale=0.5, size=600)
    )
    booster = _fit(
        "reg:squarederror", features, labels, feature_names,
        seed=seed, num_boost_round=8, max_depth=3, eta=0.3, tree_method="hist",
    )

    smallest_subnormal = float(np.float32(1.4e-45))
    negative_subnormal = float(np.float32(-5.6e-44))
    near_max = float(np.float32(3.4e38))
    first_lossy_integer = 16_777_217.0  # 2**24 + 1: the first integer float32 cannot represent exactly
    moderate = [0.0, 0.0, 0.0, 0.0, 1.5]

    rows = [
        [smallest_subnormal, 0.0, 0.0, 0.0, 1.0],
        [negative_subnormal, 0.0, 0.0, 0.0, -1.0],
        [near_max, 0.0, 0.0, 0.0, 2.0],
        [-near_max, 0.0, 0.0, 0.0, -2.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.0, -0.0, -0.0, -0.0, -0.0],
        [0.0, 0.0, 0.0, first_lossy_integer, 0.5],
        [0.0, 0.0, 0.0, -first_lossy_integer, -0.5],
        [0.0, smallest_subnormal, negative_subnormal, 0.0, 0.0],
        moderate,
    ]

    return _write_adversarial_fixture(
        "extreme_and_denormal_features",
        booster=booster, rows=rows, feature_names=feature_names,
        description=(
            "reg:squarederror fit on columns spanning float32's full dynamic range "
            "(~1e30, ~1e-30, ~1e-40, large integers, and an ordinary-scale column), so "
            "the model's own thresholds are themselves extreme. Rows exercise float32 "
            "subnormals (1.4e-45, -5.6e-44), near-max magnitude (+/-3.4e38), signed "
            "zero (0.0 and -0.0), and the first integer float32 cannot represent "
            "exactly (2**24 + 1 = 16777217)."
        ),
        seed=seed, base_score=None,
    )


# ---------------------------------------------------------------------------
# A4: the logistic clamp floor, at the output level.
# ---------------------------------------------------------------------------


def _build_logistic_clamp_floor_output() -> dict[str, Any]:
    """A4: rows whose margin falls below f32(-88.7), XGBoost's measured
    logistic clamp floor (FORMAT.md section 5.2), where predict() returns
    exactly 3.006635794144578e-39 and never 0.0.

    A large, aggressively overfit ensemble (eta=1.0, many rounds) genuinely
    reaches margins in the hundreds in either direction on its own training
    rows -- no synthetic row construction needed; XGBoost's own predict()
    on real fitted rows already crosses the floor.
    """
    seed = 903
    num_feature = 6
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(2500, num_feature))
    labels = _logistic_labels(rng, features)
    feature_names = _feature_names(num_feature)
    booster = _fit(
        "binary:logistic", features, labels, feature_names,
        seed=seed, base_score=0.987654, num_boost_round=300, max_depth=4, eta=1.0,
        tree_method="hist",
    )

    matrix = xgb.DMatrix(features, feature_names=feature_names, nthread=1)
    margin = np.asarray(booster.predict(matrix, output_margin=True), dtype=np.float32)
    floor = np.float32(-88.7)

    order = np.argsort(margin)
    lowest_indices = order[:15].tolist()
    highest_indices = order[-15:].tolist()
    rng2 = np.random.default_rng(seed + 1)
    moderate_indices = rng2.choice(len(margin), size=10, replace=False).tolist()

    chosen_indices = sorted(set(lowest_indices + highest_indices + moderate_indices))
    rows = [features[index].tolist() for index in chosen_indices]

    below_clamp_count = int(np.sum(margin[chosen_indices] < floor))
    lowest_margin_overall = float(margin.min())

    return _write_adversarial_fixture(
        "logistic_clamp_floor_output",
        booster=booster, rows=rows, feature_names=feature_names,
        description=(
            "binary:logistic, eta=1.0, 300 rounds, base_score=0.987654: an "
            "aggressively overfit ensemble whose own training rows already reach "
            "margins in the hundreds. Rows are the 15 lowest-margin, 15 "
            "highest-margin, and 10 moderate training rows, so both the logistic "
            "clamp floor (margin below f32(-88.7), output exactly "
            "3.006635794144578e-39) and its saturated-at-1.0 counterpart are "
            "represented alongside ordinary rows."
        ),
        seed=seed, base_score=0.987654,
        extra_meta={
            "logistic_clamp_floor_margin": _decimal(floor),
            "below_clamp_row_count": below_clamp_count,
            "lowest_margin_in_fixture": _decimal(float(margin[chosen_indices].min())),
            "lowest_margin_achieved_overall": _decimal(lowest_margin_overall),
        },
    )


# ---------------------------------------------------------------------------
# A5: neutralization detection.
# ---------------------------------------------------------------------------


def _build_gamma_pruned_neutralization() -> dict[str, Any]:
    """A5: a gamma-pruned model (tree_method=exact) with genuinely dead
    nodes, recording the neutralized indices in meta.

    Two independent computations of "dead", confirmed to agree: reachability
    from the root on the *exported* (already-neutralized) tree, and the raw
    ``split_indices == 2147483647`` marker read directly from the *source*
    document, before export ever touches it. ``tree_method="hist"`` never
    produces a dead node at any gamma -- it declines to grow a losing split
    rather than growing and pruning one (CLAUDE.md; measured in
    `probes/tree_structure.md`).
    """
    seed = 904
    num_feature = 6
    rng = np.random.default_rng(seed)
    features = rng.uniform(-3.0, 3.0, size=(500, num_feature))
    labels = _regression_labels(rng, features)
    feature_names = _feature_names(num_feature)
    booster = _fit(
        "reg:squarederror", features, labels, feature_names,
        seed=seed, num_boost_round=3, max_depth=5, eta=0.3,
        extra_params={"gamma": 50.0, "tree_method": "exact"},
    )

    document = _raw_document(booster)
    source_trees = _raw_source_trees(document)
    raw_deleted_marker_indices_per_tree = [
        [index for index, marker in enumerate(tree["split_indices"]) if marker == 2147483647]
        for tree in source_trees
    ]

    artifact = export_model(booster)
    dead_node_indices_per_tree = [
        sorted(set(range(len(tree["left_children"]))) - reachable_nodes(tree))
        for tree in artifact["trees"]
    ]

    rows = _sample_rows(np.random.default_rng(seed + 1), num_feature, 8, low=-3.0, high=3.0)
    return _write_adversarial_fixture(
        "gamma_pruned_neutralization",
        booster=booster, rows=rows, feature_names=feature_names,
        description=(
            "reg:squarederror, tree_method=exact, gamma=50.0: a heavily-pruned "
            "ensemble with genuinely unreachable (neutralized) nodes. "
            "meta.dead_node_indices_per_tree records the nodes unreachable from "
            "root 0 in the exported artifact; meta.raw_deleted_marker_indices_per_tree "
            "records, independently, the nodes XGBoost's own source model marks "
            "split_indices == 2147483647, read before export narrows or neutralizes "
            "anything. A reader with a wrong reachability walk would route some row "
            "into one of these slots and disagree with XGBoost's own margin."
        ),
        seed=seed, base_score=None,
        extra_meta={
            "dead_node_indices_per_tree": dead_node_indices_per_tree,
            "raw_deleted_marker_indices_per_tree": raw_deleted_marker_indices_per_tree,
        },
    )


# ---------------------------------------------------------------------------
# A6a: NaN in feature positions -- value-producing, routes by default_left.
# ---------------------------------------------------------------------------


def _build_missing_value_adversarial() -> dict[str, Any]:
    """A6a: aggressive NaN placement -- every feature missing at once, and a
    row half-missing -- alongside two rows constructed (as in the ordinary
    corpus's own missing-value fixture) to hit a chosen default_left=1 node
    and a chosen default_left=0 node deliberately.
    """
    seed = 905
    num_feature = 6
    rng = np.random.default_rng(seed)
    features = rng.uniform(-3.0, 3.0, size=(500, num_feature))
    missing_mask = rng.random(features.shape) < 0.3
    features_with_missing = features.copy()
    features_with_missing[missing_mask] = np.nan
    labels = _regression_labels(rng, features)
    feature_names = _feature_names(num_feature)
    booster = _fit(
        "reg:squarederror", features_with_missing, labels, feature_names,
        seed=seed, num_boost_round=10, max_depth=4, eta=0.3,
    )
    artifact = export_model(booster)
    trees = artifact["trees"]

    left_target, left_row = _resolve_default_left_row(trees, want_default_left=1, num_feature=num_feature)
    right_target, right_row = _resolve_default_left_row(trees, want_default_left=0, num_feature=num_feature)

    all_missing_row = [float("nan")] * num_feature
    half_missing_row = [float("nan") if index % 2 == 0 else 1.5 for index in range(num_feature)]
    ordinary_rows = _sample_rows(np.random.default_rng(seed + 1), num_feature, 3, low=-3.0, high=3.0)

    rows = [left_row, right_row, all_missing_row, half_missing_row, *ordinary_rows]

    return _write_adversarial_fixture(
        "missing_value_adversarial",
        booster=booster, rows=rows, feature_names=feature_names,
        description=(
            "reg:squarederror fit with 30% missing training values. Row 0 carries a "
            "NaN at a node whose default_left is 1; row 1 at a node whose "
            "default_left is 0. Row 2 has every feature missing (NaN routes the walk "
            "through every tree by default_left alone). Row 3 has every other "
            "feature missing. Rows 4-6 are ordinary, fully-populated rows."
        ),
        seed=seed, base_score=None,
        extra_meta={
            "default_left_1_target": {"tree_index": left_target[0], "node_index": left_target[1]},
            "default_left_0_target": {"tree_index": right_target[0], "node_index": right_target[1]},
        },
    )


def _resolve_default_left_row(
    trees: list[dict[str, Any]], want_default_left: int, num_feature: int
) -> tuple[tuple[int, int], list[float]]:
    """The first internal, non-root node with the wanted ``default_left``,
    resolved to a row carrying NaN there. Local re-implementation of
    ``corpus.py``'s ``_find_missing_value_target``/``_resolve_row`` pair,
    generalized on top of this module's own ``_resolve_row_to_value`` -- NaN
    is simply the value placed at the target feature; direction consistency
    for a shared-feature ancestor is checked via ``default_left`` rather than
    via a threshold comparison, since NaN never compares less than anything.
    """
    for tree_index, tree in enumerate(trees):
        left = tree["left_children"]
        for node_index in range(len(left)):
            if node_index == 0 or left[node_index] == -1:
                continue
            if tree["default_left"][node_index] != want_default_left:
                continue
            row = _resolve_missing_row(tree, node_index, num_feature)
            if row is not None:
                return (tree_index, node_index), row
    raise AssertionError(
        f"no internal, non-root node with default_left={want_default_left} could be "
        "resolved to a row"
    )


def _resolve_missing_row(tree: dict[str, Any], target: int, num_feature: int) -> list[float] | None:
    path = _path_to(tree, target)
    target_feature = tree["split_indices"][target]
    row = [0.0] * num_feature
    assigned: dict[int, float] = {}

    for ancestor, direction in path:
        feature = tree["split_indices"][ancestor]
        threshold = tree["node_values"][ancestor]

        if feature == target_feature:
            needed = "left" if tree["default_left"][ancestor] == 1 else "right"
            if needed != direction:
                return None
            continue

        candidate = threshold if direction == "right" else threshold - abs(threshold) - 1.0
        if feature in assigned:
            existing = assigned[feature]
            takes_left = np.float32(existing) < np.float32(threshold)
            if takes_left != (direction == "left"):
                return None
        else:
            assigned[feature] = candidate
            row[feature] = candidate

    row[target_feature] = float("nan")
    return row


# ---------------------------------------------------------------------------
# A6b: +/-inf in feature positions -- refusal only, no ground truth.
# ---------------------------------------------------------------------------


def _build_non_finite_input_refusal() -> dict[str, Any]:
    """A6b: +inf and -inf at every feature position, kept in a fixture of
    its own so a consumer cannot mistake a refusal row for a value-producing
    one. Per D022, non-finite input is a refusal case for the predictor;
    this fixture's own meta says so explicitly and no ground truth is
    computed (see ``_write_refusal_fixture``)."""
    seed = 906
    num_feature = 4
    feature_names = _feature_names(num_feature)
    rng = np.random.default_rng(seed)
    features = rng.uniform(-3.0, 3.0, size=(300, num_feature))
    labels = _regression_labels(rng, features)
    booster = _fit(
        "reg:squarederror", features, labels, feature_names,
        seed=seed, num_boost_round=8, max_depth=3, eta=0.3,
    )

    rows: list[list[float]] = []
    for position in range(num_feature):
        for sign in (1.0, -1.0):
            row = [0.5] * num_feature
            row[position] = math.inf * sign
            rows.append(row)
    rows.append([math.inf, -math.inf, 0.5, 0.5])
    rows.append([-math.inf, -math.inf, math.inf, math.inf])

    return _write_refusal_fixture(
        "non_finite_input_refusal",
        booster=booster, rows=rows, feature_names=feature_names,
        description=(
            "reg:squarederror. Every row carries +inf or -inf at one or more "
            "feature positions, scanning each column individually and then two "
            "rows with multiple non-finite entries at once. No expected_margin or "
            "expected_output is recorded: per D022, +/-inf input is a refusal case "
            "for the predictor, in both languages, and this fixture exists to pin "
            "that -- not to report a value."
        ),
        seed=seed, base_score=None,
    )


# ---------------------------------------------------------------------------
# A7: a single-node (root-is-leaf) tree, and every leaf exactly 0.0.
# ---------------------------------------------------------------------------


def _build_zero_leaf_intercept_isolation() -> dict[str, Any]:
    """A7: max_depth=0 forces every tree to be a single node whose root is a
    leaf; labels set to XGBoost's own zero-round prediction at this
    base_score make every leaf's gradient (and therefore weight) exactly
    zero. The margin on every row is then the intercept and nothing else --
    any defect in the intercept derivation is undiluted by any tree.

    binary:logistic is used deliberately rather than reg:squarederror: it is
    the objective whose intercept derivation is most delicate (the clamp-
    then-float32-log recipe of D035/D039/D040), so isolating it is worth
    more here than isolating the trivial identity transform would be.
    """
    seed = 907
    num_feature = 3
    base_score = 0.1234567  # away from 0.5 (D025's degenerate trap) and from the clamp bounds
    feature_names = _feature_names(num_feature)
    rng = np.random.default_rng(seed)
    features = rng.uniform(-3.0, 3.0, size=(80, num_feature))

    # A zero-round model at this base_score, to read XGBoost's own forward-pass
    # probability exactly -- using that (rather than the mathematical base_score
    # itself) as every label is what drives the gradient, and therefore every
    # leaf weight, to exactly zero rather than merely close to it (measured: a
    # naive base_score label leaves leaves at ~1.7e-8, not 0.0).
    zero_round_matrix = xgb.DMatrix(
        features, label=np.zeros(features.shape[0]), feature_names=feature_names, nthread=1
    )
    zero_round = xgb.train(
        {
            "objective": "binary:logistic", "max_depth": 0, "eta": 0.3,
            "nthread": 1, "seed": seed, "base_score": base_score,
        },
        zero_round_matrix, num_boost_round=0,
    )
    self_consistent_label = float(zero_round.predict(zero_round_matrix)[0])
    labels = np.full(features.shape[0], self_consistent_label)

    booster = _fit(
        "binary:logistic", features, labels, feature_names,
        seed=seed, base_score=base_score, num_boost_round=6, max_depth=0, eta=0.3,
    )

    artifact = export_model(booster)
    for tree_index, tree in enumerate(artifact["trees"]):
        if tree["left_children"] != [-1]:
            raise AssertionError(
                f"zero_leaf_intercept_isolation: tree {tree_index} is not a single "
                f"leaf node (left_children={tree['left_children']!r})"
            )
        leaf_value = tree["node_values"][0]
        if leaf_value != 0.0:
            raise AssertionError(
                f"zero_leaf_intercept_isolation: tree {tree_index}'s leaf is "
                f"{leaf_value!r}, not exactly 0.0"
            )

    ordinary_rows = _sample_rows(np.random.default_rng(seed + 1), num_feature, 4)
    nan_row = [float("nan"), 0.0, 0.0]
    extreme_row = [float(np.float32(3.4e38)), 0.0, 0.0]
    rows = [*ordinary_rows, nan_row, extreme_row]

    return _write_adversarial_fixture(
        "zero_leaf_intercept_isolation",
        booster=booster, rows=rows, feature_names=feature_names,
        description=(
            "binary:logistic, max_depth=0 (every tree is a single node whose root "
            "is a leaf), labels set to XGBoost's own zero-round prediction at this "
            "base_score so every leaf's weight is exactly 0.0. The margin on every "
            "row -- including a row with a NaN feature and one with an extreme "
            "feature value, since there is no split to reach either way -- is "
            "exactly the intercept, undiluted by any tree contribution."
        ),
        seed=seed, base_score=base_score,
        extra_meta={"self_consistent_label": self_consistent_label},
    )


_BUILDERS = (
    _build_float32_threshold_disagreement,
    _build_equality_boundary_routing,
    _build_extreme_and_denormal_features,
    _build_logistic_clamp_floor_output,
    _build_gamma_pruned_neutralization,
    _build_missing_value_adversarial,
    _build_non_finite_input_refusal,
    _build_zero_leaf_intercept_isolation,
)

#: Every fixture name this module writes, derived from the builders above so
#: there is one authority for "which fixtures exist" (mirrors ``corpus.py``'s
#: ``FIXTURE_NAMES``).
FIXTURE_NAMES: tuple[str, ...] = tuple(
    builder.__name__.removeprefix("_build_") for builder in _BUILDERS
)

#: The one fixture with no XGBoost ground truth, by design (D022 refusal
#: case). Every other fixture in ``FIXTURE_NAMES`` carries a real bit
#: pattern in every row of ``expected_margin``/``expected_output``.
REFUSAL_ONLY_FIXTURE = "non_finite_input_refusal"


def build_all() -> list[dict[str, Any]]:
    """Build and write every adversarial fixture, returning them in order."""
    return [builder() for builder in _BUILDERS]


def main() -> None:
    fixtures = build_all()
    print(f"wrote {len(fixtures)} adversarial fixtures to {ADVERSARIAL_DIR}")
    for fixture in fixtures:
        meta = fixture["meta"]
        print(f"  {meta['name']}: {meta['row_count']} rows, objective={meta['objective']!r}")


if __name__ == "__main__":
    main()
