"""The fixture corpus generator.

Every fixture this module writes carries **XGBoost's own `predict()` output**
as ground truth (`expected_margin`, `expected_output`), so a JavaScript
implementation can be verified without ever installing XGBoost. A fixture
without recorded ground truth is not a fixture -- nothing here computes an
"expected" value any way other than asking a real, freshly fitted booster.

Two encoding decisions, both because JSON has no lossless representation for
the values this project's own invariants make routine:

* **Ground truth is a uint32 hex bit pattern of the float32 value**
  (``"0x3f800000"``), never a JSON number. `survival:cox` genuinely returns
  `+inf` above a margin of about 88.72, JSON has no `Infinity` literal, and a
  decimal ground truth invites `==` comparison, under which `-0.0 == 0.0` is
  `True` for two artifacts that are not the same. `margin_decimal` and
  `output_decimal` are carried alongside for a human to skim, are explicitly
  **non-normative** (recorded as such in ``meta``), and are never the
  comparison a consumer should perform. A non-finite decimal is rendered as
  the string `"inf"` / `"-inf"` / `"nan"` rather than a bare JSON token, so
  the file stays ordinary, standards-conformant JSON throughout.
* **A missing feature value in `rows` is JSON `null`**, converted to `NaN`
  before it reaches a predictor. `NaN` has no JSON literal either, and unlike
  the ground truth fields `rows` is an *input* a future JavaScript reader
  must be able to `JSON.parse` without a custom grammar.

Every model here is fit with `nthread=1` and a fixed integer seed, and
`export_model`'s own output is a pure function of the fitted booster, so
regenerating the corpus is required to be byte-identical -- verified by
running this module twice and diffing file hashes, not asserted and hoped.

Each fixture also carries a defensive, generation-time self-check: this
module's own `walk_margin` call must reproduce XGBoost's observed margin
bit-for-bit before a file is written at all. `fixtures/tests/test_corpus.py`
performs the same check independently, against the file already on disk;
the one here exists so a broken fixture is never written in the first
place, and a disagreement is reported rather than silently baked into the
corpus.

Corpus composition is a correctness requirement, not coverage bookkeeping
(CLAUDE.md): at `base_score = 0.5` every broken logistic intercept variant
still scores 5000/5000, and Cox's estimated default collapses the intercept
to exactly `0.0`, at which point intercept placement stops mattering at all.
Every fixture below was chosen because some specific invariant cannot be
observed to fail without it -- see each builder function's docstring.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import xgboost as xgb

from xgboost_bridge.export import export_model
from generate.probe_rows import narrows_onto
from xgboost_bridge.trees import reachable_nodes, walk_margin

CORPUS_DIR = Path(__file__).resolve().parents[1] / "corpus"

__all__ = ["CORPUS_DIR", "build_all", "main"]


# ---------------------------------------------------------------------------
# Small, shared utilities. None of these hold a numerical opinion of their
# own -- ground truth always comes from `booster.predict()`, never from a
# transform reimplemented here.
# ---------------------------------------------------------------------------


def _feature_names(count: int) -> list[str]:
    """Generic, non-application-specific column names."""
    return [f"feature_{index}" for index in range(count)]


def _bits32(value: float) -> str:
    """The uint32 hex bit pattern of a float32 value, per D044."""
    return f"0x{int(np.float32(value).view(np.uint32)):08x}"


def _decimal(value: float) -> float | str:
    """The float32 value, non-normative -- a string for anything non-finite.

    A bare `Infinity`/`NaN` JSON token is non-standard and some parsers
    reject it outright; a string keeps the file ordinary JSON while still
    recording the value for a human to read.
    """
    narrowed = float(np.float32(value))
    if math.isfinite(narrowed):
        return narrowed
    return repr(narrowed)


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and value != value


def _json_row(row: Sequence[float]) -> list[float | None]:
    """`NaN` -> `null` for the wire; every other value passes through as a float."""
    return [None if _is_nan(value) else float(value) for value in row]


def _fit(
    objective: str,
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    *,
    seed: int,
    num_boost_round: int = 10,
    max_depth: int = 3,
    eta: float = 0.3,
    base_score: float | None = None,
    tree_method: str = "hist",
    extra_params: dict[str, Any] | None = None,
) -> xgb.Booster:
    """Fit one booster. `nthread=1` throughout -- determinism, not speed."""
    params: dict[str, Any] = {
        "objective": objective,
        "max_depth": max_depth,
        "eta": eta,
        "nthread": 1,
        "seed": seed,
        "tree_method": tree_method,
    }
    if base_score is not None:
        params["base_score"] = float(base_score)
    if extra_params:
        params.update(extra_params)
    dtrain = xgb.DMatrix(
        np.asarray(features, dtype=np.float64),
        label=np.asarray(labels, dtype=np.float64),
        feature_names=feature_names,
        nthread=1,
    )
    return xgb.train(params, dtrain, num_boost_round=num_boost_round)


def _regression_labels(rng: np.random.Generator, features: np.ndarray) -> np.ndarray:
    coefficients = rng.uniform(-2.0, 2.0, size=features.shape[1])
    return features @ coefficients + rng.normal(0.0, 0.5, size=features.shape[0])


def _logistic_labels(rng: np.random.Generator, features: np.ndarray) -> np.ndarray:
    coefficients = rng.uniform(-1.5, 1.5, size=features.shape[1])
    logits = features @ coefficients
    probability = 1.0 / (1.0 + np.exp(-logits))
    return (rng.random(features.shape[0]) < probability).astype(np.float64)


def _cox_labels(
    rng: np.random.Generator, features: np.ndarray, coefficients: np.ndarray | None = None
) -> np.ndarray:
    """Signed magnitude labels: positive is an event, negative is right-censored."""
    if coefficients is None:
        coefficients = rng.uniform(-1.0, 1.0, size=features.shape[1])
    hazard = np.exp(features @ coefficients)
    time = rng.exponential(1.0 / np.clip(hazard, 1e-6, None))
    event = rng.integers(0, 2, size=features.shape[0])
    return np.where(event == 1, time, -time)


def _sample_rows(
    rng: np.random.Generator, num_feature: int, count: int, low: float = -4.0, high: float = 4.0
) -> list[list[float]]:
    return rng.uniform(low, high, size=(count, num_feature)).tolist()


def _ground_truth(
    booster: xgb.Booster, rows: list[list[float]], feature_names: list[str]
) -> tuple[list[str], list[str], list[float | str], list[float | str]]:
    """Ask XGBoost, and only XGBoost, what the margin and output are.

    `DMatrix`'s default `missing=nan` is exactly the convention FORMAT.md
    section 9.3 specifies for the artifact-side walk, so a `null` in `rows`
    reaches the same missing-value branch on both sides.
    """
    matrix = xgb.DMatrix(np.asarray(rows, dtype=np.float64), feature_names=feature_names, nthread=1)
    margin = np.asarray(booster.predict(matrix, output_margin=True), dtype=np.float32)
    output = np.asarray(booster.predict(matrix), dtype=np.float32)
    margin_bits = [_bits32(value) for value in margin]
    output_bits = [_bits32(value) for value in output]
    margin_decimal = [_decimal(value) for value in margin]
    output_decimal = [_decimal(value) for value in output]
    return margin_bits, output_bits, margin_decimal, output_decimal


def _resolved_xgboost_version(booster: xgb.Booster) -> str:
    document = json.loads(booster.save_raw(raw_format="json"))
    return ".".join(str(component) for component in document["version"])


def _write_fixture(
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
    """Assemble one fixture, self-check it, and write it to `CORPUS_DIR`.

    The self-check re-walks every row with this repository's own
    `walk_margin` and requires a bit-exact match against XGBoost's observed
    margin *before* anything is written. Per CLAUDE.md, a disagreement here
    is never resolved by adjusting the expected value -- it is a defect to
    report, and this function raises rather than papering over one.
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
                f"fixture {name!r}: walk_margin disagrees with XGBoost's own margin at "
                f"generation time (row={row!r}, xgboost={expected_bits}, "
                f"walk_margin={computed_bits}); this is a defect to report, not to fix "
                "by adjusting the expected value"
            )

    _assert_probe_rows_agree_with_xgboost(name, booster, feature_names, trees, intercept)

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

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    path = CORPUS_DIR / f"{name}.json"
    path.write_text(
        json.dumps(fixture, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return fixture


# ---------------------------------------------------------------------------
# Path-resolution helpers for the missing-value fixture. These exist only to
# *construct* an input row that is known, by direct simulation, to visit a
# chosen node -- they carry no numerical opinion and are not the thing being
# tested. The thing being tested is XGBoost's own `predict()` on the row
# they produce.
# ---------------------------------------------------------------------------


def _parent_map(tree: dict[str, Any]) -> dict[int, tuple[int, str] | None]:
    left = tree["left_children"]
    right = tree["right_children"]
    parents: dict[int, tuple[int, str] | None] = {0: None}
    for index in range(len(left)):
        if left[index] != -1:
            parents[left[index]] = (index, "left")
            parents[right[index]] = (index, "right")
    return parents


def _path_to(tree: dict[str, Any], target: int) -> list[tuple[int, str]]:
    """Ancestors of `target`, root-first, with the direction taken at each."""
    parents = _parent_map(tree)
    path: list[tuple[int, str]] = []
    node = target
    while parents[node] is not None:
        ancestor, direction = parents[node]  # type: ignore[misc]
        path.append((ancestor, direction))
        node = ancestor
    path.reverse()
    return path


def _resolve_row(tree: dict[str, Any], target: int, num_feature: int) -> list[float] | None:
    """A feature row that reaches `target`, or `None` if no simple one exists.

    Each ancestor on the path independently gets a value driving the walk
    toward `target`: `threshold` exactly for "right" (equality routes right,
    per FORMAT.md section 10), and a value comfortably below it for "left".
    If two ancestors share a split feature, the second assignment must agree
    with the first or the row is infeasible and this returns `None` -- most
    importantly when an ancestor's split feature *is* `target`'s own split
    feature, in which case that ancestor reads the same `NaN` this function
    is about to place there, and is routed by its own `default_left` rather
    than by any threshold at all.
    """
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


def _find_missing_value_target(
    trees: list[dict[str, Any]], want_default_left: int, num_feature: int
) -> tuple[tuple[int, int], list[float]]:
    """The first internal, non-root node with the wanted `default_left`, resolved to a row.

    The root is excluded so the two fixture rows in
    `_build_missing_value_both_directions` are visibly distinct examples
    rather than both degenerating to the same trivial case.
    """
    for tree_index, tree in enumerate(trees):
        left = tree["left_children"]
        for node_index in range(len(left)):
            if node_index == 0 or left[node_index] == -1:
                continue
            if tree["default_left"][node_index] != want_default_left:
                continue
            row = _resolve_row(tree, node_index, num_feature)
            if row is not None:
                return (tree_index, node_index), row
    raise AssertionError(
        f"no internal, non-root node with default_left={want_default_left} could be "
        "resolved to a row; widen the model's missing-value coverage"
    )


# ---------------------------------------------------------------------------
# Fixture builders. Each one exists to make a specific invariant observable
# to fail -- see CLAUDE.md's fixture-design traps and FORMAT.md's required
# fixtures (sections 6.3 and 8.3).
# ---------------------------------------------------------------------------


def _build_reg_squarederror_base_score_low() -> dict[str, Any]:
    """`reg:squarederror`, `base_score` far below 0.5, identity intercept space.

    The identity transform makes `reg:squarederror` the objective least
    likely to hide an intercept-placement bug by accident, so it anchors the
    "far from 0.5 in both directions" requirement on the low side.
    """
    seed = 101
    num_feature = 3
    rng = np.random.default_rng(seed)
    features = rng.uniform(-4.0, 4.0, size=(250, num_feature))
    labels = _regression_labels(rng, features)
    feature_names = _feature_names(num_feature)
    base_score = -137.5
    booster = _fit(
        "reg:squarederror", features, labels, feature_names,
        seed=seed, base_score=base_score, num_boost_round=12,
    )
    rows = _sample_rows(np.random.default_rng(seed + 1), num_feature, 8)
    return _write_fixture(
        "reg_squarederror_base_score_low",
        booster=booster, rows=rows, feature_names=feature_names,
        description=(
            "reg:squarederror with base_score far below 0.5 in margin space, "
            "stressing the accumulation recipe against a large, non-trivial intercept."
        ),
        seed=seed, base_score=base_score,
    )


def _build_reg_squarederror_base_score_high() -> dict[str, Any]:
    """`reg:squarederror`, `base_score` far above 0.5 -- the opposite direction."""
    seed = 102
    num_feature = 3
    rng = np.random.default_rng(seed)
    features = rng.uniform(-4.0, 4.0, size=(250, num_feature))
    labels = _regression_labels(rng, features)
    feature_names = _feature_names(num_feature)
    base_score = 412.25
    booster = _fit(
        "reg:squarederror", features, labels, feature_names,
        seed=seed, base_score=base_score, num_boost_round=12,
    )
    rows = _sample_rows(np.random.default_rng(seed + 1), num_feature, 8)
    return _write_fixture(
        "reg_squarederror_base_score_high",
        booster=booster, rows=rows, feature_names=feature_names,
        description=(
            "reg:squarederror with base_score far above 0.5 in margin space, the "
            "opposite-direction companion to reg_squarederror_base_score_low."
        ),
        seed=seed, base_score=base_score,
    )


def _build_reg_squarederror_zero_tree() -> dict[str, Any]:
    """The required zero-tree model: the intercept is the entire output.

    `num_boost_round=0` with `base_score` passed explicitly. For
    `reg:squarederror` the link transform is the identity, so this is a
    simple margin-is-constant case -- the more delicate zero-tree,
    explicit-`base_score` case is `binary_logistic_signed_zero` below.
    """
    seed = 103
    num_feature = 3
    rng = np.random.default_rng(seed)
    features = rng.uniform(-4.0, 4.0, size=(10, num_feature))
    labels = _regression_labels(rng, features)
    feature_names = _feature_names(num_feature)
    base_score = 42.75
    booster = _fit(
        "reg:squarederror", features, labels, feature_names,
        seed=seed, base_score=base_score, num_boost_round=0,
    )
    rows = _sample_rows(np.random.default_rng(seed + 1), num_feature, 5)
    return _write_fixture(
        "reg_squarederror_zero_tree",
        booster=booster, rows=rows, feature_names=feature_names,
        description=(
            "Zero boosting rounds: the margin is the intercept alone on every row, "
            "with no tree to read a feature at all."
        ),
        seed=seed, base_score=base_score,
    )


def _build_binary_logistic_base_score_low_inside_clamp() -> dict[str, Any]:
    """`binary:logistic`, base_score far below 0.5, inside `[f32(1e-6), f32(1-1e-6)]`."""
    seed = 111
    num_feature = 3
    rng = np.random.default_rng(seed)
    features = rng.uniform(-4.0, 4.0, size=(250, num_feature))
    labels = _logistic_labels(rng, features)
    feature_names = _feature_names(num_feature)
    base_score = 0.02
    booster = _fit(
        "binary:logistic", features, labels, feature_names,
        seed=seed, base_score=base_score, num_boost_round=12,
    )
    rows = _sample_rows(np.random.default_rng(seed + 1), num_feature, 8)
    return _write_fixture(
        "binary_logistic_base_score_low_inside_clamp",
        booster=booster, rows=rows, feature_names=feature_names,
        description=(
            "binary:logistic with base_score far below 0.5 but inside the logistic "
            "clamp domain -- exercises the -log(f32(f32(1/p)-1)) intercept transform "
            "without the clamp itself firing."
        ),
        seed=seed, base_score=base_score,
    )


def _build_binary_logistic_base_score_high_inside_clamp() -> dict[str, Any]:
    """`binary:logistic`, base_score far above 0.5, inside the clamp domain."""
    seed = 112
    num_feature = 3
    rng = np.random.default_rng(seed)
    features = rng.uniform(-4.0, 4.0, size=(250, num_feature))
    labels = _logistic_labels(rng, features)
    feature_names = _feature_names(num_feature)
    base_score = 0.98
    booster = _fit(
        "binary:logistic", features, labels, feature_names,
        seed=seed, base_score=base_score, num_boost_round=12,
    )
    rows = _sample_rows(np.random.default_rng(seed + 1), num_feature, 8)
    return _write_fixture(
        "binary_logistic_base_score_high_inside_clamp",
        booster=booster, rows=rows, feature_names=feature_names,
        description=(
            "binary:logistic with base_score far above 0.5 but inside the logistic "
            "clamp domain, the high-side companion of the low_inside_clamp fixture."
        ),
        seed=seed, base_score=base_score,
    )


def _build_binary_logistic_base_score_below_clamp() -> dict[str, Any]:
    """`binary:logistic`, base_score below `f32(1e-6)` -- the clamp floor fires.

    `base_score = 1e-7` is the exact value D035 measured the unclamped
    recipe getting wrong by 2.3 in margin space; this fixture puts a real
    fitted model, not just the zero-tree oracle, at that value.
    """
    seed = 113
    num_feature = 3
    rng = np.random.default_rng(seed)
    features = rng.uniform(-4.0, 4.0, size=(250, num_feature))
    labels = _logistic_labels(rng, features)
    feature_names = _feature_names(num_feature)
    base_score = 1e-7
    booster = _fit(
        "binary:logistic", features, labels, feature_names,
        seed=seed, base_score=base_score, num_boost_round=12,
    )
    rows = _sample_rows(np.random.default_rng(seed + 1), num_feature, 8)
    return _write_fixture(
        "binary_logistic_base_score_below_clamp",
        booster=booster, rows=rows, feature_names=feature_names,
        description=(
            "binary:logistic with base_score below the logistic clamp floor "
            "f32(1e-6), so the intercept derivation's clamp must fire to match "
            "XGBoost (D035)."
        ),
        seed=seed, base_score=base_score,
    )


def _build_binary_logistic_base_score_above_clamp() -> dict[str, Any]:
    """`binary:logistic`, base_score above `f32(1-1e-6)` -- the clamp ceiling fires.

    `base_score = 0.9999999` is D035's other measured example, where the
    unclamped recipe is wrong by about 2.2 in margin space.
    """
    seed = 114
    num_feature = 3
    rng = np.random.default_rng(seed)
    features = rng.uniform(-4.0, 4.0, size=(250, num_feature))
    labels = _logistic_labels(rng, features)
    feature_names = _feature_names(num_feature)
    base_score = 0.9999999
    booster = _fit(
        "binary:logistic", features, labels, feature_names,
        seed=seed, base_score=base_score, num_boost_round=12,
    )
    rows = _sample_rows(np.random.default_rng(seed + 1), num_feature, 8)
    return _write_fixture(
        "binary_logistic_base_score_above_clamp",
        booster=booster, rows=rows, feature_names=feature_names,
        description=(
            "binary:logistic with base_score above the logistic clamp ceiling "
            "f32(1-1e-6), the high-side companion of the below_clamp fixture."
        ),
        seed=seed, base_score=base_score,
    )


def _build_binary_logistic_signed_zero() -> dict[str, Any]:
    """FORMAT.md section 6.3's required fixture: `-0.0` survives to the output.

    Zero boosting rounds and `base_score = 0.5` passed *explicitly* -- left
    at the default, `boost_from_average` stays `"1"` and XGBoost would emit
    the raw `0.5` rather than the link-transformed `-0.0` (D036), which
    would look like this fixture while testing nothing at all.
    """
    seed = 121
    num_feature = 2
    rng = np.random.default_rng(seed)
    features = rng.uniform(-4.0, 4.0, size=(10, num_feature))
    labels = (rng.random(10) < 0.5).astype(np.float64)
    feature_names = _feature_names(num_feature)
    base_score = 0.5
    booster = _fit(
        "binary:logistic", features, labels, feature_names,
        seed=seed, base_score=base_score, num_boost_round=0,
    )
    rows = _sample_rows(np.random.default_rng(seed + 1), num_feature, 5)
    return _write_fixture(
        "binary_logistic_signed_zero",
        booster=booster, rows=rows, feature_names=feature_names,
        description=(
            "The required signed-zero fixture (FORMAT.md section 6.3): zero trees "
            "and base_score=0.5 passed explicitly, so the intercept is exactly -0.0 "
            "(bit pattern 0x80000000) and is the entire output on every row."
        ),
        seed=seed, base_score=base_score,
        extra_meta={
            "required_intercept_bit_pattern": "0x80000000",
        },
    )


def _build_survival_cox_base_score_low() -> dict[str, Any]:
    """`survival:cox`, base_score far below 0.5 -- no clamp exists for Cox."""
    seed = 131
    num_feature = 3
    rng = np.random.default_rng(seed)
    features = rng.uniform(-3.0, 3.0, size=(250, num_feature))
    labels = _cox_labels(rng, features)
    feature_names = _feature_names(num_feature)
    base_score = 1e-4
    booster = _fit(
        "survival:cox", features, labels, feature_names,
        seed=seed, base_score=base_score, num_boost_round=12,
    )
    rows = _sample_rows(np.random.default_rng(seed + 1), num_feature, 8)
    return _write_fixture(
        "survival_cox_base_score_low",
        booster=booster, rows=rows, feature_names=feature_names,
        description=(
            "survival:cox with base_score far below 0.5, giving a clearly negative "
            "log(f32(base_score)) intercept -- away from the base_score=1.0 trap "
            "where the Cox intercept collapses to exactly 0.0."
        ),
        seed=seed, base_score=base_score,
    )


def _build_survival_cox_base_score_high() -> dict[str, Any]:
    """`survival:cox`, base_score far above 0.5 -- the opposite direction."""
    seed = 132
    num_feature = 3
    rng = np.random.default_rng(seed)
    features = rng.uniform(-3.0, 3.0, size=(250, num_feature))
    labels = _cox_labels(rng, features)
    feature_names = _feature_names(num_feature)
    base_score = 250.0
    booster = _fit(
        "survival:cox", features, labels, feature_names,
        seed=seed, base_score=base_score, num_boost_round=12,
    )
    rows = _sample_rows(np.random.default_rng(seed + 1), num_feature, 8)
    return _write_fixture(
        "survival_cox_base_score_high",
        booster=booster, rows=rows, feature_names=feature_names,
        description=(
            "survival:cox with base_score far above 0.5, the high-side companion "
            "of survival_cox_base_score_low."
        ),
        seed=seed, base_score=base_score,
    )


def _build_survival_cox_overflow_to_infinity() -> dict[str, Any]:
    """The required +inf fixture: a Cox margin beyond XGBoost's overflow point.

    `base_score = 8e37` (nonzero-intercept, not the `1.0` trap) combined
    with a moderately large forest pushes at least one row's margin past
    XGBoost's measured overflow point around 88.72, where `exp` in float32
    genuinely returns `+inf` -- this is the exact case D044's bit-pattern
    encoding exists for. Other rows in the same fixture stay large but
    finite, so the boundary is visible rather than the whole fixture being
    trivially infinite.
    """
    seed = 141
    num_feature = 2
    rng = np.random.default_rng(seed)
    features = rng.uniform(-5.0, 5.0, size=(300, num_feature))
    labels = _cox_labels(rng, features, coefficients=np.array([0.3, -0.2]))
    feature_names = _feature_names(num_feature)
    base_score = 8e37
    booster = _fit(
        "survival:cox", features, labels, feature_names,
        seed=seed, base_score=base_score, num_boost_round=100, max_depth=3, eta=0.5,
    )
    rows = [
        [1e6, 1e6], [-1e6, -1e6], [3.0, -2.0], [0.0, 0.0], [-3.0, 4.0], [50.0, -50.0],
        [4.9, -4.9], [-4.9, 4.9], [2.0, 2.0], [-2.0, -2.0],
    ]
    return _write_fixture(
        "survival_cox_overflow_to_infinity",
        booster=booster, rows=rows, feature_names=feature_names,
        description=(
            "survival:cox with base_score and tree contributions pushed close enough "
            "to XGBoost's measured Cox overflow point (margin ~88.72) that at least "
            "one row's expected_output is +inf while others stay large but finite."
        ),
        seed=seed, base_score=base_score,
    )


def _build_gamma_pruned_dead_nodes() -> dict[str, Any]:
    """FORMAT.md section 8.3's required fixture: a pruned model with genuine dead nodes.

    `tree_method="exact"` is required to observe `tree_param.num_deleted > 0`
    at all: the histogram-based default declines to grow a split whose gain
    is below `gamma` in the first place, leaving nothing to prune
    afterwards, while the exact grower splits first and prunes in a
    separate pass, leaving unreachable nodes behind exactly as FORMAT.md
    section 8.3 describes.
    """
    seed = 199
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

    document = json.loads(booster.save_raw(raw_format="json"))
    source_trees = document["learner"]["gradient_booster"]["model"]["trees"]
    num_deleted_per_tree = [int(tree["tree_param"]["num_deleted"]) for tree in source_trees]

    artifact = export_model(booster)
    dead_indices_per_tree = []
    for tree in artifact["trees"]:
        node_count = len(tree["left_children"])
        reachable = reachable_nodes(tree)
        dead_indices_per_tree.append(
            [index for index in range(node_count) if index not in reachable]
        )

    rows = _sample_rows(np.random.default_rng(seed + 1), num_feature, 8, low=-3.0, high=3.0)
    return _write_fixture(
        "gamma_pruned_dead_nodes",
        booster=booster, rows=rows, feature_names=feature_names,
        description=(
            "reg:squarederror fit with tree_method=exact and gamma=50.0, producing "
            "trees with genuinely unreachable (neutralized) nodes -- "
            "FORMAT.md section 8.3's required pruned-model fixture."
        ),
        seed=seed, base_score=None,
        extra_meta={
            "num_deleted_per_tree": num_deleted_per_tree,
            "dead_node_indices_per_tree": dead_indices_per_tree,
        },
    )


def _build_missing_value_both_directions() -> dict[str, Any]:
    """Both `default_left` directions, each actually taken by a constructed row.

    Two rows are built by direct path resolution (see
    `_find_missing_value_target`) to place a `NaN` at a node whose
    `default_left` is `1` and, separately, at a different node whose
    `default_left` is `0`. Both are confirmed the only way that means
    anything here: XGBoost's own `predict()` is asked for the margin on
    each row, and this module's own `walk_margin` is required (in
    `_write_fixture`'s self-check) to reproduce it bit-for-bit -- so a
    predictor that silently routed the wrong direction would fail before
    the fixture even reached disk.
    """
    seed = 151
    num_feature = 5
    rng = np.random.default_rng(seed)
    features = rng.uniform(-3.0, 3.0, size=(400, num_feature))
    missing_mask = rng.random(features.shape) < 0.25
    features_with_missing = features.copy()
    features_with_missing[missing_mask] = np.nan
    labels = _regression_labels(rng, features)
    feature_names = _feature_names(num_feature)

    booster = _fit(
        "reg:squarederror", features_with_missing, labels, feature_names,
        seed=seed, num_boost_round=8, max_depth=4, eta=0.3,
    )
    artifact = export_model(booster)
    trees = artifact["trees"]

    left_target, left_row = _find_missing_value_target(trees, want_default_left=1, num_feature=num_feature)
    right_target, right_row = _find_missing_value_target(trees, want_default_left=0, num_feature=num_feature)

    ordinary_rows = _sample_rows(np.random.default_rng(seed + 1), num_feature, 4, low=-3.0, high=3.0)
    rows = [left_row, right_row, *ordinary_rows]

    return _write_fixture(
        "missing_value_both_directions",
        booster=booster, rows=rows, feature_names=feature_names,
        description=(
            "reg:squarederror fit with 25% missing values in training data. Row 0 "
            "carries a NaN at a node whose default_left is 1 (routes left); row 1 "
            "carries a NaN at a different node whose default_left is 0 (routes "
            "right); rows 2-5 are ordinary, fully-populated rows."
        ),
        seed=seed, base_score=None,
        extra_meta={
            "default_left_1_target": {"tree_index": left_target[0], "node_index": left_target[1]},
            "default_left_0_target": {"tree_index": right_target[0], "node_index": right_target[1]},
        },
    )


def _build_single_feature_model() -> dict[str, Any]:
    """A model with exactly one feature -- the minimal `feature_names` shape."""
    seed = 161
    num_feature = 1
    rng = np.random.default_rng(seed)
    features = rng.uniform(-4.0, 4.0, size=(200, num_feature))
    labels = _logistic_labels(rng, features)
    feature_names = _feature_names(num_feature)
    booster = _fit(
        "binary:logistic", features, labels, feature_names,
        seed=seed, num_boost_round=10,
    )
    rows = _sample_rows(np.random.default_rng(seed + 1), num_feature, 6)
    return _write_fixture(
        "single_feature_model",
        booster=booster, rows=rows, feature_names=feature_names,
        description="binary:logistic with exactly one feature -- the minimal feature_names shape.",
        seed=seed, base_score=None,
    )


def _build_single_row_model() -> dict[str, Any]:
    """A fixture whose `rows` array has exactly one row -- the minimal row-count shape."""
    seed = 171
    num_feature = 4
    rng = np.random.default_rng(seed)
    features = rng.uniform(-4.0, 4.0, size=(200, num_feature))
    labels = _regression_labels(rng, features)
    feature_names = _feature_names(num_feature)
    booster = _fit(
        "reg:squarederror", features, labels, feature_names,
        seed=seed, num_boost_round=10,
    )
    rows = _sample_rows(np.random.default_rng(seed + 1), num_feature, 1)
    return _write_fixture(
        "single_row_model",
        booster=booster, rows=rows, feature_names=feature_names,
        description="reg:squarederror with exactly one prediction row -- the minimal row-count shape.",
        seed=seed, base_score=None,
    )


_BUILDERS = (
    _build_reg_squarederror_base_score_low,
    _build_reg_squarederror_base_score_high,
    _build_reg_squarederror_zero_tree,
    _build_binary_logistic_base_score_low_inside_clamp,
    _build_binary_logistic_base_score_high_inside_clamp,
    _build_binary_logistic_base_score_below_clamp,
    _build_binary_logistic_base_score_above_clamp,
    _build_binary_logistic_signed_zero,
    _build_survival_cox_base_score_low,
    _build_survival_cox_base_score_high,
    _build_survival_cox_overflow_to_infinity,
    _build_gamma_pruned_dead_nodes,
    _build_missing_value_both_directions,
    _build_single_feature_model,
    _build_single_row_model,
)

#: Every fixture name this module writes, derived from the builders above so
#: there is one authority for "which fixtures exist" rather than two lists
#: that can silently drift apart.
FIXTURE_NAMES: tuple[str, ...] = tuple(
    builder.__name__.removeprefix("_build_") for builder in _BUILDERS
)


def build_all() -> list[dict[str, Any]]:
    """Build and write every fixture in the corpus, returning them in order."""
    return [builder() for builder in _BUILDERS]



def _assert_probe_rows_agree_with_xgboost(
    name: str,
    booster: xgb.Booster,
    feature_names: list[str],
    trees: list[dict[str, Any]],
    intercept: float,
) -> None:
    """Re-check the walk on rows the fixture's own rows cannot discriminate.

    **This closes the one door every other guard in this repository leaves open.**
    Every check downstream treats ``expected_margin`` as ground truth, and it *is*
    ground truth -- it comes from ``booster.predict()``, never from this
    repository's walk. But the check above can only fire on rows that
    *distinguish* a defect, and the corpus rows do not distinguish the
    highest-value one: reverting the sample-side ``np.float32`` cast in
    ``walk_margin`` and regenerating the corpus **succeeded**, silently, because
    every corpus value is already float32-exact.

    So the sequence "change the comparison, regenerate the fixtures, watch
    everything pass" was reachable. Not because the ground truth was laundered,
    but because nothing asked XGBoost about an input that could tell.

    The rows built here are that input: for every internal node, a value that
    rounds *onto* the threshold without equalling it. Narrowed, it compares equal
    and routes RIGHT; un-narrowed, the below-variant routes LEFT -- a whole
    subtree of difference. XGBoost is asked directly, so this is an independent
    oracle rather than a re-derivation.

    Raises:
        AssertionError: the walk disagrees with XGBoost on a probe row. As above,
            that is a defect to report, never to fix by adjusting an expectation.
    """
    probes: list[list[float]] = []
    baseline = [0.0] * len(feature_names)
    for tree in trees:
        left = tree["left_children"]
        for node, child in enumerate(left):
            if child == -1:
                continue
            pair = narrows_onto(float(tree["node_values"][node]))
            if pair is None:
                continue
            column = int(tree["split_indices"][node])
            for value in pair:
                row = list(baseline)
                row[column] = value
                probes.append(row)

    if not probes:
        # A leaf-only model has no thresholds to probe. Stated rather than passed
        # over, because "no probes" and "all probes agreed" are different results.
        print(f"    {name}: no internal nodes, so no probe rows")
        return

    matrix = xgb.DMatrix(
        np.asarray(probes, dtype=np.float32), feature_names=feature_names, nthread=1
    )
    observed = np.asarray(booster.predict(matrix, output_margin=True), dtype=np.float32)
    for index, row in enumerate(probes):
        expected = _bits32(observed[index])
        computed = _bits32(walk_margin(trees, intercept, np.asarray(row, dtype=np.float64)))
        if computed != expected:
            raise AssertionError(
                f"fixture {name!r}: walk_margin disagrees with XGBoost on a "
                f"narrows-onto-threshold probe row (row={row!r}, "
                f"xgboost={expected}, walk_margin={computed}). These rows exist "
                f"because the fixture's own rows cannot detect a missing float32 "
                f"narrowing on the sample side of the comparison. This is a defect "
                f"to report, not to fix by adjusting the expected value."
            )
    print(f"    {name}: {len(probes)} narrows-onto probe rows agree with XGBoost")

def main() -> None:
    fixtures = build_all()
    print(f"wrote {len(fixtures)} fixtures to {CORPUS_DIR}")
    for fixture in fixtures:
        meta = fixture["meta"]
        print(f"  {meta['name']}: {meta['row_count']} rows, objective={meta['objective']!r}")


if __name__ == "__main__":
    main()
