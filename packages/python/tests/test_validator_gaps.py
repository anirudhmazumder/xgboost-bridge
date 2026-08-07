"""Gaps where a validator established less than its consumers assumed (D058).

Every case here was found by an adversarial read of the validators against their
callers, and every one is pinned in both directions: the refusal fires, and an
ordinary model still passes. The second half matters as much as the first — a
bound written `>= 1` instead of `> 1`, or an emptiness check that stopped being
conditional, would reject every artifact, and the resulting failures would be
scattered across unrelated tests rather than saying what happened.

The documents are mutated **real** fitted models rather than hand-built dicts, so
"well-formed apart from the defect" means what XGBoost actually serializes, not
what a test author remembered to include.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import numpy as np
import pytest
import xgboost as xgb

from xgboost_bridge import errors, objectives, trees, validate

_TREES_PATH = ("learner", "gradient_booster", "model", "trees")


def _document(*, rounds: int = 3, objective: str = "reg:squarederror") -> dict[str, Any]:
    """A real serialized model, deep-copied so a mutation cannot leak."""
    rng = np.random.default_rng(20260806)
    x = rng.normal(size=(200, 2))
    y = rng.normal(size=200)
    matrix = xgb.DMatrix(x, label=y, feature_names=["f0", "f1"])
    booster = xgb.train(
        {"objective": objective, "tree_method": "exact", "max_depth": 3, "nthread": 1,
         "seed": 20260806},
        matrix,
        num_boost_round=rounds,
        verbose_eval=False,
    )
    return json.loads(booster.save_raw(raw_format="json"))


def _first_tree(document: dict[str, Any]) -> dict[str, Any]:
    node: Any = document
    for key in _TREES_PATH:
        node = node[key]
    return node[0]


# ---------------------------------------------------------------------------
# A DAG reaching the exporter was a HANG, not a wrong number
#
# `export._leaf_reaching_rows` enumerates root-to-leaf PATHS, which equals the
# leaf count only for a tree. Measured on a hand-built forward-pointing diamond
# chain: 18 nodes 0.006s, 26 nodes 0.265s, 34 nodes 14.18s, 60 never returned.
# The reader refuses a shared child (D058); this validator did not.
# ---------------------------------------------------------------------------


def test_extract_trees_refuses_a_shared_child() -> None:
    document = _document()
    tree = _first_tree(document)
    size = len(tree["left_children"])
    assert size >= 5, "the fitted tree must be big enough to rewrite"

    # Nodes 1 and 2 both point at leaves 3 and 4. Consistently internal or leaf,
    # forward-pointing, every index in range — so nothing else refuses it.
    tree["left_children"] = [1, 3, 3, -1, -1] + [-1] * (size - 5)
    tree["right_children"] = [2, 4, 4, -1, -1] + [-1] * (size - 5)

    with pytest.raises(errors.MalformedTreeError) as caught:
        trees.extract_trees(document)
    assert "one parent" in str(caught.value)


def test_extract_trees_still_accepts_the_unmutated_model() -> None:
    """The companion. Without it, a check that rejected everything would pass the
    test above and this suite would report the breakage only indirectly."""
    extracted = trees.extract_trees(_document())
    assert len(extracted) == 3
    assert all(len(tree["left_children"]) > 0 for tree in extracted)


# ---------------------------------------------------------------------------
# `validate_source_model` is documented as a gate a caller may use directly, so
# a false pass there is a defect even when `export_model` catches it afterwards
# ---------------------------------------------------------------------------


def test_a_truncated_split_type_cannot_hide_a_categorical_node_from_validate() -> None:
    """`1 in split_type` over a TRUNCATED array searches the wrong domain.

    Measured on a real categorical model: clearing `categories_nodes` and
    `feature_types`, then truncating `split_type` to `[0]`, made this gate PASS
    while `trees.extract_trees` still refused the same document. `trees.py`
    carried the length check with a rationale and a test; the fix had been applied
    in one of the two modules that read the field.
    """
    document = _document()
    tree = _first_tree(document)
    node_count = len(tree["left_children"])
    tree["split_type"] = [0]  # one entry for a many-node tree

    with pytest.raises(errors.MalformedTreeError) as caught:
        validate.validate_source_model(document, tested_versions=("3.3.0",))
    assert caught.value.field == "split_type"
    assert str(node_count) in str(caught.value)


def test_a_full_length_split_type_still_passes() -> None:
    document = _document()
    validate.validate_source_model(document, tested_versions=("3.3.0",))


@pytest.mark.parametrize("raw", ["1.0", "", "two", None, 1.5])
def test_a_non_integer_num_feature_raises_structurally(raw: Any) -> None:
    """A bare `int()` leaked `ValueError` out of a module documenting that a
    caller gets "one of the structured exceptions in xgboost_bridge.errors". The
    same defect at the same idiom had already been found and fixed once here, for
    `best_iteration`."""
    document = _document()
    document["learner"]["learner_model_param"]["num_feature"] = raw
    with pytest.raises(errors.XGBoostBridgeError):
        validate.validate_source_model(document, tested_versions=("3.3.0",))


def test_a_wrong_typed_attributes_block_does_not_skip_the_early_stopping_refusal() -> None:
    """`attributes` as a list previously made the whole check `return None`.

    Absence legitimately means "no refusal needed" and is defaulted. A present
    value of the wrong type is not absence, and treating it as such skipped the
    D038 early-stopping refusal entirely — the silent default this module's own
    fail-loudly rule forbids.
    """
    document = _document()
    document["learner"]["attributes"] = ["best_iteration", "0"]
    with pytest.raises(errors.MalformedTreeError) as caught:
        validate.validate_source_model(document, tested_versions=("3.3.0",))
    assert caught.value.field == "attributes"


def test_an_absent_attributes_block_is_still_fine() -> None:
    document = _document()
    document["learner"].pop("attributes", None)
    validate.validate_source_model(document, tested_versions=("3.3.0",))


def test_empty_feature_names_is_refused_even_when_num_feature_is_zero() -> None:
    """FORMAT.md and D021 both say "raise if feature_names is empty", unqualified.

    The check read `and num_feature != 0`, so `feature_names: []` with
    `num_feature: "0"` passed the entire gate — the length check below it also
    passes, `0 == 0` — and export would have emitted an artifact that our own
    reader and our own published schema both refuse. XGBoost declines to configure
    a 0-feature learner, so it is unreachable from a real fit, which is exactly
    why no test caught it.
    """
    document = _document()
    document["learner"]["feature_names"] = []
    document["learner"]["learner_model_param"]["num_feature"] = "0"
    with pytest.raises(errors.XGBoostBridgeError):
        validate.validate_source_model(document, tested_versions=("3.3.0",))


# ---------------------------------------------------------------------------
# Two guards whose docstrings claimed more than they delivered
# ---------------------------------------------------------------------------


def test_the_zero_tree_oracle_compares_more_than_one_row() -> None:
    """`_single_margin` asserts the margin is constant by requiring exactly one
    distinct bit pattern — which on a size-1 array is unconditionally true.

    This is the export path for the intercept of every zero-tree model, and
    `_verify_against_source_margin` is tautological for such a model, so nothing
    downstream would have caught a non-constant margin. The docstring called the
    constancy "asserted rather than assumed"; it was assumed.
    """
    seen: list[int] = []
    original = objectives._single_margin

    def recording(margin: np.ndarray, label: str) -> np.float32:
        seen.append(int(np.asarray(margin).size))
        return original(margin, label)

    objectives._single_margin = recording  # type: ignore[assignment]
    try:
        document = _document(rounds=0)
        objectives.derive_intercept(document)
        booster_rows = objectives._observed_zero_round_margin(
            "reg:squarederror", np.float32(0.5)
        )
        assert np.isfinite(booster_rows)
    finally:
        objectives._single_margin = original  # type: ignore[assignment]

    assert seen, "the oracle helper was never reached"
    assert min(seen) >= 2, (
        f"a constancy check ran over {min(seen)} row(s); with fewer than 2 it "
        f"cannot fire"
    )


def test_node_values_cannot_be_made_writeable_again() -> None:
    """The read-only flag was reversible in one line.

    `values.flags.writeable = False` on an array that OWNS its buffer can simply
    be set back to True, so `v.flags.writeable = True; v[1] = 999.0` moved a
    threshold and the margin with it — while two docstrings said a caller "cannot
    alter the loaded model". Handing out a view of a read-only base makes the
    claim true: numpy refuses to re-enable the flag when the base is not writeable.
    """
    from xgboost_bridge import Predictor

    artifact = json.loads(
        (
            __import__("pathlib").Path(__file__).resolve().parents[3]
            / "fixtures"
            / "corpus"
            / "survival_cox_base_score_high.json"
        ).read_text()
    )["artifact"]
    values = Predictor.from_json(artifact).trees[0]["node_values"]

    assert values.flags.writeable is False
    assert values.flags.owndata is False, "must be a view, or the flag is reversible"
    with pytest.raises(ValueError):
        values.flags.writeable = True
    with pytest.raises(ValueError):
        values[0] = 999.0
    assert values.dtype == np.float32


# ---------------------------------------------------------------------------
# The READMEs name specific error attributes, and that text is published
# ---------------------------------------------------------------------------


def test_the_readme_names_attributes_that_actually_exist() -> None:
    """The Python README previously promised `field`, `value`, `expected` and
    `index` on *every* exception. No exception carries all four, and
    `FeatureKeyMismatchError` carries none of them — so a caller following the
    README read `err.field` as an AttributeError. This text ships inside the
    wheel's METADATA and renders on PyPI permanently for a version, so it is
    pinned against the classes rather than reviewed.
    """
    expected = {
        errors.MalformedTreeError: ("field", "value", "expected", "location"),
        errors.NonFiniteFeatureError: ("index", "value"),
        errors.FeatureKeyMismatchError: ("missing_keys", "extra_keys"),
    }
    readme = (
        __import__("pathlib").Path(__file__).resolve().parents[1] / "README.md"
    ).read_text()

    for cls, attributes in expected.items():
        assert cls.__name__ in readme, f"{cls.__name__} is not named in the README"
        for attribute in attributes:
            assert f"`{attribute}`" in readme, (
                f"the README does not name {cls.__name__}.{attribute}"
            )

    # And the other direction: every attribute the README names must exist.
    instances = {
        errors.MalformedTreeError: errors.MalformedTreeError("f", 1, "x", "loc"),
        errors.NonFiniteFeatureError: errors.NonFiniteFeatureError(0, float("inf")),
        errors.FeatureKeyMismatchError: errors.FeatureKeyMismatchError(("a",), ("b",)),
    }
    for cls, attributes in expected.items():
        for attribute in attributes:
            assert hasattr(instances[cls], attribute), (
                f"{cls.__name__} has no attribute {attribute}, but the README says it does"
            )
