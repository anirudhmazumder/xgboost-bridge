"""Tests for the export-time validation gate.

Every fixture below is a real model fitted with xgboost, per the brief this
suite was built from -- hand-built artifact dicts are used only for a shape
xgboost 3.3.0 cannot itself produce (the 3.4.0-dev ``weight_drop`` location,
§below), and that case says so explicitly.

Nothing here loosens, skips, or xfails a check in ``xgboost_bridge.validate``.
If a test below is inconvenient, the gate is telling us something about the
gate, not about the test.
"""

from __future__ import annotations

import json
import warnings
from typing import Any

import numpy as np
import pytest
import xgboost as xgb

from xgboost_bridge import errors
from xgboost_bridge.validate import validate_source_model

# Base training configuration shared by every fixture, per the corresponding
# probes: generic feature names, generic synthetic normal data, small trees.
_BASE_PARAMS = {"tree_method": "exact", "max_depth": 2, "eta": 0.3, "nthread": 1}


def _fit(
    params: dict[str, Any],
    *,
    num_boost_round: int = 3,
    rows: int = 200,
    cols: int = 4,
    label: np.ndarray | None = None,
    feature_names: list[str] | None | str = "auto",
    feature_types: list[str] | None = None,
    enable_categorical: bool = False,
    evals: list[Any] | None = None,
    early_stopping_rounds: int | None = None,
    seed: int = 20260804,
    columns: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit a small real model and return its parsed JSON artifact.

    ``columns``, when given, is used verbatim as the feature matrix (for
    shapes that need a specific column, e.g. a categorical one) instead of
    the generated normal data.
    """
    rng = np.random.default_rng(seed)
    x = columns if columns is not None else rng.normal(size=(rows, cols))

    if label is None:
        label_rng = np.random.default_rng(seed + 1)
        objective = params.get("objective", "reg:squarederror")
        if objective == "binary:logistic":
            label = (label_rng.random(x.shape[0]) > 0.5).astype(np.float64)
        elif objective in ("multi:softprob", "multi:softmax"):
            num_class = int(params.get("num_class", 2))
            label = label_rng.integers(0, num_class, size=x.shape[0]).astype(np.float64)
        else:
            label = label_rng.normal(size=x.shape[0])

    names = [f"f{i}" for i in range(x.shape[1])] if feature_names == "auto" else feature_names

    dtrain = xgb.DMatrix(
        x,
        label=label,
        feature_names=names,
        feature_types=feature_types,
        enable_categorical=enable_categorical,
    )

    full_params = dict(_BASE_PARAMS)
    full_params.update(params)
    full_params["seed"] = seed

    with warnings.catch_warnings():
        # dart and gblinear are deprecated upstream (probes/boosters.md); the
        # deprecation warning is expected and is not this suite's concern.
        warnings.simplefilter("ignore")
        booster = xgb.train(
            full_params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals or [],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )

    return json.loads(booster.save_raw(raw_format="json"))


def _model_version(model: dict[str, Any]) -> str:
    return ".".join(str(component) for component in model["version"])


def _tested(model: dict[str, Any]) -> frozenset[str]:
    """The trivial tested-versions set that accepts whatever we just fit."""
    return frozenset({_model_version(model)})


def _accepts(model: dict[str, Any]) -> None:
    assert validate_source_model(model, tested_versions=_tested(model)) is None


# ---------------------------------------------------------------------------
# Truth table (probes/arity_gate.md §4): each shape, and its expected verdict.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("objective", ["reg:squarederror", "binary:logistic", "survival:cox"])
def test_accepts_each_in_scope_objective(objective: str) -> None:
    model = _fit({"objective": objective})
    _accepts(model)


@pytest.mark.parametrize("objective", ["reg:squarederror", "binary:logistic", "survival:cox"])
def test_accepts_num_class_one_on_every_in_scope_objective(objective: str) -> None:
    """num_class == "1" is a genuine single-output model, not a defect (D037)."""
    model = _fit({"objective": objective, "num_class": 1})
    assert model["learner"]["learner_model_param"]["num_class"] == "1"
    _accepts(model)


def test_rejects_reg_squarederror_with_num_target_2() -> None:
    """A 2-column label matrix under an in-scope objective produces (N, 2) margins."""
    rng = np.random.default_rng(20260805)
    two_target_labels = rng.normal(size=(200, 2))
    model = _fit({"objective": "reg:squarederror"}, label=two_target_labels)
    assert model["learner"]["learner_model_param"]["num_target"] == "2"

    with pytest.raises(errors.UnsupportedModelShapeError) as excinfo:
        validate_source_model(model, tested_versions=_tested(model))
    assert excinfo.value.field == "num_target"
    assert excinfo.value.value == "2"


def test_rejects_reg_squarederror_num_target_2_even_with_num_class_1() -> None:
    """num_target is checked before num_class; the hole isn't reopened by num_class."""
    rng = np.random.default_rng(20260805)
    two_target_labels = rng.normal(size=(200, 2))
    model = _fit({"objective": "reg:squarederror", "num_class": 1}, label=two_target_labels)

    with pytest.raises(errors.UnsupportedModelShapeError) as excinfo:
        validate_source_model(model, tested_versions=_tested(model))
    assert excinfo.value.field == "num_target"


@pytest.mark.parametrize("objective", ["multi:softprob", "multi:softmax"])
def test_rejects_multiclass_objectives(objective: str) -> None:
    model = _fit({"objective": objective, "num_class": 3})
    with pytest.raises(errors.UnsupportedObjectiveError) as excinfo:
        validate_source_model(model, tested_versions=_tested(model))
    assert excinfo.value.objective == objective


@pytest.mark.parametrize("objective", ["reg:squarederror", "binary:logistic", "survival:cox"])
def test_accepts_zero_boosting_round_model(objective: str) -> None:
    """trees is present and empty, never absent, and the gate must accept it."""
    model = _fit({"objective": objective}, num_boost_round=0)
    assert model["learner"]["gradient_booster"]["model"]["trees"] == []
    _accepts(model)


def test_accepts_single_feature_model() -> None:
    model = _fit({"objective": "reg:squarederror"}, cols=1)
    assert model["learner"]["learner_model_param"]["num_feature"] == "1"
    _accepts(model)


def test_accepts_gamma_pruned_leaf_only_model() -> None:
    """min_split_loss blocks every split; the tree carries dead nodes but is exportable."""
    model = _fit({"objective": "reg:squarederror", "gamma": 1e9, "max_depth": 3})
    tree_param = model["learner"]["gradient_booster"]["model"]["trees"][0]["tree_param"]
    assert int(tree_param["num_deleted"]) > 0
    _accepts(model)


def test_accepts_num_parallel_tree_model() -> None:
    model = _fit({"objective": "binary:logistic", "num_parallel_tree": 4})
    assert model["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
        "num_parallel_tree"
    ] == "4"
    _accepts(model)


def test_accepts_hist_tree_method_model() -> None:
    model = _fit({"objective": "reg:squarederror", "tree_method": "hist"})
    _accepts(model)


def test_rejects_model_fit_from_a_bare_array_with_no_feature_names() -> None:
    model = _fit({"objective": "reg:squarederror"}, feature_names=None)
    assert model["learner"]["feature_names"] == []

    with pytest.raises(errors.MissingFeatureNamesError) as excinfo:
        validate_source_model(model, tested_versions=_tested(model))
    assert excinfo.value.num_feature == 4


def test_rejects_gblinear() -> None:
    model = _fit(
        {"objective": "binary:logistic", "booster": "gblinear", "updater": "coord_descent"}
    )
    assert model["learner"]["gradient_booster"]["name"] == "gblinear"

    with pytest.raises(errors.UnsupportedBoosterError) as excinfo:
        validate_source_model(model, tested_versions=_tested(model))
    assert excinfo.value.booster == "gblinear"


def test_rejects_dart_with_rate_drop_producing_weight_drop() -> None:
    model = _fit(
        {"objective": "binary:logistic", "booster": "dart", "rate_drop": 0.3, "skip_drop": 0.1}
    )
    assert "weight_drop" in model["learner"]["gradient_booster"]

    with pytest.raises(errors.UnsupportedBoosterError) as excinfo:
        validate_source_model(model, tested_versions=_tested(model))
    assert excinfo.value.booster == "dart"


def test_rejects_categorical_model() -> None:
    rng = np.random.default_rng(20260806)
    categorical_column = rng.integers(0, 4, size=(200, 1)).astype(np.float64)
    numeric_column = rng.normal(size=(200, 1))
    columns = np.hstack([categorical_column, numeric_column])

    model = _fit(
        {"objective": "reg:squarederror", "tree_method": "hist"},
        columns=columns,
        feature_names=["category", "number"],
        feature_types=["c", "float"],
        enable_categorical=True,
    )
    assert "c" in model["learner"]["feature_types"]

    with pytest.raises(errors.CategoricalSplitError) as excinfo:
        validate_source_model(model, tested_versions=_tested(model))
    assert "feature_types" in excinfo.value.signals


# ---------------------------------------------------------------------------
# Feature names (D021): all three conditions FORMAT.md section 11 requires,
# not only "empty". Both gaps below are reached through the ordinary public
# API -- ``booster.feature_names = [...]`` is accepted by xgboost 3.3.0 with
# no warning and serialized verbatim.
# ---------------------------------------------------------------------------


def _fit_and_rename_features(new_feature_names: list[str]) -> dict[str, Any]:
    """A real 4-column fitted model whose feature names are overwritten
    post-fit via the public ``Booster.feature_names`` setter -- the exact
    route xgboost 3.3.0 accepts silently for both a wrong length and
    duplicate names.
    """
    rng = np.random.default_rng(20260810)
    x = rng.normal(size=(200, 4))
    y = rng.normal(size=200)
    dtrain = xgb.DMatrix(x, label=y, feature_names=["f0", "f1", "f2", "f3"])
    booster = xgb.train(
        {**_BASE_PARAMS, "objective": "reg:squarederror"},
        dtrain,
        num_boost_round=3,
        verbose_eval=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        booster.feature_names = new_feature_names
    return json.loads(booster.save_raw(raw_format="json"))


def test_rejects_feature_names_whose_length_disagrees_with_num_feature() -> None:
    """F1: too few names for num_feature slips through unless length is checked.

    A live model with ``split_indices`` reaching column 3 has no name for
    that column under a strict-key policy -- FORMAT.md sections 7 and 13.
    """
    model = _fit_and_rename_features(["a", "b"])
    assert model["learner"]["feature_names"] == ["a", "b"]
    assert model["learner"]["learner_model_param"]["num_feature"] == "4"

    with pytest.raises(errors.UnsupportedModelShapeError) as excinfo:
        validate_source_model(model, tested_versions=_tested(model))
    assert excinfo.value.field == "feature_names"
    assert excinfo.value.value == 2


def test_rejects_feature_names_containing_a_duplicate() -> None:
    """F1: a duplicate name makes one key unreachable under a strict-key policy (D021)."""
    model = _fit_and_rename_features(["a", "a", "b", "c"])
    assert model["learner"]["feature_names"] == ["a", "a", "b", "c"]

    with pytest.raises(errors.UnsupportedModelShapeError) as excinfo:
        validate_source_model(model, tested_versions=_tested(model))
    assert excinfo.value.field == "feature_names"
    assert excinfo.value.value == ("a",)


# ---------------------------------------------------------------------------
# Categorical detection (D016/CategoricalSplitError signals): each of the
# three ``.get(key, default)`` reads that previously disarmed the refusal,
# pinned in isolation. FORMAT.md section 2's lesson -- unrecognized-field
# detection cannot catch a relocation or removal -- applies here exactly as
# it does to the version ceiling.
# ---------------------------------------------------------------------------


def _fit_categorical_model() -> dict[str, Any]:
    """A real categorical model (same construction as test_rejects_categorical_model)."""
    rng = np.random.default_rng(20260806)
    categorical_column = rng.integers(0, 4, size=(200, 1)).astype(np.float64)
    numeric_column = rng.normal(size=(200, 1))
    columns = np.hstack([categorical_column, numeric_column])

    return _fit(
        {"objective": "reg:squarederror", "tree_method": "hist"},
        columns=columns,
        feature_names=["category", "number"],
        feature_types=["c", "float"],
        enable_categorical=True,
    )


def test_rejects_categorical_model_when_split_type_is_relocated() -> None:
    """F2: an absent split_type must raise, never read as 'no signal here'."""
    model = _fit_categorical_model()
    for tree in model["learner"]["gradient_booster"]["model"]["trees"]:
        tree["split_type_v2"] = tree.pop("split_type")

    with pytest.raises(errors.MalformedTreeError) as excinfo:
        validate_source_model(model, tested_versions=_tested(model))
    assert excinfo.value.field == "split_type"


def test_rejects_categorical_model_when_categories_nodes_is_relocated() -> None:
    """F2: an absent categories_nodes must raise, never read as 'no signal here'."""
    model = _fit_categorical_model()
    for tree in model["learner"]["gradient_booster"]["model"]["trees"]:
        tree["categories_nodes_v2"] = tree.pop("categories_nodes")

    with pytest.raises(errors.MalformedTreeError) as excinfo:
        validate_source_model(model, tested_versions=_tested(model))
    assert excinfo.value.field == "categories_nodes"


def test_rejects_categorical_model_when_feature_types_is_deleted() -> None:
    """F2: an absent learner.feature_types must raise, never read as 'no signal here'."""
    model = _fit_categorical_model()
    del model["learner"]["feature_types"]

    with pytest.raises(errors.MalformedTreeError) as excinfo:
        validate_source_model(model, tested_versions=_tested(model))
    assert excinfo.value.field == "feature_types"


# ---------------------------------------------------------------------------
# Early stopping: absent best_iteration is fine (D038); a present-but-invalid
# one must raise a structured error rather than crash the gate itself.
# ---------------------------------------------------------------------------


def test_rejects_best_iteration_that_indexes_past_iteration_indptr() -> None:
    """F3: previously an unguarded IndexError; must now be a structured error."""
    model = _fit({"objective": "reg:squarederror"}, num_boost_round=3)
    model["learner"]["attributes"] = {"best_iteration": "99"}

    with pytest.raises(errors.MalformedTreeError) as excinfo:
        validate_source_model(model, tested_versions=_tested(model))
    assert excinfo.value.field == "best_iteration"
    assert excinfo.value.value == 99


def test_rejects_non_integer_best_iteration() -> None:
    """F3: previously an unguarded ValueError from int(); must now be structured."""
    model = _fit({"objective": "reg:squarederror"}, num_boost_round=3)
    model["learner"]["attributes"] = {"best_iteration": "not-a-number"}

    with pytest.raises(errors.MalformedTreeError) as excinfo:
        validate_source_model(model, tested_versions=_tested(model))
    assert excinfo.value.field == "best_iteration"
    assert excinfo.value.value == "not-a-number"


def test_rejects_early_stopped_model_with_iteration_indptr_deleted() -> None:
    """F4 corollary: iteration_indptr is required once best_iteration is present."""
    rng = np.random.default_rng(20260809)
    train_x = rng.normal(size=(200, 4))
    train_y = rng.normal(size=200)
    val_x = rng.normal(size=(50, 4))
    val_y = rng.normal(size=50)
    dtrain = xgb.DMatrix(train_x, label=train_y, feature_names=["f0", "f1", "f2", "f3"])
    dval = xgb.DMatrix(val_x, label=val_y, feature_names=["f0", "f1", "f2", "f3"])

    booster = xgb.train(
        {**_BASE_PARAMS, "objective": "reg:squarederror", "seed": 1},
        dtrain,
        num_boost_round=20,
        evals=[(dval, "val")],
        early_stopping_rounds=3,
        verbose_eval=False,
    )
    model = json.loads(booster.save_raw(raw_format="json"))
    tested = _tested(model)
    assert "best_iteration" in model["learner"]["attributes"]

    del model["learner"]["gradient_booster"]["model"]["iteration_indptr"]

    with pytest.raises(errors.XGBoostBridgeError):
        validate_source_model(model, tested_versions=tested)


# ---------------------------------------------------------------------------
# F4: every metadata key validate_source_model itself reads must raise a
# structured XGBoostBridgeError when absent -- never a bare KeyError,
# IndexError, or TypeError.
# ---------------------------------------------------------------------------


def _delete_path(container: Any, path: tuple[Any, ...]) -> None:
    """Delete the key/index at the end of ``path`` from a mutable, nested container."""
    for step in path[:-1]:
        container = container[step]
    del container[path[-1]]


_REQUIRED_METADATA_PATHS: tuple[tuple[Any, ...], ...] = (
    ("learner",),
    ("learner", "objective"),
    ("learner", "objective", "name"),
    ("learner", "learner_model_param"),
    ("learner", "learner_model_param", "num_target"),
    ("learner", "learner_model_param", "num_class"),
    ("learner", "learner_model_param", "num_feature"),
    ("learner", "gradient_booster"),
    ("learner", "gradient_booster", "name"),
    ("learner", "gradient_booster", "model"),
    ("learner", "gradient_booster", "model", "trees"),
    ("learner", "gradient_booster", "model", "trees", 0, "tree_param"),
    ("learner", "gradient_booster", "model", "trees", 0, "tree_param", "size_leaf_vector"),
    ("learner", "gradient_booster", "model", "trees", 0, "split_type"),
    ("learner", "gradient_booster", "model", "trees", 0, "categories_nodes"),
    ("learner", "feature_types"),
    ("learner", "feature_names"),
    ("version",),
)


@pytest.mark.parametrize(
    "path",
    _REQUIRED_METADATA_PATHS,
    ids=[".".join(str(step) for step in path) for path in _REQUIRED_METADATA_PATHS],
)
def test_every_required_metadata_key_raises_a_structured_error_when_absent(
    path: tuple[Any, ...],
) -> None:
    """Deleting any key ``validate_source_model`` reads must raise, not crash.

    ``path`` is computed from a real fitted model before the deletion, so
    the test exercises exactly the read ``validate_source_model`` performs
    on its way to a verdict -- not a hand-invented shape.
    """
    model = _fit({"objective": "reg:squarederror"}, num_boost_round=3)
    tested = _tested(model)
    _delete_path(model, path)

    with pytest.raises(errors.XGBoostBridgeError):
        validate_source_model(model, tested_versions=tested)


# ---------------------------------------------------------------------------
# F5: size_leaf_vector and num_class rejecting branches, driven directly.
# Every real vector-leaf or out-of-range-num_class model also fails an
# earlier check (num_target=="2", or the objective allow-list) first, so
# these branches never execute their raise on any real fitted model in this
# suite -- these fixtures are hand-edited from a real single-tree fit for
# that reason, and each says so.
# ---------------------------------------------------------------------------


def test_output_arity_gate_rejects_a_hand_edited_size_leaf_vector() -> None:
    """Hand-edited: no real model reaches this branch (a vector-leaf model
    always carries num_target=="2", which is checked first and would mask
    this branch). Only tree_param.size_leaf_vector is edited; everything
    else is a genuine single-tree reg:squarederror fit.
    """
    model = _fit({"objective": "reg:squarederror"}, num_boost_round=1)
    assert model["learner"]["learner_model_param"]["num_target"] == "1"
    model["learner"]["gradient_booster"]["model"]["trees"][0]["tree_param"][
        "size_leaf_vector"
    ] = "2"

    with pytest.raises(errors.UnsupportedModelShapeError) as excinfo:
        validate_source_model(model, tested_versions=_tested(model))
    assert excinfo.value.field == "size_leaf_vector"
    assert excinfo.value.value == "2"


def test_output_arity_gate_size_leaf_vector_as_a_json_integer_still_rejects() -> None:
    """Hand-edited: xgboost never serializes this field as anything but a
    quoted digit string (probes/arity_gate.md section 2). This pins that an
    integer 1 is not silently accepted as equivalent to the string "1".
    """
    model = _fit({"objective": "reg:squarederror"}, num_boost_round=1)
    model["learner"]["gradient_booster"]["model"]["trees"][0]["tree_param"][
        "size_leaf_vector"
    ] = 1

    with pytest.raises(errors.UnsupportedModelShapeError) as excinfo:
        validate_source_model(model, tested_versions=_tested(model))
    assert excinfo.value.field == "size_leaf_vector"
    assert excinfo.value.value == 1


def test_output_arity_gate_rejects_a_hand_edited_num_class() -> None:
    """Hand-edited: every real model with num_class outside {"0","1"} is a
    multi:softprob/softmax model, which the objective allow-list rejects
    first (probes/arity_gate.md section 4 truth table). Only
    learner_model_param.num_class is edited; everything else is a genuine
    single-tree binary:logistic fit.
    """
    model = _fit({"objective": "binary:logistic"}, num_boost_round=1)
    assert model["learner"]["learner_model_param"]["num_target"] == "1"
    model["learner"]["learner_model_param"]["num_class"] = "5"

    with pytest.raises(errors.UnsupportedModelShapeError) as excinfo:
        validate_source_model(model, tested_versions=_tested(model))
    assert excinfo.value.field == "num_class"
    assert excinfo.value.value == "5"


def test_output_arity_gate_num_class_as_a_json_integer_still_rejects() -> None:
    """Hand-edited: xgboost never serializes this field as anything but a
    quoted digit string. This pins that an integer 1 is not silently
    accepted as equivalent to the string "1".
    """
    model = _fit({"objective": "binary:logistic"}, num_boost_round=1)
    model["learner"]["learner_model_param"]["num_class"] = 1

    with pytest.raises(errors.UnsupportedModelShapeError) as excinfo:
        validate_source_model(model, tested_versions=_tested(model))
    assert excinfo.value.field == "num_class"
    assert excinfo.value.value == 1


# ---------------------------------------------------------------------------
# String-vs-integer comparison: the gate must never coerce a gate field to int.
# ---------------------------------------------------------------------------


def test_num_target_is_compared_as_a_string_not_an_integer() -> None:
    """A model whose serialized num_target is the string "2" trips the gate.

    If the gate ever compared ``num_target`` as an integer, ``"2" == 2`` would
    be evaluated -- and that comparison is False in Python, which would
    silently disable the check rather than trip it. This is the invariant
    documented in probes/arity_gate.md §2 and FORMAT.md §11.
    """
    rng = np.random.default_rng(20260805)
    two_target_labels = rng.normal(size=(200, 2))
    model = _fit({"objective": "reg:squarederror"}, label=two_target_labels)

    num_target = model["learner"]["learner_model_param"]["num_target"]
    assert isinstance(num_target, str)
    assert num_target == "2"

    with pytest.raises(errors.UnsupportedModelShapeError):
        validate_source_model(model, tested_versions=_tested(model))


def test_num_class_is_compared_as_a_string_not_an_integer() -> None:
    """F5: num_class's rejecting branch, driven through validate_source_model.

    Supersedes a prior version of this test that asserted only Python's own
    ``"2" == 2`` semantics and exercised no library code at all. This drives
    the actual comparison in ``_check_output_arity`` on a hand-edited model
    (see ``test_output_arity_gate_rejects_a_hand_edited_num_class`` for why
    no real model reaches this branch): if that comparison were ever
    rewritten to compare against a bare ``int``, the every-model-rejecting
    direction is caught by every "accepts" test in this suite, and this
    test additionally pins that a genuinely out-of-range value still raises.
    """
    model = _fit({"objective": "reg:squarederror"}, num_boost_round=1)
    model["learner"]["learner_model_param"]["num_class"] = "2"

    with pytest.raises(errors.UnsupportedModelShapeError) as excinfo:
        validate_source_model(model, tested_versions=_tested(model))
    assert excinfo.value.field == "num_class"
    assert excinfo.value.value == "2"


def test_size_leaf_vector_is_compared_as_a_string_not_an_integer() -> None:
    """F5: size_leaf_vector's rejecting branch, driven through validate_source_model.

    Supersedes a prior version of this test that asserted only Python's own
    ``"2" == 2`` semantics and exercised no library code at all. See
    ``test_output_arity_gate_rejects_a_hand_edited_size_leaf_vector`` for why
    no real model reaches this branch in isolation.
    """
    model = _fit({"objective": "reg:squarederror"}, num_boost_round=1)
    model["learner"]["gradient_booster"]["model"]["trees"][0]["tree_param"][
        "size_leaf_vector"
    ] = "3"

    with pytest.raises(errors.UnsupportedModelShapeError) as excinfo:
        validate_source_model(model, tested_versions=_tested(model))
    assert excinfo.value.field == "size_leaf_vector"
    assert excinfo.value.value == "3"


def test_size_leaf_vector_is_checked_on_every_tree_not_only_the_first() -> None:
    """D037 requires the per-tree check to run for **every** tree.

    ``size_leaf_vector`` lives only in each tree's ``tree_param``, never in
    ``learner_model_param``, so "check the model" has no referent and the gate
    has to iterate. A gate that iterated only as far as the first tree would
    pass every test that edits ``trees[0]`` and every model whose trees all
    agree -- which is all of them -- while admitting a mixed-arity ensemble.
    The last tree is edited here precisely because it is the one a truncated
    loop never reaches.
    """
    model = _fit({"objective": "reg:squarederror"}, num_boost_round=3)
    trees = model["learner"]["gradient_booster"]["model"]["trees"]
    assert len(trees) >= 3, "the fixture needs more than one tree to say anything"
    assert trees[0]["tree_param"]["size_leaf_vector"] == "1"

    trees[-1]["tree_param"]["size_leaf_vector"] = "2"

    with pytest.raises(errors.UnsupportedModelShapeError) as excinfo:
        validate_source_model(model, tested_versions=_tested(model))
    assert excinfo.value.field == "size_leaf_vector"
    assert excinfo.value.value == "2"


# ---------------------------------------------------------------------------
# Version ceiling.
# ---------------------------------------------------------------------------


def test_accepts_when_the_producing_version_is_tested() -> None:
    model = _fit({"objective": "reg:squarederror"})
    validate_source_model(model, tested_versions=frozenset({_model_version(model)}))


def test_rejects_when_the_producing_version_is_not_in_tested_versions() -> None:
    model = _fit({"objective": "reg:squarederror"})
    actual_version = _model_version(model)
    excluding_versions = frozenset({"0.0.1", "999.0.0"})
    assert actual_version not in excluding_versions

    with pytest.raises(errors.UnsupportedVersionError) as excinfo:
        validate_source_model(model, tested_versions=excluding_versions)
    assert excinfo.value.version == actual_version
    assert excinfo.value.supported == tuple(sorted(excluding_versions))


# ---------------------------------------------------------------------------
# Dart's single in-artifact signal, at both known JSON paths.
# ---------------------------------------------------------------------------


def test_dart_weight_drop_detected_at_the_gradient_booster_path() -> None:
    """The path weight_drop actually occupies on xgboost 3.3.0 (probes/boosters.md)."""
    model = _fit(
        {"objective": "binary:logistic", "booster": "dart", "rate_drop": 0.3, "skip_drop": 0.1}
    )
    gradient_booster = model["learner"]["gradient_booster"]
    assert "weight_drop" in gradient_booster
    assert "weight_drop" not in gradient_booster.get("model", {})

    with pytest.raises(errors.UnsupportedBoosterError) as excinfo:
        validate_source_model(model, tested_versions=_tested(model))
    assert excinfo.value.booster == "dart"


def test_dart_weight_drop_detected_at_the_relocated_model_path() -> None:
    """Simulates the 3.4.0-dev layout (probes/version_drift.md §2).

    xgboost 3.3.0 -- the only version installed here -- cannot itself
    produce ``weight_drop`` nested inside ``gradient_booster.model``; that
    relocation was only observed on a 3.4.0-dev nightly build. The model is
    otherwise a genuine fitted dart artifact; only the one field is moved by
    hand, which is exactly the shape probes/version_drift.md §2 measured
    upstream doing on its own.
    """
    model = _fit(
        {"objective": "binary:logistic", "booster": "dart", "rate_drop": 0.3, "skip_drop": 0.1}
    )
    gradient_booster = model["learner"]["gradient_booster"]
    weight_drop = gradient_booster.pop("weight_drop")
    gradient_booster["model"]["weight_drop"] = weight_drop

    assert "weight_drop" not in gradient_booster
    assert "weight_drop" in gradient_booster["model"]

    with pytest.raises(errors.UnsupportedBoosterError) as excinfo:
        validate_source_model(model, tested_versions=_tested(model))
    assert excinfo.value.booster == "dart"


# ---------------------------------------------------------------------------
# Early stopping: the predicate is ambiguity, not mere presence of best_iteration.
# ---------------------------------------------------------------------------


def test_rejects_early_stopped_model_with_ambiguous_tree_count() -> None:
    rng = np.random.default_rng(20260807)
    train_x = rng.normal(size=(200, 4))
    train_y = rng.normal(size=200)
    val_x = rng.normal(size=(50, 4))
    val_y = rng.normal(size=50)
    dtrain = xgb.DMatrix(train_x, label=train_y, feature_names=["f0", "f1", "f2", "f3"])
    dval = xgb.DMatrix(val_x, label=val_y, feature_names=["f0", "f1", "f2", "f3"])

    booster = xgb.train(
        {**_BASE_PARAMS, "objective": "reg:squarederror", "seed": 1},
        dtrain,
        num_boost_round=20,
        evals=[(dval, "val")],
        early_stopping_rounds=3,
        verbose_eval=False,
    )
    model = json.loads(booster.save_raw(raw_format="json"))

    best_iteration = int(model["learner"]["attributes"]["best_iteration"])
    iteration_indptr = model["learner"]["gradient_booster"]["model"]["iteration_indptr"]
    total_trees = len(model["learner"]["gradient_booster"]["model"]["trees"])
    assert iteration_indptr[best_iteration + 1] != total_trees

    with pytest.raises(errors.AmbiguousTreeCountError) as excinfo:
        validate_source_model(model, tested_versions=_tested(model))
    assert excinfo.value.best_iteration == best_iteration
    assert excinfo.value.total_trees == total_trees


def test_accepts_early_stopped_model_whose_best_iteration_is_unambiguous() -> None:
    """best_iteration present, but the model never actually diverged from it.

    Evaluating against the training set itself gives a monotonically
    improving metric, so the best iteration coincides with the last one and
    both readings of the tree count agree (D038). Refusing on the mere
    presence of ``best_iteration`` would incorrectly reject this model.
    """
    rng = np.random.default_rng(20260808)
    train_x = rng.normal(size=(200, 4))
    train_y = rng.normal(size=200)
    dtrain = xgb.DMatrix(train_x, label=train_y, feature_names=["f0", "f1", "f2", "f3"])

    booster = xgb.train(
        {**_BASE_PARAMS, "objective": "reg:squarederror", "seed": 1},
        dtrain,
        num_boost_round=5,
        evals=[(dtrain, "train")],
        early_stopping_rounds=10,
        verbose_eval=False,
    )
    model = json.loads(booster.save_raw(raw_format="json"))

    best_iteration = int(model["learner"]["attributes"]["best_iteration"])
    iteration_indptr = model["learner"]["gradient_booster"]["model"]["iteration_indptr"]
    total_trees = len(model["learner"]["gradient_booster"]["model"]["trees"])
    assert iteration_indptr[best_iteration + 1] == total_trees

    _accepts(model)


def test_accepts_model_with_no_best_iteration_attribute_at_all() -> None:
    model = _fit({"objective": "reg:squarederror"})
    assert "best_iteration" not in model["learner"].get("attributes", {})
    _accepts(model)
