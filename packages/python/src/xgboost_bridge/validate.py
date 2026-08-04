"""The export-time validation gate.

``validate_source_model`` decides whether a fitted XGBoost model, as parsed
from ``booster.save_raw(raw_format="json")``, may be exported at all. Every
check here exists because a model that slips through produces a confident,
plausible, *wrong* prediction rather than an error -- see ``CLAUDE.md``.

This module reads only string and integer **metadata** fields -- objective
name, arity counters, booster name, structural-signal presence, feature-name
presence, iteration bookkeeping, and the version marker. It never reads,
stores, transforms, or compares a split threshold, a leaf value, a
``base_score``, or an intercept; that is the numerical core and belongs to
the tree walk, not to this gate.

All string-typed fields on a serialized model are compared as strings, never
coerced to ``int``. ``num_class == 0`` is ``False`` against the string
``"0"`` -- an integer comparison would silently never fire, disabling the
gate rather than tripping it.

Every check raises on the first failure. Nothing here defaults, guesses, or
infers by analogy; a caller either gets ``None`` back, meaning the model may
be exported, or one of the structured exceptions in :mod:`xgboost_bridge.errors`.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .errors import (
    AmbiguousTreeCountError,
    CategoricalSplitError,
    MalformedTreeError,
    MissingFeatureNamesError,
    UnsupportedBoosterError,
    UnsupportedModelShapeError,
    UnsupportedObjectiveError,
    UnsupportedVersionError,
)

#: Objectives this version of the library supports. Exact string match only
#: -- no prefix-matching, no case normalization.
SUPPORTED_OBJECTIVES: tuple[str, ...] = (
    "reg:squarederror",
    "binary:logistic",
    "survival:cox",
)

#: Boosters this version of the library supports.
SUPPORTED_BOOSTERS: tuple[str, ...] = ("gbtree",)


def validate_source_model(
    model: dict[str, Any], *, tested_versions: frozenset[str]
) -> None:
    """Raise if ``model`` may not be exported; otherwise return ``None``.

    Args:
        model: The parsed output of ``booster.save_raw(raw_format="json")``.
        tested_versions: XGBoost version strings (e.g. ``"3.3.0"``) that have
            been empirically verified against this library. A producing
            version outside this set raises, because an untested version is
            an unrecognized input.

    The booster check runs before the output-arity check rather than after
    it. ``gblinear``'s serialized ``gradient_booster.model`` has no
    ``trees`` key at all -- reading arity fields per tree on such a model
    would raise ``KeyError`` instead of the intended
    :class:`~xgboost_bridge.errors.UnsupportedBoosterError`. Every other
    check's relative order matches the brief this module was built from.
    """
    learner = _required(model, "learner", None)

    _check_objective(learner)
    _check_booster(learner)
    _check_output_arity(learner)
    _check_categorical_splits(learner)
    _check_feature_names(learner)
    _check_early_stopping(learner)
    _check_version(model, tested_versions=tested_versions)


def _required(container: Any, key: str, location: str | None) -> Any:
    """Read ``key`` out of a JSON object, raising rather than defaulting.

    Every field this gate reads is required for the check that reads it.
    An absent field is an unrecognized artifact shape (FORMAT.md section
    13) -- a relocated or removed field, not a value this module is
    entitled to infer or silently skip past (see CLAUDE.md's "fail loudly"
    invariant). ``KeyError``, ``IndexError``, and ``TypeError`` are exactly
    the exceptions this function exists to intercept before they escape as
    unstructured failures.
    """
    if not isinstance(container, dict) or key not in container:
        raise MalformedTreeError(key, "<absent>", "the field to be present", location)
    return container[key]


def _trees(learner: dict[str, Any]) -> list[dict[str, Any]]:
    gradient_booster = _required(learner, "gradient_booster", "learner")
    model = _required(gradient_booster, "model", "learner.gradient_booster")
    return _required(model, "trees", "learner.gradient_booster.model")


def _check_objective(learner: dict[str, Any]) -> None:
    """Objective must be one of the three in-scope objectives, exactly."""
    objective = _required(learner, "objective", "learner")
    name = _required(objective, "name", "learner.objective")
    if name not in SUPPORTED_OBJECTIVES:
        raise UnsupportedObjectiveError(name, SUPPORTED_OBJECTIVES)


def _check_output_arity(learner: dict[str, Any]) -> None:
    """Reject any shape that produces more than one output per row.

    The objective allow-list alone has a hole: ``reg:squarederror`` with
    ``num_target=2`` is an in-scope objective that produces ``(N, 2)``
    margins. Arity is therefore checked on the fields that actually
    determine it, independently of the objective name.

    All three fields are JSON strings and are compared as strings.
    ``num_class`` accepting ``"1"`` is deliberate: a genuine single-output
    model can carry ``num_class == "1"``, and requiring ``"0"`` falsely
    rejects it. ``size_leaf_vector`` exists only per tree, never at the
    model level, so a zero-tree model passes this check vacuously.
    """
    model_param = _required(learner, "learner_model_param", "learner")

    num_target = _required(model_param, "num_target", "learner.learner_model_param")
    if num_target != "1":
        raise UnsupportedModelShapeError(
            "num_target",
            num_target,
            'exactly "1"',
            location="learner.learner_model_param.num_target",
        )

    num_class = _required(model_param, "num_class", "learner.learner_model_param")
    if num_class not in ("0", "1"):
        raise UnsupportedModelShapeError(
            "num_class",
            num_class,
            '"0" or "1"',
            location="learner.learner_model_param.num_class",
        )

    for index, tree in enumerate(_trees(learner)):
        location = f"learner.gradient_booster.model.trees[{index}]"
        tree_param = _required(tree, "tree_param", location)
        size_leaf_vector = _required(tree_param, "size_leaf_vector", f"{location}.tree_param")
        if size_leaf_vector != "1":
            raise UnsupportedModelShapeError(
                "size_leaf_vector",
                size_leaf_vector,
                'exactly "1"',
                location="tree_param.size_leaf_vector",
            )


def _check_booster(learner: dict[str, Any]) -> None:
    """Reject anything other than a plain tree ensemble.

    ``gradient_booster.name`` alone cannot detect dart -- dart serializes it
    as ``"gbtree"``, indistinguishable from a plain tree ensemble on that
    field. The only in-artifact dart signal is the presence of
    ``weight_drop``, and it relocated between XGBoost 3.3.0 and 3.4.0-dev,
    so both known paths are checked.
    """
    gradient_booster = _required(learner, "gradient_booster", "learner")

    booster = _required(gradient_booster, "name", "learner.gradient_booster")
    if booster not in SUPPORTED_BOOSTERS:
        raise UnsupportedBoosterError(booster, SUPPORTED_BOOSTERS)

    # Reached only for a "gbtree"-named booster (the branch above already
    # raised for anything else, including gblinear) -- every such artifact
    # carries a "model" key, so requiring it here cannot reopen the
    # gblinear-KeyError hole the check order above exists to close.
    model = _required(gradient_booster, "model", "learner.gradient_booster")
    if "weight_drop" in gradient_booster or "weight_drop" in model:
        raise UnsupportedBoosterError("dart", SUPPORTED_BOOSTERS)


def _check_categorical_splits(learner: dict[str, Any]) -> None:
    """Reject any model containing a categorical split.

    Three independent signals are measured to appear together on a
    categorical model. All three are checked and reported, because this is
    a refusal test where redundancy is free.
    """
    signals: list[str] = []
    trees = _trees(learner)

    if any(
        1
        in _required(
            tree, "split_type", f"learner.gradient_booster.model.trees[{index}]"
        )
        for index, tree in enumerate(trees)
    ):
        signals.append("split_type")

    if any(
        _required(
            tree, "categories_nodes", f"learner.gradient_booster.model.trees[{index}]"
        )
        for index, tree in enumerate(trees)
    ):
        signals.append("categories_nodes")

    if "c" in _required(learner, "feature_types", "learner"):
        signals.append("feature_types")

    if signals:
        raise CategoricalSplitError(tuple(signals))


def _check_feature_names(learner: dict[str, Any]) -> None:
    """Reject a model whose ``feature_names`` cannot support a strict-key policy.

    Three independent conditions, per FORMAT.md section 11 / D021, all
    reachable through the ordinary public API with no warning:

    * Empty while ``num_feature`` is nonzero -- a model fit from a bare
      array. A strict feature-key policy with no keys to check reads as
      enforced and is not.
    * A duplicate name -- two identical keys under a strict-key policy make
      one of them unreachable, which is the same "reads as enforced and is
      not" failure by a different route (``booster.feature_names = [...]``
      accepts duplicates with no error).
    * A length that disagrees with ``num_feature`` -- the model's declared
      feature count and its declared feature names describe two different
      shapes, and a split on the unnamed column has no key to report.
    """
    feature_names = _required(learner, "feature_names", "learner")
    model_param = _required(learner, "learner_model_param", "learner")
    num_feature = int(_required(model_param, "num_feature", "learner.learner_model_param"))

    if not feature_names and num_feature != 0:
        raise MissingFeatureNamesError(num_feature)

    duplicates = tuple(
        sorted(name for name, count in Counter(feature_names).items() if count > 1)
    )
    if duplicates:
        raise UnsupportedModelShapeError(
            "feature_names",
            duplicates,
            "no duplicate names",
            location="learner.feature_names",
        )

    if len(feature_names) != num_feature:
        raise UnsupportedModelShapeError(
            "feature_names",
            len(feature_names),
            f'a length equal to num_feature ("{num_feature}")',
            location="learner.feature_names",
        )


def _check_early_stopping(learner: dict[str, Any]) -> None:
    """Reject an early-stopped model whose effective tree count is ambiguous.

    The predicate is not "is ``best_iteration`` present" -- with
    ``early_stopping_rounds`` set but never fired, ``best_iteration`` is
    present and both readings of the tree count already agree, so refusing
    on presence alone would reject an unambiguous model. The predicate is
    whether ``trees`` actually extends past the iteration ``best_iteration``
    names.
    """
    # ``attributes`` and ``best_iteration`` genuinely are optional: their
    # absence legitimately means "no refusal needed", not an unrecognized
    # shape -- so this is the one pair of reads in this module that stays a
    # default rather than becoming a ``_required`` call. What must not
    # default is everything *downstream* of learning that a best iteration
    # was recorded: a malformed or out-of-range value there is not absence,
    # it is a present value this module refuses to interpret via a bare
    # ``int()`` / list-index crash.
    attributes = learner.get("attributes", {})
    best_iteration_raw = attributes.get("best_iteration") if isinstance(attributes, dict) else None
    if best_iteration_raw is None:
        return

    try:
        best_iteration = int(best_iteration_raw)
    except (TypeError, ValueError) as exc:
        raise MalformedTreeError(
            "best_iteration",
            best_iteration_raw,
            "an integer-valued string",
            "learner.attributes.best_iteration",
        ) from exc

    gradient_booster = _required(learner, "gradient_booster", "learner")
    model = _required(gradient_booster, "model", "learner.gradient_booster")
    iteration_indptr = _required(model, "iteration_indptr", "learner.gradient_booster.model")
    total_trees = len(_required(model, "trees", "learner.gradient_booster.model"))

    if best_iteration < 0 or best_iteration + 1 >= len(iteration_indptr):
        raise MalformedTreeError(
            "best_iteration",
            best_iteration,
            "an index i such that iteration_indptr has an entry at i + 1 "
            f"(len(iteration_indptr)={len(iteration_indptr)})",
            "learner.attributes.best_iteration",
        )

    effective_trees = iteration_indptr[best_iteration + 1]

    if effective_trees != total_trees:
        raise AmbiguousTreeCountError(best_iteration, effective_trees, total_trees)


def _check_version(model: dict[str, Any], *, tested_versions: frozenset[str]) -> None:
    """Reject a model produced by an XGBoost version this library has not verified.

    Unrecognized-field detection catches additions and cannot catch
    relocations or removals -- a missing optional field is not an unknown
    field. Only an explicit, enumerated ceiling defends against that class
    of drift, so an untested version is treated as an unrecognized input.
    """
    version_marker = _required(model, "version", None)
    version_string = ".".join(str(component) for component in version_marker)

    if version_string not in tested_versions:
        raise UnsupportedVersionError(version_string, tuple(sorted(tested_versions)))
