"""Per-objective intercept derivation, and its independent-oracle check.

This module owns the single most error-prone value in the library. The
intercept is the initial value of the margin accumulator (FORMAT.md section
10), so it is added to *every* prediction: an error here is a constant offset
on every row of every artifact, and it is a plausible number rather than a
crash.

Nothing here is derivable by reasoning. Every rule below is a measured
finding recorded under ``probes/``, and each has a cheaper-looking wrong
version that is correct on most inputs and wrong on a few:

* The **space** is per-objective and is never inferred by analogy
  (``probes/base_score.md`` sections 3-5). ``reg:squarederror`` stores the
  intercept in margin space, ``binary:logistic`` in probability space,
  ``survival:cox`` in hazard-ratio space.
* ``boost_from_average`` selects whether a link transform applies at all.
  With **zero trees** and ``boost_from_average == "1"`` XGBoost emits the
  **raw** ``base_score`` as the margin (D036, ``probes/arity_gate.md``
  section 7). Flipping that one string moves a logistic zero-tree margin
  between ``0.5`` and ``-0.0``.
* ``binary:logistic`` clamps ``p`` **before** the transform while storing it
  unclamped (D035, D039). Applying the recipe to the stored value is wrong by
  up to ``13.8`` in margin space, and is a domain error at
  ``base_score = 1 - 1e-10``, which stores as ``[1E0]``.
* Both logarithms are **float32** logarithms -- ``np.log`` of a float32, not
  a float64 logarithm narrowed afterwards (D040). The two routes disagree on
  0.055% of float32 inputs, so an ordinary sweep finds nothing; on the
  disagreeing inputs only the float32 route matches XGBoost.
* The textbook ``log(p / (1 - p))`` is **not** the logistic transform. The
  float32 ``1 / p - 1`` intermediate is (``probes/base_score.md`` section 5).

Float32 discipline is a property of this module, not a habit: the stored
decimal is snapped to float32 on parse, every arithmetic step is wrapped in
``np.float32``, and the emitted value is ``float(np.float32(v))`` so it is a
float64 that recovers exactly one float32 (FORMAT.md section 9.1).

Signed zero is reachable through an ordinary default and is never
normalized: ``binary:logistic`` at ``base_score = 0.5`` derives exactly
``-0.0``, bit pattern ``0x80000000``. Comparison anywhere in this module is
on bit patterns, because ``-0.0 == 0.0`` is ``True`` and the two are
different artifacts.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np

from .errors import (
    InterceptMismatchError,
    MalformedTreeError,
    UnsupportedObjectiveError,
)

#: Margin-space-to-output transform per objective, measured rather than
#: assumed: eleven candidate transforms were scored across 43 fitted models
#: and the winners below are float32 bit-exact on 107500/107500 rows
#: (``probes/output_transform.md``). Carried in the artifact explicitly
#: rather than derived at read time (FORMAT.md section 5); this mapping is
#: what export writes and what export cross-checks against ``objective``.
OUTPUT_TRANSFORMS: Mapping[str, str] = MappingProxyType(
    {
        "reg:squarederror": "identity",
        "binary:logistic": "sigmoid",
        "survival:cox": "exp",
    }
)

#: Objectives whose intercept space has been verified against a real fitted
#: artifact. Derived from :data:`OUTPUT_TRANSFORMS` so this module has one
#: authority for "which objectives exist", not two that can disagree. An
#: objective absent here raises; nothing is inferred by analogy.
SUPPORTED_OBJECTIVES: tuple[str, ...] = tuple(OUTPUT_TRANSFORMS)

# The logistic clamp bounds. XGBoost clamps `base_score` into this interval
# before deriving the intercept and stores the value unclamped, so the
# clamp cannot be recovered from the artifact and must be reapplied here
# (D035). Both bounds are pinned to an adjacent float32 pair by exhaustive
# search (D039, `probes/base_score_clamp.md` sections 2-4); the constants
# below sit inside the observational equivalence class of the true source
# literals -- 8 admissible float32 values for the low bound, 2 for the high
# -- and every member gives bit-identical intercepts on every float32 input,
# scoring 226/226 against XGBoost where no clamp scores 52/226.
_LOGISTIC_CLAMP_LOW = np.float32(1e-6)
_LOGISTIC_CLAMP_HIGH = np.float32(1.0) - np.float32(1e-6)

_ONE = np.float32(1.0)

# `learner_model_param.boost_from_average` is a JSON *string*. Only these
# two values have been observed; anything else is an unrecognized input and
# raises rather than being read as truthy or falsy.
_BOOST_FROM_AVERAGE_VALUES = ("0", "1")

# Labels for the zero-round oracle fit in `verify_intercept`. A zero-round
# model has no trees, so the labels cannot influence the margin -- but each
# objective validates its labels at configure time, so they must be legal.
# `survival:cox` uses the sign convention: positive is an event, negative is
# right-censored.
_ORACLE_LABELS: Mapping[str, tuple[float, ...]] = MappingProxyType(
    {
        "reg:squarederror": (0.0, 1.0),
        "binary:logistic": (0.0, 1.0),
        "survival:cox": (1.0, -2.0),
    }
)


def derive_intercept(model: dict[str, Any]) -> float:
    """Return the margin-space intercept for a parsed source model.

    Args:
        model: The parsed output of ``booster.save_raw(raw_format="json")``.

    Returns:
        The intercept as ``float(np.float32(value))`` -- a float64 that
        recovers exactly one float32, ready to emit under FORMAT.md section
        9.1 with no further rounding or formatting. Negative zero is
        returned as negative zero.

    Raises:
        UnsupportedObjectiveError: The objective has no verified intercept
            space in this version.
        MalformedTreeError: A field this derivation reads is absent, is not
            the type every probed artifact carries, or holds a value no
            probe has observed -- including a ``base_score`` that is not a
            bracketed one-element string, and a ``boost_from_average``
            outside ``{"0", "1"}``.

    The two steps are ordered and neither is optional:

    1. ``boost_from_average`` selects the space. Zero trees **and**
       ``boost_from_average == "1"`` means XGBoost emits the raw
       ``base_score``, for every objective. Any other combination applies
       the per-objective link transform. That is one cell of a measured
       four-cell table, not a heuristic (D036).
    2. Clamp, then transform. The clamp applies to ``binary:logistic``
       only; ``survival:cox`` and ``reg:squarederror`` are unclamped,
       verified across 34 and 25 values spanning the float32 range (D035,
       D039).
    """
    learner = _required(model, "learner", None)
    objective = _objective_name(learner)
    model_param = _required(learner, "learner_model_param", "learner")

    base_score = _read_base_score(model_param)
    boost_from_average = _read_boost_from_average(model_param)
    tree_count = len(_read_trees(learner))

    if tree_count == 0 and boost_from_average == "1":
        # Measured, not chosen: in this one cell XGBoost never ran the
        # prob-to-margin conversion, so the stored value *is* the margin.
        # Applying the link here gives 0.5 where XGBoost gives -0.0 for
        # logistic, and 0.5 where it gives -0.693147 for Cox.
        return _emit(base_score)

    return _emit(_link_transform(objective, base_score))


def verify_intercept(booster: Any, derived: float) -> None:
    """Raise unless ``derived`` matches XGBoost's own zero-tree margin.

    Args:
        booster: The ``xgboost.Booster`` the intercept was derived from.
        derived: The value :func:`derive_intercept` returned for that
            booster's serialized model.

    Raises:
        InterceptMismatchError: ``derived`` is not bit-identical to the
            margin XGBoost itself reports for a zero-tree model in the same
            configuration.
        UnsupportedObjectiveError: The objective is not one this version
            supports, so no oracle can be constructed for it.
        MalformedTreeError: The oracle model did not come out in the
            configuration it was asked for -- wrong tree count, wrong
            ``boost_from_average``, or a ``base_score`` that did not survive
            the round trip. That is a broken instrument, not a mismatch, and
            it is reported as such rather than compared.

    **What the oracle is, and why it cannot share the defect.** The oracle
    is XGBoost's *observed* margin, read from ``predict(output_margin=True)``
    -- never a re-derivation of the recipe in this module. An earlier
    version of this check compared a derived intercept against a second
    derivation from ``base_score``; both sides ran the same recipe, so a
    recipe error could not make it fire, and it passed the real
    ``base_score`` clamping defect that D035 fixes (D034, FORMAT.md section
    6.2). This check fails on any recipe error, including that one.

    Two oracle shapes, because "the same configuration" differs by tree
    count:

    * **Zero trees.** The booster's own margin already *is* the zero-tree
      margin, in either ``boost_from_average`` cell. It is read directly.
      Re-fitting instead would be wrong: passing ``base_score`` explicitly
      flips ``boost_from_average`` to ``"0"`` and the oracle would come
      back in link space for a model whose margin is the raw value.
    * **Trees present.** A fresh model is fitted with zero boosting rounds
      and the same objective, with ``base_score`` passed **explicitly** at
      the source model's stored value. Explicit is mandatory: left at its
      default, ``boost_from_average`` stays ``"1"`` and the oracle returns
      the raw value instead of the transform.

    Comparison is on float32 bit patterns. ``derived`` must also survive the
    float32 round trip exactly, since a value that is not float32 is not a
    value XGBoost can have produced.
    """
    document = json.loads(booster.save_raw(raw_format="json"))
    learner = _required(document, "learner", None)
    objective = _objective_name(learner)
    model_param = _required(learner, "learner_model_param", "learner")
    base_score = _read_base_score(model_param)
    tree_count = len(_read_trees(learner))

    if tree_count == 0:
        observed = _observed_margin(booster, learner)
    else:
        observed = _observed_zero_round_margin(objective, base_score)

    derived32 = np.float32(derived)
    round_trips = _bits64(np.float64(float(derived32))) == _bits64(
        np.float64(derived)
    )
    if not round_trips or _bits32(derived32) != _bits32(observed):
        raise InterceptMismatchError(derived, float(observed), objective)


def _link_transform(objective: str, base_score: np.float32) -> np.float32:
    """Apply the per-objective prob/hazard-to-margin transform."""
    if objective == "reg:squarederror":
        # Margin space already; the link is the identity. Not a shortcut --
        # `ln` is off by up to 1227 and `logit` is usually out of domain
        # here, so this objective is the one that fails loudly when
        # confused with another.
        return base_score
    if objective == "survival:cox":
        return _cox_intercept(base_score)
    if objective == "binary:logistic":
        return _logistic_intercept(base_score)
    raise UnsupportedObjectiveError(objective, SUPPORTED_OBJECTIVES)


def _cox_intercept(base_score: np.float32) -> np.float32:
    """``log(f32(base_score))``, with a **float32** logarithm. No clamp.

    ``np.log`` of a float32 returns a float32 and is the measured route:
    6947/6947 bit-exact against XGBoost, against 6912/6947 for a float64
    logarithm narrowed afterwards, and 6912/6947 for a correctly-rounded
    ``mpmath`` reference (D040). XGBoost's Cox logarithm is not correctly
    rounded, so widening or improving the logarithm makes this *worse*.

    ``base_score <= 0`` is accepted by XGBoost and yields ``-inf`` or
    ``NaN``; the state is reproduced rather than second-guessed here, and
    numpy's warnings are suppressed so the value does not depend on ambient
    warning filters. Whether such an intercept is exportable is a format
    question (FORMAT.md sections 6, 9.3), not this transform's to decide.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.float32(np.log(base_score))


def _logistic_intercept(base_score: np.float32) -> np.float32:
    """``-log(f32(f32(1/p) - 1))`` of the **clamped** ``p``, in float32.

    Three separate measured facts are load-bearing in these five lines:

    * the clamp, which XGBoost applies and does not store (D035, D039);
    * the ``1 / p - 1`` intermediate rather than the ratio ``p / (1 - p)``
      -- the textbook form is bit-wrong on 16 of 27 measured values and
      breaches the ``1e-6`` margin gate (``probes/base_score.md`` section
      5);
    * the float32 logarithm (D040).

    Clamping the input rather than the derived intercept also keeps the
    logarithm inside its domain: at ``base_score = 1.0``, which XGBoost
    accepts and stores as ``[1E0]``, the unclamped argument is exactly
    ``0.0``.
    """
    probability = base_score
    if probability < _LOGISTIC_CLAMP_LOW:
        probability = _LOGISTIC_CLAMP_LOW
    elif probability > _LOGISTIC_CLAMP_HIGH:
        probability = _LOGISTIC_CLAMP_HIGH

    reciprocal = np.float32(_ONE / probability)
    odds = np.float32(reciprocal - _ONE)
    return np.float32(-np.log(odds))


def _emit(value: np.float32) -> float:
    """Widen a float32 to the float64 that recovers exactly that float32.

    Python's shortest-round-trip repr of this float64 is what FORMAT.md
    section 9.1 requires on the wire. ``float()`` of a float32 preserves the
    sign of zero, so ``-0.0`` survives.
    """
    return float(np.float32(value))


def _objective_name(learner: dict[str, Any]) -> str:
    objective = _required(learner, "objective", "learner")
    name = _required(objective, "name", "learner.objective")
    if name not in SUPPORTED_OBJECTIVES:
        raise UnsupportedObjectiveError(name, SUPPORTED_OBJECTIVES)
    return str(name)


def _read_base_score(model_param: dict[str, Any]) -> np.float32:
    """Parse ``base_score`` and snap it to float32 in one step.

    XGBoost stores this field as a JSON **string containing a bracketed
    array** -- ``"[4.8E-1]"`` -- so it needs two parses
    (``probes/base_score.md`` section 1). A non-string, a missing bracket,
    or more than one element raises: a multi-element vector has not been
    observed and its meaning is not established.

    The snap is not cosmetic. ``float("[4.8E-1]"[1:-1])`` is ``0.48``; the
    value XGBoost holds is ``0.47999998927116394``, and the transforms
    disagree if the unsnapped float64 is used.
    """
    location = "learner.learner_model_param.base_score"
    stored = _required(model_param, "base_score", "learner.learner_model_param")

    if not isinstance(stored, str):
        raise MalformedTreeError(
            "base_score", stored, "a JSON string, as every probed artifact carries", location
        )
    if not (stored.startswith("[") and stored.endswith("]")) or len(stored) < 3:
        raise MalformedTreeError(
            "base_score", stored, "a bracketed one-element array, e.g. '[4.8E-1]'", location
        )

    inner = stored[1:-1]
    if "," in inner:
        raise MalformedTreeError(
            "base_score",
            stored,
            "exactly one element; a multi-element base_score has not been observed",
            location,
        )
    try:
        parsed = float(inner)
    except ValueError:
        raise MalformedTreeError(
            "base_score", stored, "a decimal number inside the brackets", location
        ) from None

    # A decimal outside float32's range casts to an infinity, which numpy
    # warns about. The raise below is the loud path; the warning would only
    # make it depend on ambient error state.
    with np.errstate(over="ignore"):
        snapped = np.float32(parsed)
    if not np.isfinite(snapped):
        raise MalformedTreeError(
            "base_score",
            stored,
            "a finite value; XGBoost's own parser refuses nan and inf here",
            location,
        )
    return snapped


def _read_boost_from_average(model_param: dict[str, Any]) -> str:
    """Read ``boost_from_average``, a JSON string, comparing it as a string.

    An integer comparison against ``"1"`` silently never fires, which would
    disable the space selection rather than trip it -- and the failure would
    be a wrong intercept on exactly the zero-tree models FORMAT.md section
    6.3 requires the fixture corpus to contain.
    """
    location = "learner.learner_model_param.boost_from_average"
    stored = _required(
        model_param, "boost_from_average", "learner.learner_model_param"
    )
    if stored not in _BOOST_FROM_AVERAGE_VALUES:
        raise MalformedTreeError(
            "boost_from_average",
            stored,
            'the string "0" or "1"',
            location,
        )
    return str(stored)


def _read_trees(learner: dict[str, Any]) -> list[Any]:
    """Return the tree list. A zero-round model serializes ``[]``, present
    and empty, verified on all three objectives -- so an absent key is a
    shape no probe measured and raises."""
    gradient_booster = _required(learner, "gradient_booster", "learner")
    model = _required(gradient_booster, "model", "learner.gradient_booster")
    trees = _required(model, "trees", "learner.gradient_booster.model")
    if not isinstance(trees, list):
        raise MalformedTreeError(
            "trees",
            trees,
            "a list, present and possibly empty",
            "learner.gradient_booster.model.trees",
        )
    return trees


def _observed_margin(booster: Any, learner: dict[str, Any]) -> np.float32:
    """Read a zero-tree booster's own margin -- XGBoost's observed value.

    With no trees the margin is constant across rows, so one row of
    arbitrary feature values is sufficient; the constancy is asserted rather
    than assumed.
    """
    import xgboost  # noqa: PLC0415 -- optional extra; see D010

    feature_names = _required(learner, "feature_names", "learner")
    model_param = _required(learner, "learner_model_param", "learner")
    column_count = int(_required(model_param, "num_feature", "learner.learner_model_param"))

    matrix = xgboost.DMatrix(
        np.zeros((1, column_count), dtype=np.float32),
        feature_names=list(feature_names) or None,
        nthread=1,
    )
    margin = np.asarray(booster.predict(matrix, output_margin=True), dtype=np.float32)
    return _single_margin(margin, "the booster's own zero-tree margin")


def _observed_zero_round_margin(
    objective: str, base_score: np.float32
) -> np.float32:
    """Fit a zero-round oracle model and read the margin XGBoost reports.

    ``base_score`` is passed **explicitly**, which is what flips
    ``boost_from_average`` to ``"0"`` and puts the oracle in link space. Left
    at its default the oracle would return the raw value and would agree
    with a wrong recipe (D036).
    """
    import xgboost  # noqa: PLC0415 -- optional extra; see D010

    labels = _ORACLE_LABELS.get(objective)
    if labels is None:
        raise UnsupportedObjectiveError(objective, SUPPORTED_OBJECTIVES)

    features = np.zeros((len(labels), 1), dtype=np.float32)
    matrix = xgboost.DMatrix(
        features, label=np.asarray(labels, dtype=np.float32), nthread=1
    )
    oracle = xgboost.train(
        {
            "objective": objective,
            "base_score": float(base_score),
            "nthread": 1,
        },
        matrix,
        num_boost_round=0,
    )

    document = json.loads(oracle.save_raw(raw_format="json"))
    learner = _required(document, "learner", None)
    model_param = _required(learner, "learner_model_param", "learner")
    _assert_oracle_shape(learner, model_param, base_score)

    margin = np.asarray(oracle.predict(matrix, output_margin=True), dtype=np.float32)
    return _single_margin(margin, "the zero-round oracle margin")


def _assert_oracle_shape(
    learner: dict[str, Any],
    model_param: dict[str, Any],
    base_score: np.float32,
) -> None:
    """Verify the oracle came out in the configuration it was asked for.

    Three ways the instrument could be silently wrong, each checked:
    trees present (then the margin is not the intercept),
    ``boost_from_average != "0"`` (then the margin is the raw value), and a
    ``base_score`` that did not survive the round trip (then a different
    value was measured than the one under test).
    """
    tree_count = len(_read_trees(learner))
    if tree_count != 0:
        raise MalformedTreeError(
            "trees",
            tree_count,
            "zero trees in the zero-round oracle model",
            "learner.gradient_booster.model.trees",
        )

    boost_from_average = _read_boost_from_average(model_param)
    if boost_from_average != "0":
        raise MalformedTreeError(
            "boost_from_average",
            boost_from_average,
            '"0" in the oracle model, since base_score was passed explicitly',
            "learner.learner_model_param.boost_from_average",
        )

    stored = _read_base_score(model_param)
    if _bits32(stored) != _bits32(base_score):
        raise MalformedTreeError(
            "base_score",
            float(stored),
            f"the value under test, float32 bits {_bits32(base_score)}",
            "learner.learner_model_param.base_score",
        )


def _single_margin(margin: np.ndarray, description: str) -> np.float32:
    """Return the one float32 value a zero-tree margin must be.

    Constancy is checked on **bit patterns**, not with ``==``: a NaN margin
    compares unequal to itself, so a value-based check would report one
    distinct value per row and pass exactly where it should fail.
    """
    if margin.ndim != 1 or margin.size == 0:
        raise MalformedTreeError(
            "margin", margin.shape, f"a non-empty one-dimensional {description}", None
        )
    patterns = set(margin.view(np.uint32).tolist())
    if len(patterns) != 1:
        raise MalformedTreeError(
            "margin",
            sorted(patterns),
            f"one float32 bit pattern across all rows of {description}",
            None,
        )
    return np.float32(margin[0])


def _bits32(value: np.float32) -> int:
    return int(np.float32(value).view(np.uint32))


def _bits64(value: np.float64) -> int:
    return int(np.float64(value).view(np.uint64))


def _required(container: Any, key: str, location: str | None) -> Any:
    """Read a required field, raising rather than defaulting.

    Every field this module reads is required by the derivation that reads
    it. An absent field is a shape no probe measured -- a relocation or a
    removal, which unrecognized-*field* detection structurally cannot catch
    -- not a value this module may infer.
    """
    if not isinstance(container, dict) or key not in container:
        raise MalformedTreeError(key, "<absent>", "the field to be present", location)
    return container[key]
