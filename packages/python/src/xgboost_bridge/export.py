"""Artifact assembly and deterministic serialization -- the public export entry point.

This module is the mechanical half of export. It computes no threshold, no
leaf value, and no intercept: every one of those arrives from
:mod:`xgboost_bridge.trees` and :mod:`xgboost_bridge.objectives` already
emission-ready as ``float(np.float32(x))`` (FORMAT.md section 9.1). This
module's job is to run the gate, place those values into the seven-key
envelope FORMAT.md section 3 specifies, and serialize the result
byte-identically on every call.

Nothing here rounds, reformats, or otherwise "tidies" a number. Re-emitting a
threshold at reduced precision lands on a different float32 on 2 of 341
measured values at eight significant digits (FORMAT.md section 9.1) -- often
enough to be a wrong prediction, rare enough to survive review -- so every
number that reaches :func:`to_json` is serialized by Python's ordinary
``json`` float repr and nothing else touches it.

Order of operations matters and is deliberate: the export-time gate
(:func:`xgboost_bridge.validate.validate_source_model`) runs before any
numeric value is read, so a refused model never reaches the numeric path.
Trees are extracted, the intercept is derived, and *then* the intercept is
checked for finiteness and verified against XGBoost's own observed margin --
never the other way around, because a non-finite value would otherwise reach
an oracle comparison that cannot distinguish ``NaN`` from ``NaN`` (D043).

``provenance`` carries ``xgboost_version``, ``base_score`` (the value exactly
as XGBoost stored it, with no parsing and no float32 transform applied here),
and ``exporter_version``. **No predictor reads any field of ``provenance``.**
It exists for a human inspecting an artifact after the fact; nothing on the
prediction path may ever become dependent on it, or the field silently stops
being metadata (FORMAT.md sections 6.1, 15; D020).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from typing import Any

from ._version import __version__
from .errors import (
    MalformedTreeError,
    NonFiniteInterceptError,
    UnsupportedModelShapeError,
    UnsupportedObjectiveError,
)
from .objectives import OUTPUT_TRANSFORMS, derive_intercept, verify_intercept
from .trees import extract_trees
from .validate import validate_source_model

__all__ = ["DEFAULT_TESTED_VERSIONS", "export_model", "to_json"]

#: FORMAT.md section 3 -- the seven required top-level keys, and no others.
ENVELOPE_KEYS: frozenset[str] = frozenset(
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

#: ``provenance``'s own fixed key set (FORMAT.md sections 2, 15). No
#: predictor reads any of these three fields.
PROVENANCE_KEYS: frozenset[str] = frozenset(
    {"xgboost_version", "base_score", "exporter_version"}
)

#: The only artifact format version this exporter writes. This is
#: unconditionally the integer ``1`` -- FORMAT.md section 2 is explicit that
#: this is *not* the XGBoost version, and the two must never be confused.
FORMAT_VERSION = 1

#: The enumerated version ceiling (D018), matching the list ``COMPAT.md``
#: records as actually probed. A caller that has probed a different XGBoost
#: version passes its own ``tested_versions`` rather than relying on this
#: default silently covering an untested build.
DEFAULT_TESTED_VERSIONS: frozenset[str] = frozenset({"3.3.0"})


def export_model(
    booster: Any,
    *,
    feature_names: Iterable[str] | None = None,
    tested_versions: frozenset[str] = DEFAULT_TESTED_VERSIONS,
) -> dict[str, Any]:
    """Export a fitted ``xgboost.Booster`` as a FORMAT.md-conforming artifact dict.

    Args:
        booster: A fitted ``xgboost.Booster`` (or an estimator's underlying
            booster). Read via ``booster.save_raw(raw_format="json")``, the
            same entry point every module in this package is built against.
        feature_names: Column names to use instead of the model's own. Only
            needed when the model was fit from a bare array, whose own
            ``feature_names`` serializes as ``[]`` (D021) -- passing names
            here is what makes such a model exportable at all. When given,
            these are validated exactly as the model's own names would be:
            non-empty, no duplicates, and a length agreeing with
            ``num_feature``, all enforced by the same gate check either way.
            When omitted, the model's own ``feature_names`` is used, and an
            empty one raises through that gate as it always has.

        tested_versions: The XGBoost version ceiling (D018). Defaults to
            :data:`DEFAULT_TESTED_VERSIONS`, the versions ``COMPAT.md``
            records as actually probed.

    Returns:
        A plain ``dict`` with exactly the seven keys FORMAT.md section 3
        requires. Pass it to :func:`to_json` to serialize.

    Raises:
        Any exception :func:`xgboost_bridge.validate.validate_source_model`
        raises, for a model the gate refuses.
        :class:`~xgboost_bridge.errors.UnsupportedModelShapeError`: a
            caller-supplied ``feature_names`` entry is not a string.
        :class:`~xgboost_bridge.errors.NonFiniteInterceptError`: the derived
            intercept is not finite (FORMAT.md section 9.3, D043) --
            reachable and silent upstream: ``survival:cox`` with
            ``base_score = 0.0`` derives ``-inf``, and with any negative
            ``base_score`` derives ``NaN``, both accepted by XGBoost with no
            error and no warning.
        :class:`~xgboost_bridge.errors.InterceptMismatchError`: the derived
            intercept disagrees with XGBoost's own observed zero-tree margin
            (FORMAT.md section 6.2) -- the independent-oracle check, not a
            re-derivation of this module's own recipe.

    The gate runs first, on the caller-supplied feature names already
    written into the parsed document, so a bare-array model with names
    supplied here is validated by the exact same duplicate/length/emptiness
    checks the model's own names would go through -- one check, not two that
    could disagree.
    """
    document = json.loads(booster.save_raw(raw_format="json"))

    if feature_names is not None:
        _apply_feature_name_override(document, feature_names)

    validate_source_model(document, tested_versions=tested_versions)

    trees = extract_trees(document)

    intercept = derive_intercept(document)
    if not math.isfinite(intercept):
        # Reachable and silent upstream: Cox at base_score=0.0 derives -inf, and
        # at any negative base_score derives NaN, both accepted by XGBoost with
        # no warning. The derivation reproduces them so the oracle below agrees,
        # so the refusal lives here. Note the oracle cannot catch it -- a
        # bit-pattern comparison matches NaN against NaN. See D043.
        raise NonFiniteInterceptError(
            intercept,
            document["learner"]["objective"]["name"],
            document["learner"]["learner_model_param"]["base_score"],
        )
    verify_intercept(booster, intercept)

    learner = document["learner"]
    objective_name = learner["objective"]["name"]
    output_transform = OUTPUT_TRANSFORMS.get(objective_name)
    if output_transform is None:
        raise UnsupportedObjectiveError(objective_name, tuple(OUTPUT_TRANSFORMS))

    artifact: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "objective": objective_name,
        "output_transform": output_transform,
        "intercept": intercept,
        "feature_names": list(learner["feature_names"]),
        "trees": trees,
        "provenance": _build_provenance(document),
    }
    _assert_envelope_keys(artifact)
    return artifact


def _apply_feature_name_override(document: dict[str, Any], feature_names: Iterable[str]) -> None:
    """Write caller-supplied feature names into the parsed document in place.

    Writing them into ``learner.feature_names`` *before* the gate runs, in
    the same slot the model's own names occupy, is what lets one check --
    ``validate.py``'s existing duplicate/length/emptiness rules -- cover both
    sources without being written twice. This function performs no
    threshold, leaf-value, or ``base_score`` handling; it moves a list of
    strings from one container to another.
    """
    names = list(feature_names)
    for index, name in enumerate(names):
        if not isinstance(name, str):
            raise UnsupportedModelShapeError(
                "feature_names",
                name,
                f"a string at position {index}",
                location="feature_names (caller-supplied)",
            )
    document["learner"]["feature_names"] = names


def _build_provenance(document: dict[str, Any]) -> dict[str, Any]:
    """Assemble ``provenance``, read by nothing on any prediction path.

    ``base_score`` is carried through exactly as XGBoost stored it -- no
    bracket parsing, no float32 snap, no link transform. That numeric
    handling belongs to :mod:`xgboost_bridge.objectives` alone, and this
    field's entire purpose is to record what the *source* held, not a value
    derived from it; ``intercept`` is that derived value, and it is a
    separate key of the envelope, not this one.
    """
    learner = document["learner"]
    model_param = learner["learner_model_param"]
    provenance = {
        "xgboost_version": ".".join(str(component) for component in document["version"]),
        "base_score": model_param["base_score"],
        "exporter_version": __version__,
    }
    _assert_provenance_keys(provenance)
    return provenance


def _assert_envelope_keys(artifact: dict[str, Any]) -> None:
    """Confirm ``artifact`` carries exactly the seven keys of FORMAT.md section 3.

    A self-check on this module's own construction rather than a reader's
    validation pass, but written as a real callable rather than a bare
    ``assert`` so a test can drive it directly against a hand-built envelope
    that carries an eighth key or is missing one.
    """
    present = frozenset(artifact)
    if present != ENVELOPE_KEYS:
        raise MalformedTreeError(
            "<envelope>",
            sorted(present),
            f"exactly these keys: {sorted(ENVELOPE_KEYS)}",
            None,
        )


def _assert_provenance_keys(provenance: dict[str, Any]) -> None:
    """Confirm ``provenance`` carries exactly its three enumerated keys."""
    present = frozenset(provenance)
    if present != PROVENANCE_KEYS:
        raise MalformedTreeError(
            "<provenance>",
            sorted(present),
            f"exactly these keys: {sorted(PROVENANCE_KEYS)}",
            "provenance",
        )


def to_json(artifact: dict[str, Any]) -> str:
    """Serialize ``artifact`` deterministically (FORMAT.md section 12, D008).

    Object keys are sorted lexicographically at every level, separators are
    compact with no insignificant whitespace, and the result carries exactly
    one trailing newline. Numbers are serialized by Python's ordinary
    shortest-round-trip ``float`` repr -- the same rule FORMAT.md section 9.1
    requires and the one every number in ``artifact`` already satisfies by
    construction, so no extra formatting step touches them here. Signed zero
    is preserved rather than normalized, because that is what
    ``json.dumps`` already does for a Python float: ``json.dumps(-0.0)`` is
    the string ``"-0.0"``.

    Nothing here reads the environment, the clock, or the filesystem, so two
    calls on the same ``artifact`` produce byte-identical output.
    """
    return json.dumps(artifact, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
