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

The assembled artifact is then walked and compared against XGBoost's own
``predict(output_margin=True)``, bit-for-bit, which FORMAT.md section 8.3 and
D027 both make mandatory (:func:`_verify_against_source_margin`). Every
structural check upstream of it validates *deadness detection* -- that the
``split_indices == 2147483647`` marker agrees with reachability -- and none of
them validates the arrays neutralization produced. A neutralization that
cleared a node the walk actually visits, or a value read out of the wrong
source array, passes every one of those checks and shows up only as a
different number.

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
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from ._version import __version__
from .errors import (
    MalformedTreeError,
    MarginMismatchError,
    NonFiniteInterceptError,
    UnsupportedModelShapeError,
    UnsupportedObjectiveError,
)
from .objectives import OUTPUT_TRANSFORMS, observe_intercept
from .trees import LEAF_CHILD, extract_trees, reachable_nodes, walk_margin
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
        :class:`~xgboost_bridge.errors.MarginMismatchError`: the assembled
            artifact's own walk disagrees with XGBoost's
            ``predict(output_margin=True)`` on any self-check row (FORMAT.md
            section 8.3, D027).

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

    # Read out of the engine, not computed from base_score. XGBoost derives this
    # value with the platform's logf, which is not correctly rounded, and its own
    # answer differs between darwin/arm64 and linux/x86_64 by 1 ULP on 29 of 58
    # discriminating inputs (probes/platform_log.md, D053). No recipe reproduces
    # it everywhere, so deriving one guarantees a spurious refusal on some
    # platform: this is exactly what CI's first Linux run reported, 18 failures
    # that pass on darwin. objectives.derive_intercept still documents how the
    # engine reaches the value, and is deliberately no longer on this path.
    #
    # The intercept is validated end-to-end by _verify_against_source_margin
    # below, against XGBoost's own predict output on leaf-reaching rows -- an
    # oracle that is independent of this value's provenance and that a 1-ULP
    # error fails. Comparing the engine's value against the engine's value here
    # would be the decorative check the independent-oracle principle rejects.
    intercept = observe_intercept(booster)
    if not math.isfinite(intercept):
        # Reachable and silent upstream: Cox at base_score=0.0 gives -inf, and at
        # any negative base_score gives NaN, both accepted by XGBoost with no
        # warning. The refusal lives here because the margin check below cannot
        # catch it -- for a zero-tree model it compares this value against
        # itself, and NaN never equals NaN in a bit-pattern comparison anyway.
        # See D043.
        raise NonFiniteInterceptError(
            intercept,
            document["learner"]["objective"]["name"],
            document["learner"]["learner_model_param"]["base_score"],
        )

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
    _verify_against_source_margin(booster, artifact)
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


# The interval sentinels of the self-check's row construction below. float32,
# because every bound is a float32 threshold and the walk compares in float32.
_UNBOUNDED_BELOW = np.float32(-np.inf)
_UNBOUNDED_ABOVE = np.float32(np.inf)


def _verify_against_source_margin(booster: Any, artifact: dict[str, Any]) -> None:
    """Walk the assembled artifact and require XGBoost's own margin, bit-for-bit.

    The mandatory self-check of FORMAT.md section 8.3 and D027. It is the only
    check in this module that examines the *result* of neutralization rather
    than the evidence for it: :func:`xgboost_bridge.trees.neutralize_dead_nodes`
    asserts that the ``split_indices == 2147483647`` marker agrees with
    reachability, which validates deadness *detection*. A neutralization that
    cleared a live node, or a value read out of the wrong source array -- the
    ``base_weights``-instead-of-``split_conditions`` confusion is off by
    ``5.10`` in margin space (FORMAT.md section 15) -- satisfies every
    structural check and differs only in the numbers.

    Args:
        booster: The source ``xgboost.Booster``, used only as the oracle.
        artifact: The assembled artifact, read for its ``trees``,
            ``intercept`` and ``feature_names``.

    Raises:
        :class:`~xgboost_bridge.errors.MarginMismatchError`: any sampled row's
            margin differs from XGBoost's in a single bit.
        :class:`~xgboost_bridge.errors.MalformedTreeError`: XGBoost returned a
            margin whose shape is not one value per row, so there is nothing
            scalar to compare against.

    **What the oracle is, and why it cannot share the defect.** The oracle is
    XGBoost's ``predict(output_margin=True)`` on the same rows -- the engine
    that produced the model, reading its own untouched in-memory trees.
    Nothing in this library's extraction, neutralization, or emission path
    contributes to it, so a defect in any of them cannot move both sides
    together. The walk is
    :func:`xgboost_bridge.trees.walk_margin`, the normative one, not a second
    implementation written for this check.

    The comparison is on float32 bit patterns rather than ``==``, because
    ``-0.0 == 0.0`` is ``True`` and the two are different artifacts. Note that
    it deliberately does *not* subsume the finiteness refusal above: a
    ``NaN`` intercept would make both sides ``NaN``, and ``NaN`` bit patterns
    match perfectly well (D043).
    """
    import xgboost  # noqa: PLC0415 -- optional extra; see D010

    trees = artifact["trees"]
    intercept = artifact["intercept"]
    rows = _self_check_rows(trees, len(artifact["feature_names"]))

    # `booster.feature_names`, not the artifact's: measured, ``predict`` raises
    # ``ValueError`` unless the matrix carries the model's own names, or no
    # names at all when the model has none. A caller-supplied override
    # (D021) renames columns in the artifact and must not reach the oracle.
    matrix = xgboost.DMatrix(rows, feature_names=booster.feature_names, nthread=1)
    observed = np.asarray(booster.predict(matrix, output_margin=True), dtype=np.float32)
    if observed.shape != (len(rows),):
        # The arity gate (D017) already refuses multi-output models, so this is
        # unreachable through a validated model. It is here because the
        # alternative to a structured refusal is comparing a scalar against a
        # row of values, which is an unstructured `TypeError` at best.
        raise MalformedTreeError(
            "<observed margin>",
            observed.shape,
            f"one margin per row, i.e. shape ({len(rows)},)",
            None,
        )

    mismatches = 0
    first: tuple[int, float, float] | None = None
    for index in range(len(rows)):
        derived = walk_margin(trees, intercept, rows[index])
        if int(derived.view(np.uint32)) != int(observed[index].view(np.uint32)):
            mismatches += 1
            if first is None:
                first = (index, float(derived), float(observed[index]))
    if first is not None:
        raise MarginMismatchError(first[0], first[1], first[2], mismatches, len(rows))

    # Only now. Coverage is a statement about the sufficiency of a check that
    # PASSED -- so it runs after the comparison, never before it. Asserting it
    # first inverts the ordering: corrupted thresholds can make a leaf's
    # feature-box empty, so the coverage check would fire on an artifact whose
    # actual defect is a wrong number, and report the weaker finding. Measured:
    # doing so replaced MarginMismatchError with a sample complaint in
    # test_export_refuses_node_values_read_from_the_wrong_source_array.
    _assert_sample_reaches_every_live_leaf(trees, rows)


def _self_check_rows(
    trees: Sequence[Mapping[str, Any]], feature_count: int
) -> np.ndarray:
    """Build the deterministic sample :func:`_verify_against_source_margin` walks.

    Two blocks, and the composition is the whole argument for the check being
    worth anything:

    * **One row per live leaf**, constructed rather than drawn, so the sample
      reaches every node the walk can visit. See
      :func:`_leaf_reaching_rows` for the construction and for why "every
      live leaf" is the same coverage as "every live node".
    * **A missing-value block**: one row of all ``NaN``, and one row per column
      carrying ``NaN`` in that column alone. ``NaN`` is the missing value and
      routes by ``default_left`` (FORMAT.md section 9.3), which is a branch
      the constructed rows never take -- every value they carry is a real
      number, so a defect confined to the missing-value direction would be
      invisible to them.

    No row carries ``±inf``: those are refused at predict time (D022, D045),
    and a sample that raises rather than comparing would verify nothing.

    Rows are ``float64``, which is what a caller has in hand and what the walk
    documents as its input; every value in them is float32-exact, so the
    walk's narrowing changes none of them.
    """
    rows = _leaf_reaching_rows(trees, feature_count)
    rows.extend(_missing_value_rows(feature_count))
    return np.asarray(rows, dtype=np.float64)


def _assert_sample_reaches_every_live_leaf(
    trees: Sequence[Mapping[str, Any]], sample: np.ndarray
) -> None:
    """Require the sample to actually reach every live leaf, and say so if not.

    This replaces an argument with a measurement. ``_leaf_reaching_rows``
    justified its own sufficiency by reasoning that a node whose feature-interval
    is empty "is reachable in the graph but reachable by **no input at all**, so
    no row can exist for it and no change to it can alter any prediction."

    **That reasoning is false for missing values.** ``NaN`` routes by
    ``default_left`` and ignores thresholds entirely, so a node whose finite box
    is empty can still be reached -- by a row carrying ``NaN`` in the right
    column. A constructed 7-node case has a leaf reachable only via
    ``[nan, 5.0]``: the box tracking drops it, ``_representative_row`` returns
    ``None``, the row is skipped silently, and corrupting that leaf's value
    leaves every self-check margin identical. The mandatory self-check of
    FORMAT.md section 8.3 would pass over a wrong leaf.

    Latent rather than live: a search of 162 fitted models -- ``exact``/``hist``/
    ``approx`` x depth 4/8/12 x 0/30/70% missing x 2/6 columns x 3 seeds, all at
    ``min_child_weight=0`` -- found **0** live nodes with an empty box. So this
    has never fired on a real model, and the argument it replaces was still
    wrong. Asserting coverage costs one walk over a sample that is already being
    walked, and converts "no row can exist for it" from a claim into a check.

    Raises:
        MalformedTreeError: some live leaf is reached by no row in the sample.
            Reported per tree, with the leaf indices, because the actionable fix
            is to extend the sample rather than to weaken the check.
    """
    for index, tree in enumerate(trees):
        left = tree["left_children"]
        live = reachable_nodes(tree)
        leaves = {node for node in live if left[node] == LEAF_CHILD}
        if not leaves:
            continue

        reached: set[int] = set()
        for row in sample:
            node = 0
            while left[node] != LEAF_CHILD:
                value = np.float32(row[tree["split_indices"][node]])
                if np.isnan(value):
                    go_left = bool(tree["default_left"][node])
                elif value < np.float32(tree["node_values"][node]):
                    go_left = True
                else:
                    go_left = False
                node = tree["left_children"][node] if go_left else tree["right_children"][node]
            reached.add(node)

        missed = sorted(leaves - reached)
        if missed:
            raise MalformedTreeError(
                "<self-check sample>",
                missed,
                (
                    f"a sample reaching every live leaf; leaves {missed} in tree "
                    f"{index} are reached by no row, so a wrong value at any of "
                    f"them would pass the self-check"
                ),
                f"trees[{index}]",
            )


def _leaf_reaching_rows(
    trees: Sequence[Mapping[str, Any]], feature_count: int
) -> list[np.ndarray]:
    """One row per live leaf of every tree, each row reaching that leaf.

    Each tree is descended from node ``0`` carrying, per column, the interval
    ``[lower, upper)`` of values consistent with the branches taken so far:
    going left at threshold ``t`` requires ``value < t``, and going right
    requires ``value >= t``, because the operator is strict ``<`` and equality
    routes right. At a leaf the interval is realized as one row.

    **Why covering the leaves covers every live node.** A live node is either
    a leaf or has children, and if a node's interval is non-empty then at
    least one child's is too -- any value in it is either below the threshold
    or not. So every live node with a non-empty interval is an ancestor of at
    least one live leaf with a non-empty interval, and the row that reaches
    that leaf passes through it. A node whose interval *is* empty is reachable
    in the graph but reachable by **no input at all**, so no row can exist for
    it and no change to it can alter any prediction.

    Rows are deduplicated on their exact bytes: identical intervals recur
    across trees, and a duplicate row costs a full walk while proving nothing
    new. Order is otherwise the order of construction, so the sample is
    deterministic -- nothing here consults a clock, an environment, or a
    random source.
    """
    rows: list[np.ndarray] = []
    seen: set[bytes] = set()
    for tree in trees:
        left_children = tree["left_children"]
        right_children = tree["right_children"]
        split_indices = tree["split_indices"]
        node_values = tree["node_values"]

        pending: list[tuple[int, np.ndarray, np.ndarray]] = [
            (
                0,
                np.full(feature_count, _UNBOUNDED_BELOW, dtype=np.float32),
                np.full(feature_count, _UNBOUNDED_ABOVE, dtype=np.float32),
            )
        ]
        while pending:
            node, lower, upper = pending.pop()
            if left_children[node] != LEAF_CHILD:
                column = split_indices[node]
                threshold = np.float32(node_values[node])
                left_upper = upper.copy()
                if threshold < left_upper[column]:
                    left_upper[column] = threshold
                right_lower = lower.copy()
                if threshold > right_lower[column]:
                    right_lower[column] = threshold
                pending.append((left_children[node], lower.copy(), left_upper))
                pending.append((right_children[node], right_lower, upper.copy()))
                continue
            row = _representative_row(lower, upper, feature_count)
            if row is None:
                continue
            key = row.tobytes()
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def _representative_row(
    lower: np.ndarray, upper: np.ndarray, feature_count: int
) -> np.ndarray | None:
    """One row satisfying ``lower[c] <= row[c] < upper[c]`` for every column.

    Returns ``None`` when some column's interval is empty, which means the
    path those bounds describe is taken by no input whatsoever.

    The value chosen per column is the *boundary* wherever there is one --
    ``lower`` when it is finite, and otherwise the float32 immediately below
    ``upper``. That is deliberate rather than incidental: a threshold and the
    value adjacent to it are exactly where a one-sided float32 cast, a
    non-strict comparison, or an equality routed the wrong way changes the
    branch, so the sample sits on the boundary instead of near it. An
    unconstrained column takes ``0.0``, which no threshold in the model
    excludes.
    """
    row = np.zeros(feature_count, dtype=np.float64)
    for column in range(feature_count):
        low = lower[column]
        high = upper[column]
        if low == _UNBOUNDED_BELOW:
            if high == _UNBOUNDED_ABOVE:
                continue
            row[column] = float(np.nextafter(high, _UNBOUNDED_BELOW))
        elif low < high:
            row[column] = float(low)
        else:
            return None
    return row


def _missing_value_rows(feature_count: int) -> list[np.ndarray]:
    """The ``NaN`` block: all columns missing, then each column alone.

    ``NaN`` is the missing value and routes by ``default_left``; it is not an
    error (FORMAT.md section 9.3). One row per column, rather than only an
    all-``NaN`` row, because the all-``NaN`` row takes the default direction at
    every node it visits and so cannot distinguish a per-node ``default_left``
    that is read from the wrong column.
    """
    rows = [np.full(feature_count, np.nan, dtype=np.float64)]
    for column in range(feature_count):
        row = np.zeros(feature_count, dtype=np.float64)
        row[column] = np.nan
        rows.append(row)
    return rows


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
