"""Tree extraction, dead-node neutralization, and the normative margin walk.

Three jobs, in the order the exporter performs them:

1. :func:`extract_trees` reads ``learner.gradient_booster.model.trees`` from a
   parsed ``booster.save_raw(raw_format="json")`` document and returns the
   five parallel arrays FORMAT.md section 8 specifies, with every value in
   ``node_values`` already float32-exact and ready to emit.
2. :func:`neutralize_dead_nodes` overwrites the nodes a pruned tree left
   behind, in place, per FORMAT.md section 8.3. Array lengths do not change
   and no index is renumbered.
3. :func:`walk_margin` is the normative prediction algorithm of FORMAT.md
   section 10. Everything downstream of this module -- the reference
   predictor, the parity harness -- is built on it, so every rule below is
   implemented as measured rather than as it might reasonably be written.

The numeric rules are not preferences and each traces to a measurement:

* Both sides of a threshold comparison are cast to float32, the operator is
  strict ``<``, and equality therefore routes to the right child. Casting only
  the sample value produced a 6.6-percentage-point probability error on a real
  row (``probes/float32_thresholds.md`` section 6); strict ``<`` with equality
  right held on 104/104 internal nodes of the primary model, plus 195 further
  internal nodes across seven more models -- every internal node measured
  (``probes/float32_thresholds.md`` sections 4 and 9). The 216 figure sometimes
  quoted for that model is its ``split_conditions`` length, which counts leaves
  as well: 104 of its 216 entries are thresholds (section 3).
* A node is a leaf if and only if ``left_children[i] == -1``. That is the only
  leaf test that held in every measured tree shape; ``right_children[i] == -1``
  coincides at scalar leaves but carries a block index in a vector-leaf tree
  (``probes/tree_structure.md`` sections 2 and 7g).
* ``node_values`` carries thresholds at internal nodes and outputs at leaves,
  the same overloading XGBoost's ``split_conditions`` uses, so one act of
  narrowing covers both roles (FORMAT.md section 8.1).
* The accumulator starts at the float32 intercept, trees are walked in
  serialized array order, and the accumulator is narrowed to float32 after
  every single addition. Intercept-last scored 199-2120/5000, reversed tree
  order 245-2365/5000, and a float64 running sum narrowed once at the end
  318-2541/5000 (``probes/accumulation.md`` section 6).
* Emitted values are ``float(np.float32(v))``. Rounding to 8 significant
  digits already lands 2/341 thresholds on a different float32; this route
  measured 0/341 drift (``probes/float32_thresholds.md`` sections 8c and 8d).
  Nothing in this module rounds, formats, or otherwise tidies a value.
* Signed zero is never normalized. A ``-0E0`` leaf is a real serialized value
  (``probes/tree_structure.md`` section 7a).

Failures are loud. Anything this module cannot account for -- an absent field,
arrays of unequal length, a child index that does not point forward, a
non-finite value, a vector-leaf or categorical tree, a dead-node marker that
disagrees with reachability -- raises rather than being defaulted or skipped.

Structural disagreements raise :class:`~xgboost_bridge.errors.MalformedTreeError`,
which is deliberately distinct from
:class:`~xgboost_bridge.errors.UnsupportedModelShapeError`: the latter means a
well-formed model whose output arity this version declines to support, while the
former means a tree whose shape contradicts the evidence this reader was built
from. Walking such a tree would apply assumptions already known to be false.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .errors import (
    CategoricalSplitError,
    MalformedTreeError,
    NonFiniteFeatureError,
    UnsupportedModelShapeError,
)

__all__ = [
    "DELETED_NODE_MARKER",
    "LEAF_CHILD",
    "TREE_KEYS",
    "extract_trees",
    "neutralize_dead_nodes",
    "reachable_nodes",
    "walk_margin",
]

#: ``split_indices`` value XGBoost writes at a pruned node: ``INT32_MAX``.
#: Measured on all six ``gamma`` sweeps in ``probes/tree_structure.md``
#: section 7a', where it agreed with unreachability on every node.
DELETED_NODE_MARKER = 2147483647

#: Child index that marks a leaf, in both the source model and the artifact.
LEAF_CHILD = -1

#: The five keys of an extracted tree, in the order FORMAT.md section 12
#: requires them to be emitted.
TREE_KEYS = (
    "default_left",
    "left_children",
    "node_values",
    "right_children",
    "split_indices",
)

_TREES_PATH = "learner.gradient_booster.model.trees"


def _shape_error(
    field: str, value: object, expected: str, location: str | None
) -> MalformedTreeError:
    """Build the loud failure used for every structural disagreement."""
    return MalformedTreeError(field, value, expected, location)


def _child(container: object, key: str, location: str) -> Any:
    """Read ``key`` out of a JSON object, raising if either is not there."""
    if not isinstance(container, Mapping):
        raise _shape_error(key, type(container).__name__, "a JSON object to read it from", location)
    if key not in container:
        raise _shape_error(key, "<absent>", "the field to be present", location)
    return container[key]


def _int_entries(raw: object, field: str, location: str) -> list[int]:
    """Read a JSON array of integers, rejecting anything else."""
    if not isinstance(raw, list):
        raise _shape_error(field, type(raw).__name__, "a JSON array", location)
    entries: list[int] = []
    for position, value in enumerate(raw):
        # bool is a subclass of int and is not a node index.
        if isinstance(value, bool) or not isinstance(value, int):
            raise _shape_error(
                field, value, f"an integer at position {position}", location
            )
        entries.append(int(value))
    return entries


def _float32_entries(raw: object, field: str, location: str) -> list[float]:
    """Narrow a JSON array of numbers to float32, widened back for emission.

    ``float(np.float32(v))`` is the emission rule of FORMAT.md section 9.1: the
    float32 value carried in a Python float so that ``json.dumps`` writes its
    shortest round-tripping repr. Measured drift 0/341
    (``probes/float32_thresholds.md`` section 8d).
    """
    if not isinstance(raw, list):
        raise _shape_error(field, type(raw).__name__, "a JSON array", location)
    entries: list[float] = []
    for position, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _shape_error(
                field, value, f"a JSON number at position {position}", location
            )
        narrowed = np.float32(value)
        if not np.isfinite(narrowed):
            raise _shape_error(
                field,
                value,
                f"a finite value at position {position}",
                location,
            )
        entries.append(float(narrowed))
    return entries


def _feature_count(model: object) -> int:
    """Read ``learner.learner_model_param.num_feature``.

    The field is a JSON *string*; a non-string is unrecognized rather than
    coerced, because the lesson of the export-time gate fields is that a
    silently mistyped comparison never fires.
    """
    learner = _child(model, "learner", "<document root>")
    param = _child(learner, "learner_model_param", "learner")
    raw = _child(param, "num_feature", "learner.learner_model_param")
    if not isinstance(raw, str) or not raw.isdigit():
        raise _shape_error(
            "num_feature", raw, "a string of digits", "learner.learner_model_param"
        )
    return int(raw)


def _source_trees(model: object) -> list[Any]:
    """Navigate to ``learner.gradient_booster.model.trees``."""
    learner = _child(model, "learner", "<document root>")
    booster = _child(learner, "gradient_booster", "learner")
    inner = _child(booster, "model", "learner.gradient_booster")
    trees = _child(inner, "trees", "learner.gradient_booster.model")
    if not isinstance(trees, list):
        raise _shape_error(
            "trees",
            type(trees).__name__,
            "a JSON array",
            "learner.gradient_booster.model",
        )
    return trees


def _refuse_categorical(model: object, source_trees: Sequence[Any]) -> None:
    """Raise if any of the three measured categorical signals is present.

    All three fired together on every categorical model measured
    (``probes/tree_structure.md`` section 9). All three are checked because
    this is a refusal, where a signal that never fires costs nothing and a
    signal that is missing costs a wrong number: a categorical split inverts
    the child convention and parks the smallest positive subnormal float32 in
    ``split_conditions`` instead of a threshold.

    ``split_type`` is a per-node array and its length is checked against the
    node count, for the same reason :func:`_extract_one` length-checks the five
    arrays it reads: a ``split_type`` shorter than the node count drops exactly
    the entries this refusal exists to examine. Measured on a three-node source
    tree, ``[0, 1, 0]`` refuses and a truncated ``[0]`` was *accepted* before
    that check existed. ``categories_nodes`` gets no equivalent check because it
    is not a per-node array -- its length is the number of categorical nodes and
    its entries are node indices (``probes/tree_structure.md`` section 3) -- so
    there is nothing to compare it against, and any non-empty value refuses
    whatever its length. The only way to lose that signal is an *emptied* array,
    which no length check can detect and which the other two signals cover.
    """
    signals: list[str] = []

    learner = _child(model, "learner", "<document root>")
    feature_types = _child(learner, "feature_types", "learner")
    if not isinstance(feature_types, list):
        raise _shape_error(
            "feature_types", type(feature_types).__name__, "a JSON array", "learner"
        )
    if any(entry == "c" for entry in feature_types):
        signals.append("learner.feature_types contains 'c'")

    for index, source in enumerate(source_trees):
        location = f"{_TREES_PATH}[{index}]"
        # The node count comes from left_children, the array _extract_one
        # length-checks everything else against, so both readers agree on what
        # "one entry per node" means.
        node_count = len(
            _int_entries(
                _child(source, "left_children", location), "left_children", location
            )
        )
        split_type = _int_entries(
            _child(source, "split_type", location), "split_type", location
        )
        if len(split_type) != node_count:
            raise _shape_error(
                "split_type",
                len(split_type),
                f"length {node_count}, matching left_children",
                location,
            )
        unknown = sorted({value for value in split_type if value not in (0, 1)})
        if unknown:
            raise _shape_error(
                "split_type", unknown, "every entry to be 0 (numeric) or 1 (categorical)", location
            )
        if any(value == 1 for value in split_type):
            signals.append(f"{location}.split_type contains 1")

        categories_nodes = _child(source, "categories_nodes", location)
        if not isinstance(categories_nodes, list):
            raise _shape_error(
                "categories_nodes",
                type(categories_nodes).__name__,
                "a JSON array",
                location,
            )
        if categories_nodes:
            signals.append(f"{location}.categories_nodes is non-empty")

    if signals:
        raise CategoricalSplitError(tuple(signals))


def _validate_child_links(
    left_children: Sequence[int],
    right_children: Sequence[int],
    location: str,
) -> None:
    """Pin the child-link shape measured across every scalar-leaf tree.

    At a leaf both children are ``-1``. At an internal node both children are
    valid indices strictly greater than the node's own index -- measured on
    every internal node of every tree probed (``probes/tree_structure.md``
    section 4.2), and what makes :func:`walk_margin` terminate. A leaf whose
    ``right_children`` entry is not ``-1`` is the vector-leaf signature, where
    that slot holds a block index into ``leaf_weights`` instead
    (``probes/tree_structure.md`` section 7g).
    """
    node_count = len(left_children)
    for index in range(node_count):
        left_child = left_children[index]
        right_child = right_children[index]
        if left_child == LEAF_CHILD:
            if right_child != LEAF_CHILD:
                raise _shape_error(
                    "right_children",
                    right_child,
                    f"-1 at node {index}, which left_children marks as a leaf",
                    location,
                )
            continue
        for field, child in (
            ("left_children", left_child),
            ("right_children", right_child),
        ):
            if not index < child < node_count:
                raise _shape_error(
                    field,
                    child,
                    f"an index in ({index}, {node_count}) at node {index}",
                    location,
                )
        if left_child == right_child:
            raise _shape_error(
                "right_children",
                right_child,
                f"a different index from left_children at node {index}",
                location,
            )


def _extract_one(source: object, location: str) -> dict[str, Any]:
    """Turn one source tree into the five arrays of FORMAT.md section 8."""
    tree_param = _child(source, "tree_param", location)
    param_location = f"{location}.tree_param"
    size_leaf_vector = _child(tree_param, "size_leaf_vector", param_location)
    # A JSON string, and compared as one: `== 1` against "1" is False, so an
    # integer comparison here would disable the gate rather than trip it.
    if size_leaf_vector != "1":
        # Vector leaves mean more than one value per row: an output-arity fact
        # about a well-formed model, not a malformed structure. Same error class
        # the export gate raises for arity, so a caller sees one kind.
        raise UnsupportedModelShapeError(
            "size_leaf_vector",
            size_leaf_vector,
            'the string "1" (scalar leaves)',
            param_location,
        )

    left_children = _int_entries(
        _child(source, "left_children", location), "left_children", location
    )
    right_children = _int_entries(
        _child(source, "right_children", location), "right_children", location
    )
    split_indices = _int_entries(
        _child(source, "split_indices", location), "split_indices", location
    )
    default_left = _int_entries(
        _child(source, "default_left", location), "default_left", location
    )
    node_values = _float32_entries(
        _child(source, "split_conditions", location), "split_conditions", location
    )

    node_count = len(left_children)
    if node_count == 0:
        raise _shape_error(
            "left_children", 0, "at least one node, since node 0 is the root", location
        )
    for field, entries in (
        ("right_children", right_children),
        ("split_indices", split_indices),
        ("default_left", default_left),
        ("split_conditions", node_values),
    ):
        if len(entries) != node_count:
            raise _shape_error(
                field,
                len(entries),
                f"length {node_count}, matching left_children",
                location,
            )

    for index, value in enumerate(default_left):
        if value not in (0, 1):
            raise _shape_error(
                "default_left", value, f"0 or 1 at node {index}", location
            )

    _validate_child_links(left_children, right_children, location)

    return {
        "default_left": default_left,
        "left_children": left_children,
        "node_values": node_values,
        "right_children": right_children,
        "split_indices": split_indices,
    }


def reachable_nodes(tree: Mapping[str, Any]) -> frozenset[int]:
    """Return the node indices reachable from node 0 by following children.

    Reachability is the definition of a live node (FORMAT.md section 8.3): it
    is the property the walk actually depends on. The
    ``split_indices == 2147483647`` marker is checked against it rather than
    trusted in its place.
    """
    left_children = tree["left_children"]
    right_children = tree["right_children"]
    reachable: set[int] = set()
    pending = [0]
    while pending:
        node = pending.pop()
        if node in reachable:
            continue
        reachable.add(node)
        if left_children[node] != LEAF_CHILD:
            pending.append(left_children[node])
            pending.append(right_children[node])
    return frozenset(reachable)


def _neutralize_node(tree: Mapping[str, Any], index: int) -> None:
    """Overwrite one node with the canonical safe values of FORMAT.md 8.3.

    Separate from :func:`neutralize_dead_nodes` so the test suite can apply it
    to a node the reachability walk found *live* and confirm the
    walk-versus-XGBoost comparison goes red. A neutralization that clears a
    live node is silent wrongness, so that has to be demonstrated rather than
    assumed.
    """
    tree["split_indices"][index] = 0
    tree["node_values"][index] = 0.0
    tree["left_children"][index] = LEAF_CHILD
    tree["right_children"][index] = LEAF_CHILD
    tree["default_left"][index] = 0


def neutralize_dead_nodes(
    tree: Mapping[str, Any], location: str | None = None
) -> tuple[int, ...]:
    """Neutralize every unreachable node of ``tree`` in place.

    Returns the dead indices in ascending order. Array lengths are unchanged
    and no index is renumbered, so every child reference still points where
    XGBoost pointed it -- this is the reachability walk without the
    renumbering a compaction pass would need.

    The dead set is not in general a trailing suffix: at ``gamma=50.0`` it was
    ``[31, 32, 33, 34, 37, 38, 41, 42, 45, ...]``, interleaved with live nodes
    (``probes/tree_structure.md`` section 7a').

    Raises if the reachable set disagrees with the
    ``split_indices == 2147483647`` marker. The two agreed on all six measured
    ``gamma`` sweeps, so a disagreement means the model's shape is not what any
    probe measured. This also means the function is not idempotent by design:
    it takes a freshly extracted tree, whose markers are still present.
    """
    node_count = len(tree["left_children"])
    reachable = reachable_nodes(tree)
    unreachable = tuple(
        index for index in range(node_count) if index not in reachable
    )
    marked = tuple(
        index
        for index in range(node_count)
        if tree["split_indices"][index] == DELETED_NODE_MARKER
    )
    if marked != unreachable:
        raise _shape_error(
            "split_indices",
            {"marked_deleted": marked, "unreachable_from_root": unreachable},
            "the nodes marked 2147483647 to be exactly the nodes unreachable from node 0",
            location,
        )

    for index in unreachable:
        _neutralize_node(tree, index)
    return unreachable


def extract_trees(model: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract every tree of a parsed XGBoost model as FORMAT.md section 8 arrays.

    ``model`` is the parsed output of
    ``booster.save_raw(raw_format="json")``. Returns one dict per tree, in
    serialized array order -- which is normative, since reversing it scored
    245-2365/5000 bit-exact (``probes/accumulation.md`` section 6) -- carrying
    exactly ``default_left``, ``left_children``, ``node_values``,
    ``right_children`` and ``split_indices``.

    Dead nodes are neutralized (see :func:`neutralize_dead_nodes`), so the
    returned arrays contain no out-of-range value: every ``split_indices``
    entry is in ``[0, num_feature)``.
    """
    source_trees = _source_trees(model)
    feature_count = _feature_count(model)
    _refuse_categorical(model, source_trees)

    extracted: list[dict[str, Any]] = []
    for index, source in enumerate(source_trees):
        location = f"{_TREES_PATH}[{index}]"
        tree = _extract_one(source, location)
        neutralize_dead_nodes(tree, location)
        for node, column in enumerate(tree["split_indices"]):
            if not 0 <= column < feature_count:
                raise _shape_error(
                    "split_indices",
                    column,
                    f"an index in [0, {feature_count}) at node {node}",
                    location,
                )
        extracted.append(tree)
    return extracted


def walk_margin(
    trees: Sequence[Mapping[str, Any]],
    intercept: float | np.float32,
    feature_values: Sequence[float] | np.ndarray,
) -> np.float32:
    """Compute one row's margin: the normative algorithm of FORMAT.md section 10.

    ``feature_values`` is indexable by column index, so ``split_indices``
    reaches it directly. A row of a float64 NumPy matrix -- ``matrix[i]`` -- is
    an accepted input and is what a caller has in hand; so is a list of Python
    floats. ``NaN`` is the missing value and routes by ``default_left``; both
    directions were demonstrated with ``predict()`` as the arbiter
    (``probes/tree_structure.md`` section 5).

    Every float32 narrowing below is unconditional and assumes nothing about
    what the caller already narrowed. That matters more than it reads: under
    NEP 50 a Python float is a *weak* scalar, so ``python_float < np.float32(t)``
    is evaluated in float32 whether or not the sample side was cast, while an
    ``np.float64`` -- an element of the float64 row above -- is strong and drags
    the comparison into float64. A walk that leans on the weak-scalar rule is
    correct only for as long as its caller keeps handing it Python floats.

    Correct implementation of this walk scored 5000/5000 bit-exact against
    ``predict(output_margin=True)`` at max absolute error ``0.0``, across three
    objectives and tree counts 0-1000 (``probes/accumulation.md`` section 6).

    Raises:
        :class:`~xgboost_bridge.errors.NonFiniteFeatureError`: a feature value
            is infinite. ``NaN`` is accepted -- it is the missing value.
    """
    # The whole row, before the walk: checking only at visited nodes would make
    # the same invalid input raise or not depending on which branches this
    # particular tree takes, i.e. a property of the model rather than of the
    # input. Cost is O(features) against an O(depth x trees) walk. See D022.
    # Narrow first, then refuse an infinite result. Testing the float64 for
    # `±inf` instead let a finite float64 that *becomes* infinite through this
    # library's own required narrowing straight through: `1e39` is a legal
    # float64, `f32(1e39)` is `+inf`, and the walk then compared `inf` against
    # thresholds and returned a number. Same mathematical value as an explicit
    # infinity, two different behaviours, and no error either way -- which is
    # the class of silent inconsistency this library exists to remove (D055).
    #
    # `np.isinf` rather than `not np.isfinite`, because `NaN` narrows to `NaN`
    # and must be *accepted*: it is the missing value and routes by the tree's
    # default direction.
    #
    # errstate: the cast warns "overflow encountered in cast" on exactly the
    # inputs now being refused. Emitting a numpy warning and then raising is two
    # reports of one problem, and the raise is the one a caller can act on.
    with np.errstate(over="ignore"):
        for index, value in enumerate(feature_values):
            if np.isinf(np.float32(value)):
                raise NonFiniteFeatureError(index, float(value))

    accumulator = np.float32(intercept)

    for tree in trees:
        left_children = tree["left_children"]
        right_children = tree["right_children"]
        split_indices = tree["split_indices"]
        node_values = tree["node_values"]
        default_left = tree["default_left"]

        node = 0
        while left_children[node] != LEAF_CHILD:
            feature_value = feature_values[split_indices[node]]
            threshold = node_values[node]
            if feature_value != feature_value:
                # This is the only site that reads default_left, so it is the
                # only site where an out-of-range entry could pick a direction
                # by accident -- and "not 1" would pick right, a legal
                # direction, making it a plausible wrong number rather than an
                # error. extract_trees already rejects anything but 0 or 1, but
                # this walk is public and normative, so it re-checks rather
                # than trusting the tree it was handed. The check sits inside
                # the missing-value branch because that is where the value is
                # consulted; a row that never reaches it is routed by the
                # comparison alone and is unaffected by the entry.
                if default_left[node] == 1:
                    node = left_children[node]
                elif default_left[node] == 0:
                    node = right_children[node]
                else:
                    raise _shape_error(
                        "default_left",
                        default_left[node],
                        f"0 or 1 at node {node}",
                        None,
                    )
            elif np.float32(feature_value) < np.float32(threshold):
                node = left_children[node]
            else:
                node = right_children[node]
        accumulator = np.float32(accumulator + np.float32(node_values[node]))

    return accumulator
