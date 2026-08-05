"""The reference predictor: read an artifact, produce margins and outputs.

This module is the **reader**, and the reader is where float32 discipline is
most easily lost. Every hard numeric rule already lives somewhere else --
:func:`xgboost_bridge.trees.walk_margin` is the normative walk of FORMAT.md
section 10, and :mod:`xgboost_bridge.transform` is the bundled float32
margin-to-output transform of section 5 -- so nothing here re-implements
either. What this module owns is the step *before* them, which is the one that
can quietly destroy the invariant while leaving both of them reading as
correct:

* ``node_values`` is loaded into a ``dtype=np.float32`` array **at parse
  time**, not narrowed later at the comparison site (FORMAT.md section 9.2).
  A parser that lands thresholds and leaf values as unconstrained Python
  floats hands the walk float64 data; the walk still narrows both operands, so
  the corpus looks fine, and every *other* consumer of those numbers -- a
  re-serializer, an inspection utility, an arithmetic transform -- silently
  gets the float64. Narrowing by construction makes the invariant a property
  of the data structure rather than a discipline each future reader has to
  remember (D004: parsing belongs to the numerical core).
* The ``intercept`` is narrowed to float32 on read for the same reason, and
  is never transformed in any way (D015, FORMAT.md section 6). Negative zero
  is a reachable, ordinary value here and is never normalized.

Validation is exhaustive and loud, per FORMAT.md section 13: an unrecognized
key at any level, an absent required key, a wrong JSON type, a
``format_version`` that is not exactly the integer ``1``, an enumerated field
outside its set, an objective/transform pairing that disagrees, unequal tree
array lengths, an out-of-range child or ``split_indices``, a non-finite
``node_values`` entry or ``intercept``, and an empty or duplicated
``feature_names`` all raise. Nothing defaults, nothing is guessed, nothing is
skipped (D007).

Two rules cut the other way and are just as load-bearing:

* A node **unreachable** from the root does not raise. Neutralized dead slots
  are legitimate artifact content (FORMAT.md section 8.3) and a reader that
  rejected them would reject every pruned model.
* ``objective`` is **non-operative metadata** (D028). No function on the
  prediction path reads it; ``output_transform`` alone selects the transform.
  The pairing cross-check happens once, at load, which is the field's entire
  job. The companion test suite asserts the absence of that branch at source
  level, because a field that becomes load-bearing by accident is a second
  source of truth about behaviour the first one already determines.

``NaN`` in a prediction input is the missing value and routes by
``default_left``; ``±inf`` is refused (D022, D045). Both behaviours belong to
``walk_margin`` and are deliberately not re-implemented here -- a second copy
of a refusal is a second thing that can disagree with the first.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final

import numpy as np

from .errors import (
    FeatureKeyMismatchError,
    MalformedTreeError,
    UnrecognizedFieldError,
    UnsupportedObjectiveError,
    UnsupportedVersionError,
)
from .objectives import OUTPUT_TRANSFORMS, SUPPORTED_OBJECTIVES
from .transform import OUTPUT_FUNCTIONS
from .trees import LEAF_CHILD, walk_margin

__all__ = [
    "ENVELOPE_KEYS",
    "PROVENANCE_KEYS",
    "READABLE_FORMAT_VERSION",
    "TREE_KEYS",
    "Predictor",
]

#: The seven required top-level keys of FORMAT.md section 3, and no others.
#: Stated here from the specification rather than imported from
#: :mod:`xgboost_bridge.export`: a reader that derives its key set from the
#: writer's cannot detect a writer that emits the wrong set. The companion
#: test asserts the two agree, which is a check with something to compare.
ENVELOPE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "feature_names",
        "format_version",
        "intercept",
        "objective",
        "output_transform",
        "provenance",
        "trees",
    }
)

#: ``provenance``'s own fixed key set (FORMAT.md sections 2, 15). Read by
#: nothing on any prediction path; validated anyway, because "a value of the
#: wrong JSON type" raises regardless of whether the value is operative.
PROVENANCE_KEYS: Final[frozenset[str]] = frozenset(
    {"base_score", "exporter_version", "xgboost_version"}
)

#: The five parallel arrays of FORMAT.md section 8, one entry per node.
TREE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "default_left",
        "left_children",
        "node_values",
        "right_children",
        "split_indices",
    }
)

#: The only ``format_version`` this reader accepts. Not a range, not a floor:
#: exactly the integer ``1``. The marker is the migration mechanism (D003,
#: D007), so an unrecognized value raises instead of being read
#: best-effort -- including ``0``, ``2``, the string ``"1"``, the float
#: ``1.0``, and ``True``, which is an ``int`` in Python and is not this one.
READABLE_FORMAT_VERSION: Final[int] = 1

#: The four ``node_values``-adjacent arrays that carry integers.
_INTEGER_TREE_KEYS: Final[tuple[str, ...]] = (
    "default_left",
    "left_children",
    "right_children",
    "split_indices",
)

# Depth-first search colours for the termination check below.
_UNVISITED: Final[int] = 0
_ON_PATH: Final[int] = 1
_SETTLED: Final[int] = 2


def _malformed(
    field: str, value: object, expected: str, location: str | None
) -> MalformedTreeError:
    """Build the structured failure used for every disagreement with the format.

    :class:`~xgboost_bridge.errors.MalformedTreeError` is this codebase's
    established error for "a field holds a value the specification does not
    allow" -- :mod:`xgboost_bridge.objectives` already raises it for a
    malformed ``base_score`` and an unrecognized ``boost_from_average``, not
    only for tree geometry. Its four structured attributes are what let a
    caller branch on the failure rather than parse a message.
    """
    return MalformedTreeError(field, value, expected, location)


def _require_mapping(value: object, field: str, location: str | None) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _malformed(field, type(value).__name__, "a JSON object", location)
    return value


def _check_keys(
    container: Mapping[str, Any], allowed: frozenset[str], location: str | None
) -> None:
    """Require exactly ``allowed``: no unrecognized key, no absent key.

    Unrecognized keys are reported first and in sorted order, so the same
    malformed artifact always produces the same error rather than one that
    depends on dictionary insertion order.
    """
    # Sorted by `repr` rather than by value: a hand-built mapping can carry a
    # non-string key, and comparing a string against an integer raises a
    # TypeError that would replace this module's structured refusal with an
    # unstructured one.
    unrecognized = sorted((key for key in container if key not in allowed), key=repr)
    if unrecognized:
        raise UnrecognizedFieldError(str(unrecognized[0]), location)
    for key in sorted(allowed):
        if key not in container:
            raise _malformed(key, "<absent>", "the field to be present", location)


def _read_format_version(artifact: Mapping[str, Any]) -> int:
    value = artifact["format_version"]
    # `bool` is a subclass of `int`, so `True == 1` -- excluded explicitly
    # rather than left to a comparison that would accept it.
    if isinstance(value, bool) or not isinstance(value, int) or value != READABLE_FORMAT_VERSION:
        raise UnsupportedVersionError(value, (READABLE_FORMAT_VERSION,))
    return int(value)


def _read_objective(artifact: Mapping[str, Any]) -> str:
    """Read ``objective`` and check it against the enumerated set.

    This is the *only* place in this module that reads the field, and it runs
    at load time. Nothing on the prediction path consults it (D028).
    """
    value = artifact["objective"]
    if not isinstance(value, str) or value not in SUPPORTED_OBJECTIVES:
        raise UnsupportedObjectiveError(value, SUPPORTED_OBJECTIVES)
    return value


def _read_output_transform(artifact: Mapping[str, Any], objective: str) -> str:
    """Read ``output_transform`` and require it to pair with ``objective``.

    The pairing check is the whole reason ``objective`` is carried at all
    (FORMAT.md sections 4, 13). Performing it here, once, is what keeps the
    prediction path free of it: after this function returns, the transform is
    a lookup and the objective is a label.
    """
    value = artifact["output_transform"]
    if not isinstance(value, str) or value not in OUTPUT_FUNCTIONS:
        raise _malformed(
            "output_transform",
            value,
            f"one of: {', '.join(sorted(OUTPUT_FUNCTIONS))}",
            None,
        )

    paired = OUTPUT_TRANSFORMS[objective]
    if value != paired:
        raise _malformed(
            "output_transform",
            value,
            f"{paired!r}, the transform paired with objective {objective!r}",
            None,
        )
    return value


def _read_feature_names(artifact: Mapping[str, Any]) -> tuple[str, ...]:
    """Read ``feature_names``: a non-empty list of unique strings (D021).

    Emptiness is a refusal rather than a degenerate case. A strict-key policy
    with no keys to check reads as enforced and is not, which is worse than no
    policy at all because the caller believes a typo will be caught.
    """
    value = artifact["feature_names"]
    if not isinstance(value, list):
        raise _malformed("feature_names", type(value).__name__, "a JSON array", None)
    for position, name in enumerate(value):
        if not isinstance(name, str):
            raise _malformed(
                "feature_names", name, f"a string at position {position}", None
            )
    if not value:
        raise _malformed(
            "feature_names",
            [],
            "at least one name; a strict-key policy needs keys to check",
            None,
        )
    seen: set[str] = set()
    for position, name in enumerate(value):
        if name in seen:
            raise _malformed(
                "feature_names",
                name,
                f"no duplicate names (position {position} repeats an earlier entry)",
                None,
            )
        seen.add(name)
    return tuple(value)


def _narrow(value: object, field: str, position: int | None, location: str | None) -> np.float32:
    """Narrow one JSON number to float32, raising unless it is finite.

    Overflow on the cast is silenced and then *refused*: ``1e40`` narrows to
    ``inf``, which FORMAT.md section 9.3 forbids, and the loud path is the
    raise below rather than a numpy warning whose visibility depends on
    ambient filters.
    """
    where = "" if position is None else f" at position {position}"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _malformed(field, value, f"a JSON number{where}", location)
    with np.errstate(over="ignore"):
        narrowed = np.float32(value)
    if not np.isfinite(narrowed):
        raise _malformed(
            field, value, f"a finite float32 value{where}", location
        )
    return narrowed


def _read_intercept(artifact: Mapping[str, Any]) -> np.float32:
    """Narrow ``intercept`` to float32 on read, and never transform it.

    The accumulator's initial value (FORMAT.md section 10). No ``logit``, no
    ``ln``, no ``exp``, and no objective-dependent branch touches it: under
    D015 the link space is an export-time concern with no runtime
    representation. ``-0.0`` stays ``-0.0``; it is what ``binary:logistic``
    at ``base_score = 0.5`` legitimately produces.
    """
    return _narrow(artifact["intercept"], "intercept", None, None)


def _read_provenance(artifact: Mapping[str, Any]) -> Mapping[str, str]:
    """Read ``provenance``: exactly three keys, all JSON strings.

    No prediction path reads any of them. They are validated because
    FORMAT.md section 13 makes an unrecognized key and a wrong JSON type
    refusals wherever they occur, not only where they would change a number.
    ``base_score`` here is the raw bracketed string XGBoost stored, e.g.
    ``"[6E-1]"``, and is deliberately not parsed.
    """
    provenance = _require_mapping(artifact["provenance"], "provenance", None)
    _check_keys(provenance, PROVENANCE_KEYS, "provenance")
    for key in sorted(PROVENANCE_KEYS):
        value = provenance[key]
        if not isinstance(value, str):
            raise _malformed(key, value, "a JSON string", "provenance")
    return MappingProxyType({key: provenance[key] for key in sorted(PROVENANCE_KEYS)})


def _read_integer_array(
    raw: object, field: str, location: str
) -> tuple[int, ...]:
    """Read one JSON array of integers, rejecting anything else.

    ``bool`` is excluded: FORMAT.md section 8 specifies ``default_left`` as
    ``0``/``1`` integers rather than JSON booleans, and a child index is
    never a boolean either.
    """
    if not isinstance(raw, list):
        raise _malformed(field, type(raw).__name__, "a JSON array", location)
    entries: list[int] = []
    for position, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, int):
            raise _malformed(
                field, value, f"an integer at position {position}", location
            )
        entries.append(int(value))
    return tuple(entries)


def _read_node_values(raw: object, location: str) -> np.ndarray:
    """Load ``node_values`` into a ``dtype=np.float32`` array -- the crux.

    This is FORMAT.md section 9.2's narrowing site, and its position is the
    point of this module. ``json.loads`` returns float64 unconditionally, and
    on 104/104 measured thresholds that float64 is a different number from the
    engine's float32; leaf values need the same narrowing, without which
    accumulation scored 990-3706/5000 bit-exact and breached the ``1e-6`` gate
    at ``1.07e-04`` (``probes/accumulation.md`` sections 3, 6).

    One array narrowed once covers both roles, which is exactly why the format
    keeps XGBoost's overloading of thresholds and leaf outputs in a single
    array (FORMAT.md section 8.1): two arrays would be two narrowing sites,
    one of which could be forgotten.

    The result is marked read-only. A caller inspecting the arrays through
    :attr:`Predictor.trees` therefore cannot reach in and mutate the model's
    thresholds, and the float32 dtype cannot be widened back out from under
    the walk.
    """
    if not isinstance(raw, list):
        raise _malformed("node_values", type(raw).__name__, "a JSON array", location)
    for position, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _malformed(
                "node_values", value, f"a JSON number at position {position}", location
            )

    with np.errstate(over="ignore"):
        values = np.asarray(raw, dtype=np.float32)
    finite = np.isfinite(values)
    if not bool(finite.all()):
        position = int(np.argmin(finite))
        raise _malformed(
            "node_values",
            raw[position],
            f"a finite float32 value at position {position}",
            location,
        )
    values.flags.writeable = False
    return values


def _check_child_links(
    left_children: Sequence[int],
    right_children: Sequence[int],
    location: str,
) -> None:
    """Require every child index to be a leaf marker or an in-range index.

    Both children are ``-1`` at a leaf. A leaf whose ``right_children`` entry
    is something else is the vector-leaf signature, where that slot carries a
    block index instead of a child (``probes/tree_structure.md`` section 7g)
    -- a shape v1 refuses rather than walks.

    Note what is deliberately *not* required: that a child index exceed its
    parent's. Our exporter emits forward-pointing children and
    :mod:`xgboost_bridge.trees` asserts it when reading a source model, but
    FORMAT.md section 8 does not make it normative for an artifact, so
    demanding it here would refuse a conforming artifact from another
    producer. Termination is enforced directly instead, by
    :func:`_check_reachable_subgraph_terminates`.
    """
    node_count = len(left_children)
    for index in range(node_count):
        left_child = left_children[index]
        right_child = right_children[index]
        if left_child == LEAF_CHILD:
            if right_child != LEAF_CHILD:
                raise _malformed(
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
            if not 0 <= child < node_count:
                raise _malformed(
                    field,
                    child,
                    f"-1, or an index in [0, {node_count}) at node {index}",
                    location,
                )


def _check_reachable_subgraph_terminates(
    left_children: Sequence[int],
    right_children: Sequence[int],
    location: str,
) -> None:
    """Raise if a cycle is reachable from the root.

    Every other refusal in this module is against a wrong number. This one is
    against a **hang**: ``walk_margin`` follows children until it meets a
    leaf, so a cycle among reachable nodes never terminates, and a
    non-terminating predictor is not something a caller can catch. The check
    is confined to the reachable subgraph, because FORMAT.md section 13
    forbids raising on an unreachable node whatever it contains.

    A shared subtree -- two parents pointing at one child, no cycle -- is not
    refused. It terminates, so it is not this check's business.
    """
    node_count = len(left_children)
    colour = [_UNVISITED] * node_count
    # (node, revisiting) -- the second pass over a node marks it settled,
    # which is what turns a plain reachability walk into cycle detection.
    # Iterative rather than recursive: a deep tree would otherwise depend on
    # the interpreter's recursion limit.
    pending: list[tuple[int, bool]] = [(0, False)]
    while pending:
        node, revisiting = pending.pop()
        if revisiting:
            colour[node] = _SETTLED
            continue
        if colour[node] != _UNVISITED:
            continue
        colour[node] = _ON_PATH
        pending.append((node, True))
        if left_children[node] == LEAF_CHILD:
            continue
        for child in (left_children[node], right_children[node]):
            if colour[child] == _ON_PATH:
                raise _malformed(
                    "left_children",
                    child,
                    f"a child that does not close a cycle back to node {child}; "
                    "a cycle reachable from the root would make the walk "
                    "non-terminating",
                    location,
                )
            if colour[child] == _UNVISITED:
                pending.append((child, False))


def _read_tree(raw: object, index: int, feature_count: int) -> Mapping[str, Any]:
    """Read one tree object into the five arrays FORMAT.md section 8 specifies."""
    location = f"trees[{index}]"
    tree = _require_mapping(raw, "<tree>", location)
    _check_keys(tree, TREE_KEYS, location)

    arrays = {
        key: _read_integer_array(tree[key], key, location)
        for key in _INTEGER_TREE_KEYS
    }
    node_values = _read_node_values(tree["node_values"], location)

    node_count = len(arrays["left_children"])
    if node_count == 0:
        raise _malformed(
            "left_children",
            0,
            "at least one node, since node 0 is the root",
            location,
        )
    for key in ("right_children", "split_indices", "default_left"):
        if len(arrays[key]) != node_count:
            raise _malformed(
                key,
                len(arrays[key]),
                f"length {node_count}, matching left_children",
                location,
            )
    # `len`, not `.size`: this check is about the node count and nothing else.
    # An ndarray-only spelling would make it fail as an AttributeError if
    # `_read_node_values` ever stopped returning an array, reporting a length
    # problem where the real failure is the float32 dtype -- which
    # `_read_node_values` and the test that drives it are what pin.
    if len(node_values) != node_count:
        raise _malformed(
            "node_values",
            len(node_values),
            f"length {node_count}, matching left_children",
            location,
        )

    for node, column in enumerate(arrays["split_indices"]):
        # Total, not conditional on the node being internal: neutralized dead
        # slots carry `split_indices == 0` precisely so that this check needs
        # no exception (FORMAT.md section 8.3). An exception here would have
        # to apply to every artifact rather than only pruned ones.
        if not 0 <= column < feature_count:
            raise _malformed(
                "split_indices",
                column,
                f"an index in [0, {feature_count}) at node {node}",
                location,
            )
    for node, direction in enumerate(arrays["default_left"]):
        if direction not in (0, 1):
            raise _malformed(
                "default_left", direction, f"0 or 1 at node {node}", location
            )

    _check_child_links(arrays["left_children"], arrays["right_children"], location)
    _check_reachable_subgraph_terminates(
        arrays["left_children"], arrays["right_children"], location
    )

    return MappingProxyType(
        {
            "default_left": arrays["default_left"],
            "left_children": arrays["left_children"],
            "node_values": node_values,
            "right_children": arrays["right_children"],
            "split_indices": arrays["split_indices"],
        }
    )


def _read_trees(artifact: Mapping[str, Any], feature_count: int) -> tuple[Mapping[str, Any], ...]:
    """Read ``trees`` in artifact order, which is normative (FORMAT.md 8.2).

    An empty list is valid and is not a special case: a zero-boosting-round
    model serializes ``"trees": []``, present and empty, and its margin is the
    intercept alone.
    """
    raw = artifact["trees"]
    if not isinstance(raw, list):
        raise _malformed("trees", type(raw).__name__, "a JSON array", None)
    return tuple(
        _read_tree(entry, index, feature_count) for index, entry in enumerate(raw)
    )


class Predictor:
    """A loaded artifact, ready to produce margins and outputs row by row.

    Construct with :meth:`from_json` from an already-parsed artifact. There is
    no file-reading entry point on purpose (D006): a consumer does its own
    I/O, which is what lets the JavaScript port share this shape exactly.

    Every numeric value is float32 from the moment it is read. ``margin``
    returns the accumulator of FORMAT.md section 10 untouched, and ``output``
    applies the artifact's own ``output_transform`` under float32 semantics
    with no platform transcendental anywhere on the path.

    Attributes are exposed read-only. ``node_values`` arrays are marked
    non-writeable, so an inspecting caller can read a threshold at its exact
    float32 value but cannot alter the loaded model.
    """

    __slots__ = (
        "_feature_name_set",
        "_feature_names",
        "_format_version",
        "_intercept",
        "_objective",
        "_output_function",
        "_output_transform",
        "_provenance",
        "_trees",
    )

    def __init__(self, artifact: Mapping[str, Any]) -> None:
        """Validate and load ``artifact``; see :meth:`from_json`.

        The constructor is the validating path, so there is no way to build a
        ``Predictor`` that skipped it. Field order below is deliberate: the
        envelope's key set is checked before any value is read, so an artifact
        with an eighth key or a missing key raises on that rather than on
        whatever the extra key happens to break first.
        """
        envelope = _require_mapping(artifact, "<artifact>", None)
        _check_keys(envelope, ENVELOPE_KEYS, None)

        self._format_version = _read_format_version(envelope)
        self._objective = _read_objective(envelope)
        self._output_transform = _read_output_transform(envelope, self._objective)
        self._output_function = OUTPUT_FUNCTIONS[self._output_transform]
        self._feature_names = _read_feature_names(envelope)
        self._feature_name_set = frozenset(self._feature_names)
        self._intercept = _read_intercept(envelope)
        self._provenance = _read_provenance(envelope)
        self._trees = _read_trees(envelope, len(self._feature_names))

    @classmethod
    def from_json(cls, artifact: Mapping[str, Any]) -> "Predictor":
        """Load a parsed artifact, validating it per FORMAT.md section 13.

        Args:
            artifact: The artifact as a parsed JSON object -- what
                ``json.loads`` returns, or what
                :func:`xgboost_bridge.export.export_model` produces. Not a
                path and not a string: this reader does no I/O.

        Returns:
            A :class:`Predictor` whose thresholds, leaf values and intercept
            are already float32.

        Raises:
            :class:`~xgboost_bridge.errors.UnsupportedVersionError`:
                ``format_version`` is not exactly the integer ``1``.
            :class:`~xgboost_bridge.errors.UnrecognizedFieldError`: a key at
                any level is not one this format defines.
            :class:`~xgboost_bridge.errors.UnsupportedObjectiveError`:
                ``objective`` is outside the enumerated set.
            :class:`~xgboost_bridge.errors.MalformedTreeError`: any other
                disagreement with the format -- an absent required key, a
                wrong JSON type, an ``output_transform`` that is unknown or
                does not pair with ``objective``, unequal tree array lengths,
                an out-of-range child index or ``split_indices``, a
                non-finite ``node_values`` entry or ``intercept``, an empty
                or duplicated ``feature_names``, or a cycle reachable from a
                tree's root.
        """
        return cls(artifact)

    # -- inspection ---------------------------------------------------------

    @property
    def format_version(self) -> int:
        """The artifact's format version. Always ``1`` for a loaded artifact."""
        return self._format_version

    @property
    def objective(self) -> str:
        """The objective recorded in the artifact -- **metadata only** (D028).

        Exposed for inspection. Nothing on the prediction path reads it, and
        the test suite asserts that at source level: once a prediction branches
        on this field it becomes a second source of truth about behaviour that
        ``output_transform`` already determines.
        """
        return self._objective

    @property
    def output_transform(self) -> str:
        """The name of the transform ``output`` applies to the margin."""
        return self._output_transform

    @property
    def feature_names(self) -> tuple[str, ...]:
        """The exact key set a prediction input must carry (D005)."""
        return self._feature_names

    @property
    def intercept(self) -> np.float32:
        """The float32 margin-space intercept, exactly as loaded.

        Never transformed, and never normalized: ``-0.0`` is a reachable value
        and stays ``-0.0``.
        """
        return self._intercept

    @property
    def provenance(self) -> Mapping[str, str]:
        """The artifact's provenance block. Read by no prediction path."""
        return self._provenance

    @property
    def trees(self) -> tuple[Mapping[str, Any], ...]:
        """The loaded trees in artifact order, which is normative.

        Each is a read-only mapping of the five arrays of FORMAT.md section 8.
        ``node_values`` is a non-writeable ``dtype=np.float32`` array: reading
        an entry yields the exact float32 the engine compares against, which
        is the property that makes narrowing structural rather than a
        discipline every consumer has to remember.
        """
        return self._trees

    # -- prediction ---------------------------------------------------------

    def margin(self, row: Mapping[str, float]) -> np.float32:
        """Return one row's float32 margin.

        Args:
            row: A mapping whose key set equals :attr:`feature_names`
                **exactly** -- no missing key, no extra key (D005). ``NaN`` is
                the missing value and routes by the node's ``default_left``.

        Returns:
            The margin as ``np.float32``: the accumulator of FORMAT.md section
            10, untouched after the last addition.

        Raises:
            :class:`~xgboost_bridge.errors.FeatureKeyMismatchError`: the key
                set is not exactly :attr:`feature_names`. Lenient handling
                would turn a typo into a missing-value path, which is
                legitimate model structure, so the mistake would become a
                confident wrong number instead of an error.
            :class:`~xgboost_bridge.errors.NonFiniteFeatureError`: a value is
                ``±inf`` (D022, D045). Raised by the walk, which checks the
                whole row before visiting a node, so the outcome is a property
                of the input rather than of which branches this model happens
                to take.
        """
        return walk_margin(self._trees, self._intercept, self._feature_row(row))

    def output(self, row: Mapping[str, float]) -> np.float32:
        """Return one row's float32 output: the transform applied to the margin.

        The transform is selected by ``output_transform`` alone and is the
        bundled float32 implementation (D030, D032) -- ``sigmoid`` reproduces
        XGBoost's measured clamp floor and ``exp`` reproduces its overflow to
        ``+inf``. No platform transcendental is called.

        Raises the same errors as :meth:`margin`, for the same reasons.
        """
        return self._output_function(self.margin(row))

    def margins(self, rows: Iterable[Mapping[str, float]]) -> np.ndarray:
        """Return :meth:`margin` for each row, as a ``dtype=np.float32`` array.

        A loop over the single-row path, not a second implementation of it:
        the per-row result is the normative one and this only collects them.
        A vectorized walk would be a second numeric path to keep in agreement
        with the first, and with the JavaScript port.
        """
        return self._collect(self.margin(row) for row in rows)

    def outputs(self, rows: Iterable[Mapping[str, float]]) -> np.ndarray:
        """Return :meth:`output` for each row, as a ``dtype=np.float32`` array."""
        return self._collect(self.output(row) for row in rows)

    @staticmethod
    def _collect(values: Iterable[np.float32]) -> np.ndarray:
        """Gather float32 scalars into a float32 array, changing no bit.

        Every element is already ``np.float32``, so this copies bit patterns
        rather than converting: ``-0.0``, ``+inf`` and subnormals survive.
        """
        return np.asarray(list(values), dtype=np.float32).reshape(-1)

    def _feature_row(self, row: Mapping[str, float]) -> np.ndarray:
        """Order a row's values by column index, as a strong float64 array.

        Two decisions here, both deliberate:

        * The key set is compared for **exact** equality first, before a value
          is touched, and reports missing and extra keys together so a typo --
          which is one of each -- is diagnosed as a typo.
        * The array is ``dtype=np.float64``, not float32. The walk casts both
          sides of every comparison, so the result is identical either way,
          but a *pre-narrowed* row would make the walk's sample-side cast
          unobservable: under NEP 50 a float32 operand drags the comparison
          into float32 whether or not the cast is there. Handing the walk
          strong float64 keeps that cast load-bearing, which is the regime
          FORMAT.md section 10.1 requires it to be pinned in. The conversion
          itself is exact -- every Python float is a float64.
        """
        if not isinstance(row, Mapping):
            raise _malformed(
                "<row>",
                type(row).__name__,
                "a mapping from feature name to value",
                None,
            )
        present = frozenset(row)
        missing = self._feature_name_set - present
        extra = present - self._feature_name_set
        if missing or extra:
            raise FeatureKeyMismatchError(missing, extra)

        values = np.empty(len(self._feature_names), dtype=np.float64)
        for index, name in enumerate(self._feature_names):
            values[index] = float(row[name])
        return values
