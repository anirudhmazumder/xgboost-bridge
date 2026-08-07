"""Pin the two narrowing sites inside `walk_margin`, each on its own.

Found by `tools/revert_harness.py` on its first run: reverting the threshold-side
cast, and reverting the accumulator's narrowing, both left the suite **green**.
That is the finding the harness exists to produce, and it was not a false alarm.

Why nothing noticed. Every tree the `Predictor` builds has already been narrowed at
parse time, so inside `walk_margin` the threshold is *already* `np.float32` and
`np.float32 + np.float32` stays float32 under NEP 50. Both explicit casts are
therefore no-ops on that path, and the whole 1073-test suite only ever reached
`walk_margin` through that path.

They are not no-ops in general. `walk_margin` is public, normative, and documented
as accepting `Sequence[float] | np.ndarray` -- a caller who parses an artifact with
`json.load` and hands over the raw arrays gets float64, and then both casts change
the answer. Measured, both directions, before writing these tests:

* threshold side, float64 tree: with the cast the row routes RIGHT, without it
  LEFT -- a different subtree.
* accumulator, float64 tree: without the cast the sum stays float64 to the end,
  which is the "narrowed once at the end" deviation that scored 318-2541/5000.

One subtlety runs through all of it, and it is why the un-narrowed spellings must
be `np.float64` rather than plain Python floats. Under NEP 50 a Python float is
*weakly* typed: `np.float32(x) < 0.3` is evaluated in float32, so a Python-float
threshold gets narrowed by the comparison itself and hides the defect. A
`np.float64` is strongly typed and promotes to float64. The same input, spelled two
ways, exposes or conceals the bug -- so these tests spell it deliberately.

Per CLAUDE.md these are pinned **independently**. Reverting both narrowing sites
at once pins neither, because each can absorb the other's failure.
"""

from __future__ import annotations

import numpy as np

from xgboost_bridge.trees import walk_margin


def _leaf_only_tree(leaf: float, dtype) -> dict:
    """A single leaf. No split, so the accumulator is isolated from the comparison."""
    return {
        "left_children": [-1],
        "right_children": [-1],
        "split_indices": [0],
        "node_values": np.asarray([leaf], dtype=dtype),
        "default_left": [0],
    }


def test_the_threshold_side_cast_changes_the_route_on_an_un_narrowed_tree():
    """Pins `np.float32(threshold)`; the sample-side cast is present throughout.

    The threshold is one float64 step *above* an exact float32, so narrowing it
    rounds **down** — and that is the only direction in which narrowing can move
    the boundary across a float32 value. The sample is that float32 exactly, so
    with the cast it compares equal and equality routes RIGHT, while without it the
    comparison happens in float64 where the sample is genuinely smaller and routes
    LEFT.
    """
    exact32 = np.float32(0.30000001192092896)
    threshold = np.nextafter(np.float64(exact32), np.float64(np.inf))
    assert np.float32(threshold) == exact32, "the threshold must still narrow onto exact32"
    assert threshold > np.float64(exact32), "and it must narrow downward"

    tree = {
        "left_children": [1, -1, -1],
        "right_children": [2, -1, -1],
        "split_indices": [0, 0, 0],
        # float64 deliberately: this is a caller's own parse, not a Predictor tree.
        "node_values": np.asarray([threshold, -100.0, 100.0], dtype=np.float64),
        "default_left": [0, 0, 0],
    }
    row = np.asarray([np.float64(exact32)], dtype=np.float64)

    margin = walk_margin([tree], np.float32(0.0), row)

    # RIGHT leaf. Without `np.float32(threshold)` this is -100.0.
    assert float(margin) == 100.0, "the threshold-side cast is not being applied"


def test_the_accumulator_is_narrowed_after_every_addition_not_once_at_the_end():
    """Pins the outer `np.float32(...)` around the accumulator update.

    Ten leaves of `0.4` added to an intercept of `1e7`. At that magnitude a float32
    ULP is exactly `1.0`, so each `0.4` rounds away and a per-addition narrowing
    holds the accumulator at `1e7` forever. A float64 sum keeps all ten and lands
    four ULP away — visible, deterministic, and precisely the deviation the
    accumulation recipe forbids.
    """
    trees = [_leaf_only_tree(0.4, np.float64) for _ in range(10)]

    margin = walk_margin(trees, np.float32(1e7), np.asarray([0.0], dtype=np.float64))

    assert np.float32(1e7) + np.float32(0.4) == np.float32(1e7), "premise: 0.4 rounds away"
    assert float(margin) == 1e7, "the accumulator was not narrowed after every addition"
    # Stated as a bit pattern too, because `1e7 == 1e7` would also hold for a
    # float64 accumulator that happened to land on the same value.
    assert np.float32(margin).view(np.uint32) == np.float32(1e7).view(np.uint32)
    assert np.float64(margin) != 1e7 + 10 * 0.4, "the float64 route must be excluded"


def test_a_narrowed_tree_cannot_reach_either_defect():
    """Why the suite was green: the Predictor's own path is immune.

    Asserted rather than left implicit. It explains the harness finding, and it
    documents that these two casts protect the *public API* rather than the
    predictor — which is the reason to keep them rather than collapse them away.
    """
    exact32 = np.float32(0.30000001192092896)
    threshold = np.nextafter(np.float64(exact32), np.float64(np.inf))

    # The same tree the previous test uses, narrowed the way the Predictor narrows it.
    narrowed = np.asarray([threshold, -100.0, 100.0], dtype=np.float32)
    assert narrowed[0] == exact32, "parse-time narrowing collapses the distinction"

    tree = {
        "left_children": [1, -1, -1],
        "right_children": [2, -1, -1],
        "split_indices": [0, 0, 0],
        "node_values": narrowed,
        "default_left": [0, 0, 0],
    }
    row = np.asarray([np.float64(exact32)], dtype=np.float64)
    assert float(walk_margin([tree], np.float32(0.0), row)) == 100.0

    # And the accumulator: float32 + float32 stays float32 under NEP 50, so the
    # explicit narrowing has nothing left to do on this path.
    trees = [_leaf_only_tree(0.4, np.float32) for _ in range(10)]
    assert float(walk_margin(trees, np.float32(1e7), np.asarray([0.0]))) == 1e7
