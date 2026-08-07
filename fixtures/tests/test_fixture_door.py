"""Pin the generator's narrows-onto probe, which closes the fixture door.

The door is this: every guard in the repository treats a fixture's
`expected_margin` as ground truth. It genuinely is ground truth -- the generator
takes it from `booster.predict()` and never from this repository's walk -- but
a guard only fires on an input that *distinguishes* a defect, and the corpus rows
distinguish almost nothing about the split comparison. Every value they carry is
already float32-exact, so narrowing it is a no-op.

Measured, not argued: reverting the sample-side `np.float32` cast in
`walk_margin` and regenerating the corpus **succeeded silently** before the probe
existed, and **fails** with it. That is the whole reason this file exists.

The probe therefore needs its own pinning, because its failure mode is *vacuity*.
If `_narrows_onto` ever returns `None` for everything -- one bad refactor of the
`nextafter` arithmetic would do it -- the generator prints "no internal nodes" and
passes, and nothing else in the suite would notice a guard that stopped guarding.
So this file asserts the probe still produces rows, and that the rows still
discriminate, using a hand-built tree rather than the corpus so the assertion does
not depend on which fixtures happen to exist.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate.probe_rows import narrows_onto  # noqa: E402

# One internal node splitting on feature 0, two leaves. Threshold chosen with a
# long float32 mantissa so a narrows-onto value exists; leaves far apart so a
# misroute is unmistakable rather than a low-order-bit difference.
_TREE = {
    "left_children": [1, -1, -1],
    "right_children": [2, -1, -1],
    "split_indices": [0, 0, 0],
    "node_values": np.asarray([0.30000001192092896, -100.0, 100.0], dtype=np.float32),
    "default_left": [False, False, False],
}


def _route(value: float, *, narrow_sample: bool) -> float:
    """The split comparison, with the sample-side cast under our control.

    Deliberately a re-statement of the comparison rather than a call into
    `walk_margin`: the point is to demonstrate that the *row class* separates two
    candidate implementations, and calling the real one could only ever show that
    it agrees with itself.

    `np.float64(value)`, not the bare Python float, and the distinction is not
    cosmetic. Under NEP 50 a Python float is *weakly* typed: `0.3 < np.float32(x)`
    is evaluated in **float32**, so the un-narrowed implementation silently
    narrows anyway and this test passed for the wrong reason on the first
    attempt. A `np.float64` is strongly typed and promotes the comparison to
    float64, which is what the real walk does -- its values come out of a float64
    row array. The defect is therefore invisible under one spelling of "the same"
    input and visible under the other.
    """
    threshold = np.float32(_TREE["node_values"][0])
    sample = np.float64(value)
    left = np.float32(sample) < threshold if narrow_sample else sample < threshold
    return float(_TREE["node_values"][1 if left else 2])


def test_narrows_onto_yields_values_on_both_sides_of_the_threshold():
    threshold = float(_TREE["node_values"][0])
    pair = narrows_onto(threshold)
    assert pair is not None, "no probe row for a threshold with a long mantissa"
    below, above = pair

    # float64-distinct from the threshold, on opposite sides of it...
    assert below < threshold < above
    # ...and float32-indistinguishable from it. Both halves are required: a value
    # that narrows onto the threshold but sits on the same side of it in float64
    # routes identically either way and proves nothing.
    assert np.float32(below) == np.float32(threshold)
    assert np.float32(above) == np.float32(threshold)


def test_the_probe_row_class_separates_the_two_implementations():
    """Without this, the guard could pass while testing nothing.

    `below` is the discriminating value. Narrowed it compares equal to the
    threshold, and equality routes RIGHT (CLAUDE.md, measured on 104/104 internal
    nodes). Un-narrowed it is strictly less, and routes LEFT -- a different leaf.
    """
    below, above = narrows_onto(float(_TREE["node_values"][0]))

    assert _route(below, narrow_sample=True) == 100.0
    assert _route(below, narrow_sample=False) == -100.0

    # `above` agrees under both implementations. Asserted rather than omitted:
    # it shows the class is not merely "any row disagrees", which would suggest
    # the harness is broken rather than the comparison.
    assert _route(above, narrow_sample=True) == _route(above, narrow_sample=False)


@pytest.mark.parametrize("threshold", [0.0, float("inf"), float("-inf"), float("nan")])
def test_narrows_onto_declines_thresholds_it_cannot_probe(threshold):
    """Declining is correct here; the generator reports the count and moves on.

    Zero has no neighbour that rounds onto it under this construction, and a
    non-finite threshold cannot appear in a real split. Returning `None` rather
    than raising keeps the generator's failure reserved for a real disagreement.
    """
    assert narrows_onto(threshold) is None


def test_the_corpus_actually_produces_probe_rows():
    """Guards against the vacuous case: a probe that silently covers nothing.

    Runs against the committed corpus rather than a fitted model, so it stays fast
    and does not need XGBoost. It asserts only that thresholds worth probing exist
    in quantity -- the agreement itself is checked at generation time, against
    XGBoost, which is the only oracle that can check it.
    """
    import json
    import pathlib

    corpus = pathlib.Path(__file__).resolve().parents[1] / "corpus"
    probeable = 0
    internal = 0
    for path in sorted(corpus.glob("*.json")):
        artifact = json.loads(path.read_text())["artifact"]
        for tree in artifact["trees"]:
            for node, child in enumerate(tree["left_children"]):
                if child == -1:
                    continue
                internal += 1
                if narrows_onto(float(tree["node_values"][node])) is not None:
                    probeable += 1

    assert internal > 0, "the corpus has no internal nodes at all"
    # 661 of 663 at the time of writing; the two exceptions are zero thresholds,
    # which the construction above cannot probe. Asserted as a fraction rather
    # than a literal so adding a fixture does not fail this test, and as a high
    # fraction rather than `> 0` because one probeable node is not coverage.
    assert probeable >= 0.9 * internal, f"only {probeable}/{internal} nodes probeable"
