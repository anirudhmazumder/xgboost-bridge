"""Cross-language parity at scale, on deliberately adversarial generated rows.

`run_parity.py` compares the two predictors over the 299-row fixture corpus.
This runs the *same* comparison over ~100,000 generated rows, and exists because
the corpus is small enough that a defect reachable only from a specific feature
value could sit inside it unnoticed.

**It imports the comparator rather than reimplementing it.** `python_side`,
`javascript_side` and `compare_sides` come from `run_parity`, so this script
cannot disagree with the corpus gate about what a mismatch is. All it adds is
rows.

**The rows are adversarial, not uniform, because uniform rows cannot find the
defect that matters.** The float32 split comparison is the highest-value
invariant in the library, and a value that is merely *near* a threshold routes
the same way whether the comparison casts both sides or only one. It is the
value landing **exactly on** the threshold that separates them — and drawing
feature values from a continuous distribution hits one with probability zero.
Measured: 0 of 20,000 random continuous rows detect an incorrectly-cast float32
comparison, while a single on-threshold row detects it immediately.

So for every internal node of every source artifact this generates the threshold
exactly, its two float32 neighbours, and two float64 values that **narrow onto**
it without equalling it — then spends the remaining budget on more rows of the
same kind along different paths, rather than on random values. One row in ten is
left fully random as a control.

**The narrows-onto class is the one that pins the sample-side cast**, and neither
the corpus nor an on-threshold row can do it: an on-threshold row carries a value
that is already float32-exact, so narrowing the sample is a no-op there. Measured
by reverting `Math.fround` on the sample in the JavaScript walk — the 299-row
corpus reports **1** mismatch, this reports **1279 of 20000**.

Two protections that this harness deliberately does *not* claim to pin, because
they absorb each other: the parse-time `Float32Array` and the threshold-side
`Math.fround`. Reverting either alone leaves parity at exactly `0.0`, since each
fully narrows the threshold on its own. The storage type is pinned instead by a
direct type assertion — `artifact.test.js`'s "node_values is loaded into a
Float32Array" and `test_predict.py`'s `dtype` check — which is the check that can
actually see it. See D045.

Both sides read the same generated fixture file, so they receive identical
float64 **bit patterns**; `run_parity`'s own `input_bits` check confirms that
rather than assuming it, which is what keeps this from becoming a test of two
JSON number parsers.

Usage:

    uv run python parity/run_parity_scale.py            # ~100,000 rows
    uv run python parity/run_parity_scale.py --rows 5000
    uv run python parity/run_parity_scale.py --seed 7

Exit status is 0 only if both measurement points are exactly `0.0` and the
sensitivity check confirms the comparator would have reported a defect.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_parity import (  # noqa: E402 -- deliberate: sys.path is set above
    CORPUS_DIR,
    MEASUREMENT_POINTS,
    compare_sides,
    javascript_side,
    python_side,
)

#: Source artifacts, chosen to span all three objectives and several tree
#: shapes: a deep logistic model, a Cox model, a pruned model carrying dead
#: nodes, a model whose training data had missing values in both directions, and
#: a single-feature model where every row lands on the one split column.
SOURCE_FIXTURES: tuple[str, ...] = (
    "binary_logistic_base_score_high_inside_clamp.json",
    "survival_cox_base_score_high.json",
    "gamma_pruned_dead_nodes.json",
    "missing_value_both_directions.json",
    "single_feature_model.json",
)

#: Values that are legal float64 but sit at the edges of the format. Subnormals
#: and values past the float32 range are included because narrowing them is
#: where a `Math.fround` and a `np.float32` could plausibly part company.
EXTREME_VALUES: tuple[float, ...] = (
    0.0,
    -0.0,
    5e-324,  # smallest positive float64 subnormal; narrows to +0.0
    -5e-324,
    1.401298464324817e-45,  # smallest positive float32 subnormal
    -1.401298464324817e-45,
    1.1754943508222875e-38,  # smallest normal float32
    3.4028234663852886e38,  # largest finite float32
    -3.4028234663852886e38,
    1e30,
    -1e30,
    1e39,  # past float32; narrows to +inf
    -1e39,
    2.2250738585072014e-308,
)


def _float32_neighbours(value: float) -> tuple[float, float, float]:
    """A float32 value and its two adjacent float32 values, as float64.

    Returned as the float64 that recovers exactly each float32, so the fixture's
    JSON number is unambiguous and both languages narrow it identically.
    """
    exact = np.float32(value)
    below = np.nextafter(exact, np.float32(-np.inf))
    above = np.nextafter(exact, np.float32(np.inf))
    return float(exact), float(below), float(above)


def _narrows_onto(threshold: float) -> tuple[float, float] | None:
    """Two float64 values that **narrow onto** ``threshold`` without equalling it.

    This is the row class that pins the *sample* side of the split comparison,
    and neither the fixture corpus nor an on-threshold row can do it.

    An on-threshold row carries the float64 that recovers the threshold's float32
    exactly, so narrowing the sample is a no-op and an implementation that skips
    it still routes correctly. The value that separates them is one that is
    **not** float32-representable but rounds to the threshold: narrowed, it
    compares equal and routes RIGHT; un-narrowed, a value below the threshold
    routes LEFT instead. That is a whole subtree of difference from one missing
    cast.

    Returns the below/above pair, or ``None`` where the arithmetic degenerates --
    at zero, at the float32 extremes, and wherever the midpoint is not strictly
    between the neighbours in float64.
    """
    exact = np.float32(threshold)
    if exact == 0 or not np.isfinite(exact):
        return None
    up = np.nextafter(exact, np.float32(np.inf))
    down = np.nextafter(exact, np.float32(-np.inf))
    if not (np.isfinite(up) and np.isfinite(down)):
        return None
    wide = np.float64(exact)
    # Rounding boundaries: anything strictly inside these narrows to `exact`.
    high_boundary = (wide + np.float64(up)) / 2
    low_boundary = (wide + np.float64(down)) / 2
    above = float((wide + high_boundary) / 2)
    below = float((wide + low_boundary) / 2)
    # Assert the property rather than trusting the algebra: these rows are only
    # worth generating if they really do narrow onto the threshold.
    if np.float32(above) != exact or np.float32(below) != exact:
        return None
    if not (below < float(wide) < above):
        return None
    return below, above


def _internal_nodes(artifact: dict[str, Any]) -> list[tuple[int, float]]:
    """Every internal node as ``(feature_index, threshold)``.

    Leaf iff ``left_children[i] == -1`` (FORMAT.md section 10). A leaf's
    ``node_values`` entry is a leaf value, not a threshold, and feeding it to a
    split column would generate a row that tests nothing in particular.
    """
    nodes: list[tuple[int, float]] = []
    for tree in artifact["trees"]:
        left = tree["left_children"]
        for index, child in enumerate(left):
            if child == -1:
                continue
            nodes.append((int(tree["split_indices"][index]), float(tree["node_values"][index])))
    return nodes


def _baseline_row(feature_count: int, rng: np.random.Generator) -> list[Any]:
    """A row of ordinary values, used as the background for a targeted row."""
    return [float(np.float32(v)) for v in rng.normal(size=feature_count)]


def generate_rows(
    artifact: dict[str, Any], target: int, rng: np.random.Generator
) -> tuple[list[list[Any]], dict[str, int]]:
    """Build the adversarial row set for one artifact.

    Composition is counted rather than estimated, and reported, because "the
    rows were adversarial" is the kind of claim that quietly stops being true.
    """
    feature_count = len(artifact["feature_names"])
    rows: list[list[Any]] = []
    counts = {
        "on_threshold": 0,
        "narrows_onto_threshold": 0,
        "threshold_neighbour": 0,
        "missing": 0,
        "extreme": 0,
        "random": 0,
    }

    # Thresholds grouped by the column they are compared against, so a padded
    # row can put a feature exactly on a threshold that its own column actually
    # splits on. Setting feature 2 to feature 0's threshold would land on nothing.
    by_feature: dict[int, list[float]] = {}
    for feature_index, threshold in _internal_nodes(artifact):
        if math.isfinite(threshold):
            by_feature.setdefault(feature_index, []).append(float(np.float32(threshold)))
    split_features = sorted(by_feature)

    # 1. Every internal node's threshold, exactly, plus both neighbours. Exact
    #    equality is the case that matters: the rule is strict `<` with equality
    #    routing RIGHT, so an implementation that casts only one side of the
    #    comparison diverges here and nowhere else.
    for feature_index, threshold in _internal_nodes(artifact):
        if not math.isfinite(threshold):
            continue
        exact, below, above = _float32_neighbours(threshold)
        generated: list[tuple[float, str]] = [
            (exact, "on_threshold"),
            (below, "threshold_neighbour"),
            (above, "threshold_neighbour"),
        ]
        # The pair that pins the sample-side cast. See _narrows_onto.
        narrowing = _narrows_onto(threshold)
        if narrowing is not None:
            generated.append((narrowing[0], "narrows_onto_threshold"))
            generated.append((narrowing[1], "narrows_onto_threshold"))
        for value, kind in generated:
            row = _baseline_row(feature_count, rng)
            row[feature_index] = value
            rows.append(row)
            counts[kind] += 1

    # 2. The missing-value path: one row per feature, and one all-missing row.
    #    `None` is the corpus encoding for NaN (run_parity._row_to_input).
    for feature_index in range(feature_count):
        row = _baseline_row(feature_count, rng)
        row[feature_index] = None
        rows.append(row)
        counts["missing"] += 1
    rows.append([None] * feature_count)
    counts["missing"] += 1

    # 3. Format edges, in every column.
    for value in EXTREME_VALUES:
        for feature_index in range(feature_count):
            row = _baseline_row(feature_count, rng)
            row[feature_index] = value
            rows.append(row)
            counts["extreme"] += 1

    # 4. Padding. Deliberately *not* mostly random: a threshold has only so many
    #    internal nodes, so step 1 exhausts the systematic on-threshold rows
    #    quickly, and filling the remaining budget with continuous values spends
    #    it on the one class of row measured to detect nothing (0 of 20,000).
    #
    #    Instead each padded row lands on real thresholds again, along a
    #    different path: same equality comparison, different route through the
    #    tree, different accumulation order. One in ten is left fully random as a
    #    control, so the run is not exclusively pathological and the contrast
    #    between the two classes stays visible.
    modes = ("saturate", "single", "saturate", "single", "saturate", "single", "saturate", "single", "saturate", "random")
    index = 0
    while len(rows) < target:
        mode = modes[index % len(modes)]
        index += 1
        row = _baseline_row(feature_count, rng)
        if mode == "random" or not split_features:
            rows.append(row)
            counts["random"] += 1
            continue
        narrowed = False
        if mode == "saturate":
            # Every splitting column simultaneously on one of its own
            # thresholds: the deepest equality-routing row available.
            for feature_index in split_features:
                choices = by_feature[feature_index]
                row[feature_index] = choices[int(rng.integers(len(choices)))]
        else:
            feature_index = split_features[int(rng.integers(len(split_features)))]
            choices = by_feature[feature_index]
            chosen = choices[int(rng.integers(len(choices)))]
            # Half of the single-column rows use a value that narrows onto the
            # threshold rather than equalling it, so that class scales with the
            # row budget instead of being capped at the internal-node count.
            narrowing = _narrows_onto(chosen) if rng.integers(2) else None
            if narrowing is None:
                row[feature_index] = chosen
            else:
                row[feature_index] = narrowing[int(rng.integers(2))]
                narrowed = True
        rows.append(row)
        counts["narrows_onto_threshold" if narrowed else "on_threshold"] += 1

    return rows, counts


def build_fixtures(directory: Path, total_rows: int, seed: int) -> tuple[Path, ...]:
    """Write one generated fixture per source artifact, and return their paths.

    The artifact is copied verbatim from the corpus -- the same intercept, the
    same trees, the same emitted decimals. Only the rows are new, so any
    divergence this finds is in the readers or the walk, not in a hand-built
    artifact that no exporter would produce.
    """
    per_fixture = max(1, total_rows // len(SOURCE_FIXTURES))
    paths: list[Path] = []
    composition: dict[str, int] = {}
    for offset, name in enumerate(SOURCE_FIXTURES):
        source = json.loads((CORPUS_DIR / name).read_text(encoding="utf-8"))
        artifact = source["artifact"]
        # Seed per fixture so a single fixture can be regenerated identically in
        # isolation while the whole set stays deterministic.
        rng = np.random.default_rng(seed + offset)
        rows, counts = generate_rows(artifact, per_fixture, rng)
        for key, value in counts.items():
            composition[key] = composition.get(key, 0) + value
        target = directory / f"scale_{Path(name).stem}.json"
        target.write_text(
            json.dumps({"artifact": artifact, "rows": rows}), encoding="utf-8"
        )
        paths.append(target)

    print(f"generated {sum(composition.values())} rows over {len(paths)} artifacts")
    print(f"  seed                     {seed}")
    for key in (
        "on_threshold",
        "narrows_onto_threshold",
        "threshold_neighbour",
        "missing",
        "extreme",
        "random",
    ):
        print(f"  {key:<24} {composition[key]}")
    print()
    return tuple(paths)


def _injected_sensitivity(
    python_records: dict[str, list[dict[str, Any]]],
    javascript_records: dict[str, list[dict[str, Any]]],
) -> tuple[bool, str]:
    """Confirm the comparator would report a one-bit defect at this scale.

    A parity run that has never been observed to fail is not evidence of
    anything, and "0 mismatches over 100,000 rows" is exactly the shape of
    result that a silently broken comparison also produces. So one row is
    perturbed by a single ULP and the comparison must report exactly one
    mismatch, at the right measurement point.
    """
    for point in MEASUREMENT_POINTS:
        perturbed = copy.deepcopy(python_records)
        target_fixture = next(
            (name for name, records in perturbed.items()
             if any(r.get(point) is not None for r in records)),
            None,
        )
        if target_fixture is None:
            return False, f"no fixture carried a {point} value to perturb"
        index = next(
            i for i, r in enumerate(perturbed[target_fixture]) if r.get(point) is not None
        )
        original = perturbed[target_fixture][index][point]
        moved = np.nextafter(
            np.uint32(int(original, 16)).view(np.float32), np.float32(np.inf)
        )
        perturbed[target_fixture][index][point] = f"0x{int(moved.view(np.uint32)):08x}"
        if perturbed[target_fixture][index][point] == original:
            return False, f"nextafter did not move the {point} value"

        report = compare_sides(perturbed, javascript_records)
        found = getattr(report, f"{point}_mismatches")
        if len(found) != 1:
            return False, (
                f"injecting one ULP at the {point} point produced "
                f"{len(found)} mismatches, expected exactly 1"
            )
    return True, "one ULP injected at each measurement point produced exactly one mismatch each"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000, help="approximate total rows")
    parser.add_argument("--seed", type=int, default=20260806, help="generator seed")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="parity-scale-") as directory:
        paths = build_fixtures(Path(directory), args.rows, args.seed)
        response = javascript_side(paths)
        # Warnings are counted rather than left to scroll past. The generated set
        # includes values like 1e39 that are finite as float64 and infinite once
        # narrowed to float32, and numpy emits `RuntimeWarning: overflow
        # encountered in cast` on those. That is not a parity defect -- both
        # languages agree on every one of these rows -- but a library warning on
        # input it accepts is worth a number rather than a silence.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            python_records = python_side(paths)
        javascript_records = response["fixtures"]
        report = compare_sides(python_records, javascript_records)

    warning_kinds: dict[str, int] = {}
    for entry in caught:
        key = f"{entry.category.__name__}: {entry.message}"
        warning_kinds[key] = warning_kinds.get(key, 0) + 1

    rows = sum(len(records) for records in python_records.values())
    print(f"rows compared                 {rows}")
    print(f"margin-point mismatches       {len(report.margin_mismatches)}")
    print(f"output-point mismatches       {len(report.output_mismatches)}")
    print(f"refusal disagreements         {len(report.refusal_disagreements)}")
    print(f"input-bit disagreements       {len(report.input_disagreements)}")
    print(f"objective-branch findings     {len(report.objective_findings)}")
    if report.structural:
        for note in report.structural:
            print(f"  STRUCTURAL: {note}")

    for label, found in (
        ("margin", report.margin_mismatches),
        ("output", report.output_mismatches),
    ):
        for divergence in found[:5]:
            print(f"  first {label} divergence: {divergence}")

    if warning_kinds:
        print("python-side warnings on accepted input:")
        for key, count in sorted(warning_kinds.items(), key=lambda kv: -kv[1]):
            print(f"  {count:>7}  {key}")
    else:
        print("python-side warnings         none")

    sensitive, detail = _injected_sensitivity(python_records, javascript_records)
    print(f"comparator sensitivity        {'confirmed' if sensitive else 'FAILED'} -- {detail}")

    print(f"  python {sys.version.split()[0]}  numpy {np.__version__}  node {response['node_version']}")
    print()
    if report.clean and sensitive:
        print(f"PARITY AT SCALE: 0.0 at both measurement points over {rows} rows, on bit patterns.")
        return 0
    print("PARITY AT SCALE: FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
