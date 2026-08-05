"""The high-precision reference table for the bundled float32 transform.

This file exists to give the **JavaScript** side an oracle that is independent
of the Python side. `packages/js/src/transform.ts` is a hand-written port of
`packages/python/src/xgboost_bridge/transform.py`, and comparing the two proves
only that the same code was written twice: bit-identical wrong is still wrong,
and it is invisible to a parity harness precisely because both sides agree
perfectly (D030, D034, FORMAT.md section 5.6). So the JavaScript implementation
is measured against `mpmath` at 50 digits, per function, independently — and
this module is what tabulates that reference so a Node test can consult it with
zero dependencies and without a Python interpreter in the loop.

Three things this module deliberately does **not** do:

* It does not import either implementation. Nothing here can inherit a defect
  from the code it is used to check.
* It does not call `np.exp`, `math.exp`, or any other platform transcendental
  for a reference value. `mpmath` computes in arbitrary precision with its own
  algorithms, and the result is rounded onto the float32 grid by exact integer
  arithmetic on `mpmath`'s own significand — not by narrowing twice through
  float64, which is only provably harmless for the results of the four basic
  operations and not for an arbitrary real.
* It does not treat the logistic clamp as a mathematical fact. Below the
  measured floor the specified output is not the mathematical logistic
  function, it is a value XGBoost was **measured** to return
  (`probes/output_transform.md` section 3). The `sigmoid_bits` column here is
  always the *unclamped* mathematical value; the indices where the clamp
  applies instead are listed separately, and the floor constants are
  transcribed from the probe report independently of both implementations.

**Why the clamp region is a predicate rather than a distance.** Below the floor
the implementation deliberately returns the floor value while an unclamped
reference keeps falling, so a ULP comparison reports the clamp as if it were
error — measured at 1,560,434 ULP on a first attempt at exactly this
verification (D046). The correct check there is "does it return exactly the
floor bit pattern, and never zero", which is what `sigmoid_predicate_indices`
tells the consumer to perform.

Ground truth is uint32 hex bit-pattern strings, for the same reason the fixture
corpus uses them (D044): JSON has no representation for `+inf`, `-inf`, `NaN`
or `-0.0`, all four of which occur here, and a decimal ground truth invites
`==` comparison under which `-0.0 == 0.0` is true for two values that are not
the same.

Run it directly; output is byte-identical across runs:

    uv run python fixtures/generate/reference.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from mpmath import mp

mp.dps = 50

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "corpus" / "reference"

__all__ = ["REFERENCE_DIR", "build_reference", "main"]

#: Fixed, so the sample is a property of this file rather than of the run.
SEED = 20260805

#: Number of points the table carries. A few thousand is enough for an
#: always-on test, and the boundary anchors below are never thinned away by it:
#: a smaller count thins the band scans and the pseudo-random fill instead, so
#: coverage narrows in density and never in extent.
POINT_COUNT = 6000


# ---------------------------------------------------------------------------
# Literals transcribed from probes/output_transform.md, deliberately
# independent of both implementations.
# ---------------------------------------------------------------------------

#: Section 3: the sole float32 in [-90, -88] reproducing XGBoost's observed
#: floor output, found by exhaustive scan of all 262145 candidates.
PROBE_FLOOR_MARGIN = -88.69999694824219
PROBE_FLOOR_MARGIN_BITS = 0xC2B16666

#: Section 3: the single value XGBoost's predict() returns for every margin
#: below the floor, on 1959/1959, 97/97 and 1/1 measured rows.
PROBE_FLOOR_OUTPUT = 3.006635794144578e-39
PROBE_FLOOR_OUTPUT_BITS = 0x0020BD47

#: Section 4: survival:cox has no floor, and returns +inf above this margin.
PROBE_EXP_OVERFLOW_MARGIN = 88.72283935546875

#: Margins recorded by the Python side as its worst logistic cases, at 2 ULP.
#: Carried explicitly so the table reaches them rather than relying on a draw
#: landing there: they appear at a rate near 3e-04 in a uniform sample, and a
#: sweep that does not visit where the error is largest cannot measure it, while
#: its silence reads exactly like good news (D046).
WORST_KNOWN_SIGMOID_MARGINS = (-16.635820388793945, -16.80551528930664)

#: The margin at which the Python side's exponential is furthest from correctly
#: rounded, at the edge of the argument-reduction interval.
WORST_KNOWN_EXP_MARGIN = 0.34657353162765503

#: First float32 whose logistic function is exactly 1.0.
FIRST_SATURATING_SIGMOID_MARGIN = 16.63553237915039

#: First float32 whose exponential underflows to zero in float32.
FIRST_UNDERFLOWING_EXP_MARGIN = -103.97208404541016

#: The margin at which the exponential's result crosses from normal into
#: subnormal.
SUBNORMAL_TRANSITION_MARGIN = -87.336544

MAX_FLOAT32 = float(np.finfo(np.float32).max)
MIN_NORMAL_FLOAT32 = float(np.finfo(np.float32).smallest_normal)
MIN_SUBNORMAL_FLOAT32 = float(np.finfo(np.float32).smallest_subnormal)

#: Beyond this magnitude the float32 result is decided without arithmetic: the
#: exponential overflows above a margin of 88.73 and underflows below -104, both
#: far inside this bound, so short-circuiting here cannot affect any value the
#: table reports. It exists only to keep `mpmath` away from margins whose
#: exponential has an exponent in the hundreds of millions.
DECIDED_MAGNITUDE = 1000.0

POS_INF_BITS = 0x7F800000
NEG_INF_BITS = 0xFF800000
ZERO_BITS = 0x00000000
ONE_BITS = 0x3F800000
CANONICAL_NAN_BITS = 0x7FC00000


# ---------------------------------------------------------------------------
# Bit-level helpers
# ---------------------------------------------------------------------------


def bits32(value: object) -> int:
    """The IEEE-754 single-precision encoding of `value`, as an integer."""
    with np.errstate(over="ignore", under="ignore"):
        return int(np.float32(value).view(np.uint32))


def f32_from_bits(bits: int) -> np.float32:
    return np.uint32(bits).view(np.float32)


def hex32(bits: int) -> str:
    return f"0x{bits:08x}"


def is_nan32(value: object) -> bool:
    bits = bits32(value)
    return (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0


# ---------------------------------------------------------------------------
# Exact rounding of an mpmath value onto the float32 grid
#
# Deliberately not `np.float32(float(value))`: that narrows twice, and double
# rounding through float64 is only provably harmless for the results of the four
# basic operations, not for an arbitrary real -- which is exactly what these
# functions receive.
# ---------------------------------------------------------------------------

#: A tie decided the wrong way would be an invisible off-by-one-ULP in the
#: reference itself. `mpmath` at 50 digits carries 169 bits, so its own error is
#: around 2**-169 relative -- but "around" is not a proof, so any value closer to
#: a float32 midpoint than this many quantum units raises instead of being
#: rounded. If it ever fires, the reference precision is the thing to raise.
TIE_SAFETY_MARGIN = mp.mpf(2) ** -60

_OVERFLOW_BINADE = 128
_UNDERFLOW_BINADE = -150


def _binade(value: Any) -> int:
    """The base-two exponent of `value`, that is, floor of its logarithm."""
    significand = int(value.man)
    if significand < 0:
        significand = -significand
    return significand.bit_length() - 1 + int(value.exp)


def _grid_exponent(value: Any) -> int:
    """The base-two exponent of the float32 quantum at `value`'s magnitude."""
    return max(_binade(value) - 23, -149)


def tie_distance(value: Any) -> Any:
    """Distance from the nearest float32 midpoint, in quantum units."""
    magnitude = abs(mp.mpf(value))
    if magnitude == 0 or not mp.isfinite(magnitude):
        return mp.mpf(1)
    quantum = mp.mpf(2) ** _grid_exponent(magnitude)
    position = magnitude / quantum
    fraction = position - mp.floor(position)
    return abs(fraction - mp.mpf("0.5"))


def nearest_float32(value: object) -> np.float32:
    """Correctly round an `mpmath` value onto the float32 grid.

    Exact by construction: the rounding is integer arithmetic on `mpmath`'s own
    significand and exponent, with round-half-to-even, gradual underflow into
    the subnormals and overflow to infinity all handled explicitly.
    """
    candidate = mp.mpf(value)
    if mp.isnan(candidate):
        return np.float32(np.nan)
    if candidate == mp.inf:
        return f32_from_bits(POS_INF_BITS)
    if candidate == mp.ninf:
        return f32_from_bits(NEG_INF_BITS)
    if candidate == 0:
        return np.float32(0.0)

    negative = candidate < 0
    magnitude = -candidate if negative else candidate

    binade = _binade(magnitude)
    if binade >= _OVERFLOW_BINADE:
        result = f32_from_bits(POS_INF_BITS)
        return np.float32(-result) if negative else result
    if binade < _UNDERFLOW_BINADE:
        return np.float32(-0.0) if negative else np.float32(0.0)

    distance = tie_distance(magnitude)
    if distance < TIE_SAFETY_MARGIN:
        raise AssertionError(
            "reference value is too close to a float32 midpoint to round safely "
            f"at {mp.dps} digits: {mp.nstr(magnitude, 30)}"
        )

    significand = int(magnitude.man)
    if significand < 0:
        significand = -significand
    exponent = int(magnitude.exp)
    grid = _grid_exponent(magnitude)

    shift = grid - exponent
    if shift <= 0:
        quantized = significand << (-shift)
        remainder = 0
        half = 1
    else:
        quantized = significand >> shift
        remainder = significand & ((1 << shift) - 1)
        half = 1 << (shift - 1)
    if remainder > half or (remainder == half and quantized & 1):
        quantized += 1
    if quantized >> 24:
        quantized >>= 1
        grid += 1

    scaled = np.ldexp(float(quantized), grid)
    result = f32_from_bits(POS_INF_BITS) if scaled > MAX_FLOAT32 else np.float32(scaled)
    return np.float32(-result) if negative else result


# ---------------------------------------------------------------------------
# The two reference values
# ---------------------------------------------------------------------------


def reference_exp_bits(margin: np.float32) -> int:
    """The correctly rounded float32 exponential of `margin`, as bit pattern.

    Non-finite inputs follow the specification rather than `mpmath`: `NaN`
    propagates, `+inf` gives `+inf`, `-inf` gives `+0.0`.
    """
    if is_nan32(margin):
        return CANONICAL_NAN_BITS
    value = float(margin)
    if value == float("inf") or value >= DECIDED_MAGNITUDE:
        return POS_INF_BITS
    if value == float("-inf") or value <= -DECIDED_MAGNITUDE:
        return ZERO_BITS
    return bits32(nearest_float32(mp.exp(mp.mpf(value))))


def reference_sigmoid_bits(margin: np.float32) -> int:
    """The correctly rounded float32 logistic function of `margin`, unclamped.

    This is the *mathematical* value everywhere, including below the measured
    clamp floor. A consumer must apply the clamp predicate for those margins
    rather than this column -- see `sigmoid_predicate_indices`.
    """
    if is_nan32(margin):
        return CANONICAL_NAN_BITS
    value = float(margin)
    if value == float("inf") or value >= DECIDED_MAGNITUDE:
        return ONE_BITS
    if value == float("-inf") or value <= -DECIDED_MAGNITUDE:
        return ZERO_BITS
    exact = mp.mpf(1) / (mp.mpf(1) + mp.exp(-mp.mpf(value)))
    return bits32(nearest_float32(exact))


# ---------------------------------------------------------------------------
# Sample points
# ---------------------------------------------------------------------------


def _neighbours(value: float, radius: int = 3) -> list[np.float32]:
    """`value` and its `radius` nearest float32 neighbours on each side.

    Stepping off the end of the finite range is intended, not accidental: the
    neighbours of the largest finite float32 include infinity, and the table
    wants that point. The overflow flag numpy would raise for it is therefore
    silenced rather than surfaced.
    """
    with np.errstate(over="ignore"):
        out = [np.float32(value)]
        step = np.float32(value)
        for _ in range(radius):
            step = np.nextafter(step, np.float32(np.inf))
            out.append(step)
        step = np.float32(value)
        for _ in range(radius):
            step = np.nextafter(step, np.float32(-np.inf))
            out.append(step)
    return out


#: (anchor, label) pairs. Every boundary FORMAT.md section 5.6 names, plus the
#: two inputs the Python side recorded as its worst logistic cases. Each anchor
#: contributes itself and three float32 neighbours on each side, because a
#: boundary that is only sampled exactly on the boundary cannot show an
#: off-by-one-value error in where the boundary sits.
LABELLED_ANCHORS: tuple[tuple[float, str], ...] = (
    (0.0, "zero"),
    (-0.0, "negative zero"),
    (1.0, "one"),
    (-1.0, "minus one"),
    (PROBE_EXP_OVERFLOW_MARGIN, "first float32 whose exponential overflows to +inf"),
    (88.7228, "exponential overflow boundary"),
    (FIRST_UNDERFLOWING_EXP_MARGIN, "first float32 whose exponential underflows to zero"),
    (-104.0, "beyond the exponential's underflow boundary"),
    (SUBNORMAL_TRANSITION_MARGIN, "exponential result crosses from normal into subnormal"),
    (PROBE_FLOOR_MARGIN, "the measured logistic clamp floor"),
    (-88.7, "the logistic clamp floor as written in the probe report"),
    (88.7, "the undetectable upper clamp boundary"),
    (FIRST_SATURATING_SIGMOID_MARGIN, "first float32 whose logistic value is exactly 1"),
    (17.0, "logistic saturated at exactly 1"),
    (WORST_KNOWN_EXP_MARGIN, "argument-reduction interval edge; worst known exponential"),
    (0.34657359, "argument-reduction interval edge"),
    (-0.34657359, "argument-reduction interval edge, negative"),
    (0.6931472, "the logarithm of two"),
    (-0.6931472, "the negated logarithm of two"),
    (WORST_KNOWN_SIGMOID_MARGINS[0], "worst known logistic case, 2 ULP on the Python side"),
    (WORST_KNOWN_SIGMOID_MARGINS[1], "worst known logistic case, 2 ULP on the Python side"),
    (MIN_SUBNORMAL_FLOAT32, "smallest positive subnormal float32"),
    (-MIN_SUBNORMAL_FLOAT32, "smallest negative subnormal float32"),
    (MIN_NORMAL_FLOAT32, "smallest positive normal float32"),
    (MAX_FLOAT32, "largest finite float32"),
    (-MAX_FLOAT32, "most negative finite float32"),
    (175.0, "beyond the short-circuit bound on the factored power of two"),
    (-173.0, "below the short-circuit bound on the factored power of two"),
    (float("inf"), "positive infinity"),
    (float("-inf"), "negative infinity"),
    (float("nan"), "not a number"),
)


def _band_scan(low: float, high: float, budget: int) -> list[np.float32]:
    """Representable float32 values across `[low, high]`, thinned to `budget`.

    Walks bit patterns, so the step is one representable value rather than a
    decimal increment. Both endpoints must have the same sign; a band that
    straddles zero is not a contiguous run of bit patterns and none is used.
    """
    assert float(low) * float(high) > 0.0, (low, high)
    start, stop = bits32(low), bits32(high)
    first, last = min(start, stop), max(start, stop)
    span = last - first + 1
    stride = max(1, -(-span // max(budget, 1)))
    return [f32_from_bits(bits) for bits in range(first, last + 1, stride)]


#: Bands where the answer varies fastest, each capped so no one band can
#: consume the whole sample. The cap exists because the first version of the
#: Python sweep had none: two narrow bands ate the entire budget, the random
#: fill came out empty, and a million-point sweep never visited the region
#: holding the logistic's real worst case (D046).
BANDS: tuple[tuple[float, float], ...] = (
    (88.5, 88.9),
    (-104.5, -103.5),
    (-88.0, -87.0),
    (PROBE_FLOOR_MARGIN, -87.0),
    (16.0, 17.5),
    (-17.0, -16.5),
)


def _fill(count: int) -> list[np.float32]:
    """Deterministic pseudo-random margins: a broad range, then everything.

    Three quarters land in the window where the result is finite and nonzero
    and the arithmetic is doing real work. An eighth land near zero, where the
    reduction factors out a power of two of zero and the polynomial carries the
    whole result. The last eighth are drawn from raw bit patterns, uniformly
    over every float32 there is -- subnormal margins, enormous margins, and the
    occasional NaN, none of which a range-bounded draw would ever produce.
    """
    if count <= 0:
        return []
    rng = np.random.default_rng(SEED)
    dense = count - 2 * (count // 8)
    near_zero = count // 8
    anywhere = count // 8
    points = [np.float32(v) for v in rng.uniform(-110.0, 92.0, dense)]
    points.extend(np.float32(v) for v in rng.uniform(-1.0, 1.0, near_zero))
    patterns = rng.integers(0, 1 << 32, size=anywhere, dtype=np.uint64)
    points.extend(f32_from_bits(int(bits)) for bits in patterns)
    return points


def sample_points() -> tuple[list[np.float32], dict[int, str]]:
    """The table's margins, in a fixed order, with labels for the anchors.

    Duplicates are kept rather than collapsed. A repeated margin costs a few
    bytes and removing them would make the index of every labelled anchor
    depend on whether some band scan happened to land on it.
    """
    points: list[np.float32] = []
    labels: dict[int, str] = {}

    for anchor, label in LABELLED_ANCHORS:
        for offset, point in enumerate(_neighbours(anchor)):
            if offset == 0:
                labels[len(points)] = label
            points.append(point)

    budget = max(1, POINT_COUNT // 8)
    for low, high in BANDS:
        points.extend(_band_scan(low, high, budget))
    points.extend(_fill(POINT_COUNT - len(points)))
    return points, labels


# ---------------------------------------------------------------------------
# Self-checks: the reference machinery is validated before it is trusted
# ---------------------------------------------------------------------------


def _self_check() -> None:
    """Verify the rounding oracle and the transcribed clamp constants.

    A reference that is subtly wrong moves every ULP figure downstream, and it
    would do so silently. So the pieces are checked against values that can be
    stated independently of this module.
    """
    # The clamp constants agree with each other's decimal renderings.
    assert bits32(PROBE_FLOOR_MARGIN) == PROBE_FLOOR_MARGIN_BITS, "floor margin transcription"
    assert bits32(PROBE_FLOOR_OUTPUT) == PROBE_FLOOR_OUTPUT_BITS, "floor output transcription"

    # The rounding oracle on values whose float32 image is known exactly.
    assert bits32(nearest_float32(mp.mpf(1))) == ONE_BITS
    assert bits32(nearest_float32(mp.mpf(0))) == ZERO_BITS
    assert bits32(nearest_float32(mp.inf)) == POS_INF_BITS
    assert bits32(nearest_float32(mp.ninf)) == NEG_INF_BITS
    assert bits32(nearest_float32(mp.mpf(2) ** -149)) == 0x00000001
    assert bits32(nearest_float32(mp.mpf(2) ** -151)) == ZERO_BITS
    assert bits32(nearest_float32(mp.mpf(2) ** -148)) == 0x00000002
    # An exact float32 midpoint is refused rather than rounded, because at that
    # distance `mpmath`'s own error decides the direction. Nothing the two
    # references produce lands there -- but the guard has to be shown to work,
    # since a guard that never fires and a guard that cannot fire look alike.
    try:
        nearest_float32(mp.mpf(2) ** -150)
    except AssertionError:
        pass
    else:  # pragma: no cover - the guard is asserted to fire
        raise AssertionError("the float32 midpoint guard did not fire on 2**-150")

    # The two references at inputs whose answer is fixed by definition.
    assert reference_exp_bits(np.float32(0.0)) == ONE_BITS
    assert reference_exp_bits(np.float32(-np.inf)) == ZERO_BITS
    assert reference_exp_bits(np.float32(np.inf)) == POS_INF_BITS
    assert reference_sigmoid_bits(np.float32(0.0)) == 0x3F000000
    assert reference_sigmoid_bits(np.float32(np.inf)) == ONE_BITS

    # Cross-validation of the transcribed floor: the *unclamped* logistic
    # function at exactly the floor margin must land within a couple of
    # representable values of the output XGBoost was measured to return there.
    # Exact agreement is not expected and is not required -- XGBoost's own
    # exponential is not correctly rounded -- but a typo in either constant
    # would miss by orders of magnitude, which is what this catches.
    at_floor = reference_sigmoid_bits(np.float32(PROBE_FLOOR_MARGIN))
    assert abs(at_floor - PROBE_FLOOR_OUTPUT_BITS) <= 2, (
        f"the logistic function at the measured floor margin rounds to {hex32(at_floor)}, "
        f"which is not adjacent to the measured floor output {hex32(PROBE_FLOOR_OUTPUT_BITS)}"
    )

    # Every margin strictly below the floor must have a smaller *unclamped*
    # reference than the floor output, or "the clamp is a floor" would not be
    # the right description of what is happening there.
    for margin in (-88.71, -110.90191650390625, -204.50521850585938, -748.246337890625):
        below = reference_sigmoid_bits(np.float32(margin))
        assert below < PROBE_FLOOR_OUTPUT_BITS, (margin, hex32(below))


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_reference() -> dict[str, Any]:
    """Build the reference table. Pure: the same input state, the same bytes."""
    _self_check()

    points, labels = sample_points()

    margin_bits: list[str] = []
    exp_bits: list[str] = []
    sigmoid_bits: list[str] = []
    predicate_indices: list[int] = []
    nan_indices: list[int] = []

    floor_margin = np.float32(PROBE_FLOOR_MARGIN)
    for index, margin in enumerate(points):
        margin_bits.append(hex32(bits32(margin)))
        exp_bits.append(hex32(reference_exp_bits(margin)))
        sigmoid_bits.append(hex32(reference_sigmoid_bits(margin)))
        if is_nan32(margin):
            nan_indices.append(index)
        elif margin < floor_margin:
            predicate_indices.append(index)

    return {
        "meta": {
            "name": "float32_transform_reference",
            "description": (
                "Correctly rounded float32 reference values for the bundled "
                "margin-to-output transform, computed with mpmath at 50 digits. "
                "This table is the JavaScript side's independent oracle: it shares "
                "no code with either implementation, so agreement with it is "
                "evidence of correctness rather than evidence that a port was "
                "faithful."
            ),
            "mpmath_decimal_places": mp.dps,
            "numpy_version": np.__version__,
            "seed": SEED,
            "point_count": len(points),
            "bit_pattern_encoding": (
                "Every value is the uint32 hex bit pattern of the float32 it names, "
                "for the reasons D044 gives: JSON has no representation for +inf, "
                "-inf, NaN or -0.0, all of which occur here, and a decimal ground "
                "truth invites == comparison under which -0.0 == 0.0 is true for two "
                "values that are not the same."
            ),
            "sigmoid_bits_are_unclamped": (
                "sigmoid_bits is the unclamped mathematical logistic function "
                "everywhere, including below the measured clamp floor. At the "
                "indices listed in sigmoid_predicate_indices the specified output is "
                "instead exactly sigmoid_clamp.floor_output_bits, and the correct "
                "check there is that predicate rather than a ULP distance: below the "
                "floor the implementation deliberately returns the floor while an "
                "unclamped reference keeps falling, so a distance reports the clamp "
                "as if it were error."
            ),
            "nan_indices_note": (
                "At the indices listed in nan_margin_indices the margin is NaN and "
                "both reference columns carry a canonical quiet NaN pattern. NaN bit "
                "patterns are not canonical across engines, so the correct check "
                "there is that the result is a NaN, not that it has these bits."
            ),
        },
        "sigmoid_clamp": {
            "floor_margin_bits": hex32(PROBE_FLOOR_MARGIN_BITS),
            "floor_margin_decimal": PROBE_FLOOR_MARGIN,
            "floor_output_bits": hex32(PROBE_FLOOR_OUTPUT_BITS),
            "floor_output_decimal": PROBE_FLOOR_OUTPUT,
            "source": (
                "probes/output_transform.md section 3: the sole float32 in [-90, -88] "
                "reproducing XGBoost's observed output, found by exhaustive scan of "
                "all 262145 candidates; the output was one bit pattern on 2056 "
                "measured below-floor rows and is never 0.0. Transcribed here "
                "independently of both implementations. Clamp constants are XGBoost "
                "internals and are version-sensitive (D018, D032)."
            ),
        },
        "labels": {str(index): label for index, label in sorted(labels.items())},
        "margin_bits": margin_bits,
        "exp_bits": exp_bits,
        "sigmoid_bits": sigmoid_bits,
        "sigmoid_predicate_indices": predicate_indices,
        "nan_margin_indices": nan_indices,
    }


def main() -> None:
    table = build_reference()
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = REFERENCE_DIR / "float32_transform_reference.json"
    path.write_text(
        json.dumps(table, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {table['meta']['point_count']} reference points to {path}\n"
        f"  clamp-predicate points: {len(table['sigmoid_predicate_indices'])}\n"
        f"  NaN margins: {len(table['nan_margin_indices'])}\n"
        f"  labelled anchors: {len(table['labels'])}"
    )


if __name__ == "__main__":
    main()
