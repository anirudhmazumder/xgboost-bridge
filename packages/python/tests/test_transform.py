"""Validation of the bundled margin-to-output transform.

The oracle here is **`mpmath` at 50 digits**, per function, independently. It
cannot share a defect with the thing it checks because it shares no code with
it: `mpmath` computes in arbitrary precision with its own algorithms, and the
comparison rounds that result onto the float32 grid with exact integer
arithmetic rather than by narrowing twice through float64 (D034,
FORMAT.md section 5.6).

Three oracles that would have been decorative, and are therefore not used:

* **The other language.** Two identical implementations agreeing proves only
  that the code was written twice. Bit-identical wrong is still wrong, and it
  is invisible to a parity harness precisely because both sides agree.
* **The platform's own exponential.** That is the thing being replaced. A
  comparison against it measures agreement with the wrong reference, and it
  is not correctly rounded either. It appears in `main()` for context only,
  never in an assertion.
* **A re-derivation of this module's own recipe.** An error in the recipe
  could not make such a check fire.

The clamp values are a separate case: below the sigmoid floor the specified
output is not the mathematical sigmoid, it is a value XGBoost was **measured**
to return (`probes/output_transform.md` section 3, 2056 below-floor rows over
three configurations, all one bit pattern). Those literals are written here
independently of the module and the module is asserted against them, so the
oracle for the clamp is XGBoost's observed output rather than our constant.

Reported as **max** ULP, never mean: a mean hides the one input that matters.

The always-on tests use a few thousand deterministic points including every
boundary the format names. The full sweep of at least 1e6 points per function
uses the same generators with a larger count and is run directly:

    uv run python packages/python/tests/test_transform.py

Nothing here is skipped or marked expected-to-fail.
"""

from __future__ import annotations

import ast
import inspect
import struct
import sys
from pathlib import Path

import numpy as np
from mpmath import mp

from xgboost_bridge import transform
from xgboost_bridge.transform import (
    OUTPUT_FUNCTIONS,
    SIGMOID_FLOOR_OUTPUT,
    SIGMOID_MARGIN_FLOOR,
    exp_f32,
    identity_f32,
    sigmoid_f32,
)

mp.dps = 50

# ---------------------------------------------------------------------------
# Literals transcribed from probes/output_transform.md, deliberately
# independent of the module under test.
# ---------------------------------------------------------------------------

#: Section 3: the sole float32 in [-90, -88] reproducing XGBoost's observed
#: floor output, found by exhaustive scan of all 262145 candidates.
PROBE_FLOOR_MARGIN = -88.69999694824219
PROBE_FLOOR_MARGIN_BITS = 3266405990

#: Section 3: the single value XGBoost's `predict()` returns for every margin
#: below the floor, on 1959/1959, 97/97 and 1/1 measured rows.
PROBE_FLOOR_OUTPUT = 3.006635794144578e-39
PROBE_FLOOR_OUTPUT_BITS = 2145607

#: Section 3: measured margins that fall below the floor in a real fit.
PROBE_BELOW_FLOOR_MARGINS = (-110.90191650390625, -204.50521850585938, -748.246337890625)

#: Section 4: `survival:cox` has no floor -- at this margin `predict()` is the
#: exponential's own subnormal result, not the floor value.
PROBE_COX_TAIL_MARGIN = -94.62601470947266
PROBE_COX_TAIL_OUTPUT = 8.025236305188227e-42

#: Section 4: measured Cox margins above which `predict()` is `+inf`.
PROBE_COX_INF_MARGINS = (112.06353759765625, 120.70205688476562, 134.943, 212.83)

#: The margin at which this implementation's exponential is furthest from
#: correctly rounded, at the edge of the argument-reduction interval.
WORST_KNOWN_EXP_MARGIN = 0.34657353162765503

#: Margins found by search to produce this implementation's largest logistic
#: error, carried explicitly so the reported maximum is reproducible and so
#: the always-on sample reaches it rather than relying on a draw landing
#: there. The error is the exponential's own worst case compounded with the
#: rounding of the reciprocal, and it appears at a rate near 3e-04 in a
#: uniform draw over the region where the result is small but normal.
WORST_KNOWN_SIGMOID_MARGINS = (-16.635820388793945, -16.80551528930664)

MAX_FLOAT32 = float(np.finfo(np.float32).max)
MIN_NORMAL_FLOAT32 = float(np.finfo(np.float32).smallest_normal)
MIN_SUBNORMAL_FLOAT32 = float(np.finfo(np.float32).smallest_subnormal)

# Max ULP the implementation is held to. Measured over the full 1e6-point
# sweeps; see the module docstring for how to reproduce them. A regression
# past these is a defect to diagnose, not a number to raise.
EXP_MAX_ULP = 1
SIGMOID_MAX_ULP = 2


# ---------------------------------------------------------------------------
# Bit-level helpers
# ---------------------------------------------------------------------------


def bits32(value: object) -> int:
    """The IEEE-754 single-precision encoding of `value`, as an integer."""
    with np.errstate(over="ignore", under="ignore"):
        return int(np.float32(value).view(np.uint32))


def bits64(value: float) -> int:
    """The IEEE-754 double-precision encoding of `value`, as an integer."""
    return int(struct.unpack("<Q", struct.pack("<d", float(value)))[0])


def f32_from_bits(bits: int) -> np.float32:
    return np.uint32(bits).view(np.float32)


def ordinal(value: object) -> int:
    """Map a finite-or-infinite float32 to a monotonically ordered integer.

    Adjacent representable values are adjacent integers, so the difference of
    two ordinals is the distance in ULP. Signed zero maps to distinct
    ordinals only in sign; `-0.0` and `0.0` are one apart in neither
    direction, they are the same magnitude with opposite sign, which this
    mapping renders as `0` and `-0` -- equal. Zero-sign agreement is
    therefore asserted separately with bit patterns, never through ULP.
    """
    bits = bits32(value)
    if bits & 0x80000000:
        return -(bits & 0x7FFFFFFF)
    return bits


def is_nan32(value: object) -> bool:
    bits = bits32(value)
    return (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0


# ---------------------------------------------------------------------------
# The reference: exact rounding of an mpmath value onto the float32 grid
# ---------------------------------------------------------------------------

# A tie decided the wrong way would be an invisible off-by-one-ULP in the
# reference itself. `mpmath` at 50 digits carries 169 bits, so its own error
# is around 2**-169 relative -- but "around" is not a proof, so any value
# closer to a float32 midpoint than this many quantum units raises instead of
# being rounded. It has never fired; if it does, the reference precision is
# the thing to raise, not this bound.
TIE_SAFETY_MARGIN = mp.mpf(2) ** -60


#: A magnitude at or above two to this power exceeds every float32 midpoint
#: below infinity, so it rounds to infinity. Strictly below two to the
#: negative of this is under half the smallest subnormal and rounds to zero.
#: Both are decided before any scaling, because the reference values reach
#: exponents in the hundreds of millions -- the exponential of the largest
#: float32 margin, for instance.
_OVERFLOW_BINADE = 128
_UNDERFLOW_BINADE = -150


def _binade(value: mp.mpf) -> int:
    """The base-two exponent of `value`, that is, floor of its logarithm."""
    significand = int(value.man)
    if significand < 0:
        significand = -significand
    return significand.bit_length() - 1 + int(value.exp)


def _grid_exponent(value: mp.mpf) -> int:
    """The base-two exponent of the float32 quantum at `value`'s magnitude."""
    return max(_binade(value) - 23, -149)


def nearest_float32(value: object) -> np.float32:
    """Correctly round an `mpmath` value onto the float32 grid.

    Exact by construction: the rounding is integer arithmetic on `mpmath`'s
    own significand and exponent, with round-half-to-even, gradual underflow
    into the subnormals and overflow to infinity all handled explicitly.

    Deliberately not `np.float32(float(value))`. That narrows twice, and
    double rounding through float64 is only provably harmless for the results
    of the four basic operations -- not for an arbitrary real, which is
    exactly what this function receives.
    """
    candidate = mp.mpf(value)
    if mp.isnan(candidate):
        return np.float32(np.nan)
    if candidate == mp.inf:
        return f32_from_bits(0x7F800000)
    if candidate == mp.ninf:
        return f32_from_bits(0xFF800000)
    if candidate == 0:
        return np.float32(0.0)

    negative = candidate < 0
    magnitude = -candidate if negative else candidate

    binade = _binade(magnitude)
    if binade >= _OVERFLOW_BINADE:
        result = f32_from_bits(0x7F800000)
        return np.float32(-result) if negative else result
    if binade < _UNDERFLOW_BINADE:
        return np.float32(-0.0) if negative else np.float32(0.0)

    distance = tie_distance(magnitude)
    if distance < TIE_SAFETY_MARGIN:
        raise AssertionError(
            "reference value is too close to a float32 midpoint to round "
            f"safely at {mp.dps} digits: {mp.nstr(magnitude, 30)}"
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
    result = f32_from_bits(0x7F800000) if scaled > MAX_FLOAT32 else np.float32(scaled)
    return np.float32(-result) if negative else result


def tie_distance(value: mp.mpf) -> mp.mpf:
    """Distance from the nearest float32 midpoint, in quantum units."""
    magnitude = abs(mp.mpf(value))
    if magnitude == 0 or not mp.isfinite(magnitude):
        return mp.mpf(1)
    quantum = mp.mpf(2) ** _grid_exponent(magnitude)
    position = magnitude / quantum
    fraction = position - mp.floor(position)
    return abs(fraction - mp.mpf("0.5"))


def reference_exp(margin: np.float32) -> np.float32:
    """The correctly rounded float32 exponential of `margin`.

    Non-finite inputs follow the specification rather than `mpmath`: NaN
    propagates, `+inf` gives `+inf`, `-inf` gives `+0.0`.
    """
    if is_nan32(margin):
        return np.float32(np.nan)
    return nearest_float32(mp.exp(mp.mpf(float(margin))))


def reference_sigmoid(margin: np.float32) -> np.float32:
    """The correctly rounded float32 logistic function of `margin`.

    Below the measured floor the reference is XGBoost's observed output
    (`PROBE_FLOOR_OUTPUT`), not the mathematical value -- reproducing that
    saturation is the specification (D032). Everywhere else it is `mpmath`.
    """
    if is_nan32(margin):
        return np.float32(np.nan)
    if float(margin) < PROBE_FLOOR_MARGIN:
        return np.float32(PROBE_FLOOR_OUTPUT)
    exact = mp.mpf(1) / (mp.mpf(1) + mp.exp(-mp.mpf(float(margin))))
    return nearest_float32(exact)


def ulp_error(got: object, want: object) -> int:
    """Distance in representable float32 values between `got` and `want`.

    NaN and infinity are not ULP questions: a NaN reference requires a NaN
    result, and an infinite reference requires the identical bit pattern. A
    naive ordinal subtraction would score infinity against the largest finite
    value as a single ULP and hide a saturation defect completely.
    """
    if is_nan32(want) or is_nan32(got):
        return 0 if (is_nan32(want) and is_nan32(got)) else 1 << 30
    got_bits, want_bits = bits32(got), bits32(want)
    if (got_bits & 0x7FFFFFFF) == 0x7F800000 or (want_bits & 0x7FFFFFFF) == 0x7F800000:
        return 0 if got_bits == want_bits else 1 << 30
    return abs(ordinal(got) - ordinal(want))


# ---------------------------------------------------------------------------
# Sample point generators, shared by the fast tests and the full sweep
# ---------------------------------------------------------------------------

SWEEP_SEED = 20260804


def _neighbours(value: float, radius: int = 3) -> list[np.float32]:
    """`value` and its `radius` nearest float32 neighbours on each side."""
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


#: Every boundary the format requires to be exercised (FORMAT.md section 5.6),
#: for the exponential: the overflow and underflow boundaries, the subnormal
#: transition, the non-finite values, both signed zeros, and the floor the
#: sigmoid shares with it.
def exp_boundary_points() -> list[np.float32]:
    points: list[np.float32] = [
        np.float32(0.0),
        np.float32(-0.0),
        np.float32(np.inf),
        np.float32(-np.inf),
        np.float32(np.nan),
        np.float32(1.0),
        np.float32(-1.0),
        np.float32(MIN_SUBNORMAL_FLOAT32),
        np.float32(-MIN_SUBNORMAL_FLOAT32),
        np.float32(MIN_NORMAL_FLOAT32),
        np.float32(MAX_FLOAT32),
        np.float32(-MAX_FLOAT32),
    ]
    for anchor in (
        88.7228,  # overflow to +inf
        88.72283935546875,  # first float32 whose exponential is +inf
        -87.336544,  # result crosses from normal into subnormal
        -103.97208404541016,  # first float32 whose exponential underflows to 0
        -104.0,
        PROBE_FLOOR_MARGIN,
        PROBE_COX_TAIL_MARGIN,
        0.34657359,  # the reduction interval edge
        -0.34657359,
        0.6931472,
        -0.6931472,
        175.0,  # beyond the short-circuit bound on the power of two
        -173.0,
    ):
        points.extend(_neighbours(anchor))
    points.extend(np.float32(m) for m in PROBE_COX_INF_MARGINS)
    return points


def sigmoid_boundary_points() -> list[np.float32]:
    """As above, plus the floor and the margins that saturate at exactly 1."""
    points: list[np.float32] = [
        np.float32(0.0),
        np.float32(-0.0),
        np.float32(np.inf),
        np.float32(-np.inf),
        np.float32(np.nan),
        np.float32(1.0),
        np.float32(-1.0),
        np.float32(MAX_FLOAT32),
        np.float32(-MAX_FLOAT32),
    ]
    for anchor in (
        PROBE_FLOOR_MARGIN,
        -88.0,
        -87.336544,
        -87.0,
        16.63553237915039,  # first float32 whose sigmoid is exactly 1.0
        17.0,
        88.7228,
        0.34657359,
        -0.34657359,
        *WORST_KNOWN_SIGMOID_MARGINS,
    ):
        points.extend(_neighbours(anchor))
    points.extend(np.float32(m) for m in PROBE_BELOW_FLOOR_MARGINS)
    points.append(np.float32(PROBE_COX_TAIL_MARGIN))
    return points


def _band_scan(low: float, high: float, stride: int) -> list[np.float32]:
    """Every `stride`-th representable float32 in `[low, high]`.

    Walks bit patterns, so the step is one representable value rather than a
    decimal increment. Both endpoints must have the same sign; a band that
    straddles zero is not a contiguous run of bit patterns and none is used.
    """
    assert float(low) * float(high) > 0.0, (low, high)
    start, stop = bits32(low), bits32(high)
    return [
        f32_from_bits(bits)
        for bits in range(min(start, stop), max(start, stop) + 1, stride)
    ]


def _band_within_budget(low: float, high: float, budget: int) -> list[np.float32]:
    """A band scan thinned so it contributes no more than `budget` points.

    The budget exists because the first version of these generators had none:
    two narrow bands consumed the entire sample and the random fill came out
    empty, so a sweep of a million points never visited the region where the
    logistic's actual worst case lives and reported a maximum one ULP too
    low. A sample that does not reach the inputs where the error is largest
    cannot measure it, and its silence reads exactly like good news.
    """
    span = abs(bits32(high) - bits32(low)) + 1
    stride = max(1, -(-span // max(budget, 1)))
    return _band_scan(low, high, stride)


def exp_sample_points(count: int) -> list[np.float32]:
    """Deterministic sample of at least `count` margins for the exponential.

    Boundary points come first and are never dropped, so a smaller `count`
    thins the bands and the random fill rather than narrowing coverage.
    """
    points = exp_boundary_points()
    budget = max(1, count // 8)
    points.extend(_band_within_budget(88.5, 88.9, budget))
    points.extend(_band_within_budget(-104.5, -103.5, budget))
    points.extend(_band_within_budget(-88.0, -87.0, budget))
    points.extend(_fill(count - len(points), (-110.0, 90.0)))
    return points


def sigmoid_sample_points(count: int) -> list[np.float32]:
    """Deterministic sample of at least `count` margins for the logistic."""
    points = sigmoid_boundary_points()
    budget = max(1, count // 8)
    points.extend(_band_within_budget(PROBE_FLOOR_MARGIN, -87.0, budget))
    points.extend(_band_within_budget(16.0, 17.5, budget))
    points.extend(_fill(count - len(points), (-92.0, 92.0)))
    return points


def _fill(count: int, window: tuple[float, float]) -> list[np.float32]:
    """Deterministic pseudo-random margins: a dense window, then everything.

    Three quarters land in `window`, where the result is finite and nonzero
    and the arithmetic is doing real work. An eighth land near zero, where
    the reduction picks a power of two of zero and the polynomial carries the
    whole result. The last eighth are drawn from raw bit patterns, uniformly
    over every float32 there is -- subnormal margins, huge margins, and the
    occasional NaN, none of which a range-bounded draw would ever produce.
    """
    if count <= 0:
        return []
    rng = np.random.default_rng(SWEEP_SEED)
    dense = count - 2 * (count // 8)
    near_zero = count // 8
    anywhere = count // 8
    points = [np.float32(v) for v in rng.uniform(window[0], window[1], dense)]
    points.extend(np.float32(v) for v in rng.uniform(-1.0, 1.0, near_zero))
    patterns = rng.integers(0, 1 << 32, size=anywhere, dtype=np.uint64)
    points.extend(f32_from_bits(int(bits)) for bits in patterns)
    return points


def measure(function, reference, points) -> dict:
    """Max ULP of `function` against `reference` over `points`.

    Returns the max, the input that produced it, and the full histogram. The
    max is the reported figure; the histogram is context.
    """
    histogram: dict[int, int] = {}
    worst = {"ulp": -1, "margin": None, "got": None, "want": None}
    for margin in points:
        got = function(margin)
        want = reference(margin)
        error = ulp_error(got, want)
        histogram[error] = histogram.get(error, 0) + 1
        if error > worst["ulp"]:
            worst = {
                "ulp": error,
                "margin": np.float32(margin),
                "got": np.float32(got),
                "want": np.float32(want),
            }
    return {"count": len(points), "histogram": histogram, "worst": worst}


# ---------------------------------------------------------------------------
# The oracle machinery checks itself first
# ---------------------------------------------------------------------------


def test_nearest_float32_agrees_with_a_single_narrowing_where_that_is_safe() -> None:
    """On values already exactly representable in float64, both must agree.

    A rounding oracle that is subtly wrong would move every ULP figure below
    by one and read as a defect in the implementation. Narrowing a float64
    that is itself exact is a single rounding, so `np.float32` is a valid
    cross-check here -- and only here.
    """
    rng = np.random.default_rng(11)
    values = [
        *rng.uniform(-10.0, 10.0, 500),
        *rng.uniform(-1e38, 1e38, 200),
        *(float(v) for v in rng.uniform(0.0, 1e-38, 200)),
        0.0,
        1.0,
        -1.0,
        MAX_FLOAT32,
        -MAX_FLOAT32,
        MIN_SUBNORMAL_FLOAT32,
        MIN_NORMAL_FLOAT32,
        3.5e38,
        -3.5e38,
        1e-46,
        7.0e-46,
    ]
    for value in values:
        assert bits32(nearest_float32(mp.mpf(value))) == bits32(value), repr(value)


def test_nearest_float32_refuses_an_exact_midpoint() -> None:
    """A tie is where a rounding oracle goes wrong silently, so it raises.

    The reference is a 169-bit approximation of an irrational value, and a
    genuine tie cannot occur there -- but "cannot" is a probability argument,
    so the guard is real code and is exercised here with real ties.
    """
    quantum = mp.mpf(2) ** -23
    below = mp.mpf(1) + quantum / 2  # exactly between 1.0 and its successor
    above = mp.mpf(1) + quantum + quantum / 2  # between successor and next
    assert tie_distance(below) == 0
    assert tie_distance(above) == 0
    # The guard fires on an exact tie rather than guessing; both directions
    # are then checked with the guard's threshold understood.
    for value in (below, above):
        try:
            nearest_float32(value)
        except AssertionError:
            continue
        raise AssertionError("the tie guard did not fire on an exact midpoint")


def test_the_reference_cannot_express_signed_zero_so_bits_are_used_instead() -> None:
    """A stated limitation of the oracle, pinned so it is not assumed away.

    `mpmath` has one zero, not two, so no ULP comparison against it can say
    anything about the sign of a zero result. Neither transform can return a
    negative zero -- both are strictly positive functions -- and the sign of
    the zeros they do return, plus identity's pass-through of `-0.0`, are
    asserted directly on bit patterns elsewhere in this file.
    """
    assert bits32(nearest_float32(mp.mpf(-0.0))) == 0x00000000
    assert bits32(-0.0) == 0x80000000
    assert bits32(exp_f32(np.float32(-1000.0))) == 0x00000000
    assert bits32(identity_f32(np.float32(-0.0))) == 0x80000000


def test_nearest_float32_handles_non_finite_references() -> None:
    assert bits32(nearest_float32(mp.inf)) == 0x7F800000
    assert bits32(nearest_float32(mp.ninf)) == 0xFF800000
    assert is_nan32(nearest_float32(mp.nan))


def test_ulp_error_refuses_to_score_infinity_as_one_ulp() -> None:
    """Infinity against the largest finite value is a saturation defect."""
    assert ulp_error(np.float32(np.inf), np.float32(MAX_FLOAT32)) > 1
    assert ulp_error(np.float32(MAX_FLOAT32), np.float32(np.inf)) > 1
    assert ulp_error(np.float32(np.inf), np.float32(np.inf)) == 0
    assert ulp_error(np.float32(np.nan), np.float32(np.nan)) == 0
    assert ulp_error(np.float32(np.nan), np.float32(1.0)) > 1
    assert ulp_error(np.float32(1.0), np.float32(np.nan)) > 1
    successor = np.nextafter(np.float32(1.0), np.float32(np.inf))
    assert ulp_error(successor, np.float32(1.0)) == 1


def test_sample_generators_are_deterministic_and_cover_the_boundaries() -> None:
    first = [bits32(v) for v in exp_sample_points(3000)]
    second = [bits32(v) for v in exp_sample_points(3000)]
    assert first == second
    assert len(first) >= 3000
    for required in (
        bits32(0.0),
        bits32(-0.0),
        bits32(np.inf),
        bits32(-np.inf),
        bits32(PROBE_FLOOR_MARGIN),
        bits32(88.72283935546875),
        bits32(-103.97208404541016),
    ):
        assert required in first
    assert any(is_nan32(v) for v in exp_sample_points(3000))

    sigmoid_first = [bits32(v) for v in sigmoid_sample_points(3000)]
    assert sigmoid_first == [bits32(v) for v in sigmoid_sample_points(3000)]
    assert len(sigmoid_first) >= 3000
    assert bits32(16.63553237915039) in sigmoid_first
    assert bits32(PROBE_FLOOR_MARGIN) in sigmoid_first
    for margin in WORST_KNOWN_SIGMOID_MARGINS:
        assert bits32(margin) in sigmoid_first


def test_no_band_scan_may_consume_the_whole_sample() -> None:
    """The flaw that made the first million-point sweep report a low maximum.

    Two narrow bands used the entire budget, the random fill came out empty,
    and the region holding the logistic's real worst case was never visited.
    So each band is capped and the fill is required to survive. The property
    is proportional to the requested count, so checking it at three sizes
    checks it at every size.
    """
    for count in (3000, 40_000, 200_000):
        for points, bands in (
            (
                sigmoid_sample_points(count),
                ((PROBE_FLOOR_MARGIN, -87.0), (16.0, 17.5)),
            ),
            (
                exp_sample_points(count),
                ((88.5, 88.9), (-104.5, -103.5), (-88.0, -87.0)),
            ),
        ):
            values = np.array([float(value) for value in points])
            in_any_band = np.zeros(len(values), dtype=bool)
            for low, high in bands:
                lo, hi = sorted((float(low), float(high)))
                inside = (values >= lo) & (values <= hi)
                assert int(inside.sum()) <= count // 4, (count, low, high)
                in_any_band |= inside
            outside = len(values) - int(in_any_band.sum())
            assert outside >= count // 4, (count, outside)


# ---------------------------------------------------------------------------
# Constants: every bit pattern asserted, so the port can assert the same ones
# ---------------------------------------------------------------------------

#: Name to (float32 encoding, float64 encoding). The float32 integer is what
#: the value IS; the float64 integer is how the JavaScript port necessarily
#: holds it, since a JavaScript number is a double. Both are pinned because a
#: constant transcribed as a decimal string can be re-rounded by a parser and
#: land on a neighbouring float32 without anything reading as wrong.
EXPECTED_CONSTANT_BITS = {
    "_INV_LN2": (0x3FB8AA3B, 0x3FF7154760000000),
    "_LN2_HI": (0x3F317200, 0x3FE62E4000000000),
    "_LN2_LO": (0x35BFBE8E, 0x3EB7F7D1C0000000),
    "_ROUND_MAGIC": (0x4B400000, 0x4168000000000000),
    "_C2": (0x3F000000, 0x3FE0000000000000),
    "_C3": (0x3E2AAAAB, 0x3FC5555560000000),
    "_C4": (0x3D2AAAAB, 0x3FA5555560000000),
    "_C5": (0x3C088889, 0x3F81111120000000),
    "_C6": (0x3AB60B61, 0x3F56C16C20000000),
    "_C7": (0x39500D01, 0x3F2A01A020000000),
    "_ZERO": (0x00000000, 0x0000000000000000),
    "_ONE": (0x3F800000, 0x3FF0000000000000),
    "_POS_INF": (0x7F800000, 0x7FF0000000000000),
    "_NEG_INF": (0xFF800000, 0xFFF0000000000000),
    "SIGMOID_MARGIN_FLOOR": (0xC2B16666, 0xC0562CCCC0000000),
    "SIGMOID_FLOOR_OUTPUT": (0x0020BD47, 0x37F05EA380000000),
}


def test_every_constant_has_the_intended_bit_pattern() -> None:
    for name, (single, double) in EXPECTED_CONSTANT_BITS.items():
        value = getattr(transform, name)
        assert isinstance(value, np.float32), name
        assert bits32(value) == single, f"{name}: float32 bits"
        assert bits64(float(value)) == double, f"{name}: float64 bits"


def test_no_constant_is_left_unpinned() -> None:
    """A constant added without a pinned bit pattern is an unchecked constant."""
    declared = {
        name
        for name, value in vars(transform).items()
        if isinstance(value, np.float32)
    }
    assert declared == set(EXPECTED_CONSTANT_BITS)


def test_constants_equal_their_mathematical_definitions() -> None:
    """Each constant against `mpmath`, so the bit table above is not the only check.

    The table pins transcription -- that the integers in the source are the
    integers intended, which is what the JavaScript port has to match. This
    pins meaning: that each of those integers is the correctly rounded float32
    of the value it claims to be. A table alone would happily pin a typo.
    """
    assert bits32(transform._INV_LN2) == bits32(nearest_float32(1 / mp.log(2)))
    assert bits32(transform._ROUND_MAGIC) == bits32(mp.mpf(3) * mp.mpf(2) ** 22)
    for order in range(2, 8):
        expected = nearest_float32(mp.mpf(1) / mp.factorial(order))
        assert bits32(getattr(transform, f"_C{order}")) == bits32(expected), order
    assert bits32(transform._ZERO) == 0x00000000
    assert bits32(transform._ONE) == bits32(mp.mpf(1))
    assert float(transform._POS_INF) == float(np.inf)
    assert float(transform._NEG_INF) == float(-np.inf)


def test_the_reduction_constants_are_exact_by_construction() -> None:
    """The two properties that make the argument reduction accurate.

    Both are structural, and the oracle for the second is `mpmath`, not our
    own logarithm. If either fails, the reduction loses precision at large
    margins while remaining perfectly plausible at small ones.
    """
    # The low 8 significand bits of the high part are zero, so its product
    # with a power of at most 254 in magnitude fits in float32 exactly.
    assert bits32(transform._LN2_HI) & 0xFF == 0

    # And the pair reconstructs the logarithm of two far more accurately than
    # a single float32 does.
    exact = mp.log(2)
    pair = mp.mpf(float(transform._LN2_HI)) + mp.mpf(float(transform._LN2_LO))
    single = mp.mpf(float(np.float32(0.6931471805599453)))
    pair_error = abs(pair - exact)
    single_error = abs(single - exact)
    assert pair_error / exact < mp.mpf(2) ** -40
    assert single_error / exact > mp.mpf(2) ** -30
    # The split is worth having by four orders of magnitude, which is what
    # keeps the reduction accurate once the margin is large.
    assert pair_error * (mp.mpf(2) ** 12) < single_error

    # The exactness claim itself, checked over every power the code can use.
    for power in range(-254, 255):
        product = np.float32(np.float32(power) * transform._LN2_HI)
        assert mp.mpf(float(product)) == power * mp.mpf(float(transform._LN2_HI))


def test_round_magic_makes_the_rounding_step_exact() -> None:
    """The add-and-subtract trick must return an exact integer float32."""
    magic = transform._ROUND_MAGIC
    for value in (-254.0, -250.5, -1.5, -0.5, 0.0, 0.5, 1.5, 100.25, 254.0, 253.75):
        shifted = np.float32(np.float32(value) + magic)
        recovered = np.float32(shifted - magic)
        assert float(recovered) == float(int(recovered))
        assert abs(float(recovered) - value) <= 0.5


# ---------------------------------------------------------------------------
# Structural constraints, executable rather than trusted
# ---------------------------------------------------------------------------

FORBIDDEN_SOURCE_TOKENS = (
    "math.exp",
    "np.exp",
    "numpy.exp",
    "math.pow",
    "np.power",
    "np.exp2",
    "np.expm1",
    "np.ldexp",
    "math.ldexp",
    "Math.exp",
    "import math",
    "**",
)

ALLOWED_UNWRAPPED_TARGETS = {"low_exponent", "high_exponent"}


def _transform_source() -> str:
    path = Path(inspect.getfile(transform))
    return path.read_text(encoding="utf-8")


def test_transform_calls_no_platform_transcendental() -> None:
    """D030: no platform exponential on the prediction path, in any spelling.

    A textual check, because the point is that the token does not appear at
    all -- not even in a comment that a later reader might uncomment.
    """
    source = _transform_source()
    findings = []
    for lineno, line in enumerate(source.splitlines(), start=1):
        for token in FORBIDDEN_SOURCE_TOKENS:
            if token in line:
                findings.append(f"{lineno}: {token!r} in {line.strip()!r}")
    assert not findings, "forbidden tokens in transform.py:\n" + "\n".join(findings)


def test_transform_uses_only_the_four_correctly_rounded_operations() -> None:
    """No exponentiation operator anywhere in the module, at the AST level.

    The textual check above can be defeated by whitespace; this one cannot.
    """
    tree = ast.parse(_transform_source())
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp):
            assert not isinstance(node.op, ast.Pow), ast.dump(node)
        if isinstance(node, ast.AugAssign):
            assert not isinstance(node.op, ast.Pow), ast.dump(node)


def test_every_arithmetic_step_is_a_narrowed_named_intermediate() -> None:
    """FORMAT.md section 5.5, checked structurally rather than by review.

    Two requirements at once: at most one arithmetic operation per statement,
    so no runtime can contract a fused expression into a fused multiply-add;
    and every arithmetic operation wrapped in a narrowing to float32, so
    float32 semantics is a property of the code rather than of a habit.

    A float32 predictor that drifted into float64 intermediates would be
    *more* accurate against `mpmath`, so no ULP measurement can catch that
    drift. Only this check can.
    """
    tree = ast.parse(_transform_source())
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    for name in ("identity_f32", "exp_f32", "sigmoid_f32"):
        assert name in functions, name
        for statement in ast.walk(functions[name]):
            if not isinstance(statement, (ast.Assign, ast.Return)):
                continue
            operations = [
                node for node in ast.walk(statement) if isinstance(node, ast.BinOp)
            ]
            if not operations:
                continue
            targets = {
                target.id
                for target in getattr(statement, "targets", [])
                if isinstance(target, ast.Name)
            }
            if targets & ALLOWED_UNWRAPPED_TARGETS:
                continue  # integer bookkeeping, not float arithmetic
            assert len(operations) == 1, (
                f"{name}: fused expression, {len(operations)} operations in one "
                f"statement: {ast.unparse(statement)}"
            )
            wrapped = [
                node
                for node in ast.walk(statement)
                if isinstance(node, ast.Call)
                and ast.unparse(node.func) == "np.float32"
                and len(node.args) == 1
                and node.args[0] is operations[0]
            ]
            assert wrapped, (
                f"{name}: arithmetic not narrowed to float32: "
                f"{ast.unparse(statement)}"
            )


def test_transform_does_not_vectorize() -> None:
    """A NumPy array operation is not a sequence of float32 scalar steps."""
    source = _transform_source()
    for token in ("np.array", "np.asarray", "ndarray", "np.vectorize", "dtype="):
        assert token not in source, token


# ---------------------------------------------------------------------------
# The transform registry
# ---------------------------------------------------------------------------


def test_output_functions_names_exactly_the_three_format_transforms() -> None:
    assert set(OUTPUT_FUNCTIONS) == {"identity", "sigmoid", "exp"}
    assert OUTPUT_FUNCTIONS["identity"] is identity_f32
    assert OUTPUT_FUNCTIONS["sigmoid"] is sigmoid_f32
    assert OUTPUT_FUNCTIONS["exp"] is exp_f32


def test_output_functions_raises_on_an_unrecognized_name() -> None:
    """Nothing defaults. An unknown transform is a failure, not a fallback."""
    for name in ("Sigmoid", "logistic", "softmax", "", "reg:squarederror"):
        try:
            OUTPUT_FUNCTIONS[name]
        except KeyError:
            continue
        raise AssertionError(f"unrecognized transform {name!r} did not raise")


def test_output_functions_is_not_mutable() -> None:
    try:
        OUTPUT_FUNCTIONS["exp"] = identity_f32  # type: ignore[index]
    except TypeError:
        return
    raise AssertionError("the transform registry accepted a mutation")


def test_output_functions_agree_with_the_objective_pairing() -> None:
    """Every transform the objective table names must exist here.

    Kept as a cross-check between the two modules rather than a shared
    constant, because a single source of truth here would mean a typo in one
    place changes behaviour in both without anything disagreeing.
    """
    from xgboost_bridge.objectives import OUTPUT_TRANSFORMS

    assert set(OUTPUT_TRANSFORMS.values()) <= set(OUTPUT_FUNCTIONS)


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


def test_identity_returns_the_margin_and_never_normalizes_signed_zero() -> None:
    assert bits32(identity_f32(np.float32(-0.0))) == 0x80000000
    assert bits32(identity_f32(np.float32(0.0))) == 0x00000000
    assert bits32(identity_f32(np.float32(np.inf))) == 0x7F800000
    assert bits32(identity_f32(np.float32(-np.inf))) == 0xFF800000
    assert is_nan32(identity_f32(np.float32(np.nan)))
    for value in (1.0, -1.0, 1e-45, 3.4e38, -13.473269):
        assert bits32(identity_f32(np.float32(value))) == bits32(value)


def test_identity_narrows_a_float64_argument() -> None:
    """The narrowing site is the function, so a float64 caller is still safe."""
    result = identity_f32(0.1)
    assert isinstance(result, np.float32)
    assert bits32(result) == bits32(np.float32(0.1))


# ---------------------------------------------------------------------------
# exp: specified values, non-finite behaviour, saturation
# ---------------------------------------------------------------------------


def test_exp_at_zero_is_exactly_one_for_both_signed_zeros() -> None:
    assert bits32(exp_f32(np.float32(0.0))) == 0x3F800000
    assert bits32(exp_f32(np.float32(-0.0))) == 0x3F800000


def test_exp_non_finite_behaviour() -> None:
    assert bits32(exp_f32(np.float32(np.inf))) == 0x7F800000
    assert bits32(exp_f32(np.float32(-np.inf))) == 0x00000000
    assert is_nan32(exp_f32(np.float32(np.nan)))


def test_exp_has_no_floor_and_underflows_to_a_positive_zero() -> None:
    """`survival:cox` has no clamp: the tail underflows rather than saturating.

    The measured contrast (`probes/output_transform.md` section 4) is the
    point -- at a margin of -94.63 XGBoost returns the exponential's own
    subnormal, not the sigmoid floor value.
    """
    tail = exp_f32(np.float32(PROBE_COX_TAIL_MARGIN))
    assert bits32(tail) == bits32(PROBE_COX_TAIL_OUTPUT)
    assert bits32(tail) != PROBE_FLOOR_OUTPUT_BITS

    far = exp_f32(np.float32(-200.0))
    assert bits32(far) == 0x00000000  # positive zero, not negative
    assert bits32(exp_f32(np.float32(-1000.0))) == 0x00000000
    assert bits32(exp_f32(np.float32(-MAX_FLOAT32))) == 0x00000000


def test_exp_saturates_to_positive_infinity_where_xgboost_does() -> None:
    """Measured: `+inf` on 734/2500 Cox rows above a margin of about 88.72."""
    for margin in PROBE_COX_INF_MARGINS:
        assert bits32(exp_f32(np.float32(margin))) == 0x7F800000
    assert bits32(exp_f32(np.float32(MAX_FLOAT32))) == 0x7F800000
    assert bits32(exp_f32(np.float32(88.72283935546875))) == 0x7F800000
    # And the float32 immediately below it is still finite.
    below = np.nextafter(np.float32(88.72283935546875), np.float32(-np.inf))
    assert bits32(exp_f32(below)) != 0x7F800000
    assert np.isfinite(float(exp_f32(below)))


def test_exp_is_monotonically_non_decreasing() -> None:
    """A reduction bug at a power-of-two boundary shows up as a step backwards.

    Monotonicity is not a correctness oracle and is not used as one; it
    catches a specific structural failure that a ULP figure averages away.
    """
    grid = sorted(
        {
            float(value)
            for value in (
                *np.linspace(-105.0, 89.0, 4001),
                *np.linspace(-1.0, 1.0, 501),
                *(float(v) for v in exp_boundary_points() if np.isfinite(float(v))),
            )
        }
    )
    previous = exp_f32(np.float32(grid[0]))
    for value in grid[1:]:
        current = exp_f32(np.float32(value))
        assert float(current) >= float(previous), value
        previous = current


def test_exp_is_never_negative() -> None:
    for margin in exp_sample_points(600):
        result = exp_f32(margin)
        if is_nan32(result):
            continue
        assert bits32(result) & 0x80000000 == 0, float(margin)


# ---------------------------------------------------------------------------
# sigmoid: the measured floor, saturation at one, non-finite behaviour
# ---------------------------------------------------------------------------


def test_sigmoid_floor_constants_match_the_measured_values() -> None:
    """The module's constants against the probe, not against each other."""
    assert bits32(SIGMOID_MARGIN_FLOOR) == PROBE_FLOOR_MARGIN_BITS
    assert float(SIGMOID_MARGIN_FLOOR) == PROBE_FLOOR_MARGIN
    assert bits32(SIGMOID_FLOOR_OUTPUT) == PROBE_FLOOR_OUTPUT_BITS
    assert float(SIGMOID_FLOOR_OUTPUT) == PROBE_FLOOR_OUTPUT
    # The floor output is a subnormal, which is why a float64 transform is
    # not merely imprecise there.
    assert 0.0 < float(SIGMOID_FLOOR_OUTPUT) < MIN_NORMAL_FLOAT32


def test_sigmoid_at_the_floor_returns_the_measured_bit_pattern() -> None:
    assert bits32(sigmoid_f32(SIGMOID_MARGIN_FLOOR)) == PROBE_FLOOR_OUTPUT_BITS


def test_sigmoid_below_the_floor_is_that_one_value_and_never_zero() -> None:
    """Measured on 2056 below-floor rows: one bit pattern, `n_exact_zero=0`.

    A float64 transform returns 1.5e-89 on the -204.5 row -- relative error
    1.0 against XGBoost, which is why the floor is reproduced rather than
    improved upon.
    """
    margins = [
        *PROBE_BELOW_FLOOR_MARGINS,
        -88.7,
        -88.71,
        -90.0,
        -327.9169921875,
        -MAX_FLOAT32,
        -np.inf,
        np.nextafter(np.float32(PROBE_FLOOR_MARGIN), np.float32(-np.inf)),
    ]
    for margin in margins:
        result = sigmoid_f32(np.float32(margin))
        assert bits32(result) == PROBE_FLOOR_OUTPUT_BITS, float(margin)
        assert float(result) != 0.0, float(margin)


def test_sigmoid_output_of_exactly_zero_is_unreachable() -> None:
    """The floor prevents it, so a check demanding it tests nothing.

    Scanned over the entire negative half of the float32 line by exponent,
    plus every representable value in the band around the floor.
    """
    margins = [
        *_band_scan(PROBE_FLOOR_MARGIN, -88.0, 137),
        *(np.float32(-v) for v in np.logspace(-45, 38, 4000)),
        np.float32(-np.inf),
    ]
    for margin in margins:
        assert float(sigmoid_f32(margin)) != 0.0, float(margin)


def test_sigmoid_output_of_exactly_one_is_reachable() -> None:
    """Measured on 288/2500 rows. The first such float32 is 16.635532."""
    first = np.float32(16.63553237915039)
    assert bits32(sigmoid_f32(first)) == 0x3F800000
    assert bits32(sigmoid_f32(np.float32(17.0))) == 0x3F800000
    assert bits32(sigmoid_f32(np.float32(386.6369323730469))) == 0x3F800000
    assert bits32(sigmoid_f32(np.float32(MAX_FLOAT32))) == 0x3F800000
    assert bits32(sigmoid_f32(np.float32(np.inf))) == 0x3F800000
    # And the float32 immediately below is not yet exactly one.
    below = np.nextafter(first, np.float32(-np.inf))
    assert bits32(sigmoid_f32(below)) != 0x3F800000


def test_sigmoid_at_zero_is_exactly_one_half_for_both_signed_zeros() -> None:
    assert bits32(sigmoid_f32(np.float32(0.0))) == 0x3F000000
    assert bits32(sigmoid_f32(np.float32(-0.0))) == 0x3F000000


def test_sigmoid_non_finite_behaviour() -> None:
    """`-inf` is below the floor, so the floor is what applying it means."""
    assert bits32(sigmoid_f32(np.float32(np.inf))) == 0x3F800000
    assert bits32(sigmoid_f32(np.float32(-np.inf))) == PROBE_FLOOR_OUTPUT_BITS
    assert is_nan32(sigmoid_f32(np.float32(np.nan)))


def test_sigmoid_stays_within_the_unit_interval_and_is_non_decreasing() -> None:
    grid = sorted(
        {
            float(value)
            for value in (
                *np.linspace(-95.0, 95.0, 6001),
                *np.linspace(-1.0, 1.0, 501),
                *(float(v) for v in sigmoid_boundary_points() if np.isfinite(float(v))),
            )
        }
    )
    previous = sigmoid_f32(np.float32(grid[0]))
    for value in grid[1:]:
        current = sigmoid_f32(np.float32(value))
        assert 0.0 < float(current) <= 1.0, value
        assert float(current) >= float(previous), value
        previous = current


def test_sigmoid_is_the_exponential_composed_the_way_xgboost_composes_it() -> None:
    """The winning hypothesis, verbatim: 1 / (1 + exp(-floor(margin))).

    Not a correctness check -- it is the same code -- but a pin on the
    composition, which is the part the probe identified at 35000/35000 and
    which a later refactor could plausibly rearrange into a mathematically
    equivalent form that is not bit-equivalent.
    """
    for margin in (-88.0, -10.0, -1.0, -0.5, 0.0, 0.5, 1.0, 10.0, 88.0):
        value = np.float32(margin)
        expected = np.float32(np.float32(1.0) / np.float32(np.float32(1.0) + exp_f32(-value)))
        assert bits32(sigmoid_f32(value)) == bits32(expected), margin


# ---------------------------------------------------------------------------
# The power-of-two scaling helper
# ---------------------------------------------------------------------------


def test_power_of_two_is_exact_across_the_normal_range() -> None:
    for exponent in range(-126, 128):
        value = transform._power_of_two_f32(exponent)
        assert float(value) == np.ldexp(1.0, exponent), exponent
        assert bits32(value) & 0x007FFFFF == 0, exponent


def test_power_of_two_raises_outside_the_normal_range() -> None:
    """An internal invariant, tested on its own so it is not merely implied."""
    for exponent in (-127, -149, -1000, 128, 255, 1000):
        try:
            transform._power_of_two_f32(exponent)
        except ValueError:
            continue
        raise AssertionError(f"exponent {exponent} did not raise")


def test_the_power_split_keeps_both_halves_in_the_normal_range() -> None:
    """The claim that makes the first of the two scalings exact.

    If it failed, the first scaling would round and the result would carry two
    roundings instead of one -- a fraction of a ULP, invisible except here.
    """
    for power in range(transform._MIN_POWER, transform._MAX_POWER + 1):
        low = power // 2
        high = power - low
        assert low + high == power
        assert -125 <= low <= 127, power
        assert -125 <= high <= 127, power


# ---------------------------------------------------------------------------
# The always-on ULP measurement against mpmath
# ---------------------------------------------------------------------------

FAST_SWEEP_POINTS = 3000


def test_exp_max_ulp_against_mpmath() -> None:
    result = measure(exp_f32, reference_exp, exp_sample_points(FAST_SWEEP_POINTS))
    worst = result["worst"]
    assert worst["ulp"] <= EXP_MAX_ULP, (
        f"max ULP {worst['ulp']} over {result['count']} points at margin "
        f"{float(worst['margin'])!r} (bits {bits32(worst['margin'])}): got "
        f"{float(worst['got'])!r}, want {float(worst['want'])!r}; "
        f"histogram {dict(sorted(result['histogram'].items()))}"
    )


def test_sigmoid_max_ulp_against_mpmath() -> None:
    result = measure(
        sigmoid_f32, reference_sigmoid, sigmoid_sample_points(FAST_SWEEP_POINTS)
    )
    worst = result["worst"]
    assert worst["ulp"] <= SIGMOID_MAX_ULP, (
        f"max ULP {worst['ulp']} over {result['count']} points at margin "
        f"{float(worst['margin'])!r} (bits {bits32(worst['margin'])}): got "
        f"{float(worst['got'])!r}, want {float(worst['want'])!r}; "
        f"histogram {dict(sorted(result['histogram'].items()))}"
    )


def test_exp_at_its_known_worst_input() -> None:
    """The reported maximum, at the input that produced it."""
    value = np.float32(WORST_KNOWN_EXP_MARGIN)
    error = ulp_error(exp_f32(value), reference_exp(value))
    assert error == EXP_MAX_ULP, (
        f"{WORST_KNOWN_EXP_MARGIN!r} scores {error} ULP, not {EXP_MAX_ULP}; "
        "either the implementation changed or the bound is stale"
    )


def test_sigmoid_at_its_known_worst_inputs() -> None:
    """The bound is held at the inputs it is largest at, not on average.

    Reported separately because a max taken over a sample is only as good as
    the sample: these margins are the search result, so a regression here
    cannot hide behind a draw that happens to miss them.
    """
    for margin in WORST_KNOWN_SIGMOID_MARGINS:
        value = np.float32(margin)
        error = ulp_error(sigmoid_f32(value), reference_sigmoid(value))
        assert error <= SIGMOID_MAX_ULP, f"{margin!r}: {error} ULP"
        # And the bound is tight rather than generous -- if this stops being
        # reachable, the implementation improved and the bound should follow.
        assert error == SIGMOID_MAX_ULP, (
            f"{margin!r} now scores {error} ULP; SIGMOID_MAX_ULP is stale"
        )


def test_exp_max_ulp_in_the_large_margin_region() -> None:
    """The region that a single-constant argument reduction gets wrong.

    Splitting the logarithm of two into two parts is what keeps the reduction
    accurate once the margin is large enough that the subtraction cancels
    most of it. With one float32 constant instead of two, the error here is
    hundreds of ULP while small margins stay perfect -- so this region needs
    its own measurement rather than being averaged into the sweep above.
    """
    points = [
        *_band_scan(80.0, 88.5, 3000),
        *_band_scan(-88.0, -80.0, 3000),
        *(np.float32(v) for v in np.linspace(50.0, 88.7, 400)),
        *(np.float32(v) for v in np.linspace(-103.0, -50.0, 400)),
    ]
    result = measure(exp_f32, reference_exp, points)
    worst = result["worst"]
    assert worst["ulp"] <= EXP_MAX_ULP, (
        f"max ULP {worst['ulp']} at margin {float(worst['margin'])!r}: got "
        f"{float(worst['got'])!r}, want {float(worst['want'])!r}"
    )


def test_exp_saturates_in_the_same_place_as_the_reference() -> None:
    """Scanned across the two bands where the answer changes kind.

    A ULP figure cannot express "finite where it should be infinite" -- those
    two are one ordinal apart -- so the transition into `+inf` and the
    transition into zero are checked as predicates against the reference, at
    every scanned float32 in each band, in addition to the ULP bound.
    """
    for low, high in ((88.6, 88.8), (-104.2, -103.8)):
        for margin in _band_scan(low, high, 97):
            got = exp_f32(margin)
            want = reference_exp(margin)
            assert np.isinf(float(got)) == np.isinf(float(want)), float(margin)
            assert (float(got) == 0.0) == (float(want) == 0.0), float(margin)
            assert ulp_error(got, want) <= EXP_MAX_ULP, (
                f"margin {float(margin)!r}: got {float(got)!r}, "
                f"want {float(want)!r}"
            )


# ---------------------------------------------------------------------------
# The full sweep, run directly rather than as part of the suite
# ---------------------------------------------------------------------------


def _report(label: str, result: dict, reference_name: str) -> None:
    worst = result["worst"]
    print(f"\n{label}  oracle={reference_name}  points={result['count']}")
    print(f"  ULP histogram: {dict(sorted(result['histogram'].items()))}")
    wrong = sum(count for ulp, count in result["histogram"].items() if ulp)
    print(f"  MAX ULP = {worst['ulp']}   not correctly rounded: {wrong}/{result['count']}")
    print(
        f"  worst input: {float(worst['margin'])!r}  bits={bits32(worst['margin'])}\n"
        f"    got  {float(worst['got'])!r}  bits={bits32(worst['got'])}\n"
        f"    want {float(worst['want'])!r}  bits={bits32(worst['want'])}"
    )


def main(count: int = 1_000_000) -> None:
    """Run the full sweep and print max ULP per function. Never a mean."""
    print(f"mpmath dps={mp.dps}  prec={mp.prec} bits")
    print(f"requested points per function: {count}")

    exp_points = exp_sample_points(count)
    _report("exp_f32", measure(exp_f32, reference_exp, exp_points), "mpmath 50 dps")

    sigmoid_points = sigmoid_sample_points(count)
    _report(
        "sigmoid_f32",
        measure(sigmoid_f32, reference_sigmoid, sigmoid_points),
        "mpmath 50 dps",
    )

    # Context only, and explicitly not a correctness oracle: the platform
    # exponential is the thing being replaced and is not correctly rounded
    # either. Printed so a reader can see the bundled version is in the same
    # accuracy class rather than markedly worse.
    def platform_exp(margin: np.float32) -> np.float32:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            return np.float32(np.exp(np.float32(margin)))

    _report(
        "CONTEXT ONLY: the platform float32 exponential",
        measure(platform_exp, reference_exp, exp_points),
        "mpmath 50 dps",
    )

    print("\nclamp behaviour")
    for margin in (PROBE_FLOOR_MARGIN, -88.71, -204.50521850585938, -748.246337890625, -np.inf):
        value = sigmoid_f32(np.float32(margin))
        print(f"  sigmoid_f32({margin!r}) = {float(value)!r}  bits={bits32(value)}")
    for margin in (88.72283172607422, 88.72283935546875, 120.70205688476562, np.inf):
        value = exp_f32(np.float32(margin))
        print(f"  exp_f32({margin!r}) = {float(value)!r}  bits={bits32(value)}")

    print("\nnon-finite and signed zero, as bit patterns")
    for name, function in (("exp_f32", exp_f32), ("sigmoid_f32", sigmoid_f32),
                           ("identity_f32", identity_f32)):
        for label, margin in (
            ("+inf", np.float32(np.inf)),
            ("-inf", np.float32(-np.inf)),
            ("NaN", np.float32(np.nan)),
            ("+0.0", np.float32(0.0)),
            ("-0.0", np.float32(-0.0)),
        ):
            value = function(margin)
            print(f"  {name}({label}) = {float(value)!r}  bits={bits32(value)}")

    print("\nconstants, as bit patterns")
    for name, (single, double) in EXPECTED_CONSTANT_BITS.items():
        value = getattr(transform, name)
        print(
            f"  {name:22s} float32 bits={bits32(value):>10d} (0x{single:08X})  "
            f"float64 bits={bits64(float(value)):>21d} (0x{double:016X})  "
            f"value={float(value)!r}"
        )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000)
