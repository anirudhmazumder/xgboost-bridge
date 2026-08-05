"""The bundled margin-to-output transform, built from correctly-rounded primitives.

This module is the one place in the library that does not reproduce something
XGBoost already does. Everywhere else XGBoost can arbitrate a disagreement;
here there is nothing upstream to ask, because the whole point (D030) is to
not call the platform's transcendental library at all. That makes this the
most dangerous code in the repository, and it fails in a way the check that
runs against it most often cannot see: a cross-language parity harness
reports perfect agreement when both sides are equally wrong. Bit-identical
wrong is still wrong. So validation is against an external high-precision
reference (``mpmath`` at 50 digits), per side, independently -- never against
the other language (D034, FORMAT.md section 5.6).

What is implemented, and why it is implemented rather than called:

* IEEE-754 mandates correct rounding only for ``+ - * /``, square root and
  fused multiply-add. Exponentiation is not required to be correctly rounded
  and no two platform libraries agree: measured, V8 differs from Apple's
  library on 4.2% of sigmoid and 9.6% of exponential evaluations, by up to
  2 ULP (D026 correction). Calling one on each side of the bridge makes the
  exactly-``0.0`` output parity gate unreachable by construction.
* Every operation below is therefore one of ``+ - * /`` or an exact scaling
  by a power of two. Nothing else appears on this path.

Evaluation is under float32 semantics: every intermediate is wrapped in
``np.float32(...)``, mirroring ``Math.fround(...)`` in the JavaScript port
(D032, FORMAT.md section 5.1). That is exact rather than approximate --
performing an operation in float64 and narrowing to float32 gives the same
result as the float32 operation for ``+ - * /``, because float64 carries more
than twice float32's significand, so double rounding cannot occur. It is a
property of the formats, not an observation, and it does not extend to
exponentiation -- which is precisely why the exponential here is built from
the four operations rather than called.

Three structural requirements exist because bit-identity across two
languages is lost by accident rather than by decision (FORMAT.md section
5.5):

* Each operation is a separate statement with an explicit named
  intermediate. No fused expression, so no runtime can contract one into a
  fused multiply-add and change the result. The guarantee comes from how the
  code is written, not from what CPython and V8 happen to do today.
* No vectorization. A NumPy array operation is not a sequence of float32
  scalar operations and would not port.
* Every constant is defined from its literal bit pattern, never from a
  decimal string that each language's parser rounds on its own. The
  companion test asserts each pattern as an integer, so the JavaScript port
  can assert the same integers.

XGBoost's measured clamps are reproduced (D032, ``probes/output_transform.md``
sections 3-4):

* ``sigmoid`` floors at margin ``f32(-88.7)`` and returns exactly
  ``3.006635794144578e-39`` -- never ``0.0``. The sole float32 input
  producing those bits is ``-88.69999694824219``, found by exhaustive scan of
  all 262145 float32 values in ``[-90, -88]``, and every margin below it
  returned that one bit pattern on 2056 measured rows.
* ``exp`` has no clamp. It returns ``+inf`` above margin ``88.7228``, which
  is what XGBoost does on 734/2500 measured rows, and underflows to ``0.0``
  in the far tail.

Consequently sigmoid saturation at exactly ``1`` is reachable and saturation
at exactly ``0`` is not: the floor prevents it. A check demanding the latter
is testing something XGBoost cannot produce.

Bit-exactness with XGBoost at the output is unreachable and is not a goal --
its own exponential is not correctly rounded, so an exactly-rounded
reference scores 1600/2500 against it. The gate against XGBoost is a relative
one (D033); the gate against the other language is exact equality.

Signed zero is never normalized, here or anywhere else in the library.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

import numpy as np


def _f32_from_bits(bits: int) -> np.float32:
    """Return the float32 whose IEEE-754 encoding is exactly ``bits``.

    Every constant in this module is defined this way. A decimal literal
    would be re-rounded by each language's own parser, and the two parsers
    are not required to agree; a bit pattern is the same number in both.
    """
    return np.uint32(bits).view(np.float32)


# --------------------------------------------------------------------------
# Constants, each written as the integer bit pattern of the float32 it
# denotes. The companion test asserts every one of these integers, and also
# asserts the float64 bit pattern of the same value, because the JavaScript
# port holds these as doubles that happen to be exactly float32-valued.
# --------------------------------------------------------------------------

# Reciprocal of the natural logarithm of two, in float32. Used only to pick
# the integer power of two to factor out, so its own rounding error shifts
# that integer by one at worst and never leaves the reduction interval.
_INV_LN2 = _f32_from_bits(0x3FB8AA3B)  # 1.4426950216293335

# The natural logarithm of two, split into two float32 parts.
#
# The split is what makes the reduction accurate, and the low 8 significand
# bits of the high part are zero for a specific reason: the integer factored
# out is at most 254 in magnitude, which needs 8 bits, so the product of that
# integer with a 16-significant-bit constant fits in float32's 24 bits and is
# therefore EXACT. The subsequent subtraction from the margin is exact too,
# by Sterbenz's lemma, since the two quantities are within a factor of two of
# each other. All of the reduction's rounding error is thus confined to the
# single subtraction of the low part, which is small.
_LN2_HI = _f32_from_bits(0x3F317200)  # 0.693145751953125
_LN2_LO = _f32_from_bits(0x35BFBE8E)  # 1.428606765330187e-06

# Round-to-nearest-integer by adding and removing 1.5 times 2 to the 23rd.
# At that magnitude the float32 grid spacing is exactly 1, so the addition
# rounds to an integer and the subtraction recovers it exactly. This keeps
# the rounding inside the four permitted operations rather than reaching for
# a library rounding function whose tie behaviour is another thing to pin.
_ROUND_MAGIC = _f32_from_bits(0x4B400000)  # 12582912.0

# Coefficients of the Taylor series of the exponential, from the quadratic
# term up. The reduced argument satisfies a magnitude bound of one half the
# logarithm of two, so the first omitted term contributes a relative error
# near 5.2e-09 -- about 0.04 ULP, an order of magnitude below the float32
# rounding that follows it.
_C2 = _f32_from_bits(0x3F000000)  # 0.5
_C3 = _f32_from_bits(0x3E2AAAAB)  # 0.1666666716337204
_C4 = _f32_from_bits(0x3D2AAAAB)  # 0.0416666679084301
_C5 = _f32_from_bits(0x3C088889)  # 0.008333333767950535
_C6 = _f32_from_bits(0x3AB60B61)  # 0.0013888889225199819
_C7 = _f32_from_bits(0x39500D01)  # 0.00019841270113829523

_ZERO = _f32_from_bits(0x00000000)  # 0.0
_ONE = _f32_from_bits(0x3F800000)  # 1.0
_POS_INF = _f32_from_bits(0x7F800000)  # inf
_NEG_INF = _f32_from_bits(0xFF800000)  # -inf

#: The margin below which XGBoost's ``binary:logistic`` transform saturates,
#: measured rather than derived: of all 262145 float32 values in
#: ``[-90, -88]``, exactly one reproduces the observed output bit pattern,
#: and it is this one (``probes/output_transform.md`` section 3). Clamp
#: constants are XGBoost internals, are version-sensitive in the way
#: ``weight_drop`` proved to be, and fall under the version ceiling (D018).
SIGMOID_MARGIN_FLOOR = _f32_from_bits(0xC2B16666)  # -88.69999694824219

#: The value ``sigmoid`` returns at and below :data:`SIGMOID_MARGIN_FLOOR`.
#: A float32 subnormal, and never ``0.0`` -- measured on 2056 below-floor
#: rows spanning margins from ``-88.71`` down to ``-748.25``, every one of
#: which returned this single bit pattern. Present as a named constant so a
#: reader can see the specified value, but it is not used to short-circuit:
#: the floor is applied to the margin and the transform then produces this
#: value through the same arithmetic as every other input.
SIGMOID_FLOOR_OUTPUT = _f32_from_bits(0x0020BD47)  # 3.006635794144578e-39

# Bounds on the integer power of two factored out of the margin.
#
# Beyond them the result is decided without further arithmetic, which also
# keeps the conversion to a Python integer away from a non-finite value.
# Both bounds sit far outside the range where the result is finite and
# nonzero, so no interesting input reaches them: the upper corresponds to a
# margin above 175.7, where the exponential is 1e76 and float32 has long
# since overflowed, and the lower to a margin below -172.9, where it is
# 3e-76 and float32 has long since underflowed to zero.
#
# Their real job is the splitting rule below. Within these bounds, halving
# the power leaves both halves inside ``[-125, 127]``, so both scaling
# factors are normal float32 values and the first of the two scalings can
# neither overflow nor underflow -- making it exact, and leaving the second
# scaling as the single rounding that produces the returned value.
_MAX_POWER = 254
_MIN_POWER = -250


def _power_of_two_f32(exponent: int) -> np.float32:
    """Return two raised to ``exponent`` as an exactly representable float32.

    Built from the exponent field of the IEEE-754 encoding, so it is exact by
    construction rather than by the accuracy of a general power routine --
    which is not among the operations this module is allowed to use.

    Args:
        exponent: An integer in ``[-126, 127]``, the range over which the
            result is a normal float32.

    Raises:
        ValueError: If ``exponent`` is outside that range. Unreachable from
            any margin, because the caller bounds the power first; it is an
            internal invariant, and it raises rather than returning a
            silently wrong scaling factor.
    """
    if not -126 <= exponent <= 127:
        raise ValueError(
            "power-of-two exponent outside the normal float32 range: "
            f"{exponent!r} is not in [-126, 127]"
        )
    return _f32_from_bits((exponent + 127) << 23)


def identity_f32(margin: object) -> np.float32:
    """Return the margin unchanged, narrowed to float32.

    The output transform for objectives whose prediction IS the margin. It
    exists as a named function so that the transform is always a lookup in
    :data:`OUTPUT_FUNCTIONS` and never a branch that has to remember to do
    nothing.

    Signed zero and non-finite values pass through untouched: ``-0.0`` stays
    ``-0.0`` rather than being normalized, and a NaN stays a NaN.
    """
    with np.errstate(over="ignore", under="ignore"):
        return np.float32(margin)


def exp_f32(margin: object) -> np.float32:
    """Return the exponential of ``margin`` under float32 semantics.

    The output transform for ``survival:cox``. There is no clamp: the result
    overflows to ``+inf`` above a margin of ``88.7228`` and underflows to
    ``0.0`` in the far tail, both of which XGBoost does (measured, 45000/45000
    bit-exact against ``predict()`` including 734 rows at ``+inf``).

    The margin is factored as an integer power of two times a value near one.
    The power is removed by exact scaling and the remainder is evaluated as a
    short polynomial, so every operation is one of the four that IEEE-754
    requires to be correctly rounded.

    Args:
        margin: The margin. Narrowed to float32 on entry -- this is the
            narrowing site, so a caller that passes a float64 still gets
            float32 semantics rather than a value that reads as correct and
            is a bit wrong.

    Returns:
        The float32 result. ``NaN`` in gives ``NaN`` out, ``+inf`` gives
        ``+inf``, ``-inf`` gives ``+0.0``.
    """
    # Overflow to infinity and underflow into the subnormals are specified
    # outcomes here, not error conditions -- Cox saturates to `+inf` above
    # 88.7228 and the sigmoid floor is itself a subnormal -- so the
    # corresponding floating-point flags are silenced rather than surfaced as
    # warnings. No value is changed by this; only what NumPy reports.
    with np.errstate(over="ignore", under="ignore"):
        value = np.float32(margin)

        # NaN is the one input that no comparison below can route correctly,
        # since it compares false against everything including itself.
        if value != value:
            return value
        if value == _POS_INF:
            return _POS_INF
        if value == _NEG_INF:
            return _ZERO

        # Nearest integer to the margin divided by the logarithm of two.
        quotient = np.float32(value * _INV_LN2)
        shifted = np.float32(quotient + _ROUND_MAGIC)
        power_f32 = np.float32(shifted - _ROUND_MAGIC)

        # Compared as floats, before the conversion to a Python integer, so
        # that an infinite quotient is decided here instead of raising there.
        if power_f32 > np.float32(_MAX_POWER):
            return _POS_INF
        if power_f32 < np.float32(_MIN_POWER):
            return _ZERO
        power = int(power_f32)

        # Argument reduction. `hi_product` is exact because the power needs at
        # most 8 bits and `_LN2_HI` has at most 16 significant bits;
        # `reduced_hi` is exact by Sterbenz's lemma. So the only rounding in
        # the reduction is the last subtraction, of a quantity below 4e-03.
        hi_product = np.float32(power_f32 * _LN2_HI)
        reduced_hi = np.float32(value - hi_product)
        lo_product = np.float32(power_f32 * _LN2_LO)
        reduced = np.float32(reduced_hi - lo_product)

        # Taylor series for the reduced argument, one operation per statement.
        poly = np.float32(reduced * _C7)
        poly = np.float32(_C6 + poly)
        poly = np.float32(reduced * poly)
        poly = np.float32(_C5 + poly)
        poly = np.float32(reduced * poly)
        poly = np.float32(_C4 + poly)
        poly = np.float32(reduced * poly)
        poly = np.float32(_C3 + poly)
        poly = np.float32(reduced * poly)
        poly = np.float32(_C2 + poly)
        squared = np.float32(reduced * reduced)
        tail = np.float32(squared * poly)
        fraction = np.float32(reduced + tail)
        reduced_exp = np.float32(_ONE + fraction)

        # Restore the power of two in two exact halves. The first scaling
        # cannot overflow or underflow -- the reduced result lies within
        # ``[0.7071, 1.4143]`` and each half of the power lies in
        # ``[-125, 127]`` -- so it is exact, and the second scaling is the
        # single rounding that produces the returned value, including when
        # that value is subnormal or infinite.
        low_exponent = power // 2
        high_exponent = power - low_exponent
        scaled_once = np.float32(reduced_exp * _power_of_two_f32(low_exponent))
        return np.float32(scaled_once * _power_of_two_f32(high_exponent))


def sigmoid_f32(margin: object) -> np.float32:
    """Return the logistic function of ``margin`` under float32 semantics.

    The output transform for ``binary:logistic``, reproducing XGBoost's
    measured floor: a margin below :data:`SIGMOID_MARGIN_FLOOR` is raised to
    it before the transform, so the result is exactly
    :data:`SIGMOID_FLOOR_OUTPUT` and never ``0.0``. Whether XGBoost clamps
    the input or floors the output is observationally identical and is not
    resolved (``probes/output_transform.md`` A1); the input floor is used
    here, and it reproduces the measured output bit pattern exactly.

    Saturation at exactly ``1`` is reachable, from a margin of ``16.635533``
    upward. Saturation at exactly ``0`` is not reachable, because the floor
    prevents it.

    Args:
        margin: The margin. Narrowed to float32 on entry.

    Returns:
        The float32 result. ``NaN`` in gives ``NaN`` out. ``+inf`` gives
        exactly ``1.0``; ``-inf`` is below the floor and so gives
        :data:`SIGMOID_FLOOR_OUTPUT`, which is what applying the measured
        floor to it means.
    """
    # As in the exponential, saturation is specified rather than exceptional.
    # `invalid` is silenced too, because a NaN margin is carried through the
    # arithmetic on purpose -- adding one to a NaN raises that flag, and NaN
    # propagation is the specified behaviour, not a fault to report.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        value = np.float32(margin)

        # NaN first: it compares false against the floor, so leaving it to
        # the comparison below would pass it through by accident rather than
        # by decision. It reaches the arithmetic and propagates.
        if value != value:
            floored = value
        elif value < SIGMOID_MARGIN_FLOOR:
            floored = SIGMOID_MARGIN_FLOOR
        else:
            floored = value

        negated = np.float32(-floored)
        exponential = exp_f32(negated)
        denominator = np.float32(_ONE + exponential)
        return np.float32(_ONE / denominator)


#: The three output transforms named in the artifact format (FORMAT.md
#: section 5), keyed by the exact string the artifact carries. A lookup
#: raises ``KeyError`` on anything else: nothing defaults, and no transform
#: is inferred from the objective by analogy (D007, D028).
OUTPUT_FUNCTIONS: Mapping[str, Callable[[object], np.float32]] = MappingProxyType(
    {
        "identity": identity_f32,
        "sigmoid": sigmoid_f32,
        "exp": exp_f32,
    }
)
