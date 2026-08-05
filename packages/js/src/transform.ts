/**
 * The bundled margin-to-output transform, built from correctly-rounded
 * primitives.
 *
 * This is the one module in this package that does not reproduce something
 * XGBoost already does, which makes it the most dangerous code here: it fails
 * in a way the check that runs against it most often cannot see. A
 * cross-language parity harness reports perfect agreement when both sides are
 * equally wrong, and bit-identical wrong is still wrong. So this side is
 * validated against an external high-precision reference — `mpmath` at 50
 * digits, tabulated under `fixtures/corpus/reference/` — per function,
 * independently, and never against the Python implementation it mirrors
 * (D030, D034, FORMAT.md §5.6).
 *
 * What is implemented here, and why it is implemented rather than called:
 *
 * - IEEE-754 mandates correct rounding only for `+ - * /`, square root and
 *   fused multiply-add. Exponentiation is not required to be correctly rounded
 *   and no two platform libraries agree: measured, V8 differs from Apple's
 *   library on 4.2% of logistic and 9.6% of exponential evaluations, by up to
 *   2 ULP. Calling one on each side of the bridge makes the exactly-`0.0`
 *   output parity gate unreachable by construction.
 * - Every operation below is therefore one of `+ - * /` or an exact scaling by
 *   a power of two. `Math.exp`, `Math.pow` and `**` appear nowhere in this
 *   package, and a test greps the shipped bundle to keep it that way.
 *
 * Evaluation is under float32 semantics: every intermediate is wrapped in
 * `Math.fround(...)`, mirroring `np.float32(...)` in the Python reference
 * implementation (D032, FORMAT.md §5.1). That is exact rather than
 * approximate — performing an operation in float64 and narrowing to float32
 * gives the same result as the float32 operation for `+ - * /`, because
 * float64 carries more than twice float32's significand, so double rounding
 * cannot occur. It is a property of the formats, not an observation, and it
 * does not extend to exponentiation, which is precisely why the exponential
 * here is built from the four operations rather than called.
 *
 * Three structural requirements exist because bit-identity across two
 * languages is lost by accident rather than by decision (FORMAT.md §5.5):
 *
 * - Each operation is a separate statement with an explicit named
 *   intermediate. No fused expression, so no runtime can contract one into a
 *   fused multiply-add and change the result. The guarantee comes from how the
 *   code is written, not from what V8 happens to do today.
 * - No vectorization. Scalar, explicit, boring.
 * - Every constant is defined from its literal bit pattern, never from a
 *   decimal string that each language's parser rounds on its own. The
 *   companion test asserts each pattern as the same integer the Python side
 *   pins, before anything downstream is trusted.
 *
 * XGBoost's measured clamps are reproduced (D032):
 *
 * - `sigmoid` floors at margin `f32(-88.7)` and returns exactly
 *   `3.006635794144578e-39` — never `0.0`. The sole float32 input producing
 *   those bits is `-88.69999694824219`, found by exhaustive scan of all 262145
 *   float32 values in `[-90, -88]`.
 * - `exp` has no clamp. It returns `+Infinity` above margin `88.7228` and
 *   underflows to `0.0` in the far tail, both of which XGBoost does.
 *
 * Consequently saturation of the logistic at exactly `1` is reachable and
 * saturation at exactly `0` is not: the floor prevents it.
 *
 * Bit-exactness with XGBoost at the output is unreachable and is not a goal —
 * its own exponential is not correctly rounded. The gate against XGBoost is a
 * relative one (D033); the gate against the other language is exact equality.
 *
 * Signed zero is never normalized, here or anywhere else in this package.
 */

/**
 * Scratch view used to build a float32 from its IEEE-754 encoding.
 *
 * Module-level and reused rather than allocated per call. That is safe because
 * every function that touches it writes and then immediately reads, with no
 * `await`, no generator suspension and no reentrancy in between, and
 * JavaScript gives each realm a single thread.
 */
const BIT_PATTERN_VIEW = new DataView(new ArrayBuffer(4));

/**
 * Return the float32 whose IEEE-754 encoding is exactly `bits`.
 *
 * Every constant in this module is defined this way. A decimal literal would
 * be re-rounded by each language's own parser, and the two parsers are not
 * required to agree; a bit pattern is the same number in both.
 */
function float32FromBits(bits: number): number {
  BIT_PATTERN_VIEW.setUint32(0, bits >>> 0);
  return BIT_PATTERN_VIEW.getFloat32(0);
}

// ---------------------------------------------------------------------------
// Constants, each written as the integer bit pattern of the float32 it
// denotes. The companion test asserts every one of these integers against the
// integers `packages/python/src/xgboost_bridge/transform.py` pins, and also
// asserts the float64 bit pattern of the same value, because these are held
// here as doubles that happen to be exactly float32-valued.
// ---------------------------------------------------------------------------

/**
 * Reciprocal of the natural logarithm of two, in float32. Used only to pick
 * the integer power of two to factor out, so its own rounding error shifts
 * that integer by one at worst and never leaves the reduction interval.
 */
const INV_LN2 = float32FromBits(0x3fb8aa3b); // 1.4426950216293335

/**
 * The natural logarithm of two, high part.
 *
 * The split into a high and a low part is what makes the reduction accurate,
 * and the low 8 significand bits of the high part are zero for a specific
 * reason: the integer factored out is at most 254 in magnitude, which needs 8
 * bits, so the product of that integer with a 16-significant-bit constant fits
 * in float32's 24 bits and is therefore EXACT. The subsequent subtraction from
 * the margin is exact too, by Sterbenz's lemma, since the two quantities are
 * within a factor of two of each other. All of the reduction's rounding error
 * is thus confined to the single subtraction of the low part, which is small.
 */
const LN2_HI = float32FromBits(0x3f317200); // 0.693145751953125

/** The natural logarithm of two, low part. See {@link LN2_HI}. */
const LN2_LO = float32FromBits(0x35bfbe8e); // 1.428606765330187e-06

/**
 * Round-to-nearest-integer by adding and removing 1.5 times 2 to the 23rd.
 *
 * At that magnitude the float32 grid spacing is exactly 1, so the addition
 * rounds to an integer and the subtraction recovers it exactly. This keeps the
 * rounding inside the four permitted operations rather than reaching for a
 * library rounding function whose tie behaviour is another thing to pin.
 */
const ROUND_MAGIC = float32FromBits(0x4b400000); // 12582912.0

// Coefficients of the Taylor series of the exponential, from the quadratic
// term up. The reduced argument satisfies a magnitude bound of one half the
// logarithm of two, so the first omitted term contributes a relative error
// near 5.2e-09 — about 0.04 ULP, an order of magnitude below the float32
// rounding that follows it.
const C2 = float32FromBits(0x3f000000); // 0.5
const C3 = float32FromBits(0x3e2aaaab); // 0.1666666716337204
const C4 = float32FromBits(0x3d2aaaab); // 0.0416666679084301
const C5 = float32FromBits(0x3c088889); // 0.008333333767950535
const C6 = float32FromBits(0x3ab60b61); // 0.0013888889225199819
const C7 = float32FromBits(0x39500d01); // 0.00019841270113829523

const ZERO = float32FromBits(0x00000000); // 0.0
const ONE = float32FromBits(0x3f800000); // 1.0
const POS_INF = float32FromBits(0x7f800000); // Infinity
const NEG_INF = float32FromBits(0xff800000); // -Infinity

/**
 * The margin below which XGBoost's `binary:logistic` transform saturates.
 *
 * Measured rather than derived: of all 262145 float32 values in `[-90, -88]`,
 * exactly one reproduces the observed output bit pattern, and it is this one.
 * Clamp constants are XGBoost internals, are version-sensitive in the way
 * `weight_drop` proved to be, and fall under the version ceiling (D018).
 */
export const SIGMOID_MARGIN_FLOOR = float32FromBits(0xc2b16666); // -88.69999694824219

/**
 * The value `sigmoidF32` returns at and below {@link SIGMOID_MARGIN_FLOOR}.
 *
 * A float32 subnormal, and never `0.0` — measured on 2056 below-floor rows
 * spanning margins from `-88.71` down to `-748.25`, every one of which
 * returned this single bit pattern. Present as a named constant so a reader
 * can see the specified value, but it is not used to short-circuit: the floor
 * is applied to the margin and the transform then produces this value through
 * the same arithmetic as every other input.
 */
export const SIGMOID_FLOOR_OUTPUT = float32FromBits(0x0020bd47); // 3.006635794144578e-39

/**
 * Bounds on the integer power of two factored out of the margin.
 *
 * Beyond them the result is decided without further arithmetic. Both bounds
 * sit far outside the range where the result is finite and nonzero, so no
 * interesting input reaches them: the upper corresponds to a margin above
 * 175.7, where the exponential is 1e76 and float32 has long since overflowed,
 * and the lower to a margin below -172.9, where it is 3e-76 and float32 has
 * long since underflowed to zero.
 *
 * Their real job is the splitting rule in {@link expF32}. Within these bounds,
 * halving the power leaves both halves inside `[-125, 127]`, so both scaling
 * factors are normal float32 values and the first of the two scalings can
 * neither overflow nor underflow — making it exact, and leaving the second
 * scaling as the single rounding that produces the returned value.
 */
const MAX_POWER = 254;
const MIN_POWER = -250;

/**
 * The constants above, exposed so the companion test can assert each one's bit
 * pattern against the integer the Python side pins.
 *
 * Not part of the prediction path and not intended for consumers: a decimal
 * string parsed independently by two languages is exactly the failure mode
 * these bit patterns exist to prevent, so the assertion has to be able to
 * reach them.
 */
export const TRANSFORM_CONSTANTS: Readonly<Record<string, number>> = Object.freeze({
  INV_LN2,
  LN2_HI,
  LN2_LO,
  ROUND_MAGIC,
  C2,
  C3,
  C4,
  C5,
  C6,
  C7,
  ZERO,
  ONE,
  POS_INF,
  NEG_INF,
  SIGMOID_MARGIN_FLOOR,
  SIGMOID_FLOOR_OUTPUT,
});

/**
 * Return two raised to `exponent` as an exactly representable float32.
 *
 * Built from the exponent field of the IEEE-754 encoding, so it is exact by
 * construction rather than by the accuracy of a general power routine — which
 * is not among the operations this module is allowed to use.
 *
 * @param exponent - An integer in `[-126, 127]`, the range over which the
 *   result is a normal float32.
 * @throws RangeError if `exponent` is outside that range. Unreachable from any
 *   margin, because the caller bounds the power first; it is an internal
 *   invariant, and it throws rather than returning a silently wrong scaling
 *   factor.
 */
function powerOfTwoF32(exponent: number): number {
  if (!(exponent >= -126 && exponent <= 127)) {
    throw new RangeError(
      `power-of-two exponent outside the normal float32 range: ${exponent} is not in [-126, 127]`,
    );
  }
  return float32FromBits((exponent + 127) << 23);
}

/**
 * Return the margin unchanged, narrowed to float32.
 *
 * The output transform for objectives whose prediction IS the margin. It
 * exists as a named function so that the transform is always a lookup in
 * {@link OUTPUT_FUNCTIONS} and never a branch that has to remember to do
 * nothing.
 *
 * Signed zero and non-finite values pass through untouched: `-0` stays `-0`
 * rather than being normalized, and a `NaN` stays a `NaN`.
 */
export function identityF32(margin: number): number {
  return Math.fround(margin);
}

/**
 * Return the exponential of `margin` under float32 semantics.
 *
 * The output transform for `survival:cox`. There is no clamp: the result
 * overflows to `+Infinity` above a margin of `88.7228` and underflows to `0`
 * in the far tail, both of which XGBoost does (measured, 45000/45000
 * bit-exact against `predict()` including 734 rows at `+inf`).
 *
 * The margin is factored as an integer power of two times a value near one.
 * The power is removed by exact scaling and the remainder is evaluated as a
 * short polynomial, so every operation is one of the four that IEEE-754
 * requires to be correctly rounded.
 *
 * @param margin - The margin. Narrowed to float32 on entry — this is the
 *   narrowing site, so a caller that passes an un-narrowed double still gets
 *   float32 semantics rather than a value that reads as correct and is a bit
 *   wrong.
 * @returns The float32 result. `NaN` in gives `NaN` out, `+Infinity` gives
 *   `+Infinity`, `-Infinity` gives `+0`.
 */
export function expF32(margin: number): number {
  const value = Math.fround(margin);

  // NaN is the one input that no comparison below can route correctly, since
  // it compares false against everything including itself.
  if (value !== value) {
    return value;
  }
  if (value === POS_INF) {
    return POS_INF;
  }
  if (value === NEG_INF) {
    return ZERO;
  }

  // Nearest integer to the margin divided by the logarithm of two.
  const quotient = Math.fround(value * INV_LN2);
  const shifted = Math.fround(quotient + ROUND_MAGIC);
  const powerF32 = Math.fround(shifted - ROUND_MAGIC);

  // Compared as floats, before the conversion to an integer, so that an
  // infinite quotient is decided here instead of truncating to nonsense there.
  if (powerF32 > Math.fround(MAX_POWER)) {
    return POS_INF;
  }
  if (powerF32 < Math.fround(MIN_POWER)) {
    return ZERO;
  }
  // `| 0` truncates toward zero, which is exact: `powerF32` is an integer
  // value by construction and the two comparisons above bound it to
  // [-250, 254].
  const power = powerF32 | 0;

  // Argument reduction. `hiProduct` is exact because the power needs at most 8
  // bits and `LN2_HI` has at most 16 significant bits; `reducedHi` is exact by
  // Sterbenz's lemma. So the only rounding in the reduction is the last
  // subtraction, of a quantity below 4e-03.
  const hiProduct = Math.fround(powerF32 * LN2_HI);
  const reducedHi = Math.fround(value - hiProduct);
  const loProduct = Math.fround(powerF32 * LN2_LO);
  const reduced = Math.fround(reducedHi - loProduct);

  // Taylor series for the reduced argument, one operation per statement.
  const poly7 = Math.fround(reduced * C7);
  const poly6 = Math.fround(C6 + poly7);
  const poly6r = Math.fround(reduced * poly6);
  const poly5 = Math.fround(C5 + poly6r);
  const poly5r = Math.fround(reduced * poly5);
  const poly4 = Math.fround(C4 + poly5r);
  const poly4r = Math.fround(reduced * poly4);
  const poly3 = Math.fround(C3 + poly4r);
  const poly3r = Math.fround(reduced * poly3);
  const poly2 = Math.fround(C2 + poly3r);
  const squared = Math.fround(reduced * reduced);
  const tail = Math.fround(squared * poly2);
  const fraction = Math.fround(reduced + tail);
  const reducedExp = Math.fround(ONE + fraction);

  // Restore the power of two in two exact halves. The first scaling cannot
  // overflow or underflow — the reduced result lies within [0.7071, 1.4143]
  // and each half of the power lies in [-125, 127] — so it is exact, and the
  // second scaling is the single rounding that produces the returned value,
  // including when that value is subnormal or infinite.
  //
  // `>> 1` is an arithmetic shift, so it floors rather than truncating: it
  // matches Python's `//` on negative powers, where truncation would not.
  const lowExponent = power >> 1;
  const highExponent = power - lowExponent;
  const scaledOnce = Math.fround(reducedExp * powerOfTwoF32(lowExponent));
  return Math.fround(scaledOnce * powerOfTwoF32(highExponent));
}

/**
 * Return the logistic function of `margin` under float32 semantics.
 *
 * The output transform for `binary:logistic`, reproducing XGBoost's measured
 * floor: a margin below {@link SIGMOID_MARGIN_FLOOR} is raised to it before
 * the transform, so the result is exactly {@link SIGMOID_FLOOR_OUTPUT} and
 * never `0`. Whether XGBoost clamps the input or floors the output is
 * observationally identical and is not resolved; the input floor is used here,
 * and it reproduces the measured output bit pattern exactly.
 *
 * Saturation at exactly `1` is reachable, from a margin of `16.635533` upward.
 * Saturation at exactly `0` is not reachable, because the floor prevents it.
 *
 * @param margin - The margin. Narrowed to float32 on entry.
 * @returns The float32 result. `NaN` in gives `NaN` out. `+Infinity` gives
 *   exactly `1`; `-Infinity` is below the floor and so gives
 *   {@link SIGMOID_FLOOR_OUTPUT}, which is what applying the measured floor to
 *   it means.
 */
export function sigmoidF32(margin: number): number {
  const value = Math.fround(margin);

  // NaN first: it compares false against the floor, so leaving it to the
  // comparison below would pass it through by accident rather than by
  // decision. It reaches the arithmetic and propagates.
  let floored: number;
  if (value !== value) {
    floored = value;
  } else if (value < SIGMOID_MARGIN_FLOOR) {
    floored = SIGMOID_MARGIN_FLOOR;
  } else {
    floored = value;
  }

  const negated = Math.fround(-floored);
  const exponential = expF32(negated);
  const denominator = Math.fround(ONE + exponential);
  return Math.fround(ONE / denominator);
}

/** The exact set of `output_transform` names FORMAT.md §5 defines. */
export type OutputTransformName = "identity" | "sigmoid" | "exp";

/**
 * The three output transforms named in the artifact format (FORMAT.md §5),
 * keyed by the exact string the artifact carries.
 *
 * A lookup of anything else yields `undefined` and the reader raises: nothing
 * defaults, and no transform is inferred from the objective by analogy
 * (D007, D028).
 */
export const OUTPUT_FUNCTIONS: Readonly<
  Record<OutputTransformName, (margin: number) => number>
> = Object.freeze({
  identity: identityF32,
  sigmoid: sigmoidF32,
  exp: expF32,
});

/** The `output_transform` names, in the order FORMAT.md §5 lists them. */
export const OUTPUT_TRANSFORM_NAMES: readonly OutputTransformName[] = Object.freeze([
  "identity",
  "sigmoid",
  "exp",
] as const);
