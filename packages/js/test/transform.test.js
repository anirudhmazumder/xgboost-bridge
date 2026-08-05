// Validation of the bundled margin-to-output transform.
//
// The oracle is `fixtures/corpus/reference/float32_transform_reference.json`:
// correctly rounded float32 values computed with `mpmath` at 50 digits by
// `fixtures/generate/reference.py`. It shares no code with either
// implementation, so agreement with it is evidence of correctness rather than
// evidence that the port was faithful.
//
// Three oracles that would have been decorative, and are therefore not used:
//
// * The Python implementation. Two identical implementations agreeing proves
//   only that the code was written twice. Bit-identical wrong is still wrong,
//   and it is invisible to a parity harness precisely because both sides agree.
// * `Math.exp`. That is the thing being replaced, it is not correctly rounded
//   either, and V8's differs from Apple's on 9.6% of evaluations.
// * A re-derivation of this module's own recipe. An error in the recipe could
//   not make such a check fire.
//
// The clamp region is a separate case and is checked as a **predicate**, not as
// a distance: below the measured floor the implementation deliberately returns
// the floor value while an unclamped reference keeps falling, so a ULP
// comparison reports the clamp as if it were error — measured at 1,560,434 ULP
// on a first attempt at exactly this verification (D046).
//
// Reported as **max** ULP, never mean: a mean hides the one input that matters.
//
// Imports the BUILT bundle, never `src/` (D011).
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  OUTPUT_FUNCTIONS,
  SIGMOID_FLOOR_OUTPUT,
  SIGMOID_MARGIN_FLOOR,
  TRANSFORM_CONSTANTS,
  expF32,
  identityF32,
  sigmoidF32,
} from "../dist/index.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, "..", "..", "..");
const REFERENCE_PATH = path.join(
  REPO_ROOT,
  "fixtures",
  "corpus",
  "reference",
  "float32_transform_reference.json",
);

// ---------------------------------------------------------------------------
// Bit-level instrument
// ---------------------------------------------------------------------------

const SCRATCH = new DataView(new ArrayBuffer(8));

function bits32(value) {
  SCRATCH.setFloat32(0, value);
  return SCRATCH.getUint32(0);
}

function f32FromBits(bits) {
  SCRATCH.setUint32(0, bits >>> 0);
  return SCRATCH.getFloat32(0);
}

function bits64(value) {
  SCRATCH.setFloat64(0, value);
  return SCRATCH.getBigUint64(0);
}

function hex32(bits) {
  return `0x${(bits >>> 0).toString(16).padStart(8, "0")}`;
}

function parseHex32(token) {
  return Number.parseInt(token.slice(2), 16);
}

function isNan32(value) {
  const bits = bits32(value);
  return (bits & 0x7f800000) === 0x7f800000 && (bits & 0x007fffff) !== 0;
}

/**
 * Map a finite-or-infinite float32 to a monotonically ordered integer, so the
 * difference of two ordinals is a distance in ULP.
 *
 * `-0` and `0` map to the same ordinal. Signed-zero agreement is therefore
 * asserted separately, on bit patterns, and never through this.
 */
function ordinal(value) {
  const bits = bits32(value);
  if ((bits & 0x80000000) !== 0) {
    return -(bits & 0x7fffffff);
  }
  return bits;
}

/** A distance no genuine defect can reach: "these two are not comparable". */
const INCOMPARABLE = 1 << 30;

function ulpError(got, want) {
  if (isNan32(want) || isNan32(got)) {
    return isNan32(want) && isNan32(got) ? 0 : INCOMPARABLE;
  }
  const gotBits = bits32(got);
  const wantBits = bits32(want);
  if ((gotBits & 0x7fffffff) === 0x7f800000 || (wantBits & 0x7fffffff) === 0x7f800000) {
    return gotBits === wantBits ? 0 : INCOMPARABLE;
  }
  return Math.abs(ordinal(got) - ordinal(want));
}

const REFERENCE = JSON.parse(readFileSync(REFERENCE_PATH, "utf8"));

// ---------------------------------------------------------------------------
// The instrument checks itself before anything uses it
// ---------------------------------------------------------------------------

test("the bit-level instrument round-trips, signed zero included", () => {
  assert.equal(bits32(0), 0x00000000);
  assert.equal(bits32(-0), 0x80000000);
  assert.equal(bits32(1), 0x3f800000);
  assert.equal(bits32(Infinity), 0x7f800000);
  assert.equal(bits32(-Infinity), 0xff800000);
  assert.equal(f32FromBits(0x3f800000), 1);
  assert.equal(f32FromBits(0x0020bd47), 3.006635794144578e-39);
  // `-0 === 0` is true, and they are different artifacts. This is the whole
  // reason every comparison in this suite is on bit patterns.
  assert.ok(-0 === 0);
  assert.notEqual(bits32(-0), bits32(0));
  assert.equal(hex32(bits32(-0)), "0x80000000");
  assert.equal(parseHex32("0x0020bd47"), 0x0020bd47);
});

test("ulpError refuses to score infinity or NaN as one ULP", () => {
  // Without these two guards a saturation defect reads as a one-ULP rounding
  // difference and passes forever.
  const largestFinite = f32FromBits(0x7f7fffff);
  assert.ok(ulpError(Infinity, largestFinite) > 1);
  assert.ok(ulpError(largestFinite, Infinity) > 1);
  assert.equal(ulpError(Infinity, Infinity), 0);
  assert.equal(ulpError(-Infinity, -Infinity), 0);
  assert.ok(ulpError(Infinity, -Infinity) > 1);
  assert.equal(ulpError(NaN, NaN), 0);
  assert.ok(ulpError(NaN, 1) > 1);
  assert.ok(ulpError(1, NaN) > 1);
  assert.equal(ulpError(f32FromBits(0x3f800001), 1), 1);
  assert.equal(ulpError(1, 1), 0);
  assert.ok(isNan32(NaN));
  assert.ok(!isNan32(Infinity));
  assert.equal(ordinal(f32FromBits(0x3f800001)) - ordinal(1), 1);
});

test("the reference table is present, complete, and internally consistent", () => {
  // A sweep that silently reads an empty table always passes.
  const count = REFERENCE.margin_bits.length;
  assert.ok(count >= 5000, `reference table has only ${count} points`);
  assert.equal(REFERENCE.exp_bits.length, count);
  assert.equal(REFERENCE.sigmoid_bits.length, count);
  assert.equal(REFERENCE.meta.point_count, count);
  assert.equal(REFERENCE.meta.mpmath_decimal_places, 50);
  assert.ok(REFERENCE.sigmoid_predicate_indices.length > 100);
  assert.ok(REFERENCE.nan_margin_indices.length > 0);
  assert.equal(REFERENCE.sigmoid_clamp.floor_margin_bits, "0xc2b16666");
  assert.equal(REFERENCE.sigmoid_clamp.floor_output_bits, "0x0020bd47");
  // Every boundary the format names has to be in there by label, not by hope.
  const labels = Object.values(REFERENCE.labels).join(" | ");
  for (const required of [
    "overflows to +inf",
    "underflows to zero",
    "normal into subnormal",
    "logistic clamp floor",
    "exactly 1",
    "worst known logistic case",
    "positive infinity",
    "negative infinity",
    "not a number",
    "negative zero",
  ]) {
    assert.ok(labels.includes(required), `reference table has no labelled point for ${required}`);
  }
});

// ---------------------------------------------------------------------------
// Constants: the same integers the Python side pins
// ---------------------------------------------------------------------------

// Transcribed from `packages/python/tests/test_transform.py`. Each entry is
// (float32 bit pattern, float64 bit pattern) of one constant. This is the check
// FORMAT.md §5.5 requires before anything downstream is trusted: a decimal
// string parsed independently by two languages is the failure mode the bit
// patterns exist to prevent, and it would produce a transform that is a bit
// wrong on a fraction of inputs and reads as correct everywhere else.
const PYTHON_PINNED_CONSTANT_BITS = {
  INV_LN2: [0x3fb8aa3b, 0x3ff7154760000000n],
  LN2_HI: [0x3f317200, 0x3fe62e4000000000n],
  LN2_LO: [0x35bfbe8e, 0x3eb7f7d1c0000000n],
  ROUND_MAGIC: [0x4b400000, 0x4168000000000000n],
  C2: [0x3f000000, 0x3fe0000000000000n],
  C3: [0x3e2aaaab, 0x3fc5555560000000n],
  C4: [0x3d2aaaab, 0x3fa5555560000000n],
  C5: [0x3c088889, 0x3f81111120000000n],
  C6: [0x3ab60b61, 0x3f56c16c20000000n],
  C7: [0x39500d01, 0x3f2a01a020000000n],
  ZERO: [0x00000000, 0x0000000000000000n],
  ONE: [0x3f800000, 0x3ff0000000000000n],
  POS_INF: [0x7f800000, 0x7ff0000000000000n],
  NEG_INF: [0xff800000, 0xfff0000000000000n],
  SIGMOID_MARGIN_FLOOR: [0xc2b16666, 0xc0562cccc0000000n],
  SIGMOID_FLOOR_OUTPUT: [0x0020bd47, 0x37f05ea380000000n],
};

test("every transform constant matches the integer bit pattern Python pins", () => {
  const seen = Object.keys(TRANSFORM_CONSTANTS).sort();
  const pinned = Object.keys(PYTHON_PINNED_CONSTANT_BITS).sort();
  assert.deepEqual(seen, pinned, "the two sides do not carry the same constant set");

  for (const [name, [expected32, expected64]] of Object.entries(PYTHON_PINNED_CONSTANT_BITS)) {
    const value = TRANSFORM_CONSTANTS[name];
    assert.equal(
      bits32(value),
      expected32,
      `${name}: float32 bits ${hex32(bits32(value))} != ${hex32(expected32)}`,
    );
    assert.equal(
      bits64(value),
      expected64,
      `${name}: float64 bits ${bits64(value).toString(16)} != ${expected64.toString(16)}`,
    );
    // Held as a double that is exactly float32-valued, which is what makes the
    // float64 pattern meaningful.
    if (Number.isFinite(value)) {
      assert.equal(Math.fround(value), value, `${name} is not exactly float32-valued`);
    }
  }
});

test("the exported clamp constants are the ones the constant table pins", () => {
  assert.equal(bits32(SIGMOID_MARGIN_FLOOR), 0xc2b16666);
  assert.equal(bits32(SIGMOID_FLOOR_OUTPUT), 0x0020bd47);
  assert.equal(SIGMOID_MARGIN_FLOOR, TRANSFORM_CONSTANTS.SIGMOID_MARGIN_FLOOR);
  assert.equal(SIGMOID_FLOOR_OUTPUT, TRANSFORM_CONSTANTS.SIGMOID_FLOOR_OUTPUT);
});

test("every constant also means what its name says", () => {
  // The second, independent pin. A bit-pattern check alone happily pins a
  // transcription typo; on the Python side this check is what caught one. The
  // reference values here come from the platform's own math constants and
  // factorials, which is legitimate in a test — the prohibition on
  // transcendentals is on the prediction path.
  const { INV_LN2, LN2_HI, LN2_LO, ROUND_MAGIC, C2, C3, C4, C5, C6, C7 } = TRANSFORM_CONSTANTS;

  assert.equal(INV_LN2, Math.fround(1 / Math.LN2));
  assert.equal(Math.fround(LN2_HI + LN2_LO), Math.fround(Math.LN2));
  assert.equal(ROUND_MAGIC, 1.5 * 8388608);

  const factorials = [2, 6, 24, 120, 720, 5040];
  const coefficients = [C2, C3, C4, C5, C6, C7];
  for (let index = 0; index < factorials.length; index += 1) {
    assert.equal(
      coefficients[index],
      Math.fround(1 / factorials[index]),
      `Taylor coefficient ${index + 2} is not the reciprocal of ${factorials[index]}!`,
    );
  }

  assert.equal(TRANSFORM_CONSTANTS.ZERO, 0);
  assert.equal(bits32(TRANSFORM_CONSTANTS.ZERO), 0x00000000, "ZERO must be positive zero");
  assert.equal(TRANSFORM_CONSTANTS.ONE, 1);
  assert.equal(TRANSFORM_CONSTANTS.POS_INF, Infinity);
  assert.equal(TRANSFORM_CONSTANTS.NEG_INF, -Infinity);
  assert.equal(SIGMOID_MARGIN_FLOOR, Math.fround(-88.7));
  assert.equal(SIGMOID_FLOOR_OUTPUT, 3.006635794144578e-39);
});

test("the low eight significand bits of LN2_HI are zero", () => {
  // Not cosmetic. It is what makes `power * LN2_HI` exact for every power the
  // reduction can produce, which confines all of the reduction's rounding error
  // to the single subtraction of the low part.
  assert.equal(bits32(TRANSFORM_CONSTANTS.LN2_HI) & 0xff, 0);
});

// ---------------------------------------------------------------------------
// The map, and identity
// ---------------------------------------------------------------------------

test("OUTPUT_FUNCTIONS carries exactly the three names FORMAT.md §5 defines", () => {
  assert.deepEqual(Object.keys(OUTPUT_FUNCTIONS).sort(), ["exp", "identity", "sigmoid"]);
  assert.equal(OUTPUT_FUNCTIONS.identity, identityF32);
  assert.equal(OUTPUT_FUNCTIONS.sigmoid, sigmoidF32);
  assert.equal(OUTPUT_FUNCTIONS.exp, expF32);
  assert.ok(Object.isFrozen(OUTPUT_FUNCTIONS));
  assert.equal(OUTPUT_FUNCTIONS.softplus, undefined);
});

test("identityF32 narrows and normalizes nothing", () => {
  assert.equal(bits32(identityF32(-0)), 0x80000000);
  assert.equal(bits32(identityF32(0)), 0x00000000);
  assert.equal(bits32(identityF32(Infinity)), 0x7f800000);
  assert.equal(bits32(identityF32(-Infinity)), 0xff800000);
  assert.ok(Number.isNaN(identityF32(NaN)));
  // It narrows: a double that is not float32-valued comes back as its float32.
  assert.equal(identityF32(0.1), Math.fround(0.1));
  assert.notEqual(identityF32(0.1), 0.1);
});

// ---------------------------------------------------------------------------
// Specified values at the boundaries
// ---------------------------------------------------------------------------

test("expF32 has no clamp: it overflows to +Infinity and underflows to +0", () => {
  assert.equal(bits32(expF32(0)), 0x3f800000);
  assert.equal(bits32(expF32(-0)), 0x3f800000);
  assert.equal(bits32(expF32(Infinity)), 0x7f800000);
  assert.equal(bits32(expF32(-Infinity)), 0x00000000);
  assert.ok(Number.isNaN(expF32(NaN)));

  // Measured Cox margins above which XGBoost's predict() is +inf.
  for (const margin of [88.72283935546875, 112.06353759765625, 120.70205688476562, 134.943, 212.83]) {
    assert.equal(bits32(expF32(margin)), 0x7f800000, `expF32(${margin}) should be +Infinity`);
  }
  // The float32 immediately below the overflow boundary is still finite.
  const below = f32FromBits(bits32(88.72283935546875) - 1);
  assert.notEqual(bits32(expF32(below)), 0x7f800000);
  assert.ok(Number.isFinite(expF32(below)));

  // The far tail underflows to POSITIVE zero, never negative zero.
  for (const margin of [-1000, -3.4028234663852886e38, -200]) {
    assert.equal(bits32(expF32(margin)), 0x00000000, `expF32(${margin}) should be +0`);
  }
});

test("expF32 never returns a negative value", () => {
  // A structural property a ULP figure averages away: the sign bit of the
  // result must be clear for every input, because the exponential is positive
  // and a sign flip in the reconstruction would otherwise be a plausible
  // wrong number.
  const count = REFERENCE.margin_bits.length;
  for (let index = 0; index < count; index += 1) {
    const margin = f32FromBits(parseHex32(REFERENCE.margin_bits[index]));
    if (Number.isNaN(margin)) {
      continue;
    }
    assert.equal(
      bits32(expF32(margin)) & 0x80000000,
      0,
      `expF32(${margin}) has the sign bit set`,
    );
  }
});

test("sigmoidF32 floors at the measured margin and is never 0", () => {
  // The floor bit pattern, and the fact that 0 is unreachable. Saturation of
  // the logistic at exactly 0 is not producible by XGBoost — the clamp
  // prevents it — so a check demanding it would be testing nothing.
  assert.equal(bits32(sigmoidF32(SIGMOID_MARGIN_FLOOR)), 0x0020bd47);
  assert.equal(sigmoidF32(SIGMOID_MARGIN_FLOOR), 3.006635794144578e-39);

  const belowFloor = [
    f32FromBits(bits32(SIGMOID_MARGIN_FLOOR) + 1), // one representable value further down
    -88.71,
    -110.90191650390625,
    -204.50521850585938,
    -748.246337890625,
    -3.4028234663852886e38,
    -Infinity,
  ];
  for (const margin of belowFloor) {
    assert.equal(
      bits32(sigmoidF32(margin)),
      0x0020bd47,
      `sigmoidF32(${margin}) should be the floor value`,
    );
    assert.notEqual(bits32(sigmoidF32(margin)), 0x00000000);
    assert.notEqual(bits32(sigmoidF32(margin)), 0x80000000);
    assert.notEqual(sigmoidF32(margin), 0);
  }

  // Just above the floor the transform is doing arithmetic, not returning the
  // constant — otherwise the "floor" would be a short-circuit over a region.
  const aboveFloor = f32FromBits(bits32(SIGMOID_MARGIN_FLOOR) - 1);
  assert.ok(aboveFloor > SIGMOID_MARGIN_FLOOR);
  assert.notEqual(bits32(sigmoidF32(aboveFloor)), 0x0020bd47);
});

test("sigmoidF32 saturates at exactly 1, which is reachable", () => {
  assert.equal(bits32(sigmoidF32(16.63553237915039)), 0x3f800000);
  assert.equal(bits32(sigmoidF32(17)), 0x3f800000);
  assert.equal(bits32(sigmoidF32(386.6369323730469)), 0x3f800000);
  assert.equal(bits32(sigmoidF32(3.4028234663852886e38)), 0x3f800000);
  assert.equal(bits32(sigmoidF32(Infinity)), 0x3f800000);
  // The float32 immediately below is not yet saturated, so the boundary is
  // where it is claimed to be.
  const below = f32FromBits(bits32(16.63553237915039) - 1);
  assert.notEqual(bits32(sigmoidF32(below)), 0x3f800000);
});

test("sigmoidF32 at zero is exactly one half, from either signed zero", () => {
  assert.equal(bits32(sigmoidF32(0)), 0x3f000000);
  assert.equal(bits32(sigmoidF32(-0)), 0x3f000000);
  assert.ok(Number.isNaN(sigmoidF32(NaN)));
});

// ---------------------------------------------------------------------------
// The always-on ULP measurement against the mpmath table
// ---------------------------------------------------------------------------

// Max ULP each function is held to. The Python side measured the same two
// figures over its own 1e6-point sweeps (D046). A materially worse number here
// means the port diverged and is a defect to diagnose, not a bound to raise.
const EXP_MAX_ULP = 1;
const SIGMOID_MAX_ULP = 2;

function measure(fn, wantColumn, skipIndices) {
  const count = REFERENCE.margin_bits.length;
  const histogram = new Map();
  let worst = { ulp: -1, margin: NaN, got: NaN, want: NaN, index: -1 };
  let measured = 0;
  for (let index = 0; index < count; index += 1) {
    if (skipIndices.has(index)) {
      continue;
    }
    const margin = f32FromBits(parseHex32(REFERENCE.margin_bits[index]));
    const got = fn(margin);
    const want = f32FromBits(parseHex32(REFERENCE[wantColumn][index]));
    const error = ulpError(got, want);
    histogram.set(error, (histogram.get(error) ?? 0) + 1);
    measured += 1;
    if (error > worst.ulp) {
      worst = { ulp: error, margin, got, want, index };
    }
  }
  return { measured, histogram, worst };
}

test("expF32 max ULP against the mpmath reference", () => {
  const nanIndices = new Set(REFERENCE.nan_margin_indices);
  const result = measure(expF32, "exp_bits", nanIndices);
  assert.ok(result.measured >= 5000, `only ${result.measured} points measured`);
  assert.ok(
    result.worst.ulp <= EXP_MAX_ULP,
    `expF32 max ULP ${result.worst.ulp} over ${result.measured} points at margin ` +
      `${result.worst.margin} (bits ${hex32(bits32(result.worst.margin))}): got ` +
      `${result.worst.got}, want ${result.worst.want}; histogram ` +
      `${JSON.stringify([...result.histogram])}`,
  );
  // The sweep must have found *some* rounding difference, or it is not
  // exercising the arithmetic and its silence means nothing.
  assert.ok(
    (result.histogram.get(1) ?? 0) > 0,
    "no one-ULP point found at all; the sweep is not reaching the arithmetic",
  );
});

test("sigmoidF32 max ULP against the mpmath reference, above the clamp floor", () => {
  const skip = new Set([
    ...REFERENCE.nan_margin_indices,
    ...REFERENCE.sigmoid_predicate_indices,
  ]);
  const result = measure(sigmoidF32, "sigmoid_bits", skip);
  assert.ok(result.measured >= 4000, `only ${result.measured} points measured`);
  assert.ok(
    result.worst.ulp <= SIGMOID_MAX_ULP,
    `sigmoidF32 max ULP ${result.worst.ulp} over ${result.measured} points at margin ` +
      `${result.worst.margin} (bits ${hex32(bits32(result.worst.margin))}): got ` +
      `${result.worst.got}, want ${result.worst.want}; histogram ` +
      `${JSON.stringify([...result.histogram])}`,
  );
  assert.ok(
    (result.histogram.get(2) ?? 0) > 0,
    "the sweep never reached a two-ULP point, so it cannot have measured the maximum",
  );
});

test("below the clamp floor the check is a predicate, and it holds on every point", () => {
  const floorBits = parseHex32(REFERENCE.sigmoid_clamp.floor_output_bits);
  let checked = 0;
  for (const index of REFERENCE.sigmoid_predicate_indices) {
    const margin = f32FromBits(parseHex32(REFERENCE.margin_bits[index]));
    const got = sigmoidF32(margin);
    assert.equal(
      bits32(got),
      floorBits,
      `sigmoidF32(${margin}) is ${hex32(bits32(got))}, not the floor ${hex32(floorBits)}`,
    );
    assert.notEqual(got, 0);
    checked += 1;
  }
  assert.ok(checked > 100, `only ${checked} below-floor points in the table`);
});

test("a NaN margin gives a NaN from every transform, on every tabulated point", () => {
  let checked = 0;
  for (const index of REFERENCE.nan_margin_indices) {
    const margin = f32FromBits(parseHex32(REFERENCE.margin_bits[index]));
    assert.ok(Number.isNaN(margin), "the table flagged a non-NaN margin as NaN");
    assert.ok(Number.isNaN(expF32(margin)));
    assert.ok(Number.isNaN(sigmoidF32(margin)));
    assert.ok(Number.isNaN(identityF32(margin)));
    checked += 1;
  }
  assert.ok(checked > 0);
});
