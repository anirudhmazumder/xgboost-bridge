// Imports the BUILT package, never src/. This is the point of the test: it
// exercises what a consumer actually receives — entry point, export map,
// bundler config — not the TypeScript source. See D011.
//
// Plain JavaScript on purpose. Node 20 cannot execute TypeScript, and a
// transpile step would emit a build artifact into this source directory while
// buying no type checking at all. See D014.
import test from "node:test";
import assert from "node:assert/strict";

import {
  PredictorError,
  UnsupportedObjectiveError,
  UnsupportedBoosterError,
  UnrecognizedFieldError,
  UnsupportedVersionError,
  FeatureKeyMismatchError,
} from "../dist/index.js";

const subclasses = [
  UnsupportedObjectiveError,
  UnsupportedBoosterError,
  UnrecognizedFieldError,
  UnsupportedVersionError,
  FeatureKeyMismatchError,
];

test("PredictorError is a real Error and carries a code", () => {
  const err = new PredictorError("UNSUPPORTED_OBJECTIVE", "test message");
  assert.ok(err instanceof Error);
  assert.ok(err instanceof PredictorError);
  assert.equal(err.name, "PredictorError");
  assert.equal(err.code, "UNSUPPORTED_OBJECTIVE");
  assert.equal(err.message, "test message");
});

test("every subclass exists, extends PredictorError, and instanceof holds", () => {
  for (const Ctor of subclasses) {
    assert.equal(typeof Ctor, "function");
    assert.ok(
      Object.prototype.isPrototypeOf.call(PredictorError.prototype, Ctor.prototype),
      `${Ctor.name} must extend PredictorError`,
    );
  }
});

test("error `name` survived bundling for every class", () => {
  const base = new PredictorError("UNRECOGNIZED_FIELD", "x");
  assert.equal(base.name, "PredictorError");

  const named = new Map([
    [UnsupportedObjectiveError, "UnsupportedObjectiveError"],
    [UnsupportedBoosterError, "UnsupportedBoosterError"],
    [UnrecognizedFieldError, "UnrecognizedFieldError"],
    [UnsupportedVersionError, "UnsupportedVersionError"],
    [FeatureKeyMismatchError, "FeatureKeyMismatchError"],
  ]);
  for (const [Ctor, expectedName] of named) {
    let instance;
    if (Ctor === FeatureKeyMismatchError) {
      instance = new Ctor(["a"], ["b"]);
    } else if (Ctor === UnsupportedVersionError) {
      instance = new Ctor(99);
    } else {
      instance = new Ctor("bogus");
    }
    assert.equal(instance.name, expectedName);
    assert.ok(instance instanceof PredictorError);
    assert.ok(instance instanceof Error);
  }
});

test("UnsupportedObjectiveError carries the offending objective", () => {
  const err = new UnsupportedObjectiveError("multi:softmax");
  assert.equal(err.code, "UNSUPPORTED_OBJECTIVE");
  assert.equal(err.objective, "multi:softmax");
  assert.match(err.message, /multi:softmax/);
});

test("UnsupportedBoosterError carries the offending booster", () => {
  const err = new UnsupportedBoosterError("gbmystery");
  assert.equal(err.code, "UNSUPPORTED_BOOSTER");
  assert.equal(err.booster, "gbmystery");
  assert.match(err.message, /gbmystery/);
});

test("UnrecognizedFieldError carries the offending field", () => {
  const err = new UnrecognizedFieldError("trees[0].mystery_field");
  assert.equal(err.code, "UNRECOGNIZED_FIELD");
  assert.equal(err.field, "trees[0].mystery_field");
  assert.match(err.message, /mystery_field/);
});

test("UnsupportedVersionError carries the offending version verbatim", () => {
  const err = new UnsupportedVersionError(999);
  assert.equal(err.code, "UNSUPPORTED_VERSION");
  assert.equal(err.version, 999);
  assert.match(err.message, /999/);

  const missing = new UnsupportedVersionError(undefined);
  assert.equal(missing.version, undefined);
});

test("FeatureKeyMismatchError distinguishes missing keys from extra keys", () => {
  const err = new FeatureKeyMismatchError(["a", "b"], ["c"]);
  assert.equal(err.code, "FEATURE_KEY_MISMATCH");
  assert.deepEqual(err.missing, ["a", "b"]);
  assert.deepEqual(err.extra, ["c"]);
  assert.match(err.message, /a, b/);
  assert.match(err.message, /c/);

  const onlyMissing = new FeatureKeyMismatchError(["a"], []);
  assert.deepEqual(onlyMissing.missing, ["a"]);
  assert.deepEqual(onlyMissing.extra, []);

  const onlyExtra = new FeatureKeyMismatchError([], ["z"]);
  assert.deepEqual(onlyExtra.missing, []);
  assert.deepEqual(onlyExtra.extra, ["z"]);
});

test("thrown subclass instances are still catchable as the base class", () => {
  assert.throws(
    () => {
      throw new UnsupportedBoosterError("gblinear-but-broken");
    },
    (err) => err instanceof PredictorError && err instanceof Error,
  );
});
