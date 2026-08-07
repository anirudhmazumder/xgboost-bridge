// The JavaScript side of the cross-language parity gate.
//
// Reads a request on stdin, emits one JSON response on stdout, and computes
// nothing else. It is deliberately a plain emitter rather than a comparator:
// the comparison lives on the Python side so that exactly one implementation of
// it exists, and so that a bug in the comparison cannot be written twice in a
// way that agrees with itself.
//
// Four properties matter here and each is a rule rather than a preference:
//
//   1. **It imports the BUILT bundle**, `../packages/js/dist/index.js`, never
//      `../packages/js/src/`. The specifier below is a static literal so that
//      it can be read out of this file, and the URL the loader actually
//      resolved is reported back for the harness to check.
//   2. **Every value crosses the boundary as a uint32 hex bit pattern.**
//      `JSON.stringify(-0)` emits `0`, silently destroying the sign of a value
//      that is reachable through an ordinary default -- `binary:logistic` at
//      `base_score = 0.5` has intercept `-0`. `Infinity` has no JSON number
//      representation at all and serializes as `null`, and `survival:cox`
//      genuinely returns `+Infinity` above a margin of about 88.72. Bit
//      patterns carry all four of `-0`, `+inf`, `-inf` and `NaN` with no
//      special case.
//   3. **A refusal is a result, not an absence.** A row this predictor refuses
//      is reported with the error's name at both measurement points, because a
//      row silently skipped on both sides looks exactly like a row that passed.
//   4. **The inputs are reported too**, as float64 bit patterns, so the harness
//      can establish that the two walks were fed identical values before it
//      credits any parity number. Nothing else would rule out two sides
//      agreeing about two different questions.
//
// Nothing here reads the fixtures' recorded ground truth. That is the accuracy
// gate against XGBoost, which is a different gate with a different oracle, and
// mixing them is how a tolerance leaks into a comparison that has none.

import { readFileSync } from "node:fs";

// The one and only module specifier that reaches the package under test.
const BUNDLE_SPECIFIER = "../packages/js/dist/index.js";

import { Predictor, fromJSON, loadArtifact } from "../packages/js/dist/index.js";

// One 8-byte scratch buffer, reused. `DataView` is big-endian by default, which
// is the byte order the hex rendering below assumes on both sides.
const SCRATCH = new DataView(new ArrayBuffer(8));

/** The uint32 bit pattern of `value` narrowed to float32, as `"0x3f800000"`. */
function bits32(value) {
  SCRATCH.setFloat32(0, value);
  return `0x${(SCRATCH.getUint32(0) >>> 0).toString(16).padStart(8, "0")}`;
}

/** The uint64 bit pattern of `value` as a float64, 16 hex digits. */
function bits64(value) {
  SCRATCH.setFloat64(0, value);
  return `0x${SCRATCH.getBigUint64(0).toString(16).padStart(16, "0")}`;
}

/** The float32 named by a uint32 hex bit pattern. */
function float32FromBits(token) {
  if (!/^0x[0-9a-f]{8}$/.test(token)) {
    throw new Error(`not a uint32 hex bit pattern: ${token}`);
  }
  SCRATCH.setUint32(0, Number.parseInt(token.slice(2), 16) >>> 0);
  return SCRATCH.getFloat32(0);
}

/**
 * Decode one fixture row into a feature mapping, per the corpus encoding.
 *
 * `null` is the missing value and becomes `NaN`; the strings `"inf"` and
 * `"-inf"` are the refusal fixture's input encoding, since standard JSON has no
 * literal for either. Anything else is a fixture this emitter does not
 * understand, and it throws rather than coercing -- a coerced input would be
 * compared happily against the other side and mean nothing.
 */
function rowToInput(featureNames, row) {
  if (row.length !== featureNames.length) {
    throw new Error(`row has ${row.length} values against ${featureNames.length} feature names`);
  }
  const input = {};
  for (let index = 0; index < featureNames.length; index += 1) {
    const value = row[index];
    if (value === null) {
      input[featureNames[index]] = NaN;
    } else if (value === "inf") {
      input[featureNames[index]] = Infinity;
    } else if (value === "-inf") {
      input[featureNames[index]] = -Infinity;
    } else if (typeof value === "number") {
      input[featureNames[index]] = value;
    } else {
      throw new Error(`unrecognized row encoding: ${JSON.stringify(value)}`);
    }
  }
  return input;
}

/**
 * One measurement point on one row: its bit pattern, or the refusal's name.
 *
 * Every throw is caught and named. This gate compares refusals between the two
 * sides, so a refusal has to survive as data; letting it abort the run would
 * lose the one thing worth reporting, which is *which* row disagreed.
 */
function measure(predictor, point, input) {
  try {
    return { bits: bits32(predictor[point](input)), refusal: null };
  } catch (failure) {
    const name =
      failure instanceof Error && typeof failure.name === "string"
        ? failure.name
        : String(failure);
    return { bits: null, refusal: name };
  }
}

function recordRow(predictor, relabelled, featureNames, input) {
  const record = { input_bits: featureNames.map((name) => bits64(input[name])) };
  for (const [suffix, source] of [
    ["", predictor],
    ["_relabelled", relabelled],
  ]) {
    for (const point of ["margin", "output"]) {
      const measured = measure(source, point, input);
      record[`${point}${suffix}`] = measured.bits;
      record[`${point}${suffix}_refusal`] = measured.refusal;
    }
  }
  return record;
}

async function readRequest() {
  const chunks = [];
  for await (const chunk of process.stdin) {
    chunks.push(chunk);
  }
  return JSON.parse(Buffer.concat(chunks).toString("utf8"));
}

function resolvedBundleUrl() {
  // `import.meta.resolve` reports what the loader actually resolved, which is
  // stronger evidence than re-deriving the path here would be.
  if (typeof import.meta.resolve === "function") {
    return import.meta.resolve(BUNDLE_SPECIFIER);
  }
  return new URL(BUNDLE_SPECIFIER, import.meta.url).href;
}

const request = await readRequest();

const fixtures = {};
for (const { name, path } of request.fixtures) {
  const fixture = JSON.parse(readFileSync(path, "utf8"));
  const artifact = fixture.artifact;
  const predictor = fromJSON(artifact);
  // The same artifact CONSTRUCTED with a different `objective`. No prediction
  // path reads that field (D028), so every bit below must be identical to the
  // untouched predictor's -- a behavioural check on the shipped bundle, run
  // over the whole corpus rather than one hand-built row.
  // Constructed rather than mutated after load. The instance is frozen now
  // (D058), so the old `relabelled.objective = ...` throws -- and the
  // replacement is stronger regardless: mutating the field after construction
  // cannot detect a version that reads it *during* construction and caches a
  // transform, because the cache would already be built and the numbers would
  // not move. Passing the label in exercises the path where that would show.
  const relabelled = new Predictor({
    ...loadArtifact(artifact),
    objective: request.objective_overwrite,
  });
  const featureNames = artifact.feature_names;
  fixtures[name] = fixture.rows.map((row) =>
    recordRow(predictor, relabelled, featureNames, rowToInput(featureNames, row)),
  );
}

// The transport probe. Each value goes out in two encodings: the bit pattern
// this boundary uses, and the plain JSON number it deliberately does not. The
// second one is expected to be lossy, and the harness requires that loss to show
// up -- an encoding whose failure mode is never observed is an assumption.
const transport = request.transport_probe.map((sent) => {
  const value = float32FromBits(sent);
  return { sent, echoed: bits32(value), naive_number: value };
});

process.stdout.write(
  JSON.stringify({
    node_version: process.version,
    module_specifier: BUNDLE_SPECIFIER,
    module_url: resolvedBundleUrl(),
    transport,
    fixtures,
  }),
);
