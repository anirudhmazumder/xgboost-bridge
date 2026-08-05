// The whole fixture corpus, ordinary and adversarial, replayed through the
// shipped bundle.
//
// The oracle is **XGBoost's own `predict()` output**, recorded into each fixture
// at generation time by `fixtures/generate/`. This file never recomputes it and
// never consults the Python implementation, which is why the JavaScript side
// needs no XGBoost and no Python interpreter: the corpus is the whole point.
//
// Ground truth is uint32 hex bit patterns (D044), and the comparison is on those
// bit patterns rather than on `==`. JSON has no representation for `+inf`, which
// `survival:cox` genuinely returns above a margin of about 88.72, and a decimal
// ground truth invites `==` under which `-0.0 == 0.0` is true for two values
// that are not the same.
//
// Imports the BUILT bundle, never `src/` (D011).
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { NonFiniteFeatureError, OUTPUT_FUNCTIONS, fromJSON } from "../dist/index.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const CORPUS_DIR = path.resolve(HERE, "..", "..", "..", "fixtures", "corpus");

const SCRATCH = new DataView(new ArrayBuffer(4));

function bits32(value) {
  SCRATCH.setFloat32(0, value);
  return SCRATCH.getUint32(0);
}

function f32FromBits(bits) {
  SCRATCH.setUint32(0, bits >>> 0);
  return SCRATCH.getFloat32(0);
}

function hex32(bits) {
  return `0x${(bits >>> 0).toString(16).padStart(8, "0")}`;
}

function parseHex32(token) {
  assert.match(token, /^0x[0-9a-f]{8}$/, `not a uint32 hex bit pattern: ${token}`);
  return Number.parseInt(token.slice(2), 16);
}

function loadDirectory(directory, prefix) {
  const out = [];
  for (const name of readdirSync(directory).sort()) {
    if (!name.endsWith(".json")) {
      continue;
    }
    const fixture = JSON.parse(readFileSync(path.join(directory, name), "utf8"));
    out.push([`${prefix}${name.slice(0, -5)}`, fixture]);
  }
  return out;
}

/**
 * Every fixture, ordinary then adversarial.
 *
 * `fixtures/corpus/reference/` is a sibling directory holding the `mpmath`
 * reference table, not fixtures, and is not walked: the listing here is
 * non-recursive by construction.
 */
const FIXTURES = [
  ...loadDirectory(CORPUS_DIR, ""),
  ...loadDirectory(path.join(CORPUS_DIR, "adversarial"), "adversarial/"),
];

/** Fixtures whose recorded behaviour is a refusal, so they carry no values. */
const REFUSAL_FIXTURES = FIXTURES.filter(([, fx]) => fx.meta.expected_behavior === "raise");

/** Fixtures carrying numeric ground truth. */
const VALUE_FIXTURES = FIXTURES.filter(([, fx]) => fx.meta.expected_behavior !== "raise");

/**
 * Turn a fixture row into a feature mapping.
 *
 * `null` is the missing value and becomes `NaN`. The strings `"inf"` and
 * `"-inf"` are the refusal fixture's input-side encoding, since standard JSON
 * has no literal for either (D044 addendum) — and reading them back as
 * infinities is what makes that fixture test the refusal rather than a string.
 */
function rowToInput(featureNames, row) {
  const input = {};
  for (let index = 0; index < featureNames.length; index += 1) {
    const value = row[index];
    if (value === null) {
      input[featureNames[index]] = NaN;
    } else if (value === "inf") {
      input[featureNames[index]] = Infinity;
    } else if (value === "-inf") {
      input[featureNames[index]] = -Infinity;
    } else {
      assert.equal(typeof value, "number", `unexpected row encoding: ${JSON.stringify(value)}`);
      input[featureNames[index]] = value;
    }
  }
  return input;
}

test("the corpus is present, non-trivial, and carries both kinds of fixture", () => {
  // A suite that silently reads an empty corpus passes exactly like a correct
  // implementation does.
  assert.ok(FIXTURES.length >= 20, `only ${FIXTURES.length} fixtures found under ${CORPUS_DIR}`);
  assert.ok(VALUE_FIXTURES.length >= 20);
  assert.equal(REFUSAL_FIXTURES.length, 1);
  const rows = VALUE_FIXTURES.reduce((total, [, fx]) => total + fx.rows.length, 0);
  assert.ok(rows >= 250, `only ${rows} value-bearing rows in the corpus`);
});

test("margin reproduces XGBoost's recorded margin bit-for-bit on every corpus row", () => {
  const mismatches = [];
  let rows = 0;
  for (const [name, fixture] of VALUE_FIXTURES) {
    const predictor = fromJSON(fixture.artifact);
    const featureNames = fixture.artifact.feature_names;
    for (let index = 0; index < fixture.rows.length; index += 1) {
      rows += 1;
      const input = rowToInput(featureNames, fixture.rows[index]);
      const got = bits32(predictor.margin(input));
      const want = parseHex32(fixture.expected_margin[index]);
      if (got !== want) {
        mismatches.push(`${name}[${index}]: want ${hex32(want)}, got ${hex32(got)}`);
      }
    }
  }
  assert.deepEqual(
    mismatches,
    [],
    `margin disagreed with XGBoost on ${mismatches.length}/${rows} rows`,
  );
  assert.ok(rows >= 250, `only ${rows} rows compared`);
});

// The exact set of rows where this package's output differs from XGBoost's, and
// nothing more. Every one is a `libm` difference inside the bundled exponential,
// expected by construction: bit-exactness with XGBoost at the output is
// unreachable because XGBoost's own `expf` is not correctly rounded, and an
// mpmath-exact reference scores 1600/2500 against it.
//
// Pinned as a set rather than absorbed into a tolerance, matching what D047
// recorded for the Python side. Movement in *either* direction fails, including
// an improvement, which keeps the gate a tripwire rather than a band a future
// defect could hide inside. All six fall on the two objectives that use the
// bundled exponential; the `identity` fixtures diverge on nothing, which is the
// right shape for the finding.
const EXPECTED_OUTPUT_DIVERGENCES = [
  "binary_logistic_base_score_low_inside_clamp[2]",
  "single_feature_model[0]",
  "survival_cox_base_score_low[1]",
  "survival_cox_base_score_low[2]",
  "survival_cox_overflow_to_infinity[0]",
  "survival_cox_overflow_to_infinity[8]",
];

/** FORMAT.md §5.7 / D033: the output gate is RELATIVE, and it is `1e-6`. */
const OUTPUT_RELATIVE_GATE = 1e-6;

test("output matches XGBoost except on an exactly pinned set of libm differences", () => {
  const divergences = [];
  let worst = { relative: -1, where: "none" };
  let rows = 0;
  let exact = 0;

  for (const [name, fixture] of VALUE_FIXTURES) {
    const predictor = fromJSON(fixture.artifact);
    const featureNames = fixture.artifact.feature_names;
    for (let index = 0; index < fixture.rows.length; index += 1) {
      rows += 1;
      const where = `${name}[${index}]`;
      const input = rowToInput(featureNames, fixture.rows[index]);
      const got = predictor.output(input);
      const wantBits = parseHex32(fixture.expected_output[index]);
      const want = f32FromBits(wantBits);

      // NaN is always a failure, on either side, with no exception: it compares
      // unequal to everything including itself, so a naive harness silently
      // *skips* exactly these rows.
      assert.ok(!Number.isNaN(got), `${where}: this package produced NaN`);
      assert.ok(!Number.isNaN(want), `${where}: ground truth is NaN`);

      if (bits32(got) === wantBits) {
        exact += 1;
        continue;
      }

      // Infinity must match as a bit pattern and is never divided.
      assert.ok(
        Number.isFinite(got) && Number.isFinite(want),
        `${where}: infinity mismatch, want ${hex32(wantBits)}, got ${hex32(bits32(got))}`,
      );
      // Where XGBoost's value is zero, bit equality is required rather than a
      // ratio — and it already failed above, so reaching here is a defect.
      assert.notEqual(want, 0, `${where}: ground truth is zero and the bits differ`);

      const relative = Math.abs((got - want) / want);
      divergences.push(where);
      if (relative > worst.relative) {
        worst = { relative, where };
      }
      assert.ok(
        relative <= OUTPUT_RELATIVE_GATE,
        `${where}: relative error ${relative} exceeds ${OUTPUT_RELATIVE_GATE}`,
      );
    }
  }

  assert.deepEqual(
    divergences.sort(),
    EXPECTED_OUTPUT_DIVERGENCES,
    `the pinned divergence set moved; max relative error ${worst.relative} at ${worst.where}`,
  );
  assert.equal(exact, rows - EXPECTED_OUTPUT_DIVERGENCES.length);
  assert.ok(worst.relative < 1e-7, `max relative error ${worst.relative} at ${worst.where}`);
});

test("output is the transform named by output_transform, on every corpus row", () => {
  // Behavioural confirmation that the transform is selected by
  // `output_transform` and by nothing else (D028), over the whole corpus rather
  // than a hand-built case.
  for (const [name, fixture] of VALUE_FIXTURES) {
    const predictor = fromJSON(fixture.artifact);
    const transform = OUTPUT_FUNCTIONS[fixture.artifact.output_transform];
    assert.equal(typeof transform, "function", `${name}: unknown output_transform`);
    const featureNames = fixture.artifact.feature_names;
    for (let index = 0; index < fixture.rows.length; index += 1) {
      const input = rowToInput(featureNames, fixture.rows[index]);
      assert.equal(
        bits32(predictor.output(input)),
        bits32(transform(predictor.margin(input))),
        `${name}[${index}]`,
      );
    }
  }
});

test("the refusal fixture raises on every one of its rows", () => {
  // No numeric ground truth is recorded for it by design: upstream is
  // inconsistent about infinite inputs — it raises through `DMatrix` and treats
  // them as ordinary comparable values through `inplace_predict` — so XGBoost's
  // own `predict()` is not an oracle here. This package picks one behaviour and
  // this fixture pins it (D022, D045).
  assert.equal(REFUSAL_FIXTURES.length, 1);
  for (const [name, fixture] of REFUSAL_FIXTURES) {
    const predictor = fromJSON(fixture.artifact);
    const featureNames = fixture.artifact.feature_names;
    assert.ok(fixture.rows.length >= 10, `${name}: only ${fixture.rows.length} rows`);
    for (let index = 0; index < fixture.rows.length; index += 1) {
      const input = rowToInput(featureNames, fixture.rows[index]);
      // The row must actually contain an infinity, or the fixture is testing
      // nothing and the refusal below would be about something else.
      assert.ok(
        Object.values(input).some((value) => value === Infinity || value === -Infinity),
        `${name}[${index}] carries no infinity`,
      );
      assert.throws(
        () => predictor.margin(input),
        NonFiniteFeatureError,
        `${name}[${index}] must raise from margin`,
      );
      assert.throws(
        () => predictor.output(input),
        NonFiniteFeatureError,
        `${name}[${index}] must raise from output`,
      );
    }
    // Ground truth is absent, and that is recorded rather than inferred.
    assert.equal(fixture.meta.ground_truth, "none");
    assert.ok(fixture.expected_margin.every((value) => value === null));
    assert.ok(fixture.expected_output.every((value) => value === null));
  }
});

// ---------------------------------------------------------------------------
// The specific corpus properties the invariants turn on
// ---------------------------------------------------------------------------

function fixture(name) {
  const found = FIXTURES.find(([candidate]) => candidate === name);
  assert.ok(found, `fixture ${name} is missing from the corpus`);
  return found[1];
}

test("the signed-zero fixture keeps -0 all the way to the output", () => {
  // The only configuration in which `-0` survives to the output rather than
  // being absorbed by the first addition: zero trees, base_score = 0.5 passed
  // explicitly for `binary:logistic`.
  const fx = fixture("binary_logistic_signed_zero");
  const predictor = fromJSON(fx.artifact);
  assert.equal(bits32(predictor.intercept), 0x80000000);
  assert.equal(predictor.trees.length, 0);
  const featureNames = fx.artifact.feature_names;
  for (let index = 0; index < fx.rows.length; index += 1) {
    const input = rowToInput(featureNames, fx.rows[index]);
    assert.equal(bits32(predictor.margin(input)), 0x80000000, `row ${index}`);
    assert.equal(parseHex32(fx.expected_margin[index]), 0x80000000);
    // `-0 === 0` is true; the bit patterns are what distinguish them.
    assert.ok(predictor.margin(input) === 0);
    assert.ok(Object.is(predictor.margin(input), -0));
  }
});

test("the Cox overflow fixture reproduces +Infinity as a bit pattern", () => {
  const fx = fixture("survival_cox_overflow_to_infinity");
  const predictor = fromJSON(fx.artifact);
  const featureNames = fx.artifact.feature_names;
  let infinite = 0;
  for (let index = 0; index < fx.rows.length; index += 1) {
    if (parseHex32(fx.expected_output[index]) !== 0x7f800000) {
      continue;
    }
    const input = rowToInput(featureNames, fx.rows[index]);
    assert.equal(bits32(predictor.output(input)), 0x7f800000, `row ${index}`);
    infinite += 1;
  }
  assert.ok(infinite > 0, "the fixture records no +inf row");
  assert.ok(infinite < fx.rows.length, "every row is +inf, so the boundary is invisible");
});

test("the logistic clamp fixture reproduces the floor and is never 0", () => {
  const fx = fixture("adversarial/logistic_clamp_floor_output");
  const predictor = fromJSON(fx.artifact);
  const featureNames = fx.artifact.feature_names;
  let floored = 0;
  for (let index = 0; index < fx.rows.length; index += 1) {
    if (parseHex32(fx.expected_output[index]) !== 0x0020bd47) {
      continue;
    }
    const input = rowToInput(featureNames, fx.rows[index]);
    const output = predictor.output(input);
    assert.equal(bits32(output), 0x0020bd47, `row ${index}`);
    assert.notEqual(output, 0);
    assert.notEqual(bits32(output), 0x00000000);
    assert.notEqual(bits32(output), 0x80000000);
    floored += 1;
  }
  assert.ok(floored > 0, "the fixture records no below-floor row");
});

test("the pruned fixture walks past its neutralized dead slots", () => {
  // A neutralized dead node is indistinguishable from a leaf carrying 0, so the
  // reader must accept it and the walk must never visit it. A pruned-model
  // fixture that a broken walk still passes would be decorative, which is why
  // the corpus builds one where a dead node is the target of a live node's
  // stale link.
  for (const name of ["gamma_pruned_dead_nodes", "adversarial/gamma_pruned_neutralization"]) {
    const fx = fixture(name);
    const predictor = fromJSON(fx.artifact);
    const dead = fx.meta.dead_node_indices_per_tree;
    assert.ok(dead, `${name}: no dead-node indices recorded`);
    assert.ok(
      dead.some((indices) => indices.length > 0),
      `${name}: no tree has a dead node, so the fixture tests nothing`,
    );
    const featureNames = fx.artifact.feature_names;
    for (let index = 0; index < fx.rows.length; index += 1) {
      const input = rowToInput(featureNames, fx.rows[index]);
      assert.equal(
        bits32(predictor.margin(input)),
        parseHex32(fx.expected_margin[index]),
        `${name}[${index}]`,
      );
    }
  }
});

test("every fixture's artifact loads without a single refusal", () => {
  // A refusal on a fixture the exporter produced would mean the reader and the
  // writer disagree about the format, which no numeric test would surface.
  for (const [name, fx] of FIXTURES) {
    assert.doesNotThrow(() => fromJSON(fx.artifact), `${name} must load`);
    const predictor = fromJSON(fx.artifact);
    assert.equal(predictor.formatVersion, 1);
    assert.deepEqual([...predictor.featureNames], fx.artifact.feature_names);
    assert.equal(predictor.trees.length, fx.artifact.trees.length);
  }
});
