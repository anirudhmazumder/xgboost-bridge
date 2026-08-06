// The normative walk of FORMAT.md §10, and the input contract around it.
//
// Each measured rule gets a case that fails if the rule is broken, built so the
// *wrong* implementation produces a different bit pattern rather than a value
// that merely looks off. Where a hand-built artifact is used it is stated why:
// several of these invariants are only observable at magnitudes where float32
// addition stops being associative, which no fitted model happens to visit.
//
// Imports the BUILT bundle, never `src/` (D011).
import test from "node:test";
import assert from "node:assert/strict";

import {
  FeatureKeyMismatchError,
  MalformedArtifactError,
  NonFiniteFeatureError,
  OUTPUT_FUNCTIONS,
  Predictor,
  PredictorError,
  fromJSON,
  loadArtifact,
} from "../dist/index.js";
// The whole export surface, for the API-surface checks at the end of this file.
import * as BUNDLE from "../dist/index.js";

const SCRATCH = new DataView(new ArrayBuffer(4));

function bits32(value) {
  SCRATCH.setFloat32(0, value);
  return SCRATCH.getUint32(0);
}

function hex32(bits) {
  return `0x${(bits >>> 0).toString(16).padStart(8, "0")}`;
}

/** An artifact with one feature and one single-leaf tree per `leaves` entry. */
function stumpArtifact(intercept, leaves) {
  return {
    feature_names: ["feature_0"],
    format_version: 1,
    intercept,
    objective: "reg:squarederror",
    output_transform: "identity",
    provenance: {
      base_score: "[0E0]",
      exporter_version: "0.1.0.dev0",
      xgboost_version: "3.3.0",
    },
    trees: leaves.map((leaf) => ({
      default_left: [0],
      left_children: [-1],
      node_values: [leaf],
      right_children: [-1],
      split_indices: [0],
    })),
  };
}

/** A one-split tree: below `threshold` goes left, otherwise right. */
function splitArtifact(threshold, leftLeaf, rightLeaf, defaultLeft) {
  return {
    feature_names: ["feature_0"],
    format_version: 1,
    intercept: 0,
    objective: "reg:squarederror",
    output_transform: "identity",
    provenance: {
      base_score: "[0E0]",
      exporter_version: "0.1.0.dev0",
      xgboost_version: "3.3.0",
    },
    trees: [
      {
        default_left: [defaultLeft, 0, 0],
        left_children: [1, -1, -1],
        node_values: [threshold, leftLeaf, rightLeaf],
        right_children: [2, -1, -1],
        split_indices: [0, 0, 0],
      },
    ],
  };
}

// ---------------------------------------------------------------------------
// FORMAT.md §16, worked through
// ---------------------------------------------------------------------------

const WORKED_EXAMPLE = {
  feature_names: ["feature_a", "feature_b"],
  format_version: 1,
  intercept: 0.40546515583992004,
  objective: "binary:logistic",
  output_transform: "sigmoid",
  provenance: {
    base_score: "[6E-1]",
    exporter_version: "0.1.0.dev0",
    xgboost_version: "3.3.0",
  },
  trees: [
    {
      default_left: [1, 0, 0],
      left_children: [1, -1, -1],
      node_values: [0.5, -0.25, 0.75],
      right_children: [2, -1, -1],
      split_indices: [0, 0, 0],
    },
    {
      default_left: [0],
      left_children: [-1],
      node_values: [0.125],
      right_children: [-1],
      split_indices: [0],
    },
  ],
};

test("the worked example of FORMAT.md §16 reproduces, margin and output", () => {
  const predictor = fromJSON(WORKED_EXAMPLE);
  const row = { feature_a: 0.25, feature_b: 9.0 };

  // Step 1: the accumulator starts at the float32 intercept.
  assert.equal(predictor.intercept, Math.fround(0.40546515583992004));
  // Steps 2-3: left at the root, then the second tree's single leaf.
  const margin = predictor.margin(row);
  assert.equal(margin, Math.fround(0.28046516));
  assert.equal(bits32(margin), bits32(Math.fround(0.28046516)));
  // Step 4: the FLOAT32 logistic value the specification prints, not the
  // float64 one — both narrow to 0x3f11d541, so an implementer comparing
  // against the wrong decimal concludes they have a bug.
  const output = predictor.output(row);
  assert.equal(output, 0.5696602463722229);
  assert.equal(bits32(output), 0x3f11d541);
  assert.notEqual(output, 0.5696602593994496);
});

test("a feature no split reads must still be present in the input", () => {
  // `feature_b` is never consulted by any node of the worked example, and
  // omitting it raises rather than being treated as missing. That is the whole
  // point of D005: a missing value is legitimate model structure, so lenient
  // handling turns a typo into a confident wrong number.
  const predictor = fromJSON(WORKED_EXAMPLE);
  assert.throws(
    () => predictor.margin({ feature_a: 0.25 }),
    (error) => {
      assert.ok(error instanceof FeatureKeyMismatchError);
      assert.deepEqual([...error.missing], ["feature_b"]);
      assert.deepEqual([...error.extra], []);
      return true;
    },
  );
});

// ---------------------------------------------------------------------------
// Strict feature keys (D005)
// ---------------------------------------------------------------------------

test("the input key set must equal feature_names exactly", () => {
  const predictor = fromJSON(WORKED_EXAMPLE);
  const good = { feature_a: 0.25, feature_b: 9.0 };
  assert.ok(Number.isFinite(predictor.margin(good)));

  const cases = [
    ["missing one", { feature_a: 0.25 }, ["feature_b"], []],
    ["missing both", {}, ["feature_a", "feature_b"], []],
    ["one extra", { ...good, feature_c: 1 }, [], ["feature_c"]],
    // A typo is one missing key and one extra key, and is diagnosed as both.
    ["a typo", { feature_a: 0.25, featureb: 9.0 }, ["feature_b"], ["featureb"]],
  ];
  for (const [description, row, missing, extra] of cases) {
    assert.throws(
      () => predictor.margin(row),
      (error) => {
        assert.ok(error instanceof FeatureKeyMismatchError, description);
        assert.deepEqual([...error.missing], missing, description);
        assert.deepEqual([...error.extra], extra, description);
        assert.ok(error instanceof PredictorError);
        return true;
      },
      description,
    );
  }
});

test("a row that is not an object of numbers raises", () => {
  const predictor = fromJSON(WORKED_EXAMPLE);
  for (const row of [null, undefined, 5, "x", [0.25, 9.0]]) {
    assert.throws(
      () => predictor.margin(row),
      PredictorError,
      `row = ${JSON.stringify(row)}`,
    );
  }
  // A numeric string is not a number. Coercing it would make `undefined`
  // coerce to NaN, which is this format's *missing value* — a silent
  // wrong-number path.
  assert.throws(
    () => predictor.margin({ feature_a: "0.25", feature_b: 9.0 }),
    MalformedArtifactError,
  );
  assert.throws(
    () => predictor.margin({ feature_a: null, feature_b: 9.0 }),
    MalformedArtifactError,
  );
});

// ---------------------------------------------------------------------------
// Missing values and infinities (D022, D045)
// ---------------------------------------------------------------------------

test("NaN is the missing value and routes by default_left, both directions", () => {
  const left = fromJSON(splitArtifact(0.5, -1, 1, 1));
  assert.equal(left.margin({ feature_0: NaN }), -1, "default_left = 1 must route left");

  const right = fromJSON(splitArtifact(0.5, -1, 1, 0));
  assert.equal(right.margin({ feature_0: NaN }), 1, "default_left = 0 must route right");

  // And the comparison path is unaffected: an ordinary value still routes by
  // the threshold whichever way `default_left` points.
  assert.equal(left.margin({ feature_0: 0.25 }), -1);
  assert.equal(right.margin({ feature_0: 0.25 }), -1);
  assert.equal(left.margin({ feature_0: 0.75 }), 1);
  assert.equal(right.margin({ feature_0: 0.75 }), 1);
});

test("infinity in the input raises, and NaN in the same input does not", () => {
  const predictor = fromJSON(WORKED_EXAMPLE);
  for (const value of [Infinity, -Infinity]) {
    for (const [name, index] of [
      ["feature_a", 0],
      ["feature_b", 1],
    ]) {
      const row = { feature_a: 0.25, feature_b: 9.0 };
      row[name] = value;
      assert.throws(
        () => predictor.margin(row),
        (error) => {
          assert.ok(error instanceof NonFiniteFeatureError);
          assert.equal(error.index, index);
          assert.equal(error.feature, name);
          assert.equal(error.value, value);
          return true;
        },
        `${name} = ${value}`,
      );
    }
  }
  // NaN is accepted at every position.
  assert.ok(Number.isFinite(predictor.margin({ feature_a: NaN, feature_b: NaN })));
});

test("the infinity check covers the whole row, not only the columns a node reads", () => {
  // `feature_b` is read by no split of the worked example. A lazy check would
  // let this row through, making the same invalid input raise or not depending
  // on the model rather than on the input (D045).
  const predictor = fromJSON(WORKED_EXAMPLE);
  assert.throws(
    () => predictor.margin({ feature_a: 0.25, feature_b: Infinity }),
    (error) => error instanceof NonFiniteFeatureError && error.feature === "feature_b",
  );
});

// ---------------------------------------------------------------------------
// The comparison: both sides cast, strict `<`, equality routes RIGHT
// ---------------------------------------------------------------------------

test("equality routes RIGHT, on a threshold the input hits exactly", () => {
  const predictor = fromJSON(splitArtifact(0.5, -1, 1, 0));
  assert.equal(predictor.margin({ feature_0: 0.5 }), 1, "value == threshold must go right");
  // One representable value below goes left; one above goes right.
  assert.equal(predictor.margin({ feature_0: Math.fround(0.5) - 6e-8 }), -1);
  assert.equal(predictor.margin({ feature_0: 0.5000001 }), 1);
});

test("the sample side of the comparison is cast to float32", () => {
  // The threshold is the float32 nearest 0.1, emitted at full float64 precision
  // exactly as FORMAT.md §9.1 requires. The double 0.1 is strictly *below* it,
  // so an uncast sample routes LEFT; its float32 is exactly equal to it, so a
  // cast sample routes RIGHT. One row, two different answers, and casting only
  // the threshold is what produced a 6.6-percentage-point probability error on
  // a real row.
  const threshold = Math.fround(0.1);
  assert.equal(threshold, 0.10000000149011612);
  assert.ok(0.1 < threshold, "the double 0.1 must sit below its own float32");

  const predictor = fromJSON(splitArtifact(threshold, -1, 1, 0));
  assert.equal(
    predictor.margin({ feature_0: 0.1 }),
    1,
    "f32(0.1) equals the threshold, so the row routes right",
  );
});

test("a threshold that is not float32-exact is narrowed before the comparison", () => {
  // A hand-edited or third-party artifact can carry a threshold no float32
  // represents. `16777217` narrows to `16777216`, and the sample `16777216.5`
  // narrows to `16777216` as well — equal, so RIGHT. Compared against the
  // un-narrowed float64 threshold, `16777216 < 16777217` is true and the row
  // would go LEFT. This is what the reader's `Float32Array` is for (D045).
  const predictor = fromJSON(splitArtifact(16777217, -1, 1, 0));
  assert.equal(predictor.trees[0].nodeValues[0], 16777216);
  assert.equal(predictor.margin({ feature_0: 16777216.5 }), 1);
});

// ---------------------------------------------------------------------------
// Accumulation: intercept first, artifact order, narrow after every add
// ---------------------------------------------------------------------------

// 2**24, the magnitude at which the float32 grid spacing becomes 2 and float32
// addition stops being associative. Every ordering rule below is invisible at
// ordinary magnitudes, which is exactly why a corpus of fitted models cannot
// pin them on its own.
const TWO_POW_24 = 16777216;

test("the intercept is the accumulator's initial value, not a final addend", () => {
  // Intercept first: f32(2**24 + 1) = 2**24, twice over, so the answer is 2**24.
  // Intercept last: (1 + 1) = 2, then 2**24 + 2 = 2**24 + 2, which is a
  // different bit pattern. Measured cost of getting this wrong on real models:
  // 199-2120/5000 rows bit-exact.
  const predictor = fromJSON(stumpArtifact(TWO_POW_24, [1, 1]));
  const margin = predictor.margin({ feature_0: 0 });
  assert.equal(bits32(margin), 0x4b800000, `got ${hex32(bits32(margin))}`);
  assert.equal(margin, TWO_POW_24);
  assert.notEqual(bits32(margin), bits32(TWO_POW_24 + 2));
});

test("trees are walked in artifact order, and the order changes the answer", () => {
  // Forward: 0 + 2**24 = 2**24, then +1 twice, each rounding back to 2**24.
  // Reversed: 0 + 1 + 1 = 2, then + 2**24 = 2**24 + 2. Measured cost of
  // reversing on real models: 245-2365/5000 rows bit-exact.
  const forward = fromJSON(stumpArtifact(0, [TWO_POW_24, 1, 1]));
  const reversed = fromJSON(stumpArtifact(0, [1, 1, TWO_POW_24]));
  const row = { feature_0: 0 };
  assert.equal(bits32(forward.margin(row)), 0x4b800000);
  assert.equal(bits32(reversed.margin(row)), 0x4b800001);
  assert.notEqual(bits32(forward.margin(row)), bits32(reversed.margin(row)));
});

test("the accumulator is narrowed to float32 after every single addition", () => {
  // A float64 running sum narrowed once at the end gives 2**24 + 2 here,
  // because the intermediate 2**24 + 1 survives in float64. Narrowing after
  // each add discards it, twice. Measured cost of the float64 sum on real
  // models: 318-2541/5000 rows bit-exact.
  const predictor = fromJSON(stumpArtifact(TWO_POW_24, [1, 1]));
  const margin = predictor.margin({ feature_0: 0 });
  assert.equal(bits32(margin), 0x4b800000);
  assert.notEqual(bits32(margin), bits32(Math.fround(TWO_POW_24 + 1 + 1)));
  assert.equal(bits32(Math.fround(TWO_POW_24 + 1 + 1)), 0x4b800001);
});

test("leaf values are narrowed on read, and it changes the accumulated margin", () => {
  // `16777217` is an ordinary finite double that no float32 represents. Narrowed
  // to 16777216 it disappears into the accumulator; carried as float64 it does
  // not. Measured cost of skipping the narrowing on real models: 990-3706/5000
  // rows bit-exact, breaching the 1e-6 gate at 1.07e-04.
  const predictor = fromJSON(stumpArtifact(1, [16777217]));
  assert.equal(predictor.trees[0].nodeValues[0], TWO_POW_24);
  const margin = predictor.margin({ feature_0: 0 });
  assert.equal(bits32(margin), 0x4b800000);
  assert.notEqual(bits32(margin), bits32(Math.fround(1 + 16777217)));
});

// ---------------------------------------------------------------------------
// `objective` is not operative (D028)
// ---------------------------------------------------------------------------

test("no prediction method mentions `objective` in its own source", () => {
  // The shipped bundle's function text, not the TypeScript source. A no-op
  // `if (objective === ...)` inserted into any of these fails here. The
  // whole-bundle scan in bundle.test.js covers a branch moved into a helper,
  // and the behavioural check below covers the case where the branch is
  // obfuscated past both scans.
  for (const name of ["margin", "output", "walk", "featureRow"]) {
    const method = Predictor.prototype[name];
    assert.equal(typeof method, "function", `Predictor.prototype.${name} is missing`);
    assert.ok(
      !method.toString().includes("objective"),
      `Predictor.prototype.${name} references \`objective\``,
    );
  }
});

test("output_transform alone selects the transform, for every valid pairing", () => {
  // Behavioural: the value `output` returns is bit-for-bit the transform named
  // by `output_transform` applied to the margin. Nothing about the objective
  // enters, and a substituted transform changes the bits.
  const pairs = [
    ["reg:squarederror", "identity"],
    ["binary:logistic", "sigmoid"],
    ["survival:cox", "exp"],
  ];
  for (const [objective, transform] of pairs) {
    const artifact = stumpArtifact(0.25, [0.5, -0.125]);
    artifact.objective = objective;
    artifact.output_transform = transform;
    const predictor = fromJSON(artifact);
    for (const value of [-3, 0, 0.5, 12]) {
      const row = { feature_0: value };
      const margin = predictor.margin(row);
      assert.equal(
        bits32(predictor.output(row)),
        bits32(OUTPUT_FUNCTIONS[transform](margin)),
        `${objective}/${transform} at ${value}`,
      );
    }
    assert.equal(predictor.objective, objective);
    assert.equal(predictor.outputTransform, transform);
  }
});

// ---------------------------------------------------------------------------
// The transform lookup is own-property-only, and the constructor is public
// ---------------------------------------------------------------------------

// Every name here resolves on `Object.prototype`, so on an ordinary object
// literal the lookup succeeded and handed back something that is not a
// transform. `"constructor"` was the worst of them: `output` returned a boxed
// `Number`, which serializes as the right number and does the right arithmetic
// while failing `Object.is` and any bit-pattern comparison against it. That is
// exactly the shape of wrongness this package exists to refuse.
const PROTOTYPE_CHAIN_NAMES = [
  "constructor",
  "toString",
  "valueOf",
  "hasOwnProperty",
  "__proto__",
  "isPrototypeOf",
  "propertyIsEnumerable",
  "toLocaleString",
];

test("the transform table has a null prototype, so inherited names do not resolve", () => {
  // Pins the table itself, independently of the constructor guard below: any
  // other consumer of `OUTPUT_FUNCTIONS` gets `undefined` for these names
  // rather than a function. This mirrors the Python side, where
  // `OUTPUT_FUNCTIONS` is a `MappingProxyType` whose miss raises.
  assert.equal(Object.getPrototypeOf(OUTPUT_FUNCTIONS), null);
  for (const name of PROTOTYPE_CHAIN_NAMES) {
    assert.equal(OUTPUT_FUNCTIONS[name], undefined, `OUTPUT_FUNCTIONS["${name}"] resolved`);
    assert.equal(
      Object.prototype.hasOwnProperty.call(OUTPUT_FUNCTIONS, name),
      false,
      `OUTPUT_FUNCTIONS has an own "${name}"`,
    );
  }
  // And the three real names are unaffected.
  assert.deepEqual(Object.keys(OUTPUT_FUNCTIONS).sort(), ["exp", "identity", "sigmoid"]);
  assert.ok(Object.isFrozen(OUTPUT_FUNCTIONS));
});

test("the public constructor refuses an outputTransform it does not implement", () => {
  // Pins the guard, independently of the table's prototype: `Predictor` and its
  // constructor are exported, and `fromJSON` is not the only door. Unlike
  // `fromJSON` this path receives an already-loaded artifact, so the guard
  // covers the one field the constructor actually uses.
  const loaded = loadArtifact(WORKED_EXAMPLE);
  for (const name of [...PROTOTYPE_CHAIN_NAMES, "softplus", "", "identity ", "IDENTITY"]) {
    assert.throws(
      () => new Predictor({ ...loaded, outputTransform: name }),
      (error) => {
        assert.ok(error instanceof MalformedArtifactError, `${name}: wrong error type`);
        assert.equal(error.code, "MALFORMED_ARTIFACT");
        assert.equal(error.field, "output_transform");
        assert.equal(error.value, name);
        return true;
      },
      `outputTransform "${name}" was accepted`,
    );
  }
  for (const value of [undefined, null, 0, 1, {}, [], Symbol.iterator]) {
    assert.throws(
      () => new Predictor({ ...loaded, outputTransform: value }),
      MalformedArtifactError,
      `outputTransform ${String(value)} was accepted`,
    );
  }
});

test("the three transforms FORMAT.md §5 defines still work through the constructor", () => {
  const loaded = loadArtifact(WORKED_EXAMPLE);
  const row = { feature_a: 0.25, feature_b: 9.0 };
  for (const name of ["identity", "sigmoid", "exp"]) {
    const predictor = new Predictor({ ...loaded, outputTransform: name });
    const margin = predictor.margin(row);
    const output = predictor.output(row);
    assert.equal(typeof output, "number", `${name} returned a ${typeof output}`);
    // `Object.is`, not `==`: a boxed `Number` passes `==` against the primitive
    // it wraps, which is how the defect read as correct.
    assert.ok(
      Object.is(output, OUTPUT_FUNCTIONS[name](margin)),
      `${name} did not return the table's own value`,
    );
    assert.equal(bits32(output), bits32(OUTPUT_FUNCTIONS[name](margin)));
  }
});

test("a rejected transform yields no object at all, so nothing wrong can be called", () => {
  // The refusal happens before any field is stored: there is no half-built
  // predictor whose `margin` works and whose `output` does not.
  const loaded = loadArtifact(WORKED_EXAMPLE);
  let built;
  try {
    built = new Predictor({ ...loaded, outputTransform: "constructor" });
  } catch (error) {
    assert.ok(error instanceof PredictorError);
  }
  assert.equal(built, undefined);
});

test("the objective is exposed for inspection and is only ever a label", () => {
  const predictor = fromJSON(WORKED_EXAMPLE);
  assert.equal(predictor.objective, "binary:logistic");
  // Overwriting the label cannot change a prediction, because no prediction
  // path reads it. (A frozen field would make this unassertable, which is why
  // it is a plain data property.)
  const row = { feature_a: 0.25, feature_b: 9.0 };
  const before = bits32(predictor.output(row));
  predictor.objective = "survival:cox";
  assert.equal(bits32(predictor.output(row)), before);
  assert.equal(bits32(predictor.margin(row)), bits32(Math.fround(0.28046516)));
});

// ---------------------------------------------------------------------------
// The API surface
// ---------------------------------------------------------------------------

test("there is no fromFile, in any spelling (D006)", () => {
  // Filesystem access is unavailable in browsers and differs across edge
  // runtimes, so a loader would either add a dependency or split the bundle by
  // runtime. Consumers do their own I/O.
  assert.equal(typeof fromJSON, "function");
  for (const name of ["fromFile", "loadFile", "readFile", "fromPath", "load"]) {
    assert.equal(Predictor[name], undefined, `Predictor.${name} must not exist`);
    assert.equal(Predictor.prototype[name], undefined, `Predictor#${name} must not exist`);
  }
});

test("the bundle exports no module-level file-reading entry point either (D006)", () => {
  // The check above covers `Predictor.fromFile` and `Predictor#fromFile`. It
  // does not cover the spelling a contributor would actually reach for — a
  // plain exported function beside `fromJSON` — and `index.ts` re-exports
  // everything `predict.ts` declares, so such a function reaches consumers
  // with nothing objecting. Measured: adding `export function fromFile` turned
  // no test red before this one existed.
  const exported = Object.keys(BUNDLE).sort();
  assert.ok(exported.includes("fromJSON"), "fromJSON is the only entry point");

  const forbidden = exported.filter((name) => /file|path|fs$|directory|disk/i.test(name));
  assert.deepEqual(
    forbidden,
    [],
    `the bundle exports a file-reading entry point: ${forbidden.join(", ")}`,
  );
  for (const name of ["fromFile", "loadFile", "readFile", "fromPath", "load", "readArtifact"]) {
    assert.equal(BUNDLE[name], undefined, `the bundle must not export ${name}`);
  }
  // The scan is only worth anything if it would fire on the real thing.
  assert.ok(/file|path|fs$|directory|disk/i.test("fromFile"));
  assert.ok(/file|path|fs$|directory|disk/i.test("loadArtifactFromPath"));
});

test("no prediction method mentions `provenance` in its own source (D015)", () => {
  // The mirror of the `objective` check above, for the other non-operative
  // field. `provenance.base_score` holds the value XGBoost stored *unclamped
  // and untransformed*, so anything derived from it on a prediction path is
  // wrong by up to 13.8 in margin space (D035). The artifact carries exactly
  // one operative numeric intercept and this is not it.
  for (const name of ["margin", "output", "walk", "featureRow"]) {
    const method = Predictor.prototype[name];
    assert.equal(typeof method, "function", `Predictor.prototype.${name} is missing`);
    for (const token of ["provenance", "base_score", "exporter_version", "xgboost_version"]) {
      assert.ok(
        !method.toString().includes(token),
        `Predictor.prototype.${name} references \`${token}\``,
      );
    }
  }

  // And the behavioural half, since a source scan can be evaded: the block is
  // replaced after loading, so its load-time validation still ran on the real
  // value, and every number must be bit-identical afterwards.
  const predictor = fromJSON(WORKED_EXAMPLE);
  const row = { feature_a: 0.25, feature_b: 9.0 };
  const before = [bits32(predictor.margin(row)), bits32(predictor.output(row))];
  Object.defineProperty(predictor, "provenance", {
    value: { base_score: "corrupted", exporter_version: "corrupted", xgboost_version: "corrupted" },
    configurable: true,
  });
  assert.deepEqual([bits32(predictor.margin(row)), bits32(predictor.output(row))], before);
});

test("fromJSON takes a parsed object and never a string or a path", () => {
  assert.throws(() => fromJSON(JSON.stringify(WORKED_EXAMPLE)), PredictorError);
  assert.throws(() => fromJSON("./artifact.json"), PredictorError);
  // And the ordinary path: exactly what `JSON.parse` returns.
  const predictor = fromJSON(JSON.parse(JSON.stringify(WORKED_EXAMPLE)));
  assert.equal(bits32(predictor.margin({ feature_a: 0.25, feature_b: 9.0 })), 0x3e8f9921);
});
