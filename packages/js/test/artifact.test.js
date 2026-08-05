// The reader's refusals (FORMAT.md §13) and its structural narrowing (§9.2).
//
// Every case below is a wrong-number path, not a limitation. This package
// exists because the standard conversion route fails silently, so anything the
// reader cannot account for raises: nothing defaults, nothing is guessed,
// nothing is skipped (D007).
//
// Two rules cut the other way and get their own tests, because getting either
// backwards rejects legitimate models:
//
// * A node unreachable from the root must NOT raise. Neutralized dead slots are
//   legitimate artifact content and a reader that rejected them would reject
//   every pruned model.
// * `"trees": []` is valid. A zero-boosting-round model serializes it, and its
//   margin is the intercept alone.
//
// Imports the BUILT bundle, never `src/` (D011).
import test from "node:test";
import assert from "node:assert/strict";

import {
  MalformedArtifactError,
  PredictorError,
  UnrecognizedFieldError,
  UnsupportedObjectiveError,
  UnsupportedVersionError,
  fromJSON,
} from "../dist/index.js";

const SCRATCH = new DataView(new ArrayBuffer(4));

function bits32(value) {
  SCRATCH.setFloat32(0, value);
  return SCRATCH.getUint32(0);
}

/** FORMAT.md §16's worked example: the base every case below mutates. */
function baseArtifact() {
  return {
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
}

/** Apply `mutate` to a fresh copy of the base artifact and return it. */
function mutated(mutate) {
  const artifact = baseArtifact();
  mutate(artifact);
  return artifact;
}

function assertRefused(artifact, ErrorClass, description) {
  assert.throws(
    () => fromJSON(artifact),
    (error) => {
      assert.ok(
        error instanceof ErrorClass,
        `${description}: expected ${ErrorClass.name}, got ${error.name}: ${error.message}`,
      );
      // Every failure this package raises is catchable as one type.
      assert.ok(error instanceof PredictorError);
      assert.ok(error instanceof Error);
      assert.ok(typeof error.code === "string" && error.code.length > 0);
      return true;
    },
    description,
  );
}

test("the base artifact loads, so every refusal below is about the mutation", () => {
  const predictor = fromJSON(baseArtifact());
  assert.equal(predictor.formatVersion, 1);
  assert.equal(predictor.objective, "binary:logistic");
  assert.equal(predictor.outputTransform, "sigmoid");
  assert.deepEqual([...predictor.featureNames], ["feature_a", "feature_b"]);
  assert.equal(predictor.trees.length, 2);
});

// ---------------------------------------------------------------------------
// format_version
// ---------------------------------------------------------------------------

test("format_version must be exactly the integer 1", () => {
  for (const version of [0, 2, -1, 1.5, "1", "1.0", true, false, null, [1], { major: 1 }]) {
    assertRefused(
      mutated((a) => {
        a.format_version = version;
      }),
      UnsupportedVersionError,
      `format_version = ${JSON.stringify(version)}`,
    );
  }
});

test("format_version absent raises rather than defaulting", () => {
  assertRefused(
    mutated((a) => {
      delete a.format_version;
    }),
    MalformedArtifactError,
    "format_version absent",
  );
});

// ---------------------------------------------------------------------------
// Keys: unrecognized at any level, absent at any level
// ---------------------------------------------------------------------------

test("an unrecognized key raises at every level of the document", () => {
  const cases = [
    [
      "top level",
      mutated((a) => {
        a.base_score = 0.6;
      }),
      "base_score",
    ],
    [
      "provenance",
      mutated((a) => {
        a.provenance.fitted_at = "2026-08-05";
      }),
      "provenance.fitted_at",
    ],
    [
      "a tree",
      mutated((a) => {
        a.trees[0].loss_changes = [1, 2, 3];
      }),
      "trees[0].loss_changes",
    ],
  ];
  for (const [where, artifact, expectedField] of cases) {
    assert.throws(
      () => fromJSON(artifact),
      (error) => {
        assert.ok(error instanceof UnrecognizedFieldError, `${where}: got ${error.name}`);
        assert.equal(error.field, expectedField);
        return true;
      },
      `unrecognized key at ${where}`,
    );
  }
});

test("every required top-level key is required", () => {
  for (const key of [
    "feature_names",
    "intercept",
    "objective",
    "output_transform",
    "provenance",
    "trees",
  ]) {
    assertRefused(
      mutated((a) => {
        delete a[key];
      }),
      MalformedArtifactError,
      `${key} absent`,
    );
  }
});

test("every required provenance and tree key is required", () => {
  for (const key of ["base_score", "exporter_version", "xgboost_version"]) {
    assertRefused(
      mutated((a) => {
        delete a.provenance[key];
      }),
      MalformedArtifactError,
      `provenance.${key} absent`,
    );
  }
  for (const key of [
    "default_left",
    "left_children",
    "node_values",
    "right_children",
    "split_indices",
  ]) {
    assertRefused(
      mutated((a) => {
        delete a.trees[0][key];
      }),
      MalformedArtifactError,
      `trees[0].${key} absent`,
    );
  }
});

// ---------------------------------------------------------------------------
// Wrong JSON types
// ---------------------------------------------------------------------------

test("a numeric field carried as a string raises", () => {
  assertRefused(
    mutated((a) => {
      a.intercept = "0.40546515583992004";
    }),
    MalformedArtifactError,
    "intercept as a string",
  );
  assertRefused(
    mutated((a) => {
      a.trees[0].node_values[0] = "0.5";
    }),
    MalformedArtifactError,
    "a node_values entry as a string",
  );
  assertRefused(
    mutated((a) => {
      a.trees[0].left_children[0] = "1";
    }),
    MalformedArtifactError,
    "a child index as a string",
  );
});

test("a container of the wrong JSON type raises", () => {
  const cases = [
    ["the artifact itself is an array", [baseArtifact()]],
    ["the artifact itself is null", null],
    ["the artifact itself is a string", "{}"],
    [
      "feature_names is an object",
      mutated((a) => {
        a.feature_names = { 0: "feature_a" };
      }),
    ],
    [
      "trees is an object",
      mutated((a) => {
        a.trees = { 0: baseArtifact().trees[0] };
      }),
    ],
    [
      "a tree is an array",
      mutated((a) => {
        a.trees[0] = [1, 2, 3];
      }),
    ],
    [
      "provenance is an array",
      mutated((a) => {
        a.provenance = [];
      }),
    ],
    [
      "node_values is a number",
      mutated((a) => {
        a.trees[0].node_values = 0.5;
      }),
    ],
    [
      "a provenance value is a number",
      mutated((a) => {
        a.provenance.xgboost_version = 3.3;
      }),
    ],
  ];
  for (const [description, artifact] of cases) {
    assertRefused(artifact, MalformedArtifactError, description);
  }
});

test("a boolean is not an integer and not a number", () => {
  // `true == 1` in JavaScript's loose comparison, so a reader that used one
  // would accept a boolean where the format specifies an integer.
  assertRefused(
    mutated((a) => {
      a.trees[0].default_left[0] = true;
    }),
    MalformedArtifactError,
    "default_left carrying a boolean",
  );
  assertRefused(
    mutated((a) => {
      a.intercept = true;
    }),
    MalformedArtifactError,
    "intercept carrying a boolean",
  );
});

test("a non-integral child index, split index or default_left raises", () => {
  assertRefused(
    mutated((a) => {
      a.trees[0].left_children[0] = 1.5;
    }),
    MalformedArtifactError,
    "a fractional child index",
  );
  assertRefused(
    mutated((a) => {
      a.trees[0].split_indices[0] = 0.5;
    }),
    MalformedArtifactError,
    "a fractional split index",
  );
  assertRefused(
    mutated((a) => {
      a.trees[0].default_left[0] = 2;
    }),
    MalformedArtifactError,
    "default_left outside {0, 1}",
  );
});

// ---------------------------------------------------------------------------
// The enumerations and their pairing
// ---------------------------------------------------------------------------

test("objective must be one of the three the format enumerates", () => {
  for (const objective of [
    "multi:softmax",
    "multi:softprob",
    "rank:pairwise",
    "binary:logitraw",
    "BINARY:LOGISTIC",
    "binary:logistic ",
    "",
  ]) {
    assertRefused(
      mutated((a) => {
        a.objective = objective;
      }),
      UnsupportedObjectiveError,
      `objective = ${JSON.stringify(objective)}`,
    );
  }
  // No prefix matching, no case normalization, no trimming.
  assertRefused(
    mutated((a) => {
      a.objective = 7;
    }),
    MalformedArtifactError,
    "objective as a number",
  );
});

test("output_transform must be one of the three the format enumerates", () => {
  for (const transform of ["relu", "softplus", "log", "SIGMOID", ""]) {
    assertRefused(
      mutated((a) => {
        a.output_transform = transform;
      }),
      MalformedArtifactError,
      `output_transform = ${JSON.stringify(transform)}`,
    );
  }
});

test("a valid transform paired with the wrong objective raises", () => {
  // Each of these is individually legal and the pair is not. Without the
  // cross-check, an artifact could name an objective and then apply a different
  // objective's transform, which is a silent wrong number by construction.
  const mismatches = [
    ["binary:logistic", "identity"],
    ["binary:logistic", "exp"],
    ["reg:squarederror", "sigmoid"],
    ["reg:squarederror", "exp"],
    ["survival:cox", "sigmoid"],
    ["survival:cox", "identity"],
  ];
  for (const [objective, transform] of mismatches) {
    assertRefused(
      mutated((a) => {
        a.objective = objective;
        a.output_transform = transform;
      }),
      MalformedArtifactError,
      `${objective} paired with ${transform}`,
    );
  }
  // And every correct pairing loads.
  for (const [objective, transform] of [
    ["binary:logistic", "sigmoid"],
    ["reg:squarederror", "identity"],
    ["survival:cox", "exp"],
  ]) {
    const predictor = fromJSON(
      mutated((a) => {
        a.objective = objective;
        a.output_transform = transform;
      }),
    );
    assert.equal(predictor.outputTransform, transform);
  }
});

// ---------------------------------------------------------------------------
// feature_names
// ---------------------------------------------------------------------------

test("feature_names must be non-empty and free of duplicates", () => {
  assertRefused(
    mutated((a) => {
      a.feature_names = [];
      a.trees = [];
    }),
    MalformedArtifactError,
    "feature_names empty",
  );
  assertRefused(
    mutated((a) => {
      a.feature_names = ["feature_a", "feature_a"];
    }),
    MalformedArtifactError,
    "feature_names with a duplicate",
  );
  assertRefused(
    mutated((a) => {
      a.feature_names = ["feature_a", 1];
    }),
    MalformedArtifactError,
    "feature_names with a non-string entry",
  );
});

// ---------------------------------------------------------------------------
// Tree geometry
// ---------------------------------------------------------------------------

test("the five tree arrays must have identical length", () => {
  for (const key of [
    "right_children",
    "split_indices",
    "default_left",
    "node_values",
  ]) {
    assertRefused(
      mutated((a) => {
        a.trees[0][key] = a.trees[0][key].slice(0, 2);
      }),
      MalformedArtifactError,
      `trees[0].${key} one entry short`,
    );
    assertRefused(
      mutated((a) => {
        a.trees[0][key] = [...a.trees[0][key], a.trees[0][key][0]];
      }),
      MalformedArtifactError,
      `trees[0].${key} one entry long`,
    );
  }
});

test("a tree with zero nodes raises, because node 0 is the root", () => {
  assertRefused(
    mutated((a) => {
      a.trees[0] = {
        default_left: [],
        left_children: [],
        node_values: [],
        right_children: [],
        split_indices: [],
      };
    }),
    MalformedArtifactError,
    "a tree with no nodes",
  );
});

test("a child index out of range raises", () => {
  for (const child of [3, 99, -2, -1.0e9]) {
    assertRefused(
      mutated((a) => {
        a.trees[0].left_children[0] = child;
      }),
      MalformedArtifactError,
      `left_children[0] = ${child}`,
    );
    assertRefused(
      mutated((a) => {
        a.trees[0].right_children[0] = child;
      }),
      MalformedArtifactError,
      `right_children[0] = ${child}`,
    );
  }
});

test("a leaf whose right child is not -1 raises", () => {
  // The vector-leaf signature: at a leaf of a vector-leaf tree that slot
  // carries a block index rather than a child. v1 refuses the shape rather than
  // walking it.
  assertRefused(
    mutated((a) => {
      a.trees[0].right_children[1] = 2;
    }),
    MalformedArtifactError,
    "a leaf carrying a right child",
  );
});

test("split_indices outside [0, feature_names.length) raises", () => {
  for (const column of [2, 7, -1, 2147483647]) {
    assertRefused(
      mutated((a) => {
        a.trees[0].split_indices[0] = column;
      }),
      MalformedArtifactError,
      `split_indices[0] = ${column}`,
    );
  }
  // 2147483647 is XGBoost's dead-node marker. It is legitimate in a *source*
  // model and must never appear in an artifact, because the exporter
  // neutralizes dead slots to 0 precisely so this check needs no exception.
});

test("a cycle reachable from the root raises rather than hanging", () => {
  // Every child index here is in range, so no other check catches it. A
  // non-terminating predictor is worse for a caller than a raise, because it
  // cannot be caught.
  assertRefused(
    mutated((a) => {
      a.trees[0] = {
        default_left: [0, 0, 0],
        left_children: [1, 0, -1],
        node_values: [0.5, 0.25, 0.75],
        right_children: [2, 2, -1],
        split_indices: [0, 1, 0],
      };
    }),
    MalformedArtifactError,
    "a two-node cycle reachable from the root",
  );
  assertRefused(
    mutated((a) => {
      a.trees[0] = {
        default_left: [0],
        left_children: [0],
        node_values: [0.5],
        right_children: [0],
        split_indices: [0],
      };
    }),
    MalformedArtifactError,
    "a root that is its own child",
  );
});

// ---------------------------------------------------------------------------
// Non-finite values
// ---------------------------------------------------------------------------

test("a non-finite intercept or node_values entry raises, including after narrowing", () => {
  // 1e40 is a perfectly ordinary double and becomes Infinity in float32. A
  // reader that checked finiteness before narrowing would accept it and then
  // accumulate an infinity.
  for (const value of [1e40, -1e40, Infinity, -Infinity, NaN, 1e400]) {
    assertRefused(
      mutated((a) => {
        a.intercept = value;
      }),
      MalformedArtifactError,
      `intercept = ${value}`,
    );
    assertRefused(
      mutated((a) => {
        a.trees[0].node_values[2] = value;
      }),
      MalformedArtifactError,
      `node_values[2] = ${value}`,
    );
  }
  // The largest finite float32 is fine; one binade above it is not.
  assert.doesNotThrow(() =>
    fromJSON(
      mutated((a) => {
        a.intercept = 3.4028234663852886e38;
      }),
    ),
  );
});

// ---------------------------------------------------------------------------
// What must NOT raise
// ---------------------------------------------------------------------------

test("a node unreachable from the root does not raise, whatever it holds", () => {
  // Neutralized dead slots are legitimate content (FORMAT.md §8.3) and are
  // indistinguishable from a leaf carrying 0. Rejecting them would reject every
  // pruned model.
  const predictor = fromJSON(
    mutated((a) => {
      a.trees[0] = {
        default_left: [1, 0, 0, 0, 0],
        left_children: [1, -1, -1, -1, -1],
        node_values: [0.5, -0.25, 0.75, 0, 0],
        right_children: [2, -1, -1, -1, -1],
        split_indices: [0, 0, 0, 0, 0],
      };
    }),
  );
  assert.equal(predictor.trees[0].nodeValues.length, 5);
  // And the walk still produces the reachable answer.
  assert.ok(Number.isFinite(predictor.margin({ feature_a: 0.25, feature_b: 9 })));
});

test("an empty trees array is valid and the margin is the intercept alone", () => {
  const predictor = fromJSON(
    mutated((a) => {
      a.trees = [];
    }),
  );
  assert.equal(predictor.trees.length, 0);
  assert.equal(
    bits32(predictor.margin({ feature_a: 0.25, feature_b: 9 })),
    bits32(predictor.intercept),
  );
});

// ---------------------------------------------------------------------------
// Structural narrowing — the point of the reader (FORMAT.md §9.2)
// ---------------------------------------------------------------------------

test("node_values is loaded into a Float32Array, not an array of doubles", () => {
  // Narrowing has to be a property of the data structure rather than a
  // discipline at the comparison site. If it lived at the comparison site, any
  // other consumer of a threshold — a re-serializer, an inspection utility, an
  // arithmetic transform — would silently get the float64 back.
  const predictor = fromJSON(baseArtifact());
  for (const tree of predictor.trees) {
    assert.ok(
      tree.nodeValues instanceof Float32Array,
      `node_values is ${tree.nodeValues.constructor.name}, not Float32Array`,
    );
    assert.ok(tree.leftChildren instanceof Int32Array);
    assert.ok(tree.rightChildren instanceof Int32Array);
    assert.ok(tree.splitIndices instanceof Int32Array);
    assert.ok(tree.defaultLeft instanceof Uint8Array);
  }
});

test("a threshold read back through the public view is the engine's float32", () => {
  // The observable consequence of the structural narrowing, and the check that
  // an un-narrowed reader fails: the float64 `JSON.parse` produced is a
  // *different number*, visible here as a different value and a different bit
  // pattern.
  const artifact = mutated((a) => {
    a.trees[0].node_values = [0.1, 0.2, 0.30000000000000004];
  });
  const predictor = fromJSON(artifact);
  const values = predictor.trees[0].nodeValues;
  assert.equal(values[0], Math.fround(0.1));
  assert.notEqual(values[0], 0.1);
  assert.equal(bits32(values[0]), 0x3dcccccd);
  assert.equal(values[1], Math.fround(0.2));
  assert.notEqual(values[1], 0.2);
  assert.equal(values[2], Math.fround(0.30000000000000004));
});

test("the intercept is narrowed on read, never transformed, and keeps its sign", () => {
  const narrowed = fromJSON(
    mutated((a) => {
      a.intercept = 0.1;
    }),
  );
  assert.equal(narrowed.intercept, Math.fround(0.1));
  assert.notEqual(narrowed.intercept, 0.1);

  // `binary:logistic` at base_score = 0.5 legitimately produces exactly -0.
  // Signed zero is never normalized.
  const signedZero = fromJSON(
    mutated((a) => {
      a.intercept = -0;
      a.trees = [];
    }),
  );
  assert.equal(bits32(signedZero.intercept), 0x80000000);
  assert.equal(bits32(signedZero.margin({ feature_a: 1, feature_b: 2 })), 0x80000000);
  // And the identity of the two zeros is not something `===` can see.
  assert.ok(signedZero.intercept === 0);
  assert.ok(Object.is(signedZero.intercept, -0));
});
