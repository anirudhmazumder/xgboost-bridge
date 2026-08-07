// The JavaScript half of the wrong-but-well-formed catalogue.
//
// `packages/python/tests/test_schema_scope.py` measures the published JSON
// Schema's coverage and the Python reader's refusals. This measures the same
// artifacts against *this* reader, and the two files are deliberately separate
// rather than sharing a data file: a shared catalogue lets one reader's blind
// spot hide inside the other's suite, which is the same mistake as validating one
// language against the other instead of against an oracle. Each side constructs
// its own cases from its own reading of FORMAT.md.
//
// What is being pinned: the schema accepts 10 of these 12, and both readers must
// reject all 12. The schema's `$id` is a public URL, so a third party can
// validate against it and then walk the artifact themselves — every case the
// schema accepts is one where they get a pass and this library would not have.
//
// One case is worth naming because it is the reason this file exists rather than
// only the Python one: `sharedChildIsADagNotATree`. Child indices in range,
// reachable subgraph acyclic, every node consistently internal or leaf — and two
// parents pointing at one child, so it is a directed acyclic graph and not a
// tree. Both readers accepted it and returned the *same* plausible margin, which
// means cross-language parity could never have caught it. Agreement is not
// correctness; that is the whole premise of this project's gate structure.

import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { PredictorError, fromJSON } from "../dist/index.js";

const BASE = JSON.parse(
  readFileSync(
    new URL("../../../fixtures/corpus/survival_cox_base_score_high.json", import.meta.url),
    "utf8",
  ),
).artifact;

const clone = () => JSON.parse(JSON.stringify(BASE));

function withTree(changes) {
  const artifact = clone();
  Object.assign(artifact.trees[0], changes);
  return artifact;
}

function withArtifact(changes) {
  return Object.assign(clone(), changes);
}

/** A 401-digit integer literal: valid JSON, and `Infinity` once parsed as float64. */
const HUGE = Number("1" + "0".repeat(400));

const CATALOGUE = {
  childIndexOutOfRange: () => {
    const children = [...BASE.trees[0].left_children];
    children[0] = children.length + 50;
    return withTree({ left_children: children });
  },
  childNegativeButNotLeafMarker: () => {
    const children = [...BASE.trees[0].left_children];
    children[0] = -2; // -1 is the leaf marker; -2 means nothing
    return withTree({ left_children: children });
  },
  perTreeArraysOfUnequalLength: () =>
    withTree({ node_values: BASE.trees[0].node_values.slice(0, -1) }),
  splitIndexBeyondFeatureCount: () => {
    const indices = [...BASE.trees[0].split_indices];
    indices[0] = BASE.feature_names.length + 7;
    return withTree({ split_indices: indices });
  },
  defaultLeftOutsideZeroOne: () => {
    const flags = [...BASE.trees[0].default_left];
    flags[0] = 2;
    return withTree({ default_left: flags });
  },
  duplicateFeatureNames: () => {
    const names = [...BASE.feature_names];
    names[1] = names[0];
    return withArtifact({ feature_names: names });
  },
  featureNamesShorterThanSplitIndices: () =>
    withArtifact({ feature_names: [BASE.feature_names[0]] }),
  cycleBetweenTwoNodes: () =>
    withTree({
      left_children: [1, 0],
      right_children: [1, 0],
      split_indices: [0, 0],
      node_values: [0.5, 0.5],
      default_left: [0, 0],
    }),
  sharedChildIsADagNotATree: () => {
    // Nodes 1 and 2 both point at leaves 3 and 4. Every node is unambiguously
    // internal or leaf, nothing is out of range, and the walk terminates.
    const artifact = clone();
    artifact.trees = [
      {
        left_children: [1, 3, 3, -1, -1],
        right_children: [2, 4, 4, -1, -1],
        split_indices: [0, 0, 0, 0, 0],
        node_values: [0.1, 0.2, 0.3, 1.5, 2.5],
        default_left: [0, 0, 0, 0, 0],
      },
    ];
    return artifact;
  },
  objectiveAndTransformDisagree: () =>
    withArtifact({ objective: "reg:squarederror", output_transform: "sigmoid" }),
  interceptBeyondFloat64: () => withArtifact({ intercept: HUGE }),
  nodeValueBeyondFloat64: () => {
    const values = [...BASE.trees[0].node_values];
    values[0] = HUGE;
    return withTree({ node_values: values });
  },
};

test("the catalogue is at least as large as the documented figure", () => {
  assert.ok(Object.keys(CATALOGUE).length >= 12);
});

for (const [name, build] of Object.entries(CATALOGUE)) {
  test(`the reader rejects a wrong-but-well-formed artifact: ${name}`, () => {
    let threw = false;
    let error;
    try {
      fromJSON(build());
    } catch (caught) {
      threw = true;
      error = caught;
    }
    assert.ok(threw, `${name} was ACCEPTED; a wrong number is reachable through this reader`);
    // Structurally, through the documented hierarchy — not as a bare TypeError.
    // A consumer catching PredictorError must catch all of these, or the
    // documented contract is narrower than the code.
    assert.ok(
      error instanceof PredictorError,
      `${name} threw ${error?.constructor?.name}, which is outside PredictorError`,
    );
  });
}

test("the DAG case really is well-formed by every other rule", () => {
  // The point of this entry is that it passes everything else. If a future edit
  // made it fail on shape instead, the test above would still pass and would no
  // longer be testing in-degree at all.
  const artifact = CATALOGUE.sharedChildIsADagNotATree();
  const tree = artifact.trees[0];
  const size = tree.left_children.length;

  for (const key of ["right_children", "split_indices", "node_values", "default_left"]) {
    assert.equal(tree[key].length, size, `${key} must agree in length`);
  }
  for (let node = 0; node < size; node += 1) {
    const left = tree.left_children[node];
    const right = tree.right_children[node];
    // Consistently internal or consistently leaf — no half-leaf, which is a
    // different defect and would mask this one.
    assert.equal(
      left === -1,
      right === -1,
      `node ${node} must be wholly internal or wholly a leaf`,
    );
    if (left !== -1) {
      assert.ok(left >= 0 && left < size && right >= 0 && right < size, "children in range");
      assert.ok(tree.split_indices[node] < artifact.feature_names.length, "feature in range");
    }
  }

  // And a node genuinely has two parents, which is the defect itself.
  const inDegree = new Array(size).fill(0);
  for (let node = 0; node < size; node += 1) {
    if (tree.left_children[node] === -1) continue;
    inDegree[tree.left_children[node]] += 1;
    inDegree[tree.right_children[node]] += 1;
  }
  assert.ok(
    inDegree.some((count) => count > 1),
    "the case must actually contain a shared child, or it tests nothing",
  );
});
