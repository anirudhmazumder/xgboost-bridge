// Structural cross-check between the JSON Schema at
// `schema/xgboost-bridge-v1.schema.json` and the fixture corpus, without a
// JavaScript schema-validation library -- this package has zero runtime
// dependencies (D009) and gains no dev dependency for this file either.
//
// This file asserts, by hand, the same structural invariants the schema
// encodes: the envelope's seven keys, each tree's five keys, provenance's
// three keys, format_version, feature_names, and default_left. Reading the
// schema's own key sets with `node:fs` and comparing them against what every
// fixture actually carries is what keeps the schema and the corpus from
// drifting apart unnoticed -- a schema that quietly grew or shrank a key
// list would fail here even though no fixture predictor ever calls it.
//
// `fixtures/corpus/reference/` holds the `mpmath` transform reference table,
// not an artifact, and is not walked -- both directory listings below are
// non-recursive, mirroring `corpus.test.js`'s own `loadDirectory` comment.
//
// Reading `dist/` is unaffected: this file imports nothing from the library
// at all, because there is nothing here for the library's own exports to
// check -- the comparison is between two files on disk (the schema and the
// fixtures), neither of which is `dist/`. `node:fs` for that disk read is a
// Node built-in, not a dependency, and test files may use it freely (D014).
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, "..", "..", "..");
const SCHEMA_PATH = path.join(REPO_ROOT, "schema", "xgboost-bridge-v1.schema.json");
const CORPUS_DIR = path.join(REPO_ROOT, "fixtures", "corpus");
const ADVERSARIAL_DIR = path.join(CORPUS_DIR, "adversarial");

const SCHEMA = JSON.parse(readFileSync(SCHEMA_PATH, "utf8"));

/** FORMAT.md §3 -- the seven required top-level artifact keys. */
const ENVELOPE_KEYS = [
  "format_version",
  "objective",
  "output_transform",
  "intercept",
  "feature_names",
  "trees",
  "provenance",
].sort();

/** FORMAT.md §8 -- the five required keys of every tree object. */
const TREE_KEYS = ["default_left", "left_children", "node_values", "right_children", "split_indices"].sort();

/** FORMAT.md §16 -- the three required keys of `provenance`. */
const PROVENANCE_KEYS = ["base_score", "exporter_version", "xgboost_version"].sort();

function sortedKeys(object) {
  return Object.keys(object).sort();
}

function loadArtifacts(directory) {
  const artifacts = [];
  for (const name of readdirSync(directory).sort()) {
    if (!name.endsWith(".json")) {
      continue;
    }
    const fixture = JSON.parse(readFileSync(path.join(directory, name), "utf8"));
    artifacts.push([name.slice(0, -5), fixture.artifact]);
  }
  return artifacts;
}

/** Every fixture artifact under `fixtures/corpus/` and its `adversarial/` subdirectory. */
const ARTIFACTS = [
  ...loadArtifacts(CORPUS_DIR).map(([name, artifact]) => [name, artifact]),
  ...loadArtifacts(ADVERSARIAL_DIR).map(([name, artifact]) => [`adversarial/${name}`, artifact]),
];

test("the corpus is non-empty", () => {
  assert.ok(ARTIFACTS.length > 0, `no fixture artifacts found under ${CORPUS_DIR}`);
});

test("exactly 23 fixture artifacts are present", () => {
  // Not a magic number: 15 direct fixtures plus 8 adversarial fixtures,
  // per fixtures/generate/corpus.py's FIXTURE_NAMES and
  // fixtures/generate/adversarial.py. A silent gain or loss here means the
  // corpus and this test's other assertions have drifted apart.
  assert.equal(ARTIFACTS.length, 23);
});

test("the schema's own envelope key set matches the seven required keys", () => {
  assert.deepEqual(sortedKeys(SCHEMA.properties), ENVELOPE_KEYS);
  assert.deepEqual([...SCHEMA.required].sort(), ENVELOPE_KEYS);
});

test("the schema's own tree key set matches the five required keys", () => {
  const treeSchema = SCHEMA.$defs.tree;
  assert.deepEqual(sortedKeys(treeSchema.properties), TREE_KEYS);
  assert.deepEqual([...treeSchema.required].sort(), TREE_KEYS);
});

test("the schema's own provenance key set matches the three required keys", () => {
  const provenanceSchema = SCHEMA.properties.provenance;
  assert.deepEqual(sortedKeys(provenanceSchema.properties), PROVENANCE_KEYS);
  assert.deepEqual([...provenanceSchema.required].sort(), PROVENANCE_KEYS);
});

test("every fixture artifact has exactly the seven envelope keys", () => {
  for (const [name, artifact] of ARTIFACTS) {
    assert.deepEqual(sortedKeys(artifact), ENVELOPE_KEYS, `${name}: envelope key mismatch`);
  }
});

test("every tree in every fixture artifact has exactly the five tree keys", () => {
  for (const [name, artifact] of ARTIFACTS) {
    artifact.trees.forEach((tree, index) => {
      assert.deepEqual(sortedKeys(tree), TREE_KEYS, `${name}: trees[${index}] key mismatch`);
    });
  }
});

test("every fixture artifact's provenance has exactly the three provenance keys", () => {
  for (const [name, artifact] of ARTIFACTS) {
    assert.deepEqual(sortedKeys(artifact.provenance), PROVENANCE_KEYS, `${name}: provenance key mismatch`);
  }
});

test("every fixture artifact has format_version === 1", () => {
  for (const [name, artifact] of ARTIFACTS) {
    assert.equal(artifact.format_version, 1, `${name}: format_version`);
    assert.equal(typeof artifact.format_version, "number", `${name}: format_version type`);
  }
});

test("every fixture artifact's feature_names is a non-empty array of unique strings", () => {
  for (const [name, artifact] of ARTIFACTS) {
    const names = artifact.feature_names;
    assert.ok(Array.isArray(names), `${name}: feature_names is not an array`);
    assert.ok(names.length > 0, `${name}: feature_names is empty`);
    for (const entry of names) {
      assert.equal(typeof entry, "string", `${name}: feature_names entry ${JSON.stringify(entry)} is not a string`);
    }
    assert.equal(new Set(names).size, names.length, `${name}: feature_names contains a duplicate`);
  }
});

test("every default_left entry, in every tree, in every fixture artifact, is 0 or 1", () => {
  for (const [name, artifact] of ARTIFACTS) {
    artifact.trees.forEach((tree, treeIndex) => {
      tree.default_left.forEach((value, nodeIndex) => {
        assert.ok(
          value === 0 || value === 1,
          `${name}: trees[${treeIndex}].default_left[${nodeIndex}] = ${JSON.stringify(value)}, expected 0 or 1`,
        );
      });
    });
  }
});

test("every fixture artifact's provenance.base_score is a string", () => {
  for (const [name, artifact] of ARTIFACTS) {
    assert.equal(
      typeof artifact.provenance.base_score,
      "string",
      `${name}: provenance.base_score is ${typeof artifact.provenance.base_score}, expected string`,
    );
  }
});

test("the schema's objective description states it is non-operative metadata", () => {
  const description = SCHEMA.properties.objective.description.toLowerCase();
  assert.ok(description.includes("non-operative"), description);
  assert.ok(description.includes("export-time"), description);
  assert.ok(description.includes("predictor") && description.includes("branch"), description);
});

test("the schema's intercept description states it is the operative value", () => {
  const description = SCHEMA.properties.intercept.description.toLowerCase();
  assert.ok(description.includes("single operative numeric value"), description);
  assert.ok(description.includes("provenance.base_score is read by nothing"), description);
});
