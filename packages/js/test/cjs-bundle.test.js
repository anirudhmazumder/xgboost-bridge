// The CommonJS bundle, executed.
//
// `dist/index.cjs` ships in the npm tarball and the export map routes `require()`
// to it, and no test in this suite had ever run it -- every other test imports the
// ESM build. That is not a hypothetical gap: the audit already found one real CJS
// defect, an export map whose `require` condition pointed at a `types` entry that
// did not resolve, and it was found by hand rather than by a test.
//
// `tools/clean_install_js.sh` does predict through this entry point, and it runs in
// CI as a permanent job -- so the path is covered. It is covered from *outside* the
// repository, though, against an installed tarball. That is the right place for a
// packaging check and the wrong place for the only execution of a shipped bundle:
// someone running `npm test` alone gets no signal, and the failure mode of an
// untested build output is that it rots silently between releases.
//
// This asserts the two bundles are not merely both present but numerically
// identical, which is the property a dual-format package actually promises.

import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";

import * as ESM from "../dist/index.js";

const require = createRequire(import.meta.url);
const CJS = require("../dist/index.cjs");

const FIXTURE = JSON.parse(
  readFileSync(
    new URL("../../../fixtures/corpus/survival_cox_base_score_high.json", import.meta.url),
    "utf8",
  ),
);

const bits32 = (value) => {
  const view = new DataView(new ArrayBuffer(4));
  view.setFloat32(0, value);
  return view.getUint32(0);
};

function rowsOf(artifact) {
  return FIXTURE.rows.map((row) =>
    Object.fromEntries(
      artifact.feature_names.map((name, index) => [
        name,
        row[index] === null ? NaN : row[index],
      ]),
    ),
  );
}

test("the CommonJS bundle exposes the same named exports as the ESM bundle", () => {
  const esm = Object.keys(ESM).filter((k) => k !== "default").sort();
  const cjs = Object.keys(CJS).filter((k) => k !== "default").sort();
  assert.deepEqual(cjs, esm, "the two builds must present the same surface");
  assert.ok(esm.length > 0, "the ESM surface must not be empty, or this compares nothing");
});

test("the CommonJS bundle predicts, and agrees with ESM on every bit", () => {
  const artifact = FIXTURE.artifact;
  const viaCjs = CJS.fromJSON(artifact);
  const viaEsm = ESM.fromJSON(artifact);

  let compared = 0;
  for (const row of rowsOf(artifact)) {
    assert.equal(bits32(viaCjs.margin(row)), bits32(viaEsm.margin(row)), "margin");
    assert.equal(bits32(viaCjs.output(row)), bits32(viaEsm.output(row)), "output");
    compared += 1;
  }
  assert.ok(compared >= 5, `only ${compared} rows compared`);
});

test("the CommonJS bundle is bit-exact against the committed ground truth", () => {
  // Not just "agrees with the other build" -- two builds of the same mistake
  // agree perfectly. This compares against XGBoost's recorded output.
  const artifact = FIXTURE.artifact;
  const predictor = CJS.fromJSON(artifact);
  const rows = rowsOf(artifact);
  let exact = 0;
  rows.forEach((row, index) => {
    if (bits32(predictor.margin(row)) === Number.parseInt(FIXTURE.expected_margin[index], 16)) {
      exact += 1;
    }
  });
  assert.equal(exact, rows.length, `${exact}/${rows.length} margins bit-exact through require()`);
});

test("the CommonJS bundle refuses what the ESM bundle refuses", () => {
  // Refusal semantics are the contract, so a dual-format package must not have
  // two of them. Each case is one the readers were changed to refuse recently.
  const artifact = FIXTURE.artifact;
  const cases = [
    ["non-finite in float32", (m) => {
      const p = m.fromJSON(artifact);
      const row = Object.fromEntries(artifact.feature_names.map((n) => [n, 0.5]));
      row[artifact.feature_names[0]] = 1e39;
      p.margin(row);
    }],
    ["a shared child, i.e. a DAG", (m) => {
      const bad = JSON.parse(JSON.stringify(artifact));
      bad.trees = [{
        left_children: [1, 3, 3, -1, -1],
        right_children: [2, 4, 4, -1, -1],
        split_indices: [0, 0, 0, 0, 0],
        node_values: [0.1, 0.2, 0.3, 1.5, 2.5],
        default_left: [0, 0, 0, 0, 0],
      }];
      m.fromJSON(bad);
    }],
    ["an unrecognised field", (m) => m.fromJSON({ ...artifact, surprise: 1 })],
  ];

  for (const [label, run] of cases) {
    let cjsCode = null;
    let esmCode = null;
    try { run(CJS); } catch (error) { cjsCode = error.code; }
    try { run(ESM); } catch (error) { esmCode = error.code; }
    assert.ok(cjsCode !== null, `CJS accepted ${label}`);
    assert.equal(cjsCode, esmCode, `the two builds disagree on ${label}`);
  }
});

test("the CommonJS bundle carries no import or require of its own", () => {
  // Zero runtime dependencies (D009) has to hold for the file `require()`
  // actually loads, not only for the ESM one the other tests import.
  const source = readFileSync(new URL("../dist/index.cjs", import.meta.url), "utf8");
  const requires = [...source.matchAll(/\brequire\s*\(\s*["']([^"']+)["']\s*\)/g)].map((m) => m[1]);
  assert.deepEqual(requires, [], `dist/index.cjs requires: ${requires.join(", ")}`);
  const imports = [...source.matchAll(/\bfrom\s*["']([^"']+)["']/g)].map((m) => m[1]);
  assert.deepEqual(imports, [], `dist/index.cjs imports: ${imports.join(", ")}`);
});

// ---------------------------------------------------------------------------
// `--tag rc` in release.yml is load-bearing (D061)
//
// Measured on npm 12.0.1 with a throwaway manifest: `publishConfig.registry` is
// applied — a sentinel registry appears in the publish target — while
// `publishConfig.tag: "rc"` is not: `npm publish` still reports `with tag latest`.
// An earlier comment in this repository claimed the manifest field made the flag
// redundant. It does not.
//
// So the flag is the only thing keeping a release candidate off the `latest`
// dist-tag, which is what `npm i xgboost-predictor` resolves to. Removing it as
// redundant is a plausible future cleanup with a user-visible, hard-to-reverse
// consequence, so it is pinned here rather than left to a comment.
// ---------------------------------------------------------------------------

test("release.yml publishes the RC under a non-default dist-tag", () => {
  const workflow = readFileSync(
    new URL("../../../.github/workflows/release.yml", import.meta.url),
    "utf8",
  );
  // Lines that *invoke* npm publish, not lines that mention it. Prose mentions
  // live in `#` comments and in a YAML block scalar (the `targets` input
  // description), and an earlier version of this filter caught the latter --
  // trimming and testing for a leading `#` does not exclude a block scalar.
  const publishLines = workflow
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.startsWith("npm publish"));

  assert.equal(publishLines.length, 1, `expected one npm publish line, got ${publishLines.length}`);
  assert.match(
    publishLines[0],
    /--tag\s+rc\b/,
    "npm publish must carry --tag rc; publishConfig.tag does not cover for it",
  );
  assert.match(publishLines[0], /--provenance\b/, "the release publish must request provenance");
});

test("the manifest declares a prerelease version while the rc tag is in force", () => {
  const manifest = JSON.parse(
    readFileSync(new URL("../package.json", import.meta.url), "utf8"),
  );
  // If the version ever stops being a prerelease, `--tag rc` becomes wrong rather
  // than protective, and this says so instead of silently publishing 1.0.0 to rc.
  const isPrerelease = /-(?:rc|alpha|beta)\./.test(manifest.version);
  assert.equal(
    isPrerelease,
    true,
    `version ${manifest.version} is not a prerelease, so --tag rc in release.yml ` +
      `would hide a final release behind a non-default tag`,
  );
});

test("release.yml passes npm publish an unambiguous path, not a bare relative one", () => {
  // `npm publish dist/x.tgz` does not publish a file: npm parses a bare `a/b`
  // spec as the GitHub shorthand `github:a/b`. The first dispatch of release.yml
  // failed on exactly this, with `git ls-remote ssh://git@github.com/dist/...` and
  // "Permission denied (publickey)" -- an error about SSH keys from a step that
  // touches no git. Reproduced locally: `dist/x.tgz` gives
  // `Refusing to fetch "github:dist/x.tgz"`; `./dist/x.tgz` and an absolute path
  // both resolve to the file.
  const workflow = readFileSync(
    new URL("../../../.github/workflows/release.yml", import.meta.url),
    "utf8",
  );
  const publishLine = workflow
    .split("\n")
    .map((line) => line.trim())
    .find((line) => line.startsWith("npm publish"));
  assert.ok(publishLine, "no npm publish invocation found");

  // The argument must be a variable, and that variable must be built from an
  // absolute path -- `$(cd dist && pwd)` -- rather than a bare `ls dist/*.tgz`.
  assert.match(publishLine, /npm publish "\$[A-Z_]+"/, "the spec must be a quoted variable");
  assert.match(
    workflow,
    /TARBALL="\$\(cd dist && pwd\)/,
    "the tarball path must be absolute, or npm reads it as a github: spec",
  );
  assert.doesNotMatch(
    workflow,
    /TARBALL=\$\(ls dist\/\*\.tgz\)/,
    "a bare relative glob is the form that failed",
  );
});
