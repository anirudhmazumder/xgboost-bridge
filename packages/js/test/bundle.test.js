// Structural checks on the source and on the shipped bundle.
//
// These exist because no numeric measurement can replace them:
//
// * A drift to a platform exponential would score *better* against the mpmath
//   reference, so accuracy cannot detect it. Only a scan can (D046).
// * A branch on `objective` can be a no-op, in which case every prediction is
//   still correct and the field has silently become a second source of truth
//   about behaviour `output_transform` already determines (D028).
// * A runtime dependency is not visible in any prediction. Zero JavaScript
//   runtime dependencies is non-negotiable — not "few", zero (D009).
//
// Comments are stripped before scanning, because this file's own prose and the
// source's own documentation both name the forbidden tokens: `Math.exp` and `**`
// appear in the explanation of why they are banned, and a scan that could not
// tell a comment from code would have to be either useless or unwritable. The
// stripper is itself tested below, since a stripper that removed everything
// would make every scan pass.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PACKAGE_ROOT = path.resolve(HERE, "..");
const SRC_DIR = path.join(PACKAGE_ROOT, "src");
const DIST_DIR = path.join(PACKAGE_ROOT, "dist");

/**
 * Remove block and line comments, leaving code and string literals in place.
 *
 * Deliberately simple: it tracks string, template and regular-expression
 * literals well enough not to mistake a `//` inside one for a comment, and it
 * replaces each comment with a newline so line numbers survive.
 */
function stripComments(source) {
  let out = "";
  let index = 0;
  const length = source.length;
  while (index < length) {
    const character = source[index];
    const next = source[index + 1];
    if (character === "/" && next === "/") {
      while (index < length && source[index] !== "\n") {
        index += 1;
      }
      out += "\n";
    } else if (character === "/" && next === "*") {
      index += 2;
      while (index < length && !(source[index] === "*" && source[index + 1] === "/")) {
        if (source[index] === "\n") {
          out += "\n";
        }
        index += 1;
      }
      index += 2;
    } else if (character === '"' || character === "'" || character === "`") {
      const quote = character;
      out += character;
      index += 1;
      while (index < length && source[index] !== quote) {
        if (source[index] === "\\") {
          out += source[index];
          index += 1;
        }
        if (index < length) {
          out += source[index];
          index += 1;
        }
      }
      out += quote;
      index += 1;
    } else {
      out += character;
      index += 1;
    }
  }
  return out;
}

function filesUnder(directory, suffixes) {
  const out = [];
  for (const name of readdirSync(directory).sort()) {
    const full = path.join(directory, name);
    if (statSync(full).isDirectory()) {
      out.push(...filesUnder(full, suffixes));
    } else if (suffixes.some((suffix) => name.endsWith(suffix))) {
      out.push(full);
    }
  }
  return out;
}

// Every platform transcendental and every exponentiation spelling. Named
// individually rather than matched by a loose pattern, so a new one has to be
// added here deliberately rather than slipping through a regex.
const FORBIDDEN_TOKENS = [
  "Math.exp",
  "Math.expm1",
  "Math.pow",
  "Math.log",
  "Math.log2",
  "Math.log10",
  "Math.log1p",
  "Math.E",
  "Math.LN2",
  "Math.LOG2E",
  "Math.sinh",
  "Math.cosh",
  "Math.tanh",
  "**",
];

test("the comment stripper removes comments and keeps code", () => {
  // A stripper that removed everything would make every scan below pass.
  const sample = [
    "// Math.exp in a line comment",
    "/* Math.pow and ** in a block comment */",
    'const url = "https://example.invalid/a//b";',
    "const contracted = 2 ** 3;",
    "const kept = Math.fround(1);",
  ].join("\n");
  const stripped = stripComments(sample);
  assert.ok(!stripped.includes("Math.exp"), "line comment survived");
  assert.ok(!stripped.includes("Math.pow"), "block comment survived");
  assert.ok(stripped.includes("2 ** 3"), "code was removed along with the comments");
  assert.ok(stripped.includes("Math.fround(1)"), "code was removed along with the comments");
  assert.ok(stripped.includes("https://example.invalid/a//b"), "a string literal was truncated");
  // And the stripper finds a real violation, which is the only reason to run it.
  assert.ok(FORBIDDEN_TOKENS.some((token) => stripped.includes(token)));
});

test("no platform exponential and no exponentiation operator in src/", () => {
  const files = filesUnder(SRC_DIR, [".ts"]);
  assert.ok(files.length >= 5, `only ${files.length} source files found under ${SRC_DIR}`);
  const findings = [];
  for (const file of files) {
    const stripped = stripComments(readFileSync(file, "utf8"));
    const lines = stripped.split("\n");
    for (let index = 0; index < lines.length; index += 1) {
      for (const token of FORBIDDEN_TOKENS) {
        if (lines[index].includes(token)) {
          findings.push(`${path.relative(PACKAGE_ROOT, file)}:${index + 1}: ${token}`);
        }
      }
    }
  }
  assert.deepEqual(findings, [], `forbidden tokens on the prediction path:\n${findings.join("\n")}`);
});

test("no platform exponential and no exponentiation operator in the shipped bundle", () => {
  // The source scan is not enough on its own: a bundler transform, a polyfill,
  // or a downlevelling pass could introduce one, and the bundle is what a
  // consumer actually receives.
  const files = filesUnder(DIST_DIR, [".js", ".cjs"]);
  assert.ok(files.length >= 2, `expected an ESM and a CJS bundle, found ${files.length}`);
  const findings = [];
  for (const file of files) {
    const stripped = stripComments(readFileSync(file, "utf8"));
    for (const token of FORBIDDEN_TOKENS) {
      if (stripped.includes(token)) {
        findings.push(`${path.relative(PACKAGE_ROOT, file)}: ${token}`);
      }
    }
  }
  assert.deepEqual(findings, [], `forbidden tokens in the bundle:\n${findings.join("\n")}`);
});

test("Math.fround is the only Math member the prediction path uses", () => {
  // The positive form of the check above. `fround` is float32 narrowing, which
  // is a rounding of a value the four permitted operations already produced,
  // not a transcendental.
  const permitted = new Set(["fround", "abs", "floor", "min", "max"]);
  const used = new Set();
  for (const file of filesUnder(SRC_DIR, [".ts"])) {
    const stripped = stripComments(readFileSync(file, "utf8"));
    for (const match of stripped.matchAll(/Math\.([A-Za-z0-9_]+)/g)) {
      used.add(match[1]);
    }
  }
  assert.ok(used.has("fround"), "the source does not narrow to float32 at all");
  for (const member of used) {
    assert.ok(permitted.has(member), `Math.${member} is used on the prediction path`);
  }
  // In fact only `fround` is used today; recorded so a later addition is a
  // deliberate edit here rather than a silent one.
  assert.deepEqual([...used].sort(), ["fround"]);
});

test("nothing in the shipped bundle branches on `objective`", () => {
  // A branch here would make `objective` a second source of truth about
  // behaviour `output_transform` already determines, and two fields that must
  // agree where only one is validated is how a silent divergence starts (D028).
  // The pairing cross-check is a table lookup at load time and is deliberately
  // written so that no comparison stands next to the field.
  const patterns = [
    /\bobjective\b\s*(?:===|!==|==|!=|<|>)/,
    /(?:===|!==|==|!=)\s*[A-Za-z0-9_$.[\]"']*objective\b/,
    /switch\s*\([^)]*\bobjective\b/,
    /\bobjective\b\s*\?/,
    /\bif\s*\([^)]*\bobjective\b/,
    /\bobjective\b\s*(?:&&|\|\|)/,
    /(?:&&|\|\|)\s*[A-Za-z0-9_$.]*objective\b/,
    /\bobjective\b\.(?:startsWith|endsWith|includes|indexOf|match|test)/,
  ];
  const findings = [];
  for (const file of filesUnder(DIST_DIR, [".js", ".cjs"])) {
    const stripped = stripComments(readFileSync(file, "utf8"));
    for (const pattern of patterns) {
      const match = pattern.exec(stripped);
      if (match) {
        findings.push(`${path.relative(PACKAGE_ROOT, file)}: ${match[0]}`);
      }
    }
  }
  assert.deepEqual(findings, [], `a branch on \`objective\`:\n${findings.join("\n")}`);
  // And the scan is only meaningful if it would fire on a real violation.
  for (const violation of [
    'if (objective === "binary:logistic") { return 1; }',
    "switch (this.objective) { default: break; }",
    'const t = objective == "survival:cox" ? 1 : 2;',
    'if (loaded.objective) { return 0; }',
    'x = objective.startsWith("binary");',
  ]) {
    assert.ok(
      patterns.some((pattern) => pattern.test(violation)),
      `the scan would not have caught: ${violation}`,
    );
  }
});

test("the package declares zero runtime dependencies", () => {
  const manifest = JSON.parse(readFileSync(path.join(PACKAGE_ROOT, "package.json"), "utf8"));
  assert.deepEqual(manifest.dependencies, {}, "runtime dependencies must be empty");
  assert.equal(Object.keys(manifest.dependencies).length, 0);
  assert.equal(manifest.peerDependencies, undefined, "a peer dependency is still a dependency");
  assert.equal(manifest.optionalDependencies, undefined);
  assert.equal(manifest.bundleDependencies, undefined);
  assert.equal(manifest.bundledDependencies, undefined);
});

test("the shipped bundle imports nothing at all", () => {
  // The other half of zero dependencies: an empty `dependencies` block with a
  // bare `import` of something the runtime is expected to provide would still
  // break a browser or an edge runtime.
  for (const file of filesUnder(DIST_DIR, [".js", ".cjs"])) {
    const stripped = stripComments(readFileSync(file, "utf8"));
    const findings = [
      ...stripped.matchAll(/\bimport\s*[({'"]/g),
      ...stripped.matchAll(/\brequire\s*\(/g),
      ...stripped.matchAll(/\bfrom\s*['"]/g),
    ].map((match) => match[0]);
    assert.deepEqual(
      findings,
      [],
      `${path.relative(PACKAGE_ROOT, file)} reaches outside the bundle: ${findings.join(", ")}`,
    );
  }
});

test("no source file reads the filesystem or the environment", () => {
  // `fromFile` is refused by D006, and the reason generalizes: anything that
  // touches `node:fs`, `process`, or a global that only exists in one runtime
  // splits the bundle by runtime.
  const findings = [];
  for (const file of filesUnder(SRC_DIR, [".ts"])) {
    const stripped = stripComments(readFileSync(file, "utf8"));
    for (const token of ["node:fs", "node:path", "require(", "process.", "globalThis", "__dirname"]) {
      if (stripped.includes(token)) {
        findings.push(`${path.relative(PACKAGE_ROOT, file)}: ${token}`);
      }
    }
  }
  assert.deepEqual(findings, [], `runtime-specific access in src/:\n${findings.join("\n")}`);
});

test("every test file imports the built bundle and never src/ (D011)", () => {
  // Testing `src/` verifies code that is not what ships: a wrong entry point, a
  // broken export map, a dropped declaration file or an accidentally
  // externalized module is invisible to a source-level suite and breaks every
  // consumer.
  const testFiles = filesUnder(HERE, [".test.js"]);
  assert.ok(testFiles.length >= 5, `only ${testFiles.length} test files found`);
  let importingDist = 0;
  for (const file of testFiles) {
    const source = readFileSync(file, "utf8");
    assert.ok(
      !/from\s+['"][^'"]*\/src\//.test(source),
      `${path.basename(file)} imports from src/`,
    );
    if (/from\s+['"]\.\.\/dist\/index\.js['"]/.test(source)) {
      importingDist += 1;
    }
  }
  assert.ok(importingDist >= 4, `only ${importingDist} test files import the bundle`);
});

test("the built bundle carries the files the export map promises", () => {
  const manifest = JSON.parse(readFileSync(path.join(PACKAGE_ROOT, "package.json"), "utf8"));
  // `exports["."]` is a map of CONDITION -> either a path or a nested map, because
  // `types` is declared per condition: a require()-mode TypeScript consumer must
  // resolve dist/index.d.cts, not the ESM dist/index.d.ts. Walk it rather than
  // reading fixed keys, so the test follows the map instead of one shape of it.
  const fromExports = Object.values(manifest.exports["."]).flatMap((value) =>
    typeof value === "string" ? [value] : Object.values(value),
  );
  const promised = [manifest.types, manifest.main, manifest.module, ...fromExports];

  // Every declaration flavour must be reachable from the map, not merely present
  // on disk: dist/index.d.cts shipped for weeks while nothing pointed at it, and
  // `tsc` under node16 resolution rejected the package as a result.
  assert.ok(fromExports.includes("./dist/index.d.ts"), "ESM types must be in the export map");
  assert.ok(fromExports.includes("./dist/index.d.cts"), "CJS types must be in the export map");

  for (const relative of promised) {
    assert.equal(typeof relative, "string", `export map yielded a non-path: ${relative}`);
    const full = path.join(PACKAGE_ROOT, relative);
    assert.ok(statSync(full).isFile(), `${relative} is promised by package.json and is missing`);
  }
});
