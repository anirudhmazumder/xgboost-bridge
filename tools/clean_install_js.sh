#!/usr/bin/env bash
#
# Pack the npm tarball, verify its CONTENTS, then install it into a project
# containing nothing else and predict from it through both module systems.
#
# Why this exists as its own check: every JavaScript test imports
# ../dist/index.js by relative path from inside this repository. That exercises
# the bundle but never the PACKAGE -- not the export map, not the `files` list,
# not what a consumer actually receives. Two real defects lived exactly there:
# the tarball shipped no README and no LICENSE, and dist/index.cjs (the
# require() entry point) was never executed by any test at all.
#
# Runs identically on a laptop and in CI. No arguments.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "== building and packing =="
# Install the dev dependencies first. This script must be self-contained: on a
# fresh checkout there is no packages/js/node_modules, so `npm run build` would
# invoke tsup and exit 127. That is exactly how this failed on its first CI run
# while passing locally, where node_modules already existed.
#
# `npm ci` rather than `npm install`, so the build being verified is the build
# the lockfile describes rather than whatever resolves today.
npm --prefix "$REPO/packages/js" ci --silent --no-audit --no-fund
# stderr is NOT swallowed here. The first version discarded it, which turned a
# one-line "tsup: not found" into an opaque exit code.
npm --prefix "$REPO/packages/js" run build >/dev/null
(cd "$REPO/packages/js" && npm pack --pack-destination "$WORK" >/dev/null)
TARBALL="$(find "$WORK" -name '*.tgz' | head -1)"
echo "   tarball: $(basename "$TARBALL")"

echo
echo "== verifying tarball CONTENTS =="
tar -tzf "$TARBALL" | sed 's/^/     /'
node - "$TARBALL" <<'JS'
const { execFileSync } = require("node:child_process");
const tarball = process.argv[2];
const names = execFileSync("tar", ["-tzf", tarball], { encoding: "utf8" })
  .split("\n").filter(Boolean).map((n) => n.replace(/^package\//, ""));

// Everything a consumer needs to import or predict. The export map in
// package.json promises all four dist entries; a missing one is a resolution
// failure on someone else's machine and a green suite on ours.
const required = [
  "package.json",
  "dist/index.js",      // import
  "dist/index.cjs",     // require
  "dist/index.d.ts",    // types, import condition
  "dist/index.d.cts",   // types, require condition
  "README.md",          // the npm page renders this
  "LICENSE",            // declared MIT; must actually ship
];
const missing = required.filter((r) => !names.includes(r));
if (missing.length) {
  console.error(`FAIL: tarball is missing ${JSON.stringify(missing)}`);
  process.exit(1);
}
console.log(`   all ${required.length} required files present`);

// Nothing that should not ship.
const forbidden = names.filter((n) => n.startsWith("src/") || n.startsWith("test/") || n.endsWith(".tsbuildinfo"));
if (forbidden.length) {
  console.error(`FAIL: tarball ships ${JSON.stringify(forbidden)}`);
  process.exit(1);
}
console.log("   no src/ or test/ paths in the tarball (the source itself ships inside dist/*.map -- see the note above)");

// Sourcemaps are a decision (D056), not a default, so the properties that make
// shipping them safe are asserted rather than re-checked by hand each release:
// they must be present, they must carry the source inline (the `sources` entries
// are relative `../src/*.ts` paths that are NOT shipped, so a map without
// `sourcesContent` would be dead weight), and they must contain no absolute
// path -- which is what would turn "we ship our source" into "we ship the
// maintainer's directory layout".
const maps = names.filter((n) => n.endsWith(".map"));
if (maps.length !== 2) {
  console.error(`FAIL: expected 2 sourcemaps in the tarball, found ${maps.length}`);
  process.exit(1);
}
for (const mapName of maps) {
  // Entries are prefixed with `package/` inside the tarball, as the export-map
  // check below also relies on; `names` has that prefix stripped.
  const raw = execFileSync("tar", ["-xzOf", tarball, `package/${mapName}`], { encoding: "utf8" });
  const map = JSON.parse(raw);
  if (!Array.isArray(map.sourcesContent) || map.sourcesContent.length === 0) {
    console.error(`FAIL: ${mapName} has no sourcesContent, so its mappings are dead`);
    process.exit(1);
  }
  const absolute = (map.sources || []).filter((src) => src.startsWith("/") || /^[A-Za-z]:\\/.test(src));
  if (absolute.length > 0) {
    console.error(`FAIL: ${mapName} carries absolute paths: ${absolute.join(", ")}`);
    process.exit(1);
  }
  if (map.sourceRoot) {
    console.error(`FAIL: ${mapName} sets sourceRoot (${map.sourceRoot}), which can leak a local layout`);
    process.exit(1);
  }
  const leaked = raw.match(/\/(Users|home)\/[A-Za-z0-9_.-]+/g);
  if (leaked) {
    console.error(`FAIL: ${mapName} contains a local filesystem path: ${leaked[0]}`);
    process.exit(1);
  }
}
console.log(`   ${maps.length} sourcemaps ship deliberately (D056): source inline, no absolute paths`);

// The export map must point only at files that are actually inside.
const pkgRaw = execFileSync("tar", ["-xzOf", tarball, "package/package.json"], { encoding: "utf8" });
const pkg = JSON.parse(pkgRaw);
// The map is per-condition (types nested inside import/require), so flatten one
// level rather than assuming every value is a string.
const targets = Object.values(pkg.exports["."])
  .flatMap((v) => (typeof v === "string" ? [v] : Object.values(v)))
  .map((t) => t.replace(/^\.\//, ""));
const dangling = targets.filter((t) => !names.includes(t));
if (dangling.length) {
  console.error(`FAIL: export map points at files not in the tarball: ${JSON.stringify(dangling)}`);
  process.exit(1);
}
console.log(`   export map resolves entirely inside the tarball: ${targets.join(", ")}`);

const runtimeDeps = Object.keys(pkg.dependencies || {}).length;
if (runtimeDeps !== 0) {
  console.error(`FAIL: expected 0 runtime dependencies, found ${runtimeDeps}`);
  process.exit(1);
}
console.log("   declared runtime dependencies: 0");
JS

echo
echo "== installing into an empty project =="
mkdir -p "$WORK/consumer"
cd "$WORK/consumer"
printf '{"name":"consumer","private":true,"version":"0.0.0","type":"module"}\n' > package.json
npm install --silent --no-audit --no-fund "$TARBALL" >/dev/null 2>&1
cp "$REPO/fixtures/corpus/binary_logistic_signed_zero.json" fixture.json

# Zero runtime dependencies must hold after a real install, not only in the
# manifest: this is what the consumer's node_modules actually contains.
INSTALLED=$(node -p "Object.keys(require('./node_modules/xgboost-predictor/package.json').dependencies||{}).length")
echo "   installed package declares $INSTALLED runtime dependencies"
[ "$INSTALLED" -eq 0 ] || { echo "FAIL: non-zero runtime dependencies"; exit 1; }

echo
echo "== predicting through the ESM entry point =="
cat > esm.mjs <<'JS'
import { fromJSON } from "xgboost-predictor";
import { readFileSync } from "node:fs";

const fixture = JSON.parse(readFileSync("fixture.json", "utf8"));
const predictor = fromJSON(fixture.artifact);
const buf = new Float32Array(1);
const bits = new Uint32Array(buf.buffer);
const asBits = (v) => { buf[0] = v; return "0x" + bits[0].toString(16).padStart(8, "0"); };

let ok = 0;
fixture.rows.forEach((row, i) => {
  const values = Object.fromEntries(
    fixture.artifact.feature_names.map((n, k) => [n, row[k] === null ? NaN : row[k]]),
  );
  if (asBits(predictor.margin(values)) === fixture.expected_margin[i]) ok += 1;
});
console.log(`   margin bit-exact vs XGBoost ground truth: ${ok}/${fixture.rows.length}`);
if (ok !== fixture.rows.length) { console.error("FAIL: ESM entry point"); process.exit(1); }
JS
node esm.mjs

echo
echo "== predicting through the CommonJS entry point =="
cat > cjs.cjs <<'JS'
const { fromJSON } = require("xgboost-predictor");
const { readFileSync } = require("node:fs");

const fixture = JSON.parse(readFileSync("fixture.json", "utf8"));
const predictor = fromJSON(fixture.artifact);
const buf = new Float32Array(1);
const bits = new Uint32Array(buf.buffer);
const asBits = (v) => { buf[0] = v; return "0x" + bits[0].toString(16).padStart(8, "0"); };

let ok = 0;
fixture.rows.forEach((row, i) => {
  const values = Object.fromEntries(
    fixture.artifact.feature_names.map((n, k) => [n, row[k] === null ? NaN : row[k]]),
  );
  if (asBits(predictor.margin(values)) === fixture.expected_margin[i]) ok += 1;
});
console.log(`   margin bit-exact vs XGBoost ground truth: ${ok}/${fixture.rows.length}`);
if (ok !== fixture.rows.length) { console.error("FAIL: CommonJS entry point"); process.exit(1); }
JS
node cjs.cjs

echo
echo "== compiling a CommonJS TypeScript consumer against the package =="
# dist/index.d.cts shipped but was unreachable from the export map, so tsc under
# node16 resolution rejected a require()-mode file with TS1479. No runtime test
# can see a condition-specific types mismatch; this pins it.
printf '{"compilerOptions":{"module":"node16","moduleResolution":"node16","noEmit":true,"strict":true,"types":[]}}\n' > tsconfig.json
cat > consumer.cts <<'TS'
import { fromJSON } from "xgboost-predictor";
const use: typeof fromJSON = fromJSON;
void use;
TS
# Pinned exactly, and no install scripts. This line runs inside release.yml's
# verify job, so an unpinned range meant the compiler used to validate a
# consumer build was whatever the registry served that day.
npm install --silent --no-audit --no-fund --ignore-scripts typescript@5.9.3 >/dev/null 2>&1
./node_modules/.bin/tsc -p tsconfig.json
echo "   tsc (module=node16, require mode) resolved the package cleanly"

echo
echo "JAVASCRIPT CLEAN INSTALL: OK"
