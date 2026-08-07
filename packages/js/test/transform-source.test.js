// FORMAT.md §5.5, enforced structurally on the TypeScript side.
//
// The rule: "Each operation is a separate statement with an explicit named
// intermediate. Do not write a fused expression and rely on neither runtime
// contracting it into an FMA. The guarantee must come from how the code is
// written, not from what the runtimes happen to do today."
//
// The Python side has enforced this with an AST scan since D046. The TypeScript
// side had nothing, and that gap is unusually sharp: a fused
// `Math.fround(a * b + c)` would leave **every** JavaScript ULP test green,
// because an FMA is *more* accurate than two rounded operations — it would score
// BETTER against the mpmath reference. Nothing in this suite would notice. The
// only check that would is the cross-language parity harness, which lives in the
// Python suite, so anyone running `npm test` alone would see a passing, more
// accurate, and wrong implementation.
//
// Wrong, not merely different: the two predictors must agree to exactly 0.0, and
// float32 semantics are simulable only because each `+ - * /` is separately
// rounded (FORMAT.md §5.1). A contracted multiply-add rounds once, so the JS
// result would differ from Python's in the last bit and parity would break — or,
// worse, hold on the corpus and break on a user's row.
//
// This uses the real TypeScript AST via the compiler API rather than a text
// scan. `typescript` is already a dev dependency for `tsc`, so no new dependency
// is introduced, and a comment or a string containing `*` cannot fool it.

import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const ts = require("typescript");

const SOURCE_PATH = new URL("../src/transform.ts", import.meta.url);
const SOURCE = readFileSync(SOURCE_PATH, "utf8");

/** The arithmetic operators FORMAT.md §5.1 permits, one per statement. */
const ARITHMETIC = new Set([
  ts.SyntaxKind.PlusToken,
  ts.SyntaxKind.MinusToken,
  ts.SyntaxKind.AsteriskToken,
  ts.SyntaxKind.SlashToken,
  ts.SyntaxKind.AsteriskAsteriskToken,
]);

function parse() {
  return ts.createSourceFile("transform.ts", SOURCE, ts.ScriptTarget.ES2020, true);
}

/**
 * Every statement in the file, paired with the arithmetic operators it contains.
 *
 * Counted per *statement* rather than per expression, because that is what the
 * rule constrains: one operation, one named intermediate.
 */
function statementsWithArithmetic() {
  const file = parse();
  const found = [];

  const visitStatement = (statement) => {
    const operators = [];
    const walk = (node) => {
      // Do not descend into a nested block or function body: `visit` reaches
      // those statements on their own, and counting them here would make a
      // function *declaration* -- which is itself a statement -- appear to
      // contain every operation in its body. That is exactly what the first
      // version of this scan did, reporting expF32 as one 25-operator statement.
      if (ts.isBlock(node) || ts.isFunctionLike(node)) {
        return;
      }
      if (ts.isBinaryExpression(node) && ARITHMETIC.has(node.operatorToken.kind)) {
        operators.push(node.operatorToken.getText(file));
      }
      // A unary minus is a sign, not an operation, and `-x` is exact in IEEE-754.
      ts.forEachChild(node, walk);
    };
    ts.forEachChild(statement, walk);
    if (operators.length > 0) {
      const { line } = file.getLineAndCharacterOfPosition(statement.getStart(file));
      found.push({
        line: line + 1,
        operators,
        text: statement.getText(file).replace(/\s+/g, " ").slice(0, 100),
      });
    }
  };

  const visit = (node) => {
    if (ts.isBlock(node) || ts.isSourceFile(node) || ts.isModuleBlock(node)) {
      node.statements.forEach(visitStatement);
    }
    ts.forEachChild(node, visit);
  };
  visit(file);
  return found;
}

test("transform.ts parses, so a scan that found nothing would be noticed", () => {
  const file = parse();
  assert.equal(file.parseDiagnostics.length, 0, "transform.ts must parse cleanly");
  const statements = statementsWithArithmetic();
  // The whole point of the file is arithmetic. If this were 0, the scan below
  // would pass vacuously -- the failure mode the Python side's scan also guards.
  assert.ok(
    statements.length >= 20,
    `only ${statements.length} arithmetic statements found; the scan is not reaching the code`,
  );
});

test("no statement in transform.ts contains more than one arithmetic operation", () => {
  const offenders = statementsWithArithmetic().filter((s) => s.operators.length > 1);
  assert.deepEqual(
    offenders,
    [],
    "FORMAT.md §5.5 requires one operation per statement with a named intermediate.\n" +
      offenders
        .map((o) => `  line ${o.line}: ${o.operators.join(" and ")} in \`${o.text}\``)
        .join("\n"),
  );
});

test("the scan detects a fused multiply-add, which is the case it exists for", () => {
  // Verify-by-reverting, inline: the guard is only worth having if it fires. A
  // fused expression is injected into a copy of the AST input and must be caught.
  const fused = SOURCE.replace(
    "export function expF32",
    "function __fusedProbe(a: number, b: number, c: number): number {\n" +
      "  return Math.fround(a * b + c);\n" +
      "}\n\nexport function expF32",
  );
  assert.notEqual(fused, SOURCE, "the injection anchor must exist");

  const file = ts.createSourceFile("probe.ts", fused, ts.ScriptTarget.ES2020, true);
  let worst = 0;
  const visitStatement = (statement) => {
    let count = 0;
    const walk = (node) => {
      if (ts.isBlock(node) || ts.isFunctionLike(node)) return;
      if (ts.isBinaryExpression(node) && ARITHMETIC.has(node.operatorToken.kind)) count += 1;
      ts.forEachChild(node, walk);
    };
    ts.forEachChild(statement, walk);
    worst = Math.max(worst, count);
  };
  const visit = (node) => {
    if (ts.isBlock(node) || ts.isSourceFile(node) || ts.isModuleBlock(node)) {
      node.statements.forEach(visitStatement);
    }
    ts.forEachChild(node, visit);
  };
  visit(file);
  assert.ok(worst >= 2, "the scan failed to see a fused multiply-add it was shown");
});

test("no platform transcendental is called anywhere in transform.ts", () => {
  // §5.4. The AST catches the call form specifically, so `Math.exp` inside a
  // comment or a string cannot trip it and cannot hide from it either.
  const file = parse();
  const forbidden = new Set(["exp", "log", "pow", "expm1", "log1p", "sinh", "cosh", "tanh"]);
  const calls = [];
  const walk = (node) => {
    if (
      ts.isCallExpression(node) &&
      ts.isPropertyAccessExpression(node.expression) &&
      node.expression.expression.getText(file) === "Math" &&
      forbidden.has(node.expression.name.getText(file))
    ) {
      const { line } = file.getLineAndCharacterOfPosition(node.getStart(file));
      calls.push(`line ${line + 1}: Math.${node.expression.name.getText(file)}`);
    }
    ts.forEachChild(node, walk);
  };
  ts.forEachChild(file, walk);
  assert.deepEqual(calls, [], `transform.ts must call no platform transcendental:\n${calls.join("\n")}`);
});
