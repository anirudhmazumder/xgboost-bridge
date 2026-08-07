/**
 * The artifact reader: types, load-time validation, and structural narrowing.
 *
 * This is where float32 discipline is most easily lost, and losing it here is
 * invisible downstream. `JSON.parse` returns float64 unconditionally, and on
 * 104/104 measured thresholds that float64 is a *different number* from the
 * one XGBoost's engine compares against. So `node_values` is loaded into a
 * `Float32Array` **at parse time** rather than narrowed later at the comparison
 * site (FORMAT.md §9.2, D004, D047): narrowing becomes a property of the data
 * structure instead of a discipline every future reader has to remember. If it
 * lived at the comparison site, any *other* consumer of a threshold — a
 * re-serializer, an inspection utility, an arithmetic transform — would
 * silently get the float64 back.
 *
 * The same reasoning applies to `intercept`, which is narrowed on read and
 * never transformed in any way (D015, FORMAT.md §6). Negative zero is a
 * reachable, ordinary value here — `binary:logistic` at `base_score = 0.5`
 * produces exactly `-0` — and is never normalized.
 *
 * Validation is exhaustive and loud, per FORMAT.md §13. Nothing defaults,
 * nothing is guessed, nothing is skipped (D007). Two rules cut the other way
 * and are just as load-bearing:
 *
 * - A node **unreachable** from the root does not raise. Neutralized dead slots
 *   are legitimate artifact content (FORMAT.md §8.3), they are indistinguishable
 *   from a leaf carrying `0`, the walk never visits them, and a reader that
 *   rejected them would reject every pruned model.
 * - `objective` is **non-operative metadata** (D028). It is read exactly once,
 *   here, for the cross-check against `output_transform`, and no prediction path
 *   consults it.
 *
 * One refusal goes beyond §13's enumerated list: a cycle in the subgraph
 * reachable from a tree's root. Every child index can be in range and still
 * form a cycle, at which point the walk never terminates — and a
 * non-terminating predictor is a worse outcome for a caller than a raise, since
 * it cannot be caught. The check is confined to the reachable subgraph so that
 * §13's rule about unreachable nodes is untouched.
 */

import {
  PredictorError,
  UnrecognizedFieldError,
  UnsupportedObjectiveError,
  UnsupportedVersionError,
} from "./errors.js";
import { OUTPUT_FUNCTIONS, type OutputTransformName } from "./transform.js";
import type { ErrorCode } from "./types.js";

// ---------------------------------------------------------------------------
// Error codes
// ---------------------------------------------------------------------------

// `ErrorCode` in `types.ts` enumerates five codes and does not yet carry one
// for a structurally malformed artifact or for a refused feature value. The
// runtime code strings below are the truthful ones: a caller that switches on
// `code` reaches its default branch and handles the failure loudly, which is
// strictly better than being told a malformed tree was an unrecognized field.
// The single assertion per code is the whole of the type-level debt, and it is
// confined to these two lines.

/**
 * Raised when an artifact contradicts the format this reader was built from.
 *
 * Covers an absent required key, a value of the wrong JSON type, an
 * `output_transform` that is unknown or does not pair with `objective`, tree
 * arrays of unequal length, an out-of-range child index or `split_indices`, a
 * non-finite `node_values` entry or `intercept`, an empty or duplicated
 * `feature_names`, a cycle reachable from a tree's root, and a prediction input
 * that is not a mapping of numbers.
 *
 * The four structured attributes are the point: FORMAT.md §13 requires a caller
 * to be able to branch on *which* key, *which* index and *what was expected*
 * rather than parsing a message string.
 */
export class MalformedArtifactError extends PredictorError {
  /** The field that disagrees with the format. */
  public readonly field: string;
  /** What was found there, verbatim where that is representable. */
  public readonly value: unknown;
  /** What the format requires instead. */
  public readonly expected: string;
  /** Where in the document the field sits, or `undefined` at the top level. */
  public readonly location: string | undefined;

  constructor(field: string, value: unknown, expected: string, location?: string) {
    const where = location === undefined ? "" : ` in ${location}`;
    super(
      "MALFORMED_ARTIFACT",
      `Malformed artifact: ${field}${where} is ${describe(value)}; expected ${expected}.`,
    );
    this.name = "MalformedArtifactError";
    this.field = field;
    this.value = value;
    this.expected = expected;
    this.location = location;
  }
}

/**
 * Raised when a prediction input carries `+Infinity` or `-Infinity` (D022,
 * D045).
 *
 * `NaN` is *not* an error: it is the missing value and routes by
 * `default_left`. Infinity is refused because upstream is genuinely
 * inconsistent about it — it raises through `DMatrix` and is treated as an
 * ordinary comparable value through `inplace_predict`, so the same input yields
 * two different predictions depending on the call path. This package picks one
 * behaviour and pins it.
 *
 * The whole row is checked before the walk begins. A lazy check would make the
 * same invalid input raise or not depending on which branches a particular
 * model happens to take, turning the outcome into a property of the model
 * instead of the input.
 */
export class NonFiniteFeatureError extends PredictorError {
  /** Column index of the offending value. */
  public readonly index: number;
  /** Feature name of the offending value. */
  public readonly feature: string;
  /** The offending value, `Infinity` or `-Infinity`. */
  public readonly value: number;

  constructor(index: number, feature: string, value: number) {
    super(
      "NON_FINITE_FEATURE",
      `Non-finite feature value: "${feature}" (column ${index}) is ${String(value)}. ` +
        "NaN is the missing value and is accepted; infinity is refused.",
    );
    this.name = "NonFiniteFeatureError";
    this.index = index;
    this.feature = feature;
    this.value = value;
  }
}

/** A short, safe rendering of an arbitrary parsed JSON value for a message. */
function describe(value: unknown): string {
  if (value === undefined) {
    return "<absent>";
  }
  if (typeof value === "number") {
    // `JSON.stringify(-0)` emits `0`, destroying the sign of a value this
    // format treats as distinct. `String(-0)` does too, so the sign is spelled
    // out rather than formatted.
    if (Object.is(value, -0)) {
      return "-0";
    }
    return String(value);
  }
  if (typeof value === "string") {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `an array of length ${value.length}`;
  }
  if (value === null) {
    return "null";
  }
  if (typeof value === "object") {
    return "an object";
  }
  return `${typeof value} ${String(value)}`;
}

// ---------------------------------------------------------------------------
// The format's fixed key sets and enumerations
// ---------------------------------------------------------------------------

/**
 * The only `format_version` this reader accepts. Not a range and not a floor:
 * exactly the integer `1`. The marker is the migration mechanism (D003, D007),
 * so an unrecognized value raises instead of being read best-effort.
 */
export const READABLE_FORMAT_VERSION = 1;

/** The seven required top-level keys of FORMAT.md §3, and no others. */
export const ENVELOPE_KEYS: readonly string[] = Object.freeze([
  "feature_names",
  "format_version",
  "intercept",
  "objective",
  "output_transform",
  "provenance",
  "trees",
]);

/** `provenance`'s own fixed key set (FORMAT.md §2, §15). Read by no prediction path. */
export const PROVENANCE_KEYS: readonly string[] = Object.freeze([
  "base_score",
  "exporter_version",
  "xgboost_version",
]);

/** The five parallel arrays of FORMAT.md §8, one entry per node. */
export const TREE_KEYS: readonly string[] = Object.freeze([
  "default_left",
  "left_children",
  "node_values",
  "right_children",
  "split_indices",
]);

/** The objectives FORMAT.md §4 enumerates. Anything else raises. */
export const SUPPORTED_OBJECTIVES: readonly string[] = Object.freeze([
  "binary:logistic",
  "reg:squarederror",
  "survival:cox",
]);

/**
 * The measured objective/transform pairing of FORMAT.md §5.
 *
 * A table rather than a chain of comparisons, deliberately: this is the one
 * place `objective` is consulted at all, and keeping it a lookup is what makes
 * the "no branch on `objective`" rule (D028) checkable in the shipped bundle.
 */
const PAIRED_TRANSFORM: Readonly<Record<string, OutputTransformName>> = Object.freeze({
  "binary:logistic": "sigmoid",
  "reg:squarederror": "identity",
  "survival:cox": "exp",
});

/** Child index that marks a leaf. A node is a leaf **iff** its left child is this. */
export const LEAF_CHILD = -1;

// ---------------------------------------------------------------------------
// Loaded shapes
// ---------------------------------------------------------------------------

/**
 * One tree, loaded. `nodeValues` is a `Float32Array`, which is the crux: it
 * carries the split threshold at an internal node and the output value at a
 * leaf (FORMAT.md §8.1), so one act of construction narrows both roles.
 */
export interface LoadedTree {
  readonly leftChildren: Int32Array;
  readonly rightChildren: Int32Array;
  readonly splitIndices: Int32Array;
  readonly nodeValues: Float32Array;
  readonly defaultLeft: Uint8Array;
}

/** A validated artifact with every numeric value already float32. */
export interface LoadedArtifact {
  readonly formatVersion: number;
  readonly objective: string;
  readonly outputTransform: OutputTransformName;
  readonly featureNames: readonly string[];
  readonly intercept: number;
  readonly provenance: Readonly<Record<string, string>>;
  readonly trees: readonly LoadedTree[];
}

// ---------------------------------------------------------------------------
// Primitive readers
// ---------------------------------------------------------------------------

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function requireObject(
  value: unknown,
  field: string,
  location?: string,
): Record<string, unknown> {
  if (!isPlainObject(value)) {
    throw new MalformedArtifactError(field, value, "a JSON object", location);
  }
  return value;
}

/**
 * Require exactly `allowed`: no unrecognized key, no absent key.
 *
 * Unrecognized keys are reported first and in sorted order, so the same
 * malformed artifact always produces the same error rather than one that
 * depends on insertion order. An unrecognized key gets
 * {@link UnrecognizedFieldError}, which is what that class is for; an absent
 * one gets {@link MalformedArtifactError}, because a missing optional field is
 * not an unknown field and conflating the two is how a relocation goes
 * unnoticed (D018).
 */
function checkKeys(
  container: Record<string, unknown>,
  allowed: readonly string[],
  location?: string,
): void {
  const permitted = new Set(allowed);
  const unrecognized = Object.keys(container)
    .filter((key) => !permitted.has(key))
    .sort();
  const first = unrecognized[0];
  if (first !== undefined) {
    throw new UnrecognizedFieldError(location === undefined ? first : `${location}.${first}`);
  }
  for (const key of allowed) {
    if (!Object.prototype.hasOwnProperty.call(container, key)) {
      throw new MalformedArtifactError(key, undefined, "the field to be present", location);
    }
  }
}

function requireArray(value: unknown, field: string, location?: string): unknown[] {
  if (!Array.isArray(value)) {
    throw new MalformedArtifactError(field, value, "a JSON array", location);
  }
  return value as unknown[];
}

/**
 * Read one JSON array of integers, rejecting anything else.
 *
 * Booleans are excluded: FORMAT.md §8 specifies `default_left` as `0`/`1`
 * integers rather than JSON booleans, and a child index is never a boolean
 * either. A non-integral number is rejected too — `1.5` is not a node index.
 */
function readIntegerArray(raw: unknown, field: string, location: string): number[] {
  const entries = requireArray(raw, field, location);
  const out: number[] = new Array<number>(entries.length);
  for (let position = 0; position < entries.length; position += 1) {
    const value = entries[position];
    if (typeof value !== "number" || !Number.isInteger(value)) {
      throw new MalformedArtifactError(
        field,
        value,
        `an integer at position ${position}`,
        location,
      );
    }
    out[position] = value;
  }
  return out;
}

/**
 * Narrow one JSON number to float32, raising unless the result is finite.
 *
 * The finiteness check is deliberately *after* the narrowing: `1e40` is a
 * perfectly finite double and becomes `Infinity` in float32, which FORMAT.md
 * §9.3 forbids.
 */
function narrow(
  value: unknown,
  field: string,
  position: number | undefined,
  location: string | undefined,
): number {
  const where = position === undefined ? "" : ` at position ${position}`;
  if (typeof value !== "number") {
    throw new MalformedArtifactError(field, value, `a JSON number${where}`, location);
  }
  const narrowed = Math.fround(value);
  if (!Number.isFinite(narrowed)) {
    throw new MalformedArtifactError(
      field,
      value,
      `a finite float32 value${where}`,
      location,
    );
  }
  return narrowed;
}

/**
 * Load `node_values` into a `Float32Array` — the narrowing site of FORMAT.md
 * §9.2, and the reason this module exists.
 *
 * Every entry is type-checked before construction and finiteness is checked
 * after it, on the narrowed value. Leaf values need the narrowing exactly as
 * much as thresholds do: without it, accumulation scored 990–3706/5000
 * bit-exact and breached the `1e-6` gate at `1.07e-04`.
 */
function readNodeValues(raw: unknown, location: string): Float32Array {
  const entries = requireArray(raw, "node_values", location);
  for (let position = 0; position < entries.length; position += 1) {
    if (typeof entries[position] !== "number") {
      throw new MalformedArtifactError(
        "node_values",
        entries[position],
        `a JSON number at position ${position}`,
        location,
      );
    }
  }
  const values = new Float32Array(entries.length);
  for (let position = 0; position < entries.length; position += 1) {
    values[position] = entries[position] as number;
  }
  for (let position = 0; position < values.length; position += 1) {
    if (!Number.isFinite(values[position] as number)) {
      throw new MalformedArtifactError(
        "node_values",
        entries[position],
        `a finite float32 value at position ${position}`,
        location,
      );
    }
  }
  return values;
}

// ---------------------------------------------------------------------------
// Envelope fields
// ---------------------------------------------------------------------------

function readFormatVersion(envelope: Record<string, unknown>): number {
  const value = envelope["format_version"];
  if (typeof value !== "number" || !Number.isInteger(value) || value !== READABLE_FORMAT_VERSION) {
    throw new UnsupportedVersionError(value);
  }
  return value;
}

/**
 * Read `objective` and check it against the enumerated set.
 *
 * This is the *only* place in this package that reads the field, and it runs at
 * load time. Nothing on the prediction path consults it (D028).
 */
function readObjective(envelope: Record<string, unknown>): string {
  const value = envelope["objective"];
  // A non-string is the wrong JSON type, which is a different failure from a
  // string naming an objective this version does not implement. The two carry
  // different error classes so a caller can tell them apart.
  if (typeof value !== "string") {
    throw new MalformedArtifactError("objective", value, "a JSON string");
  }
  if (!SUPPORTED_OBJECTIVES.includes(value)) {
    throw new UnsupportedObjectiveError(value);
  }
  return value;
}

/**
 * Read `output_transform` and require it to pair with `objective`.
 *
 * The pairing check is the whole reason `objective` is carried at all
 * (FORMAT.md §4, §13). Performing it here, once, is what keeps the prediction
 * path free of it: after this function returns, the transform is a lookup and
 * the objective is a label.
 */
function readOutputTransform(
  envelope: Record<string, unknown>,
  objective: string,
): OutputTransformName {
  const value = envelope["output_transform"];
  const names = Object.keys(OUTPUT_FUNCTIONS).sort();
  if (typeof value !== "string" || !names.includes(value)) {
    throw new MalformedArtifactError(
      "output_transform",
      value,
      `one of: ${names.join(", ")}`,
    );
  }
  const paired = PAIRED_TRANSFORM[objective];
  if (paired === undefined) {
    throw new MalformedArtifactError(
      "objective",
      objective,
      "an objective with a recorded output_transform pairing",
    );
  }
  if (value !== paired) {
    throw new MalformedArtifactError(
      "output_transform",
      value,
      `"${paired}", the transform paired with objective "${objective}"`,
    );
  }
  return value as OutputTransformName;
}

/**
 * Read `feature_names`: a non-empty array of unique strings (D021).
 *
 * Emptiness is a refusal rather than a degenerate case. A strict-key policy
 * with no keys to check reads as enforced and is not, which is worse than no
 * policy at all because the caller believes a typo will be caught.
 */
function readFeatureNames(envelope: Record<string, unknown>): string[] {
  const entries = requireArray(envelope["feature_names"], "feature_names");
  const names: string[] = new Array<string>(entries.length);
  for (let position = 0; position < entries.length; position += 1) {
    const name = entries[position];
    if (typeof name !== "string") {
      throw new MalformedArtifactError(
        "feature_names",
        name,
        `a string at position ${position}`,
      );
    }
    names[position] = name;
  }
  if (names.length === 0) {
    throw new MalformedArtifactError(
      "feature_names",
      entries,
      "at least one name; a strict-key policy needs keys to check",
    );
  }
  const seen = new Set<string>();
  for (let position = 0; position < names.length; position += 1) {
    const name = names[position] as string;
    if (seen.has(name)) {
      throw new MalformedArtifactError(
        "feature_names",
        name,
        `no duplicate names (position ${position} repeats an earlier entry)`,
      );
    }
    seen.add(name);
  }
  return names;
}

/**
 * Read `provenance`: exactly three keys, all JSON strings.
 *
 * No prediction path reads any of them. They are validated because FORMAT.md
 * §13 makes an unrecognized key and a wrong JSON type refusals wherever they
 * occur, not only where they would change a number. `base_score` here is the
 * raw bracketed string XGBoost stored, e.g. `"[6E-1]"`, and is deliberately
 * not parsed.
 */
function readProvenance(envelope: Record<string, unknown>): Record<string, string> {
  const provenance = requireObject(envelope["provenance"], "provenance");
  checkKeys(provenance, PROVENANCE_KEYS, "provenance");
  const out: Record<string, string> = {};
  for (const key of PROVENANCE_KEYS) {
    const value = provenance[key];
    if (typeof value !== "string") {
      throw new MalformedArtifactError(key, value, "a JSON string", "provenance");
    }
    out[key] = value;
  }
  return Object.freeze(out);
}

// ---------------------------------------------------------------------------
// Trees
// ---------------------------------------------------------------------------

/**
 * Require every child index to be a leaf marker or an in-range index.
 *
 * Both children are `-1` at a leaf. A leaf whose `right_children` entry is
 * something else is the vector-leaf signature, where that slot carries a block
 * index instead of a child — a shape v1 refuses rather than walks.
 *
 * Note what is deliberately *not* required: that a child index exceed its
 * parent's. FORMAT.md §8 does not make that normative for an artifact, so
 * demanding it here would refuse a conforming artifact from another producer.
 * Termination is enforced directly instead, by
 * {@link checkReachableSubgraphTerminates}.
 */
function checkChildLinks(
  leftChildren: readonly number[],
  rightChildren: readonly number[],
  location: string,
): void {
  const nodeCount = leftChildren.length;
  for (let index = 0; index < nodeCount; index += 1) {
    const leftChild = leftChildren[index] as number;
    const rightChild = rightChildren[index] as number;
    if (leftChild === LEAF_CHILD) {
      if (rightChild !== LEAF_CHILD) {
        throw new MalformedArtifactError(
          "right_children",
          rightChild,
          `-1 at node ${index}, which left_children marks as a leaf`,
          location,
        );
      }
      continue;
    }
    if (!(leftChild >= 0 && leftChild < nodeCount)) {
      throw new MalformedArtifactError(
        "left_children",
        leftChild,
        `-1, or an index in [0, ${nodeCount}) at node ${index}`,
        location,
      );
    }
    if (!(rightChild >= 0 && rightChild < nodeCount)) {
      throw new MalformedArtifactError(
        "right_children",
        rightChild,
        `-1, or an index in [0, ${nodeCount}) at node ${index}`,
        location,
      );
    }
  }
}

// Depth-first search colours for the termination check below.
const UNVISITED = 0;
const ON_PATH = 1;
const SETTLED = 2;

/**
 * Raise if a cycle is reachable from the root.
 *
 * Every other refusal in this module is against a wrong number. This one is
 * against a **hang**: the walk follows children until it meets a leaf, so a
 * cycle among reachable nodes never terminates, and a non-terminating
 * predictor is not something a caller can catch. Confined to the reachable
 * subgraph, because FORMAT.md §13 forbids raising on an unreachable node
 * whatever it contains.
 *
 * A shared subtree — two parents pointing at one child, no cycle — is not
 * refused. It terminates, so it is not this check's business.
 */
// Exported so the `Predictor` constructor can establish the same property. That
// constructor is public and is not required to go through `fromJSON`, and a
// cyclic child set is the one malformed input whose consequence is a hang rather
// than a throw -- "not something a caller can catch". `ArrayLike<number>` rather
// than `readonly number[]` so it accepts both the raw parsed arrays here and the
// `Int32Array`s a `LoadedTree` carries; the body uses only `.length` and index
// access.
export function checkReachableSubgraphTerminates(
  leftChildren: ArrayLike<number>,
  rightChildren: ArrayLike<number>,
  location: string,
): void {
  const colour = new Uint8Array(leftChildren.length);
  // [node, revisiting] — the second pass over a node marks it settled, which
  // is what turns a plain reachability walk into cycle detection. Iterative
  // rather than recursive: a deep tree would otherwise depend on the engine's
  // stack depth.
  const pending: [number, boolean][] = [[0, false]];
  while (pending.length > 0) {
    const frame = pending.pop() as [number, boolean];
    const node = frame[0];
    if (frame[1]) {
      colour[node] = SETTLED;
      continue;
    }
    if (colour[node] !== UNVISITED) {
      continue;
    }
    colour[node] = ON_PATH;
    pending.push([node, true]);
    if (leftChildren[node] === LEAF_CHILD) {
      continue;
    }
    for (const child of [leftChildren[node] as number, rightChildren[node] as number]) {
      if (colour[child] === ON_PATH) {
        throw new MalformedArtifactError(
          "left_children",
          child,
          `a child that does not close a cycle back to node ${child}; a cycle ` +
            "reachable from the root would make the walk non-terminating",
          location,
        );
      }
      if (colour[child] === UNVISITED) {
        pending.push([child, false]);
      }
    }
  }
}

/** Read one tree object into the five arrays FORMAT.md §8 specifies. */
function readTree(raw: unknown, index: number, featureCount: number): LoadedTree {
  const location = `trees[${index}]`;
  const tree = requireObject(raw, "<tree>", location);
  checkKeys(tree, TREE_KEYS, location);

  const leftChildren = readIntegerArray(tree["left_children"], "left_children", location);
  const rightChildren = readIntegerArray(tree["right_children"], "right_children", location);
  const splitIndices = readIntegerArray(tree["split_indices"], "split_indices", location);
  const defaultLeft = readIntegerArray(tree["default_left"], "default_left", location);
  const nodeValues = readNodeValues(tree["node_values"], location);

  const nodeCount = leftChildren.length;
  if (nodeCount === 0) {
    throw new MalformedArtifactError(
      "left_children",
      0,
      "at least one node, since node 0 is the root",
      location,
    );
  }
  for (const [field, entries] of [
    ["right_children", rightChildren],
    ["split_indices", splitIndices],
    ["default_left", defaultLeft],
  ] as const) {
    if (entries.length !== nodeCount) {
      throw new MalformedArtifactError(
        field,
        entries.length,
        `length ${nodeCount}, matching left_children`,
        location,
      );
    }
  }
  if (nodeValues.length !== nodeCount) {
    throw new MalformedArtifactError(
      "node_values",
      nodeValues.length,
      `length ${nodeCount}, matching left_children`,
      location,
    );
  }

  for (let node = 0; node < nodeCount; node += 1) {
    // Total, not conditional on the node being internal: neutralized dead
    // slots carry `split_indices == 0` precisely so this check needs no
    // exception (FORMAT.md §8.3). An exception here would have to apply to
    // every artifact rather than only pruned ones.
    const column = splitIndices[node] as number;
    if (!(column >= 0 && column < featureCount)) {
      throw new MalformedArtifactError(
        "split_indices",
        column,
        `an index in [0, ${featureCount}) at node ${node}`,
        location,
      );
    }
    const direction = defaultLeft[node] as number;
    if (direction !== 0 && direction !== 1) {
      throw new MalformedArtifactError(
        "default_left",
        direction,
        `0 or 1 at node ${node}`,
        location,
      );
    }
  }

  checkChildLinks(leftChildren, rightChildren, location);
  checkReachableSubgraphTerminates(leftChildren, rightChildren, location);

  return {
    leftChildren: Int32Array.from(leftChildren),
    rightChildren: Int32Array.from(rightChildren),
    splitIndices: Int32Array.from(splitIndices),
    nodeValues,
    defaultLeft: Uint8Array.from(defaultLeft),
  };
}

/**
 * Read `trees` in artifact order, which is normative (FORMAT.md §8.2).
 *
 * An empty array is valid and is not a special case: a zero-boosting-round
 * model serializes `"trees": []`, present and empty, and its margin is the
 * intercept alone.
 */
function readTrees(envelope: Record<string, unknown>, featureCount: number): LoadedTree[] {
  const entries = requireArray(envelope["trees"], "trees");
  return entries.map((entry, index) => readTree(entry, index, featureCount));
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/**
 * Validate a parsed artifact and load it with every numeric value narrowed to
 * float32.
 *
 * Field order below is deliberate: the envelope's key set is checked before any
 * value is read, so an artifact with an eighth key or a missing key raises on
 * that rather than on whatever the extra key happens to break first.
 *
 * @param artifact - The artifact as a parsed JSON object — what `JSON.parse`
 *   returns. Not a path and not a string: this reader does no I/O (D006).
 * @throws {MalformedArtifactError} on any disagreement with FORMAT.md §13.
 */
export function loadArtifact(artifact: unknown): LoadedArtifact {
  const envelope = requireObject(artifact, "<artifact>");
  checkKeys(envelope, ENVELOPE_KEYS);

  const formatVersion = readFormatVersion(envelope);
  const objective = readObjective(envelope);
  const outputTransform = readOutputTransform(envelope, objective);
  const featureNames = readFeatureNames(envelope);
  const intercept = narrow(envelope["intercept"], "intercept", undefined, undefined);
  const provenance = readProvenance(envelope);
  const trees = readTrees(envelope, featureNames.length);

  return Object.freeze({
    formatVersion,
    objective,
    outputTransform,
    featureNames: Object.freeze(featureNames),
    intercept,
    provenance,
    trees: Object.freeze(trees),
  });
}
