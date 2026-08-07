/**
 * The predictor: the normative margin walk of FORMAT.md §10, plus the bundled
 * output transform.
 *
 * Every rule below was measured, and every deviation from it loses. The table
 * of costs, from `probes/accumulation.md` §6 and `probes/float32_thresholds.md`
 * §6:
 *
 * | Rule | Cost of the alternative |
 * |---|---|
 * | Both sides of the comparison cast to float32 | One-sided: a 6.6-percentage-point probability error on a real row |
 * | Strict `<`, equality routes RIGHT | Measured on 104/104 internal nodes of the primary model plus 195 more across seven others |
 * | Intercept is the accumulator's initial value | Intercept last: 199–2120/5000 bit-exact |
 * | Trees in artifact order | Reversed: 245–2365/5000 |
 * | Narrow after every single add | A float64 sum narrowed once at the end: 318–2541/5000 |
 * | Leaf values narrowed on read | Un-narrowed: 990–3706/5000 |
 *
 * Correct implementation scores 5000/5000 bit-exact against
 * `predict(output_margin=True)` at max absolute error `0.0`.
 *
 * Two properties of this file are worth stating because they are easy to lose:
 *
 * - **`Math.fround` on both sides of the comparison, not just the sample.** The
 *   threshold arrives already narrowed, from the `Float32Array` the reader built
 *   (FORMAT.md §9.2), so the threshold-side cast is idempotent for artifacts
 *   *this* exporter produced. It stays because it is the defence for artifacts
 *   it did not produce — hand-edited, third-party, or a future revision — and
 *   because a walk whose correctness depends on a property of its input is a
 *   walk that is correct until someone changes the input (D045).
 * - **`objective` is never read here.** `output_transform` alone selects the
 *   transform (D028). The companion test greps the shipped bundle to keep it
 *   that way: once a prediction path branches on `objective`, that field becomes
 *   a second source of truth about behaviour the first one already determines.
 *
 * There is no `fromFile` (D006). Filesystem access is unavailable in browsers
 * and differs across edge runtimes, so a loader would either add a dependency
 * or split the bundle by runtime. Consumers do their own I/O and call
 * {@link fromJSON}.
 */

import { FeatureKeyMismatchError } from "./errors.js";
import {
  LEAF_CHILD,
  MalformedArtifactError,
  NonFiniteFeatureError,
  checkReachableSubgraphIsATree,
  loadArtifact,
  type LoadedArtifact,
  type LoadedTree,
} from "./artifact.js";
import {
  OUTPUT_FUNCTIONS,
  OUTPUT_TRANSFORM_NAMES,
  type OutputTransformName,
} from "./transform.js";
import type { PredictionInput } from "./types.js";

/**
 * Read `index` from a typed array, raising rather than returning `undefined`.
 *
 * Every index the walk uses is range-checked at load time, so this cannot fire
 * on a validated artifact. It exists because the failure it prevents is
 * specifically the silent kind: an out-of-range typed-array read yields
 * `undefined`, `Math.fround(undefined)` is `NaN`, and `NaN` is this format's
 * *missing value* — so the walk would route down a legitimate branch and return
 * a confident wrong number instead of failing.
 */
function elementAt(
  array: Int32Array | Uint8Array | Float32Array | Float64Array,
  index: number,
): number {
  const value = array[index];
  if (value === undefined) {
    throw new MalformedArtifactError(
      "<node index>",
      index,
      `an index in [0, ${array.length})`,
    );
  }
  return value;
}

/**
 * A loaded model, ready to produce margins and outputs row by row.
 *
 * Construct with {@link fromJSON}. Every numeric value is float32 from the
 * moment it was read: `intercept` is the exact float32 the accumulator starts
 * at, and each tree's thresholds and leaf values live in a `Float32Array`.
 */
export class Predictor {
  /** The artifact's format version. Always `1` for a loaded artifact. */
  public readonly formatVersion: number;

  /**
   * The objective recorded in the artifact — **metadata only** (D028).
   *
   * Exposed for inspection. No prediction path reads it, and the test suite
   * asserts that against the shipped bundle.
   */
  public readonly objective: string;

  /** The name of the transform {@link output} applies to the margin. */
  public readonly outputTransform: OutputTransformName;

  /** The exact key set a prediction input must carry (D005). */
  public readonly featureNames: readonly string[];

  /**
   * The float32 margin-space intercept, exactly as loaded.
   *
   * Never transformed, and never normalized: `-0` is a reachable value —
   * `binary:logistic` at `base_score = 0.5` produces it — and stays `-0`.
   */
  public readonly intercept: number;

  /** The artifact's provenance block. Read by no prediction path. */
  public readonly provenance: Readonly<Record<string, string>>;

  /** The loaded trees in artifact order, which is normative (FORMAT.md §8.2). */
  public readonly trees: readonly LoadedTree[];

  private readonly featureNameSet: ReadonlySet<string>;
  private readonly outputFunction: (margin: number) => number;

  /**
   * Load an already-validated artifact.
   *
   * `fromJSON` is the ordinary entry point and validates everything; this
   * constructor is public, so it validates the one field it *uses*.
   * `outputTransform` must name a transform this package implements, as an
   * **own** property of {@link OUTPUT_FUNCTIONS}. Without that check a name
   * such as `"constructor"` or `"valueOf"` resolved through the prototype
   * chain and `output` returned a boxed `Number`: it serializes as the right
   * number and arithmetic on it gives the right number, while an `Object.is`
   * or bit-pattern comparison against it fails. A wrong number that looks
   * right is the one outcome this package refuses to produce, so an
   * unrecognized name throws instead.
   *
   * @throws {MalformedArtifactError} if `outputTransform` is not one of the
   *   three names FORMAT.md §5 defines.
   */
  constructor(loaded: LoadedArtifact) {
    // Before any field is stored: an unrecognized transform means this object
    // has no honest `output`, so it never comes into existence.
    if (!Object.prototype.hasOwnProperty.call(OUTPUT_FUNCTIONS, loaded.outputTransform)) {
      throw new MalformedArtifactError(
        "output_transform",
        loaded.outputTransform,
        `one of: ${OUTPUT_TRANSFORM_NAMES.join(", ")}`,
      );
    }

    // The walk below assumes every path from a root reaches a leaf. `fromJSON`
    // establishes that, but this constructor is public and a caller can hand it
    // a hand-built `LoadedArtifact` -- and a cycle there does not throw, it
    // spins forever, which is the one failure mode a caller cannot catch. So the
    // property is established here rather than assumed, for the same reason the
    // transform name is checked above: a walk whose correctness depends on a
    // property of its input is correct until someone changes the input.
    //
    // This repeats the check `fromJSON` already ran -- one extra O(nodes) pass at
    // load time, on a path that is not the hot one. The alternative is a flag
    // recording whether validation happened, which is the kind of coupling that
    // goes stale silently.
    loaded.trees.forEach((tree, index) => {
      checkReachableSubgraphIsATree(
        tree.leftChildren,
        tree.rightChildren,
        `trees[${index}]`,
      );
    });

    this.formatVersion = loaded.formatVersion;
    this.objective = loaded.objective;
    this.outputTransform = loaded.outputTransform;
    this.featureNames = loaded.featureNames;
    this.intercept = loaded.intercept;
    this.provenance = loaded.provenance;
    this.trees = loaded.trees;
    this.featureNameSet = new Set(loaded.featureNames);
    // Selected once, from `output_transform`, by an own-property lookup the
    // guard above has already established resolves. After this line the
    // transform is a function this object holds and nothing consults a name
    // again.
    this.outputFunction = OUTPUT_FUNCTIONS[loaded.outputTransform];
  }

  /**
   * Return one row's float32 margin: the accumulator of FORMAT.md §10,
   * untouched after the last addition.
   *
   * @param row - A mapping whose key set equals {@link featureNames}
   *   **exactly** — no missing key, no extra key (D005). `NaN` is the missing
   *   value and routes by the node's `default_left`.
   * @throws {FeatureKeyMismatchError} if the key set is not exactly
   *   {@link featureNames}. Lenient handling would turn a typo into a
   *   missing-value path, which is legitimate model structure, so the mistake
   *   would become a confident wrong number instead of an error.
   * @throws {NonFiniteFeatureError} if any value is `±Infinity` (D022, D045).
   * @throws {MalformedArtifactError} if `row` is not an object, or a value is
   *   not a number.
   */
  public margin(row: PredictionInput): number {
    return this.walk(this.featureRow(row));
  }

  /**
   * Return one row's float32 output: the transform applied to the margin.
   *
   * The transform is the bundled float32 implementation (D030, D032) —
   * `sigmoid` reproduces XGBoost's measured clamp floor and `exp` reproduces
   * its overflow to `+Infinity`. No platform transcendental is called.
   *
   * Throws the same errors as {@link margin}, for the same reasons.
   */
  public output(row: PredictionInput): number {
    return this.outputFunction(this.margin(row));
  }

  /**
   * The normative walk of FORMAT.md §10, over an already-ordered row.
   *
   * The accumulator is initialized with the float32 intercept **before any
   * tree**, trees are visited in artifact order, and the accumulator is
   * narrowed to float32 **after every single addition**.
   */
  private walk(values: Float64Array): number {
    let accumulator = Math.fround(this.intercept);

    for (const tree of this.trees) {
      let node = 0;
      // Leaf iff the left child is -1. This is the only leaf test measured to
      // hold in every observed tree shape: `right_children[i] === -1`
      // coincides at scalar leaves but carries a block index in a vector-leaf
      // tree, so a test that is accidentally correct would be a latent bug.
      while (elementAt(tree.leftChildren, node) !== LEAF_CHILD) {
        const value = elementAt(values, elementAt(tree.splitIndices, node));
        if (value !== value) {
          // NaN is the missing value. `default_left` is validated to 0 or 1 at
          // load, and re-checked here because "not 1" would pick a legal
          // direction and so produce a plausible wrong number rather than an
          // error.
          const direction = elementAt(tree.defaultLeft, node);
          if (direction === 1) {
            node = elementAt(tree.leftChildren, node);
          } else if (direction === 0) {
            node = elementAt(tree.rightChildren, node);
          } else {
            throw new MalformedArtifactError(
              "default_left",
              direction,
              `0 or 1 at node ${node}`,
            );
          }
        } else {
          // BOTH sides cast, strict `<`, equality therefore routes RIGHT.
          const sample = Math.fround(value);
          const threshold = Math.fround(elementAt(tree.nodeValues, node));
          if (sample < threshold) {
            node = elementAt(tree.leftChildren, node);
          } else {
            node = elementAt(tree.rightChildren, node);
          }
        }
      }
      const leafValue = Math.fround(elementAt(tree.nodeValues, node));
      accumulator = Math.fround(accumulator + leafValue);
    }

    return accumulator;
  }

  /**
   * Order a row's values by column index, as a `Float64Array`.
   *
   * Three decisions here, all deliberate:
   *
   * - The key set is compared for **exact** equality first, before a value is
   *   touched, and missing and extra keys are reported together so a typo —
   *   which is one of each — is diagnosed as a typo.
   * - The whole row is checked for `±Infinity` before the walk starts. A lazy
   *   check would make the same invalid input raise or not depending on which
   *   branches this particular model takes, i.e. a property of the model rather
   *   than of the input. The cost is O(features) against an O(depth × trees)
   *   walk (D045).
   * - The array is `Float64Array`, not `Float32Array`. The walk casts both sides
   *   of every comparison, so the result is identical either way, but a
   *   *pre-narrowed* row would make the walk's sample-side cast unobservable,
   *   and that cast is the highest-value invariant in this codebase.
   */
  private featureRow(row: PredictionInput): Float64Array {
    if (typeof row !== "object" || row === null || Array.isArray(row)) {
      throw new MalformedArtifactError(
        "<row>",
        row,
        "an object mapping feature name to number",
      );
    }

    const missing: string[] = [];
    for (const name of this.featureNames) {
      if (!Object.prototype.hasOwnProperty.call(row, name)) {
        missing.push(name);
      }
    }
    const extra = Object.keys(row)
      .filter((key) => !this.featureNameSet.has(key))
      .sort();
    if (missing.length > 0 || extra.length > 0) {
      throw new FeatureKeyMismatchError(missing, extra);
    }

    const values = new Float64Array(this.featureNames.length);
    for (let index = 0; index < this.featureNames.length; index += 1) {
      const name = this.featureNames[index] as string;
      const value = (row as Record<string, unknown>)[name];
      if (typeof value !== "number") {
        throw new MalformedArtifactError(`<row>.${name}`, value, "a number");
      }
      // Narrow first, then refuse an infinite result. Testing `value` itself for
      // `±Infinity` let a finite float64 that *becomes* infinite through this
      // library's own required narrowing straight through: `1e39` is a legal
      // float64, `Math.fround(1e39)` is `Infinity`, and the walk then compared
      // `Infinity` against thresholds and returned a number. Same mathematical
      // value as an explicit infinity, two different behaviours, no error
      // either way (D055).
      //
      // `NaN` is deliberately not caught here: `Math.fround(NaN)` is `NaN`,
      // which is neither `Infinity` nor `-Infinity`, so the missing value still
      // reaches the walk and routes by the tree's default direction.
      const narrowed = Math.fround(value);
      if (narrowed === Infinity || narrowed === -Infinity) {
        throw new NonFiniteFeatureError(index, name, value);
      }
      values[index] = value;
    }
    return values;
  }
}

/**
 * Load a parsed artifact and return a {@link Predictor}.
 *
 * @param artifact - The artifact as a parsed JSON object — what `JSON.parse`
 *   returns. There is deliberately no path-taking or string-taking variant
 *   (D006): consumers perform their own I/O.
 * @throws {UnsupportedVersionError} if `format_version` is not exactly the
 *   integer `1`.
 * @throws {UnrecognizedFieldError} if a key at any level is not one this format
 *   defines.
 * @throws {UnsupportedObjectiveError} if `objective` is outside the enumerated
 *   set.
 * @throws {MalformedArtifactError} on any other disagreement with FORMAT.md
 *   §13.
 */
export function fromJSON(artifact: unknown): Predictor {
  return new Predictor(loadArtifact(artifact));
}
