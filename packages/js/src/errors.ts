import type { ErrorCode } from "./types.js";

/**
 * Base class for every error this package raises.
 *
 * This package fails loudly rather than silently: an unrecognized
 * objective, booster, artifact field, version marker, or a feature-key
 * mismatch each raise a specific subclass instead of defaulting, guessing,
 * or skipping. Catch this type to handle every failure mode at once, or
 * catch a subclass to handle one specifically.
 */
export class PredictorError extends Error {
  /** Machine-readable discriminant for this failure mode. */
  public readonly code: ErrorCode;

  constructor(code: ErrorCode, message: string) {
    super(message);
    this.name = "PredictorError";
    this.code = code;
  }
}

/** Raised when an artifact declares an objective this package does not implement. */
export class UnsupportedObjectiveError extends PredictorError {
  /** The unrecognized objective string, verbatim from the artifact. */
  public readonly objective: string;

  constructor(objective: string) {
    super("UNSUPPORTED_OBJECTIVE", `Unsupported objective: "${objective}".`);
    this.name = "UnsupportedObjectiveError";
    this.objective = objective;
  }
}

/** Raised when an artifact declares a booster type this package does not implement. */
export class UnsupportedBoosterError extends PredictorError {
  /** The unrecognized booster string, verbatim from the artifact. */
  public readonly booster: string;

  constructor(booster: string) {
    super("UNSUPPORTED_BOOSTER", `Unsupported booster: "${booster}".`);
    this.name = "UnsupportedBoosterError";
    this.booster = booster;
  }
}

/** Raised when an artifact contains a field this package does not recognize. */
export class UnrecognizedFieldError extends PredictorError {
  /** The unrecognized field's path within the artifact. */
  public readonly field: string;

  constructor(field: string) {
    super("UNRECOGNIZED_FIELD", `Unrecognized artifact field: "${field}".`);
    this.name = "UnrecognizedFieldError";
    this.field = field;
  }
}

/**
 * Raised when an artifact's version marker is missing, malformed, or
 * outside the range this package supports.
 */
export class UnsupportedVersionError extends PredictorError {
  /** The version value found in the artifact, verbatim. */
  public readonly version: unknown;

  constructor(version: unknown) {
    super(
      "UNSUPPORTED_VERSION",
      `Unsupported artifact version: ${JSON.stringify(version)}.`,
    );
    this.name = "UnsupportedVersionError";
    this.version = version;
  }
}

/**
 * Raised when prediction input keys do not exactly match a model's
 * declared feature names. `missing` and `extra` are reported as separate
 * lists so the two failure modes stay distinguishable to the caller.
 */
export class FeatureKeyMismatchError extends PredictorError {
  /** Feature names the model requires that were absent from the input. */
  public readonly missing: readonly string[];
  /** Input keys that do not correspond to any declared feature name. */
  public readonly extra: readonly string[];

  constructor(missing: readonly string[], extra: readonly string[]) {
    super(
      "FEATURE_KEY_MISMATCH",
      `Feature key mismatch: missing [${missing.join(", ")}], extra [${extra.join(", ")}].`,
    );
    this.name = "FeatureKeyMismatchError";
    this.missing = missing;
    this.extra = extra;
  }
}
