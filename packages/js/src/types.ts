// Type-only declarations. No runtime values, no artifact format shape —
// the artifact format has not been designed yet.

/**
 * Feature values for a single prediction, keyed by the model's exact
 * feature names. Callers must supply exactly the declared keys: no
 * missing keys, no extra keys.
 */
export type PredictionInput = Record<string, number>;

/**
 * Machine-readable discriminant for every error this package can raise.
 * Kept in sync with the error classes in `errors.ts`.
 */
export type ErrorCode =
  | "UNSUPPORTED_OBJECTIVE"
  | "UNSUPPORTED_BOOSTER"
  | "UNRECOGNIZED_FIELD"
  | "UNSUPPORTED_VERSION"
  | "FEATURE_KEY_MISMATCH";
