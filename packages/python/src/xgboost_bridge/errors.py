"""Exception hierarchy for xgboost-bridge failure modes.

Every error this library raises -- during export, artifact parsing, or
prediction -- is an instance of :class:`XGBoostBridgeError` or one of the
subclasses below. Each subclass corresponds to exactly one failure mode
named in ``CLAUDE.md`` and ``DECISIONS.md`` (see D007: fail loudly on
anything unrecognized). Nothing here defaults, guesses, or infers by
analogy; a caller either gets a value it can act on, or an exception it can
inspect programmatically -- never a plausible wrong number.

This module contains no logic beyond constructing exceptions and
formatting their messages. It does not decide *when* an error is raised --
that decision belongs to the code that detects the failure and imports
these classes.
"""

from __future__ import annotations


class XGBoostBridgeError(Exception):
    """Base class for every exception raised by xgboost-bridge.

    Catch this to handle any failure the library can raise without
    enumerating the specific subclasses below. Never raised directly --
    the library always raises one of the subclasses so a caller can inspect
    the structured attributes describing what went wrong, not just parse a
    message string.
    """


class UnsupportedObjectiveError(XGBoostBridgeError):
    """Raised when a model's objective is not one this library supports.

    Per D007, an unrecognized objective raises rather than falling back to
    an approximation. Guessing a link function or ``base_score`` transform
    for an objective this library has not verified against a real fitted
    model would reintroduce the exact silent-failure mode this project
    exists to prevent.

    Attributes:
        objective: The objective value reported by the model or artifact
            (e.g. ``"reg:squarederror"``), whatever its type turned out to
            be.
        supported: The objectives this version of the library recognizes.
    """

    def __init__(self, objective: object, supported: tuple[str, ...]) -> None:
        self.objective = objective
        self.supported = supported
        super().__init__(
            f"unsupported objective {objective!r}; supported objectives "
            f"are {', '.join(supported)}"
        )


class UnsupportedBoosterError(XGBoostBridgeError):
    """Raised when a model's booster type is not one this library supports.

    A single serialized field is not sufficient to identify a booster type
    on its own -- DART, for instance, reports ``gradient_booster.name`` as
    ``"gbtree"``, indistinguishable from a plain tree ensemble on that
    field alone. Code that raises this error is expected to have already
    checked every signal it has, not just one.

    Attributes:
        booster: The booster type value reported by the model or artifact.
        supported: The booster types this version of the library
            recognizes.
    """

    def __init__(self, booster: object, supported: tuple[str, ...]) -> None:
        self.booster = booster
        self.supported = supported
        super().__init__(
            f"unsupported booster {booster!r}; supported boosters are "
            f"{', '.join(supported)}"
        )


class UnrecognizedFieldError(XGBoostBridgeError):
    """Raised when an artifact contains a field this library does not know.

    An unrecognized field is a hard error rather than something to ignore:
    silently dropping it could discard information that changes the
    prediction, and there is no way to tell from inside this library
    whether an unknown field is cosmetic or load-bearing.

    Attributes:
        field: The name of the unrecognized field.
        location: Where in the artifact the field was found (for example a
            dotted or indexed path such as ``"trees[3].split_conditions"``),
            or ``None`` if the field was at the artifact's top level.
    """

    def __init__(self, field: str, location: str | None = None) -> None:
        self.field = field
        self.location = location
        where = f" at {location}" if location else ""
        super().__init__(f"unrecognized artifact field {field!r}{where}")


class UnsupportedVersionError(XGBoostBridgeError):
    """Raised when an artifact's version marker is out of the supported range.

    The version marker is this library's migration mechanism: the artifact
    format reserves no structural space for unimplemented features, so an
    artifact produced by an older or newer format revision than this
    library understands must raise rather than be interpreted under an
    unverified assumption about what changed.

    Attributes:
        version: The version marker found in the artifact.
        supported: The version marker(s) this version of the library can
            read.
    """

    def __init__(self, version: object, supported: tuple[object, ...]) -> None:
        self.version = version
        self.supported = supported
        super().__init__(
            f"unsupported artifact version {version!r}; supported "
            f"versions are {', '.join(str(v) for v in supported)}"
        )


class FeatureKeyMismatchError(XGBoostBridgeError):
    """Raised when prediction input's feature keys don't exactly match the model.

    Feature keys must match the model exactly: no keys missing, no keys
    the model doesn't recognize. Lenient handling would silently convert a
    misspelled feature name into a missing-value path -- a legitimate model
    structure repurposed by accident -- producing a confident, plausible,
    wrong prediction instead of an error.

    Attributes:
        missing_keys: Feature names the model requires that were absent
            from the input. Empty if none were missing.
        extra_keys: Feature names present in the input that the model does
            not recognize. Empty if there were none.
    """

    def __init__(
        self,
        missing_keys: frozenset[str] = frozenset(),
        extra_keys: frozenset[str] = frozenset(),
    ) -> None:
        if not missing_keys and not extra_keys:
            raise ValueError(
                "FeatureKeyMismatchError requires at least one missing "
                "or extra key"
            )
        self.missing_keys = missing_keys
        self.extra_keys = extra_keys
        parts = []
        if missing_keys:
            parts.append(f"missing keys: {sorted(missing_keys)}")
        if extra_keys:
            parts.append(f"extra keys: {sorted(extra_keys)}")
        super().__init__(f"feature key mismatch ({'; '.join(parts)})")


class UnsupportedModelShapeError(XGBoostBridgeError):
    """Raised when a model's output arity is not scalar-per-row.

    An objective-name allow-list alone is not sufficient. ``reg:squarederror``
    with ``num_target=2`` is a supported objective that produces ``(N, 2)``
    margins; a scalar predictor accepts such a model and returns confident
    wrong numbers. So arity is checked separately, on the fields that
    actually determine it. See D017 and D037.

    All four gate fields are JSON *strings* in the serialized model, and
    ``field`` / ``value`` record them as found rather than coerced -- an
    integer comparison against a string silently never fires, which would
    disable the gate rather than trip it.

    Attributes:
        field: The gate field that failed (e.g. ``"num_target"``).
        value: The value found, as it appeared in the model.
        expected: A human-readable description of what was required.
        location: Where the field was read from, or ``None`` for a per-tree
            field with no single location.
    """

    def __init__(
        self,
        field: str,
        value: object,
        expected: str,
        location: str | None = None,
    ) -> None:
        self.field = field
        self.value = value
        self.expected = expected
        self.location = location
        where = f" at {location}" if location else ""
        super().__init__(
            f"unsupported model shape: {field}={value!r}{where}, expected {expected}"
        )


class CategoricalSplitError(XGBoostBridgeError):
    """Raised when a model contains a categorical split.

    Categorical features are out of scope for format version 1, and reading
    one as numeric is a wrong-number path for two independent reasons: a
    categorical split *inverts* the child convention -- the in-set branch is
    the right child, the opposite of a numeric split -- and its entry in
    ``split_conditions`` is the smallest positive subnormal float32 rather
    than a threshold.

    Attributes:
        signals: The detection signals that fired. Three independent signals
            exist and all are checked, because this is a refusal test where
            redundancy costs nothing.
    """

    def __init__(self, signals: tuple[str, ...]) -> None:
        self.signals = signals
        super().__init__(
            "model contains categorical splits, which format version 1 does "
            f"not support (signals: {', '.join(signals)})"
        )


class MissingFeatureNamesError(XGBoostBridgeError):
    """Raised when a model carries no feature names.

    A model fit from a bare array serializes ``feature_names`` as ``[]``
    while ``num_feature`` is nonzero. Exporting it would produce an artifact
    whose strict-key policy has no keys to be strict about -- which reads as
    enforced and is not, so a caller's typo would go undetected. The caller
    must supply names explicitly. See D021.

    Attributes:
        num_feature: The feature count the model reports, so the caller
            knows how many names to supply.
    """

    def __init__(self, num_feature: int) -> None:
        self.num_feature = num_feature
        super().__init__(
            f"model carries no feature names but reports {num_feature} "
            "features; supply feature names explicitly to export it"
        )


class AmbiguousTreeCountError(XGBoostBridgeError):
    """Raised when an early-stopped model's effective tree count is ambiguous.

    The correct tree count is not a property of the model. The same file
    loaded as a bare ``Booster`` uses every tree; loaded through a
    scikit-learn estimator it uses only the first ``best_iteration + 1``
    iterations. Measured divergence between the two readings reaches 1.55 in
    margin space, and no field in the serialized model distinguishes them.
    Choosing either reading would be silently wrong for half of callers, on
    every row. See D038.

    Attributes:
        best_iteration: The recorded best iteration.
        effective_trees: Tree count implied by truncating at
            ``best_iteration``, taken from ``iteration_indptr``.
        total_trees: Tree count actually present in the model.
    """

    def __init__(
        self, best_iteration: int, effective_trees: int, total_trees: int
    ) -> None:
        self.best_iteration = best_iteration
        self.effective_trees = effective_trees
        self.total_trees = total_trees
        super().__init__(
            f"model stopped early at iteration {best_iteration} "
            f"({effective_trees} trees) but carries {total_trees} trees, and "
            "which count applies depends on how the model is loaded rather "
            "than on the model itself; slice it explicitly with "
            f"booster[0:{best_iteration + 1}] and export the result"
        )


class InterceptMismatchError(XGBoostBridgeError):
    """Raised when the derived intercept disagrees with XGBoost's own output.

    Validated against XGBoost's observed zero-tree margin rather than
    against a re-derivation of the export recipe. A check that re-derives
    its own recipe cannot fire on a recipe error, which is the class of
    error it exists to catch -- an earlier version of this check was blind
    in exactly that way and passed a real ``base_score`` clamping defect.
    See D034.

    Attributes:
        derived: The intercept this library computed.
        observed: The margin XGBoost produced for a zero-tree model with the
            same configuration.
        objective: The objective in play, since the transform is
            per-objective.
    """

    def __init__(self, derived: object, observed: object, objective: str) -> None:
        self.derived = derived
        self.observed = observed
        self.objective = objective
        super().__init__(
            f"derived intercept {derived!r} does not match XGBoost's observed "
            f"zero-tree margin {observed!r} for objective {objective!r}"
        )


class MalformedTreeError(XGBoostBridgeError):
    """Raised when a tree's structure is not what any probe measured.

    Distinct from :class:`UnsupportedModelShapeError`, which is about output
    *arity* -- a model that is well-formed but produces more than one value
    per row. This error means the tree representation itself does not match
    the layout recorded in ``probes/tree_structure.md``: arrays of unequal
    length, an absent field, a child index that does not point forward, a
    non-finite threshold, or a dead-node marker that disagrees with
    reachability.

    Such a disagreement is not a model this library declines to support; it
    is a model whose shape contradicts the evidence the reader was built
    from. Continuing would mean walking a structure under assumptions
    already known to be false, which is how a plausible wrong number is
    produced.

    Attributes:
        field: The field whose value contradicted the expected layout.
        value: The value found.
        expected: A human-readable description of what the layout requires.
        location: Where the field was read from, or ``None`` if not
            applicable.
    """

    def __init__(
        self,
        field: str,
        value: object,
        expected: str,
        location: str | None = None,
    ) -> None:
        self.field = field
        self.value = value
        self.expected = expected
        self.location = location
        where = f" at {location}" if location else ""
        super().__init__(
            f"malformed tree structure: {field}={value!r}{where}, expected {expected}"
        )


class NonFiniteInterceptError(XGBoostBridgeError):
    """Raised when the derived margin intercept is not a finite number.

    Reachable through ordinary parameters, and XGBoost does not announce it:
    ``survival:cox`` with ``base_score = 0.0`` derives an intercept of
    ``-inf``, and with any negative ``base_score`` derives ``NaN``. Both are
    accepted at fit time with no error and no warning.

    The derivation reproduces these values deliberately -- the export-time
    oracle compares against XGBoost's own observed margin, so a derivation
    that "fixed" them would disagree with the oracle. The refusal therefore
    belongs at export rather than in the derivation.

    Worth noting why the oracle alone does not catch this: a bit-pattern
    comparison matches ``NaN`` against ``NaN`` perfectly well. What catches
    it is the finiteness requirement, not the equality check.

    Attributes:
        intercept: The non-finite value that was derived.
        objective: The objective in play, since the transform is
            per-objective.
        base_score: The stored ``base_score`` that produced it, verbatim, so
            a caller can see which input was degenerate.
    """

    def __init__(
        self, intercept: object, objective: str, base_score: object
    ) -> None:
        self.intercept = intercept
        self.objective = objective
        self.base_score = base_score
        super().__init__(
            f"derived a non-finite intercept {intercept!r} for objective "
            f"{objective!r} from base_score {base_score!r}; a finite "
            "intercept is required"
        )


class NonFiniteFeatureError(XGBoostBridgeError):
    """Raised when a prediction input carries an infinite feature value.

    ``NaN`` is **not** an error: it is the missing value, and it routes by the
    node's ``default_left``. Only infinities raise.

    Upstream is genuinely inconsistent here, which is why this library picks
    one behaviour and pins it (D022): ``±inf`` raises through ``DMatrix`` but
    is treated as an ordinary comparable value through ``inplace_predict``, so
    the same input yields two different predictions from XGBoost depending on
    the call path. Surfacing that class of divergence rather than silently
    inheriting one side of it is the reason this library exists.

    The whole row is checked before the walk begins, deliberately. Checking
    lazily -- only at the nodes actually visited -- would mean the same
    invalid input raises or not depending on which branches the tree happens
    to take, making the outcome a property of the model rather than of the
    input.

    Attributes:
        index: Column index of the offending value.
        value: The value found, so a caller can tell ``+inf`` from ``-inf``.
    """

    def __init__(self, index: int, value: float) -> None:
        self.index = index
        self.value = value
        super().__init__(
            f"feature at index {index} is {value!r}; infinite feature values "
            "are refused (NaN is the missing value and is accepted)"
        )
