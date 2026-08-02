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
