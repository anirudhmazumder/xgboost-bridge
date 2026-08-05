"""Export trained XGBoost models as portable JSON artifacts.

The artifact format is specified in ``FORMAT.md``; the invariants that make it
correct are in ``CLAUDE.md``. Both are worth reading before relying on
anything here, because this library's failure mode of concern is a plausible
wrong number rather than an exception.

Importing this module does **not** require XGBoost. Reading a fitted model
does, which is why ``xgboost`` is an optional extra (D010) and is imported
inside the functions that need it rather than at module scope.
"""

from __future__ import annotations

from ._version import __version__
from .errors import (
    AmbiguousTreeCountError,
    CategoricalSplitError,
    FeatureKeyMismatchError,
    InterceptMismatchError,
    MalformedTreeError,
    MissingFeatureNamesError,
    NonFiniteInterceptError,
    UnrecognizedFieldError,
    UnsupportedBoosterError,
    UnsupportedModelShapeError,
    UnsupportedObjectiveError,
    UnsupportedVersionError,
    XGBoostBridgeError,
)
from .export import FORMAT_VERSION, export_model, to_json
from .objectives import OUTPUT_TRANSFORMS, SUPPORTED_OBJECTIVES

__all__ = [
    "FORMAT_VERSION",
    "OUTPUT_TRANSFORMS",
    "SUPPORTED_OBJECTIVES",
    # Errors, every one a subclass of XGBoostBridgeError so a caller can catch
    # the base class or branch on the specific failure.
    "AmbiguousTreeCountError",
    "CategoricalSplitError",
    "FeatureKeyMismatchError",
    "InterceptMismatchError",
    "MalformedTreeError",
    "MissingFeatureNamesError",
    "NonFiniteInterceptError",
    "UnrecognizedFieldError",
    "UnsupportedBoosterError",
    "UnsupportedModelShapeError",
    "UnsupportedObjectiveError",
    "UnsupportedVersionError",
    "XGBoostBridgeError",
    "__version__",
    "export_model",
    "to_json",
]
