"""Scaffold tests for the xgboost_bridge package.

Phase 1 has no numerical core yet, so there is nothing to test about
predictions, thresholds, or base_score. What exists is the package's
import surface and its error hierarchy -- this suite pins both down so
later phases build on a foundation that is already under test. Nothing
here should ever be loosened, skipped, or xfail'd: if one of these
assertions gets in the way later, that is a signal, not an obstacle.
"""

from __future__ import annotations

import pytest

import xgboost_bridge
from xgboost_bridge import errors

# Every error class this phase declares, and the attributes its docstring
# promises callers can inspect programmatically.
ERROR_CLASSES_WITH_ATTRIBUTES: dict[str, tuple[str, ...]] = {
    "UnsupportedObjectiveError": ("objective", "supported"),
    "UnsupportedBoosterError": ("booster", "supported"),
    "UnrecognizedFieldError": ("field", "location"),
    "UnsupportedVersionError": ("version", "supported"),
    "FeatureKeyMismatchError": ("missing_keys", "extra_keys"),
}


def test_package_imports() -> None:
    assert xgboost_bridge is not None


def test_package_version_is_a_nonempty_string() -> None:
    assert isinstance(xgboost_bridge.__version__, str)
    assert xgboost_bridge.__version__ != ""


def test_base_error_exists_and_is_an_exception() -> None:
    assert issubclass(errors.XGBoostBridgeError, Exception)


@pytest.mark.parametrize("name", sorted(ERROR_CLASSES_WITH_ATTRIBUTES))
def test_error_class_exists_and_subclasses_the_base(name: str) -> None:
    cls = getattr(errors, name)
    assert issubclass(cls, errors.XGBoostBridgeError)
    assert cls is not errors.XGBoostBridgeError


@pytest.mark.parametrize("name", sorted(ERROR_CLASSES_WITH_ATTRIBUTES))
def test_error_class_is_publicly_reachable(name: str) -> None:
    public_names = {n for n in dir(errors) if not n.startswith("_")}
    assert name in public_names


def test_unsupported_objective_error_carries_its_attributes() -> None:
    err = errors.UnsupportedObjectiveError("rank:made_up", ("binary:logistic",))
    assert err.objective == "rank:made_up"
    assert err.supported == ("binary:logistic",)
    assert isinstance(err, errors.XGBoostBridgeError)


def test_unsupported_booster_error_carries_its_attributes() -> None:
    err = errors.UnsupportedBoosterError("dart_impostor", ("gbtree",))
    assert err.booster == "dart_impostor"
    assert err.supported == ("gbtree",)


def test_unrecognized_field_error_carries_its_attributes() -> None:
    err = errors.UnrecognizedFieldError("mystery_field", location="trees[0]")
    assert err.field == "mystery_field"
    assert err.location == "trees[0]"


def test_unrecognized_field_error_location_defaults_to_none() -> None:
    err = errors.UnrecognizedFieldError("mystery_field")
    assert err.location is None


def test_unsupported_version_error_carries_its_attributes() -> None:
    err = errors.UnsupportedVersionError(7, (1, 2))
    assert err.version == 7
    assert err.supported == (1, 2)


def test_feature_key_mismatch_error_distinguishes_missing_from_extra() -> None:
    only_missing = errors.FeatureKeyMismatchError(missing_keys=frozenset({"a"}))
    assert only_missing.missing_keys == frozenset({"a"})
    assert only_missing.extra_keys == frozenset()

    only_extra = errors.FeatureKeyMismatchError(extra_keys=frozenset({"b"}))
    assert only_extra.missing_keys == frozenset()
    assert only_extra.extra_keys == frozenset({"b"})

    both = errors.FeatureKeyMismatchError(
        missing_keys=frozenset({"a"}), extra_keys=frozenset({"b"})
    )
    assert both.missing_keys == frozenset({"a"})
    assert both.extra_keys == frozenset({"b"})


def test_feature_key_mismatch_error_requires_at_least_one_key() -> None:
    with pytest.raises(ValueError):
        errors.FeatureKeyMismatchError()
