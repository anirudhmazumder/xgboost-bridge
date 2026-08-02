"""Scaffold tests for the fixture-generation workspace member.

The fixture corpus itself does not exist yet -- this suite has nothing of
its own to test. What it can genuinely assert, from this side of the
workspace, is that the `xgboost-bridge` package it will depend on to build
and verify fixtures resolves correctly in this environment: it imports,
declares a real version, and exposes the error hierarchy that fixture
generation and verification code will eventually raise against. A
placeholder that asserts nothing would be worse than no test at all.
"""

from __future__ import annotations

import xgboost_bridge
from xgboost_bridge import errors

ERROR_CLASSES_WITH_ATTRIBUTES: dict[str, tuple[str, ...]] = {
    "UnsupportedObjectiveError": ("objective", "supported"),
    "UnsupportedBoosterError": ("booster", "supported"),
    "UnrecognizedFieldError": ("field", "location"),
    "UnsupportedVersionError": ("version", "supported"),
    "FeatureKeyMismatchError": ("missing_keys", "extra_keys"),
}


def test_xgboost_bridge_is_importable_from_the_fixtures_environment() -> None:
    assert xgboost_bridge is not None


def test_xgboost_bridge_declares_a_nonempty_version() -> None:
    assert isinstance(xgboost_bridge.__version__, str)
    assert xgboost_bridge.__version__ != ""


def test_error_hierarchy_is_reachable_from_the_fixtures_environment() -> None:
    assert issubclass(errors.XGBoostBridgeError, Exception)
    for name, attributes in ERROR_CLASSES_WITH_ATTRIBUTES.items():
        cls = getattr(errors, name)
        assert issubclass(cls, errors.XGBoostBridgeError)
        assert cls is not errors.XGBoostBridgeError
        # The class must accept exactly its documented attributes as
        # constructor arguments -- confirms the signature fixture code will
        # eventually call against matches what's documented, without
        # asserting anything about when or why it gets raised.
        instance = cls(*(f"placeholder-{attr}" for attr in attributes))
        for attr in attributes:
            assert hasattr(instance, attr)
