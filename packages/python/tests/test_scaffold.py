"""Scaffold tests for the xgboost_bridge package.

The scaffold has no numerical core yet, so there is nothing to test about
predictions, thresholds, or base_score. What exists is the package's
import surface and its error hierarchy -- this suite pins both down so
later phases build on a foundation that is already under test. Nothing
here should ever be loosened, skipped, or xfail'd: if one of these
assertions gets in the way later, that is a signal, not an obstacle.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import xgboost_bridge
from xgboost_bridge import errors

REPO_ROOT = Path(__file__).resolve().parents[3]

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


# ---------------------------------------------------------------------------
# The published package's runtime needs numpy only; xgboost is an extra (D010).
# ---------------------------------------------------------------------------

#: A child process that refuses to import xgboost, then reads an artifact and
#: predicts. Run out of process because this suite imports xgboost in other
#: modules, so an in-process block would come too late to prove anything.
_WITHOUT_THE_EXPORT_EXTRA = """
import sys


class Refuse:
    def find_module(self, name, path=None):
        return self.find_spec(name, path)

    def find_spec(self, name, path=None, target=None):
        if name == "xgboost" or name.startswith("xgboost."):
            raise ImportError("the export extra is not installed (simulated)")
        return None


sys.meta_path.insert(0, Refuse())

import json

import numpy as np

import xgboost_bridge
from xgboost_bridge.predict import Predictor

with open(sys.argv[1], encoding="utf-8") as handle:
    fixture = json.load(handle)

predictor = Predictor.from_json(fixture["artifact"])
row = dict(zip(predictor.feature_names, fixture["rows"][0]))
margin = predictor.margin(row)
output = predictor.output(row)
print(json.dumps({
    "margin": int(np.float32(margin).view(np.uint32)),
    "output": int(np.float32(output).view(np.uint32)),
    "imported": "xgboost" not in sys.modules,
}))
"""


def test_the_package_imports_and_predicts_with_the_export_extra_absent() -> None:
    """Reading an artifact and predicting must not require xgboost (D010).

    The base distribution declares ``numpy`` only; xgboost moves to an
    ``export`` extra, so a consumer who never exports is not made to install a
    large native dependency. That promise is kept by importing xgboost inside
    the two functions that need it rather than at module scope -- an
    arrangement a routine tidy-up reverses, at which point ``import
    xgboost_bridge`` fails on every base install while the whole suite here
    stays green, because every environment this repository tests in has xgboost.

    The margin is checked against the fixture's recorded XGBoost bit pattern,
    which is an oracle this library contributes nothing to, so the child has to
    produce the right number and not merely avoid raising. The **output** is
    checked against this process's own value instead, deliberately: XGBoost's
    recorded output differs from the bundled transform by one bit on a pinned
    set of rows, by construction, because its own exponential is not correctly
    rounded (D032, D047) -- and the row below is one of them. The transform's
    correctness is pinned against ``mpmath`` elsewhere; what is under test here
    is only that removing the extra changes no number.
    """
    fixture_path = REPO_ROOT / "fixtures" / "corpus" / "single_feature_model.json"
    assert fixture_path.is_file(), f"fixture corpus is missing: {fixture_path}"
    expected = json.loads(fixture_path.read_text(encoding="utf-8"))

    completed = subprocess.run(
        [sys.executable, "-c", _WITHOUT_THE_EXPORT_EXTRA, str(fixture_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        "importing xgboost_bridge or predicting required xgboost:\n"
        f"{completed.stderr}"
    )

    reported = json.loads(completed.stdout)
    assert reported["imported"], "xgboost reached sys.modules despite the block"
    assert reported["margin"] == int(expected["expected_margin"][0], 16)

    import numpy as np  # noqa: PLC0415 -- only this assertion needs it

    from xgboost_bridge.predict import Predictor  # noqa: PLC0415

    predictor = Predictor.from_json(expected["artifact"])
    row = dict(zip(predictor.feature_names, expected["rows"][0]))
    assert reported["output"] == int(np.float32(predictor.output(row)).view(np.uint32))
