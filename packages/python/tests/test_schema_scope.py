"""What the published JSON Schema does *not* catch, measured and pinned.

`VERIFICATION.md` and D051 both state that the schema "enforces roughly a third
of what FORMAT.md §13 requires" and that "10 of 11 wrong-but-well-formed
artifacts validate against it". Those numbers came from an audit's prose and had
**no committed artifact behind them** — so in a document whose first line
promises every figure is reproducible from a clone, that one was not, and the gap
could widen with no signal.

This module is the catalogue. Each entry is an artifact that is *structurally*
well-formed — right keys, right JSON types, right array shapes — and violates
FORMAT.md semantically. For each, three questions are measured rather than
asserted from memory:

1. Does the published schema accept it?
2. Does the Python reader reject it, and does it do so *structurally* -- through
   the documented exception hierarchy rather than as a bare `OverflowError`?

The JavaScript reader is covered by the mirror of this catalogue in
`packages/js/test/`. The two are deliberately separate files rather than a shared
data file: a shared catalogue would make one reader's blind spot invisible in the
other's suite, which is the same mistake as validating one language against the
other instead of against an oracle.

Why this matters, precisely: the schema's `$id` is a public URL, so a third party
can validate against it and then walk the artifact with their own code. Every
entry the schema accepts is a case where that third party gets a pass from a
document calling itself normative and then reads values this project would have
refused. Neither shipped reader is affected — that is what makes it a
documentation-and-trust problem rather than a wrong-number problem — and it is
exactly the recurring defect class this repository keeps finding: **a validator
that establishes less than its consumers assume.**

The counts are pinned as an exact set, not a floor. If someone strengthens the
schema, this fails and tells them to update the record; if someone weakens it,
this fails too. Movement in either direction is a signal, which is the same
tripwire discipline the six accepted output rows use.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from xgboost_bridge import Predictor, errors

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPO_ROOT / "schema" / "xgboost-bridge-v1.schema.json"
SCHEMA: dict[str, Any] = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

#: A real exported artifact with trees, used as the base every entry mutates.
#: Real rather than hand-built so that "well-formed" means what the exporter
#: actually produces, not what a test author remembered to include.
BASE: dict[str, Any] = json.loads(
    (REPO_ROOT / "fixtures" / "corpus" / "survival_cox_base_score_high.json").read_text(
        encoding="utf-8"
    )
)["artifact"]


def _mutate(**changes: Any) -> dict[str, Any]:
    artifact = copy.deepcopy(BASE)
    artifact.update(changes)
    return artifact


def _with_tree(**tree_changes: Any) -> dict[str, Any]:
    artifact = copy.deepcopy(BASE)
    artifact["trees"][0].update(tree_changes)
    return artifact


def _first_tree_size() -> int:
    return len(BASE["trees"][0]["left_children"])


def _child_out_of_range() -> dict[str, Any]:
    size = _first_tree_size()
    children = list(BASE["trees"][0]["left_children"])
    children[0] = size + 50  # in-type, in-array, out of range
    return _with_tree(left_children=children)


def _child_negative_but_not_the_leaf_marker() -> dict[str, Any]:
    children = list(BASE["trees"][0]["left_children"])
    children[0] = -2  # -1 is the leaf marker; -2 means nothing
    return _with_tree(left_children=children)


def _unequal_array_lengths() -> dict[str, Any]:
    values = list(BASE["trees"][0]["node_values"])
    return _with_tree(node_values=values[:-1])  # one short of the others


def _split_index_beyond_feature_count() -> dict[str, Any]:
    indices = list(BASE["trees"][0]["split_indices"])
    indices[0] = len(BASE["feature_names"]) + 7
    return _with_tree(split_indices=indices)


def _default_left_not_boolean_valued() -> dict[str, Any]:
    flags = list(BASE["trees"][0]["default_left"])
    flags[0] = 2  # the field is 0/1; 2 is neither
    return _with_tree(default_left=flags)


def _duplicate_feature_names() -> dict[str, Any]:
    names = list(BASE["feature_names"])
    names[1] = names[0]
    return _mutate(feature_names=names)


def _feature_names_length_disagrees_with_split_indices() -> dict[str, Any]:
    return _mutate(feature_names=[BASE["feature_names"][0]])


def _cycle_between_two_nodes() -> dict[str, Any]:
    # Both internal, pointing at each other: the walk never reaches a leaf.
    return _with_tree(
        left_children=[1, 0], right_children=[1, 0], split_indices=[0, 0],
        node_values=[0.5, 0.5], default_left=[0, 0],
    )


def _shared_child_makes_it_a_dag_not_a_tree() -> dict[str, Any]:
    # Node 3 is the left child of BOTH node 1 and node 2. Acyclic, every index
    # in range, terminates -- and it is not a tree. A shared subtree yields a
    # plausible prediction, which is the worst available outcome.
    artifact = copy.deepcopy(BASE)
    # Nodes 1 and 2 both point at leaves 3 and 4. Every node is unambiguously
    # internal or leaf -- an earlier version of this case had node 1 as
    # left=3, right=-1, a *half-leaf*, so both readers rejected it for that
    # reason and it tested nothing about in-degree at all.
    artifact["trees"] = [
        {
            "left_children": [1, 3, 3, -1, -1],
            "right_children": [2, 4, 4, -1, -1],
            "split_indices": [0, 0, 0, 0, 0],
            "node_values": [0.1, 0.2, 0.3, 1.5, 2.5],
            "default_left": [0, 0, 0, 0, 0],
        }
    ]
    return artifact


def _objective_transform_mismatch() -> dict[str, Any]:
    return _mutate(objective="reg:squarederror", output_transform="sigmoid")


def _intercept_too_large_for_float64() -> dict[str, Any]:
    # A 401-digit integer literal: valid JSON, under CPython's parse ceiling.
    return _mutate(intercept=int("1" + "0" * 400))


def _node_value_too_large_for_float64() -> dict[str, Any]:
    values = list(BASE["trees"][0]["node_values"])
    values[0] = int("1" + "0" * 400)
    return _with_tree(node_values=values)


def _unknown_xgboost_version() -> dict[str, Any]:
    artifact = copy.deepcopy(BASE)
    artifact["provenance"]["xgboost_version"] = "99.0.0"
    return artifact


#: The catalogue. Each is well-formed in shape and wrong in meaning.
WRONG_BUT_WELL_FORMED: dict[str, Any] = {
    "child_index_out_of_range": _child_out_of_range(),
    "child_negative_but_not_leaf_marker": _child_negative_but_not_the_leaf_marker(),
    "per_tree_arrays_of_unequal_length": _unequal_array_lengths(),
    "split_index_beyond_feature_count": _split_index_beyond_feature_count(),
    "default_left_outside_zero_one": _default_left_not_boolean_valued(),
    "duplicate_feature_names": _duplicate_feature_names(),
    "feature_names_shorter_than_split_indices": (
        _feature_names_length_disagrees_with_split_indices()
    ),
    "cycle_between_two_nodes": _cycle_between_two_nodes(),
    "shared_child_is_a_dag_not_a_tree": _shared_child_makes_it_a_dag_not_a_tree(),
    "objective_and_transform_disagree": _objective_transform_mismatch(),
    "intercept_beyond_float64": _intercept_too_large_for_float64(),
    "node_value_beyond_float64": _node_value_too_large_for_float64(),
}

#: Deliberately NOT in the catalogue above, and the reason is the interesting
#: part. An artifact naming an unprobed XGBoost version is wrong-but-well-formed
#: and the schema accepts it -- but the reader accepts it too, correctly: the
#: enumerated version ceiling is an *export*-time gate (D018), and FORMAT.md §2
#: states that `provenance.xgboost_version` "is recorded but never used for
#: inference". Putting it in a reader-rejection catalogue would be a category
#: error, and asserting the reader refuses it would pin behaviour the format
#: explicitly disclaims.
EXPORT_TIME_ONLY: dict[str, Any] = {
    "unprobed_xgboost_version": _unknown_xgboost_version(),
}


def _schema_accepts(artifact: dict[str, Any]) -> bool:
    try:
        jsonschema.validate(artifact, SCHEMA)
    except jsonschema.ValidationError:
        return False
    return True


def _python_reader_rejects(artifact: dict[str, Any]) -> bool:
    try:
        Predictor.from_json(artifact)
    except errors.XGBoostBridgeError:
        return True
    except Exception:  # noqa: BLE001
        # An unstructured escape is still a rejection, but it is a contract
        # breach and is reported separately below rather than counted as a pass.
        return True
    return False


def _python_reader_rejects_structurally(artifact: dict[str, Any]) -> bool:
    try:
        Predictor.from_json(artifact)
    except errors.XGBoostBridgeError:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


def test_the_catalogue_is_at_least_as_large_as_the_documented_figure() -> None:
    """D051 cited eleven. Fewer than that would mean this file records less than
    the prose it exists to make reproducible."""
    assert len(WRONG_BUT_WELL_FORMED) + len(EXPORT_TIME_ONLY) >= 11


@pytest.mark.parametrize("name", sorted(WRONG_BUT_WELL_FORMED))
def test_every_catalogue_entry_is_structurally_well_formed(name: str) -> None:
    """The catalogue must contain *semantic* violations only.

    An entry the schema rejects on shape teaches nothing about the gap -- it
    would inflate the "schema catches it" count for the wrong reason. The three
    that are legitimately shape-detectable are named here rather than left to
    coincidence.
    """
    # Measured. A first pass guessed these wrong in all three slots: the schema
    # does catch duplicate feature names (it carries `uniqueItems`), and it does
    # NOT catch either oversized integer, because a 401-digit integer literal is
    # still `type: number` to JSON Schema. Guessing what a validator checks is
    # the mistake this whole module is about.
    shape_detectable = {
        "default_left_outside_zero_one",
        "duplicate_feature_names",
    }
    accepted = _schema_accepts(WRONG_BUT_WELL_FORMED[name])
    if name in shape_detectable:
        return
    assert accepted, (
        f"{name} is rejected by the schema on shape, so it does not belong in a "
        f"catalogue of semantic violations -- move it or reclassify it"
    )


@pytest.mark.parametrize("name", sorted(WRONG_BUT_WELL_FORMED))
def test_the_python_reader_rejects_every_catalogue_entry(name: str) -> None:
    """The gate. Whatever the schema does or does not catch, the shipped reader
    must refuse all of it -- that is what keeps a schema weakness from becoming a
    wrong number for anyone using this package."""
    artifact = WRONG_BUT_WELL_FORMED[name]
    assert _python_reader_rejects(artifact), (
        f"{name} was ACCEPTED by the Python reader; a wrong number is reachable"
    )


@pytest.mark.parametrize("name", sorted(WRONG_BUT_WELL_FORMED))
def test_the_python_reader_rejects_structurally_not_by_accident(name: str) -> None:
    """A refusal that escapes as a bare `ValueError` or `OverflowError` is
    outside the documented contract, so `except XGBoostBridgeError` misses it.
    One such case was found and fixed for oversized integers; this keeps the
    whole catalogue honest about it."""
    artifact = WRONG_BUT_WELL_FORMED[name]
    assert _python_reader_rejects_structurally(artifact), (
        f"{name} raised something outside XGBoostBridgeError"
    )


def test_the_schema_gap_is_pinned_as_an_exact_set() -> None:
    """The measurement D051 recorded in prose, now reproducible from a clone.

    Pinned as a set rather than a count so the failure message says *which* case
    moved. Strengthening the schema fails this test on purpose: the record is
    what has to change with it.
    """
    accepted = {
        name for name, artifact in WRONG_BUT_WELL_FORMED.items() if _schema_accepts(artifact)
    }
    expected = {
        "child_index_out_of_range",
        "child_negative_but_not_leaf_marker",
        "per_tree_arrays_of_unequal_length",
        "split_index_beyond_feature_count",
        "feature_names_shorter_than_split_indices",
        "cycle_between_two_nodes",
        "shared_child_is_a_dag_not_a_tree",
        "objective_and_transform_disagree",
        "intercept_beyond_float64",
        "node_value_beyond_float64",
    }
    assert accepted == expected, (
        "the published schema's coverage moved.\n"
        f"  newly accepted (schema got weaker): {sorted(accepted - expected)}\n"
        f"  newly rejected (schema got stronger): {sorted(expected - accepted)}\n"
        "Update VERIFICATION.md and this set together."
    )
    print(
        f"schema accepts {len(accepted)}/{len(WRONG_BUT_WELL_FORMED)} "
        f"wrong-but-well-formed artifacts; the Python reader rejects all "
        f"{len(WRONG_BUT_WELL_FORMED)}"
    )


def test_the_export_gate_refuses_what_the_reader_deliberately_does_not() -> None:
    """The version ceiling lives at export, so the reader must NOT enforce it.

    Both halves are asserted, because either alone is misleading: if the reader
    started refusing this, artifacts already in the wild would stop loading on an
    upgrade; if the exporter stopped refusing it, the ceiling would be gone. See
    D018 and FORMAT.md section 2.
    """
    artifact = EXPORT_TIME_ONLY["unprobed_xgboost_version"]
    assert _schema_accepts(artifact), "premise: the schema does not check this either"

    # The reader loads it, deliberately.
    predictor = Predictor.from_json(artifact)
    assert predictor.provenance["xgboost_version"] == "99.0.0"

    # And the export-side gate is what refuses it.
    from xgboost_bridge import validate

    document = {
        "version": [99, 0, 0],
        "learner": {
            "objective": {"name": "reg:squarederror"},
            "learner_model_param": {
                "base_score": "[5E-1]",
                "num_target": "1",
                "num_class": "0",
                "boost_from_average": "1",
                "num_feature": "1",
            },
            "feature_names": ["f0"],
            "gradient_booster": {"name": "gbtree", "model": {"trees": []}},
        },
    }
    with pytest.raises(errors.XGBoostBridgeError):
        validate.validate_source_model(document, tested_versions=("3.3.0",))
