"""Verification of `schema/xgboost-bridge-v1.schema.json` against FORMAT.md section 3.

Two independent jobs, kept separate because a schema that merely fails to
reject anything would pass a corpus just as well as a correct one:

1. **Acceptance** -- every fixture artifact in the corpus (ordinary and
   adversarial) validates. The oracle here is the corpus itself, which is
   already independently verified against XGBoost's own output elsewhere
   (`fixtures/tests/test_corpus.py`, `fixtures/tests/test_adversarial.py`);
   this suite asks a different question of the same files -- does the shape
   match the specification -- and does not re-derive their numeric content.
2. **Rejection** -- deliberately mutated copies of a valid artifact must be
   rejected, one mutation per structural rule this schema is supposed to
   enforce. A schema that accepts everything is invisible to job 1 and only
   job 2 can catch it.

`fixtures/corpus/reference/` holds the `mpmath` transform reference table,
not an artifact, and is deliberately not walked (non-recursive glob),
mirroring `packages/js/test/corpus.test.js`'s own `loadDirectory` comment.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPO_ROOT / "schema" / "xgboost-bridge-v1.schema.json"
CORPUS_DIR = REPO_ROOT / "fixtures" / "corpus"
ADVERSARIAL_DIR = CORPUS_DIR / "adversarial"


def _load_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


SCHEMA: dict[str, Any] = _load_schema()


def _load_fixture_artifacts() -> dict[str, dict[str, Any]]:
    """Every fixture artifact under `fixtures/corpus/` and its `adversarial/`
    subdirectory, keyed by a name that disambiguates the two directories.

    Non-recursive by construction in both directories, exactly like
    `packages/js/test/corpus.test.js`'s `loadDirectory` -- so `reference/`,
    a sibling holding non-artifact data, is never walked.
    """
    artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(CORPUS_DIR.glob("*.json")):
        fixture = json.loads(path.read_text(encoding="utf-8"))
        artifacts[path.stem] = fixture["artifact"]
    for path in sorted(ADVERSARIAL_DIR.glob("*.json")):
        fixture = json.loads(path.read_text(encoding="utf-8"))
        artifacts[f"adversarial/{path.stem}"] = fixture["artifact"]
    return artifacts


FIXTURE_ARTIFACTS: dict[str, dict[str, Any]] = _load_fixture_artifacts()

#: An artifact with a non-empty `trees` array, needed by the mutation that
#: adds a sixth key to a tree object -- the signed-zero fixture's `trees` is
#: deliberately empty (D036) and has no tree object to mutate.
_ARTIFACT_WITH_TREES: dict[str, Any] | None = next(
    (artifact for artifact in FIXTURE_ARTIFACTS.values() if artifact["trees"]),
    None,
)


def test_schema_file_is_a_valid_draft_2020_12_schema() -> None:
    """The oracle is the draft's own meta-schema, via the checker jsonschema
    ships for it -- not a re-derivation of this schema's own rules."""
    jsonschema.Draft202012Validator.check_schema(SCHEMA)


def test_schema_declares_draft_2020_12_and_has_an_id() -> None:
    assert SCHEMA["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert isinstance(SCHEMA.get("$id"), str) and SCHEMA["$id"]


#: Hosts whose namespace is bound to the repository account itself. A `$id` outside
#: these is only as trustworthy as whoever holds the domain.
_ID_ALLOWED_PREFIXES = (
    "https://raw.githubusercontent.com/anirudhmazumder/xgboost-bridge/",
    "https://github.com/anirudhmazumder/xgboost-bridge/",
)


def test_schema_id_is_hosted_where_the_project_actually_controls_the_namespace() -> None:
    """`$id` was `https://xgboost-bridge.dev/...`, a domain nobody had registered.

    An identifier is not required to resolve, so nothing broke and no test could
    notice. The exposure is ownership, not resolution: the repository is public,
    so the unregistered domain in a published schema was an open invitation --
    anyone could register it and serve a *different* document at this project's
    canonical `$id`, and a consumer dereferencing it would have no signal.

    Pinning the host rather than the exact string leaves the path free to change
    with the schema version, and still fails the moment `$id` moves somewhere the
    account behind this repository does not control.
    """
    schema_id = SCHEMA["$id"]
    assert schema_id.startswith(_ID_ALLOWED_PREFIXES), (
        f"schema $id {schema_id!r} is not under a namespace this project controls; "
        f"expected one of {_ID_ALLOWED_PREFIXES}"
    )
    assert schema_id.endswith(SCHEMA_PATH.name), (
        f"schema $id {schema_id!r} does not end in this file's name, {SCHEMA_PATH.name!r}"
    )


def test_fixture_corpus_is_non_empty() -> None:
    """A suite that silently found zero fixtures must fail, not pass quietly."""
    assert FIXTURE_ARTIFACTS, f"no fixture artifacts found under {CORPUS_DIR}"


def test_fixture_with_trees_exists_for_the_tree_mutation_below() -> None:
    assert _ARTIFACT_WITH_TREES is not None, "no fixture artifact has a non-empty trees array"


@pytest.mark.parametrize("name", sorted(FIXTURE_ARTIFACTS))
def test_every_fixture_artifact_validates(name: str) -> None:
    jsonschema.validate(instance=FIXTURE_ARTIFACTS[name], schema=SCHEMA)


# ---------------------------------------------------------------------------
# Description content: FORMAT.md and D028/D015 require these two facts to be
# stated in the schema itself, not only in prose documents a reader of the
# artifact format might never open.
# ---------------------------------------------------------------------------


def test_objective_description_states_it_is_non_operative_metadata() -> None:
    description = SCHEMA["properties"]["objective"]["description"]
    lowered = description.lower()
    assert "non-operative" in lowered, description
    assert "export-time" in lowered, description
    assert "predictor" in lowered and "branch" in lowered, description


def test_intercept_description_states_it_is_the_operative_value() -> None:
    description = SCHEMA["properties"]["intercept"]["description"]
    lowered = description.lower()
    assert "single operative numeric value" in lowered, description
    assert "provenance.base_score is read by nothing" in lowered, description


# ---------------------------------------------------------------------------
# Rejection: a schema is only worth having if it goes red on real violations.
# ---------------------------------------------------------------------------


def _assert_rejected(instance: Any, *, why: str) -> None:
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=instance, schema=SCHEMA)


def _valid_artifact() -> dict[str, Any]:
    """A deep copy of a real, known-valid fixture artifact -- never a
    hand-built stand-in, so every mutation below starts from something the
    corpus test above has already shown the schema accepts."""
    name = sorted(FIXTURE_ARTIFACTS)[0]
    return copy.deepcopy(FIXTURE_ARTIFACTS[name])


def test_rejects_an_eighth_top_level_key() -> None:
    artifact = _valid_artifact()
    artifact["unrecognized_extra_key"] = "anything"
    _assert_rejected(artifact, why="additionalProperties: false at the top level")


def test_rejects_a_missing_required_top_level_key() -> None:
    artifact = _valid_artifact()
    del artifact["trees"]
    _assert_rejected(artifact, why="trees is one of the seven required envelope keys")


def test_rejects_format_version_two() -> None:
    artifact = _valid_artifact()
    artifact["format_version"] = 2
    _assert_rejected(artifact, why="format_version must be const 1")


def test_rejects_a_tree_with_a_sixth_key() -> None:
    assert _ARTIFACT_WITH_TREES is not None
    artifact = copy.deepcopy(_ARTIFACT_WITH_TREES)
    artifact["trees"][0]["unrecognized_extra_key"] = [0]
    _assert_rejected(artifact, why="additionalProperties: false on the tree object")


def test_rejects_provenance_base_score_as_a_number() -> None:
    artifact = _valid_artifact()
    artifact["provenance"]["base_score"] = 0.5
    _assert_rejected(artifact, why="provenance.base_score must be a JSON string, per FORMAT.md section 16")


def test_rejects_empty_feature_names() -> None:
    artifact = _valid_artifact()
    artifact["feature_names"] = []
    _assert_rejected(artifact, why="feature_names is required to be non-empty, per FORMAT.md section 7")
