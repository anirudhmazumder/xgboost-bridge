"""Pin `docs/DECISION_INDEX.md` to the two sources it is generated from.

The index exists so that someone about to edit `walk_margin` sees the entries
governing it before they touch it. That only works if it is current, and the
failure mode of a stale index is the bad one: it does not look stale. It looks
like a complete list that happens not to mention the constraint you are about to
break.

So four things are asserted here, and they are different assertions rather than
one repeated:

1. the committed file matches a fresh generation, byte for byte;
2. every decision it cites exists, so no row is a dead link;
3. the symbols that carry the numerical core are actually covered, because an
   index that generated cleanly and covers nothing would pass (1) and (2);
4. both directions of the join still contribute, because either one silently
   collapsing to zero would also pass (1), (2) and (3).

(4) is the assertion this file exists for. The first three check the artifact; the
fourth checks that the mechanism producing it has not quietly become half of
itself -- the same shape as the fixture-door probe, whose failure mode is also
vacuity rather than error.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
INDEX = REPO_ROOT / "docs" / "DECISION_INDEX.md"
DECISIONS = REPO_ROOT / "docs" / "DECISIONS.md"
GENERATOR = REPO_ROOT / "tools" / "build_decision_index.py"


def _load_generator():
    """Import the generator by path; `tools/` is not an installed package."""
    spec = importlib.util.spec_from_file_location("build_decision_index", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_decision_index"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator():
    return _load_generator()


def test_the_committed_index_is_not_stale(generator):
    """Byte-identical, which is the only comparison that catches a partial edit.

    Fails after any change to the source symbols *or* to `DECISIONS.md`, which is
    the point: an entry added without regenerating leaves an index that is
    confidently incomplete.
    """
    assert INDEX.exists(), "run: uv run python tools/build_decision_index.py"
    assert INDEX.read_text() == generator.generate(), (
        "docs/DECISION_INDEX.md is stale. Regenerate:\n"
        "  uv run python tools/build_decision_index.py"
    )


def test_every_cited_decision_exists(generator):
    """No dead links, in either direction of the join.

    A `D0nn` reference in a source comment for an entry that was never written --
    or was renumbered -- would render a link to nothing. Checked against
    `DECISIONS.md` itself rather than against the generator's own parse, so a bug
    in the parser cannot satisfy this by agreeing with itself.
    """
    existing = set(re.findall(r"^## (D\d{3})", DECISIONS.read_text(), re.MULTILINE))
    cited = set(re.findall(r"\[(D\d{3})\]", INDEX.read_text()))

    assert cited, "the index cites no decisions at all"
    assert cited <= existing, f"cited but not present in DECISIONS.md: {sorted(cited - existing)}"


# The numerical core, by name. If any of these stops being covered, the index has
# lost the rows that matter most -- these are the functions where a wrong number
# is silent. Listed explicitly because "some symbols are covered" is not the
# property worth having.
LOAD_BEARING = (
    "walk_margin",
    "extract_trees",
    "observe_intercept",
    "export_model",
)


@pytest.mark.parametrize("symbol", LOAD_BEARING)
def test_the_numerical_core_is_covered(symbol):
    row = re.search(rf"^\| `{re.escape(symbol)}` \|.*$", INDEX.read_text(), re.MULTILINE)
    assert row, f"{symbol} has no row in the decision index"
    assert re.findall(r"\[(D\d{3})\]", row.group(0)), f"{symbol}'s row cites nothing"


def test_walk_margin_carries_the_float32_and_intercept_decisions():
    """The two entries a maintainer must not miss at this particular cursor.

    D004 is the indivisible-numerical-core entry that makes the both-sides cast
    normative; D053 is why the accumulator is seeded with a value read out of the
    engine rather than computed. Neither names `walk_margin` in its body -- they
    predate it -- so this row exists only because the *code* cites them. That is
    exactly the case the mention-only version of this generator missed, and
    asserting it here keeps the code-side direction honest at the one site where
    losing it would cost the most.
    """
    row = re.search(r"^\| `walk_margin` \|.*$", INDEX.read_text(), re.MULTILINE)
    cited = set(re.findall(r"\[(D\d{3})\]", row.group(0)))
    assert {"D004", "D053"} <= cited, f"walk_margin cites {sorted(cited)}"


def test_both_directions_of_the_join_contribute(generator):
    """Neither half may silently collapse to zero.

    Measured by asking each direction what it finds on its own. The mention side
    reads `DECISIONS.md`; the code side reads the source comments. A refactor that
    broke either -- a regex that stops matching, a path glob that stops resolving
    -- would leave a smaller index that still regenerates consistently and still
    passes every other test in this file.
    """
    decisions = generator.parse_decisions(DECISIONS.read_text())
    mentioned = {
        decision.id for decision in decisions if decision.mentions
    }
    from_code = {
        identifier for ids in generator.collect_code_citations().values() for identifier in ids
    }

    assert len(mentioned) >= 20, f"the mention side found only {len(mentioned)} entries"
    assert len(from_code) >= 20, f"the code side found only {len(from_code)} entries"

    # And the two are not the same set -- if they were, one would be redundant and
    # the index would not need both.
    assert from_code - mentioned or mentioned - from_code
