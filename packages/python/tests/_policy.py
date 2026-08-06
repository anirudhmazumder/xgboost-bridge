"""One definition of which paths are the historical record, shared by every check.

Two checks in this suite scan committed prose: the vocabulary scrub, and the
specifier comparison in ``test_export``. Both face the same question — what does
a check do when the text it would flag is a *record of the thing being flagged*?

D052 answered it inconsistently in a single commit. The specifier test excluded
``docs/DECISIONS.md`` so a decision entry could quote a withdrawn dependency
range. The scrub did not exclude it, flagged that same file for naming the
identifiers a fixed blind spot used to miss, and the prose was rewritten to
satisfy the check. Same file, same category of content, opposite rule.

The rule, decided once and applied by both:

    **The historical record is exempt from any check that would otherwise edit
    it.**

A decision entry, a probe transcript and a superseded finding are all evidence.
Their value is that they say what was true at the time, including the wrong
value, the withdrawn specifier, and the identifier that used to slip. A check
that forces evidence to be rewritten to stay green does not make the repository
more correct; it makes the record less accurate, and it does so silently,
because the edit looks like a passing build.

This is scoped deliberately and is not a general escape hatch:

* It covers ``docs/DECISIONS.md`` and everything under ``probes/`` — the two
  places this project stores what it measured and when.
* It covers **prose scanning only**. Nothing here exempts any file from a
  behavioural test, and neither of the two checks that consult this module can
  affect a prediction.
* Every other tracked file is scanned exactly as before. ``FORMAT.md``,
  ``COMPAT.md``, ``VERIFICATION.md`` and both package READMEs are normative
  documents a user acts on, not records of what was once believed, and they stay
  in scope for both checks.

The narrower reading — exempt from the *specifier* check but not the scrub —
was rejected because it is the inconsistency that prompted this module, and
because the failure mode it produces is the worse of the two: a decision record
edited to keep a check green is a record that no longer records.

Consumers load this module by path. pytest runs with ``--import-mode=importlib``
(see the root ``pyproject.toml`` and the reason recorded there), which
deliberately does not put a test file's own directory on ``sys.path``, so a plain
``import _policy`` from a sibling test module fails. The short
``spec_from_file_location`` load at the top of each consumer is the cost of
having one definition instead of two that can drift apart.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["HISTORICAL_RECORD_PATHS", "is_historical_record"]

#: Repository-relative. A file is exempt if it equals one of these or is under it.
HISTORICAL_RECORD_PATHS: tuple[Path, ...] = (
    Path("docs") / "DECISIONS.md",
    Path("probes"),
)


def is_historical_record(relative_path: Path) -> bool:
    """True if ``relative_path`` (repo-relative) is part of the historical record."""
    for exempt in HISTORICAL_RECORD_PATHS:
        if relative_path == exempt:
            return True
        if exempt in relative_path.parents:
            return True
    return False
