"""Vocabulary scrub: this is a general-purpose library, so no application-specific
vocabulary appears anywhere — variable names and documentation included.

Executable rather than a manual grep, per D029. A check whose output has to be
interpreted by hand is a check that gets waved through once it is inconvenient.

Two term classes, because a single bare-word pattern is not usable:

*Unambiguous* terms are domain nouns and public dataset names with no ordinary
technical meaning. They match on a word boundary.

*Ambiguous* terms are words with a legitimate everyday or engineering sense —
"churn" as in code churn, "target" as in build target. They match only in a form
that indicates modelling use: joined into a compound identifier, or adjacent to a
modelling noun. The first version of this scrub flagged the phrase "incidental
churn on master" as a domain term, which is exactly the false positive that
teaches reviewers to ignore the check.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
from pathlib import Path


def _load_policy():
    """`_policy.py` holds the one definition of the historical-record exemption.

    Loaded by path because pytest runs under `--import-mode=importlib`, which
    does not put this file's directory on `sys.path`. See `_policy` for the rule
    and why it is shared with `test_export` rather than restated here.
    """
    path = Path(__file__).resolve().with_name("_policy.py")
    spec = importlib.util.spec_from_file_location("_xgboost_bridge_test_policy", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_POLICY = _load_policy()

REPO_ROOT = Path(__file__).resolve().parents[3]

SCANNED_SUFFIXES = {".py", ".ts", ".js", ".mjs", ".cjs", ".md", ".toml", ".json", ".yml", ".yaml"}

# This file necessarily contains every term it searches for.
SELF = Path(__file__).resolve()

EXCLUDED_NAMES = {"package-lock.json", "uv.lock"}

UNAMBIGUOUS = [
    # Domain nouns with no ordinary technical sense.
    "patient", "patients", "clinician", "clinical", "diagnosis", "diagnoses",
    "comorbidity", "readmission", "readmissions", "biopsy", "tumor", "tumour",
    "mortgage", "borrower", "policyholder", "claimant", "creditworthiness",
    "delinquency", "underwriting",
    # Named public datasets.
    "titanic", "mnist", "imagenet", "cifar",
]

# Matched only in modelling form. Ordinary technical usage is allowed.
AMBIGUOUS = [
    "churn", "fraud", "fraudulent", "customer", "customers", "subscriber",
    "loan", "loans", "cancer", "insurance", "salary", "spam", "applicant",
    "employee", "iris", "housing",
]

# A modelling noun adjacent to an ambiguous term marks it as domain vocabulary.
MODELLING_CONTEXT = (
    r"predict\w*|model\w*|classif\w*|dataset|corpus|label\w*|target\w*|"
    r"score\w*|rate|probabilit\w*|risk|detect\w*|churn|feature\w*"
)

_UNAMBIGUOUS_ALT = "|".join(UNAMBIGUOUS)
_UNAMBIGUOUS_CAP_ALT = "|".join(term.capitalize() for term in UNAMBIGUOUS)
_AMBIGUOUS_ALT = "|".join(AMBIGUOUS)
_AMBIGUOUS_CAP_ALT = "|".join(term.capitalize() for term in AMBIGUOUS)

# `\b` is the wrong boundary here: `_` is a word character, so `\bpatient\b` misses
# `patient_id`. But "not preceded by a letter" is too strong on its own — it is blind
# to a term in the middle of a camelCase identifier, where the preceding character is
# always a letter. `getPatientCount`, `numPatients` and `loadPatientData` all passed a
# scrub built only on that guard, and the guard's own comment claimed otherwise.
#
# So two entry forms, deliberately separate:
#
#   segment start   not preceded by a letter, term matched case-insensitively —
#                   `patient`, `patient_id`, `Patient`, `TITANIC`, `titanicRows`
#   camelCase hump  preceded by a lowercase letter or digit, term Capitalized so the
#                   hump is real — `getPatientCount`, `numPatients`
#
# The hump form must stay case-SENSITIVE. A global `re.IGNORECASE` would make the
# capitalized alternation match a lowercase run and turn `impatient` into a hit, which
# is the false positive that teaches reviewers to ignore the check (D029). Scoped
# `(?i:...)` is therefore used for the case-insensitive part only, never a global flag.
#
# A trailing optional `s` catches the plural of a term listed in the singular
# (`numTumors`), and `(?![a-z])` still stops `patient` from matching inside a longer
# lowercase word.
UNAMBIGUOUS_RE = re.compile(
    r"(?:"
    r"(?<![A-Za-z])(?i:" + _UNAMBIGUOUS_ALT + r")s?(?![a-z])"
    r"|(?<=[a-z0-9])(?:" + _UNAMBIGUOUS_CAP_ALT + r")s?(?![a-z])"
    r")"
)

# Compound identifier forms, all four of them. The `_`/`-` spellings are
# case-insensitive; the two camelCase spellings are not, for the reason above.
#
#   customer_churn, fraud-score   separator compound, term either side
#   customerChurn                 hump with the term second
#   churnRate                     term first, immediately followed by a hump
#
# `churnRate` is the case an earlier revision of this comment claimed to cover while
# the pattern required a separator and did not match it at all.
AMBIGUOUS_IDENTIFIER_RE = re.compile(
    r"(?:"
    r"(?i:\b(?:\w+[_-](?:" + _AMBIGUOUS_ALT + r")|(?:" + _AMBIGUOUS_ALT + r")[_-]\w+)\b)"
    r"|(?<=[a-z0-9])(?:" + _AMBIGUOUS_CAP_ALT + r")s?(?![a-z])"
    r"|(?<![A-Za-z])(?i:" + _AMBIGUOUS_ALT + r")s?(?=[A-Z])"
    r")"
)

# Adjacent-word form: "customer churn", "churn prediction".
AMBIGUOUS_ADJACENT_RE = re.compile(
    r"\b(?:(?:" + MODELLING_CONTEXT + r")\s+(?:" + _AMBIGUOUS_ALT + r")"
    r"|(?:" + _AMBIGUOUS_ALT + r")\s+(?:" + MODELLING_CONTEXT + r"))\b",
    re.IGNORECASE,
)


def _candidate_files() -> list[Path]:
    """Files git would track: staged, committed, or untracked-and-not-ignored."""
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    files = []
    for line in result.stdout.splitlines():
        path = REPO_ROOT / line
        if path.suffix.lower() not in SCANNED_SUFFIXES:
            continue
        if path.name in EXCLUDED_NAMES:
            continue
        if path.resolve() == SELF:
            continue
        # The historical record is exempt from checks that would otherwise edit
        # it -- the same rule `test_export`'s specifier check applies, from the
        # same definition. See `_policy`.
        if _POLICY.is_historical_record(Path(line)):
            continue
        if path.is_file():
            files.append(path)
    return files


def test_scrub_covers_a_meaningful_number_of_files() -> None:
    """A scrub that silently scans nothing always passes."""
    files = _candidate_files()
    assert len(files) >= 15, f"scrub found only {len(files)} files; expected the repo corpus"


def test_no_application_specific_vocabulary() -> None:
    findings: list[str] = []

    for path in _candidate_files():
        rel = path.relative_to(REPO_ROOT)
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
        ):
            for pattern, kind in (
                (UNAMBIGUOUS_RE, "domain term"),
                (AMBIGUOUS_IDENTIFIER_RE, "domain term in identifier"),
                (AMBIGUOUS_ADJACENT_RE, "domain term in modelling context"),
            ):
                match = pattern.search(line)
                if match:
                    findings.append(f"{rel}:{lineno}: {kind} {match.group(0)!r}")
                    break

    assert not findings, "application-specific vocabulary found:\n" + "\n".join(findings)


def test_scrub_detects_what_it_claims_to_detect() -> None:
    """The scrub is only worth running if it goes red on real violations.

    Verified here directly rather than trusted, since a pattern that matches
    nothing passes the suite exactly like a clean repository does.
    """
    must_flag = [
        "patient_id = 3",
        "const titanicRows = load()",
        "customer_churn = predict(x)",
        "churn prediction threshold",
        "readmission risk",
        "def fraud_score(row): ...",
        # camelCase mid-identifier: the preceding character is a letter, so the
        # segment-start guard alone is blind to every one of these. Each passed
        # the scrub before the hump form was added.
        "def getPatientCount(rows): ...",
        "const numPatients = rows.length",
        "loadPatientData(handle)",
        "const churnRate = 0.2",
        "const customerChurn = predict(x)",
        "let numTumors = 0",
    ]
    for sample in must_flag:
        hit = (
            UNAMBIGUOUS_RE.search(sample)
            or AMBIGUOUS_IDENTIFIER_RE.search(sample)
            or AMBIGUOUS_ADJACENT_RE.search(sample)
        )
        assert hit, f"scrub failed to flag a real violation: {sample!r}"

    must_not_flag = [
        "incidental churn on master",  # the original false positive
        "the build target is es2020",
        "narrow the accumulator after every add",
        "score = margin + intercept",
        "feature_names must be unique",
        "rate_drop and skip_drop are dart parameters",
        # The hump form is case-sensitive on purpose. Under a global IGNORECASE
        # the capitalized alternation would match this lowercase run and the
        # scrub would flag ordinary English.
        "the impatient reader will skip this section",
        "reallocation of the buffer is not required",
    ]
    for sample in must_not_flag:
        hit = (
            UNAMBIGUOUS_RE.search(sample)
            or AMBIGUOUS_IDENTIFIER_RE.search(sample)
            or AMBIGUOUS_ADJACENT_RE.search(sample)
        )
        assert hit is None, f"scrub false-positived on ordinary usage: {sample!r} -> {hit.group(0)!r}"
