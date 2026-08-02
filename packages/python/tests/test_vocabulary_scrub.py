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

import re
import subprocess
from pathlib import Path

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

# `\b` is the wrong boundary here: `_` is a word character, so `\bpatient\b` misses
# `patient_id`, and a trailing `\b` also misses the camelCase `titanicRows`. Instead:
# not preceded by a letter, and not followed by a LOWERCASE letter. The term itself is
# matched case-insensitively via a scoped flag so the lookahead stays case-sensitive —
# a global re.IGNORECASE would make `[a-z]` match `R` and break the camelCase case.
UNAMBIGUOUS_RE = re.compile(
    r"(?<![A-Za-z])(?i:" + "|".join(UNAMBIGUOUS) + r")(?![a-z])"
)

# Compound identifier form: customer_churn, churnRate, fraud-score.
AMBIGUOUS_IDENTIFIER_RE = re.compile(
    r"\b(?:\w+[_-]("
    + "|".join(AMBIGUOUS)
    + r")|("
    + "|".join(AMBIGUOUS)
    + r")[_-]\w+)\b",
    re.IGNORECASE,
)

# Adjacent-word form: "customer churn", "churn prediction".
AMBIGUOUS_ADJACENT_RE = re.compile(
    r"\b(?:(?:" + MODELLING_CONTEXT + r")\s+(?:" + "|".join(AMBIGUOUS) + r")"
    r"|(?:" + "|".join(AMBIGUOUS) + r")\s+(?:" + MODELLING_CONTEXT + r"))\b",
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
    ]
    for sample in must_not_flag:
        hit = (
            UNAMBIGUOUS_RE.search(sample)
            or AMBIGUOUS_IDENTIFIER_RE.search(sample)
            or AMBIGUOUS_ADJACENT_RE.search(sample)
        )
        assert hit is None, f"scrub false-positived on ordinary usage: {sample!r} -> {hit.group(0)!r}"
