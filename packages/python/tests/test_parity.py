"""The cross-language parity gate: exactly `0.0` at two measurement points.

This drives `parity/run_parity.py` over the **whole** fixture corpus -- every
row of every fixture, ordinary and adversarial -- as part of the ordinary suite.
Nothing is sampled, nothing is skipped, and there is no fast path: 299 rows and
one subprocess is not enough work to justify a shortcut, and a gate that runs
only in a separate job is a gate that stops running.

What this file does *not* establish is worth stating first, because a reader who
mistakes it for a correctness check will over-trust it. Cross-language agreement
is not evidence of correctness -- two implementations that agree prove only that
the same code was written twice, and two sides that are equally wrong agree
perfectly. Correctness is established by two other gates with independent
oracles: Python against XGBoost's own recorded `predict()` (`fixtures/tests/`,
`test_predict.py`) and each transform against `mpmath` at 50 digits, per side
independently. **No tolerance from either of those gates appears here**, and one
test below exists to keep it that way.

The gate itself:

* the **margin** point -- exactly `0.0`, uint32 bit patterns, no tolerance;
* the **output** point, after the output transform -- the same.

Around those, four things a naive harness gets wrong and this one is checked
against:

* `-0.0 == 0.0` is `True` and `-0 === 0` is `true`, so comparison is on bit
  patterns. The corpus carries a fixture whose margin is `-0.0` on every row for
  exactly this reason, and a test below asserts both that it agrees and that a
  value comparison would not have distinguished it.
* Rows both predictors refuse are part of the gate. A row silently skipped on
  both sides is indistinguishable from a row that passed, so refusal is a
  status the two sides must *agree* on.
* The transport must not lose bits. It is verified on `-0.0`, both infinities
  and `NaN`, and the naive JSON-number encoding is exercised on the same values
  in the same round trip to show what it would have lost.
* The harness must be able to fail. Three tests inject a deliberate one-ULP
  difference and require the comparison to name the measurement point, the
  fixture and the row -- because a parity harness that has never been observed
  to fail is not evidence of anything.
"""

from __future__ import annotations

import copy
import json
import re
import sys
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PARITY_DIR = _REPO_ROOT / "parity"
if str(_PARITY_DIR) not in sys.path:
    sys.path.insert(0, str(_PARITY_DIR))

from run_parity import (  # noqa: E402
    EMITTER_PATH,
    JS_BUNDLE,
    MEASUREMENT_POINTS,
    OBJECTIVE_OVERWRITE,
    ParityReport,
    bits32_of,
    compare_sides,
    corpus_fixture_paths,
    fixture_display_name,
    float32_from_bits,
    render,
    run_parity,
)

_RUN_PARITY_SOURCE = _PARITY_DIR / "run_parity.py"

#: The fixture whose margin is `-0.0` on every row (FORMAT.md section 6.3): a
#: zero-tree `binary:logistic` model at `base_score = 0.5` passed explicitly.
_SIGNED_ZERO_FIXTURE = "binary_logistic_signed_zero"
_NEGATIVE_ZERO = "0x80000000"
_POSITIVE_ZERO = "0x00000000"
_POSITIVE_INFINITY = "0x7f800000"
#: `sigmoid` of either signed zero, which is where the signed-zero fixture's
#: output point lands: the sign is observable at the margin point and nowhere
#: after it.
_ONE_HALF = "0x3f000000"

#: The measured `binary:logistic` clamp floor -- a subnormal, and never `0.0`.
_CLAMP_FLOOR = "0x0020bd47"


@cache
def _report() -> ParityReport:
    """One full-corpus run, shared by every test in this file.

    Cached because it spawns a subprocess and walks the corpus twice per side
    (once with `objective` overwritten). Every test below reads the same run, so
    they cannot disagree about what was measured.
    """
    return run_parity()


@cache
def _corpus_row_counts() -> dict[str, int]:
    """Row counts read straight from the fixture files.

    The expected totals are derived from the corpus rather than hard-coded, so
    adding a fixture does not require editing a number here -- while a corpus
    that silently *lost* rows still fails, because the harness's count is
    compared against this one.
    """
    counts: dict[str, int] = {}
    for path in corpus_fixture_paths():
        fixture = json.loads(path.read_text(encoding="utf-8"))
        counts[fixture_display_name(path)] = len(fixture["rows"])
    return counts


def _rows(report: ParityReport, fixture: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assert fixture in report.python_records, f"{fixture} is missing from the run"
    return report.python_records[fixture], report.javascript_records[fixture]


def _one_ulp_away(token: str) -> str:
    """The bit pattern one ULP from `token`, staying finite.

    Incrementing the uint32 pattern is a one-ULP step in float32's
    sign-magnitude ordering, which is what makes this a *minimal* injected
    defect rather than an obvious one. `+inf` is stepped downwards instead, so
    the injected value never becomes `NaN` -- a NaN would be caught by being a
    NaN rather than by the bit comparison under test.
    """
    bits = int(token, 16)
    for candidate in ((bits + 1) & 0xFFFFFFFF, (bits - 1) & 0xFFFFFFFF):
        if np.isfinite(np.uint32(candidate).view(np.float32)[()]):
            return f"0x{candidate:08x}"
    raise AssertionError(f"no finite neighbour for {token}")


def _first_value_row(report: ParityReport) -> tuple[str, int]:
    """The first row that produced values on both sides."""
    for fixture, records in report.javascript_records.items():
        for index, record in enumerate(records):
            if record["margin"] is not None:
                return fixture, index
    raise AssertionError("the corpus produced no value rows at all")


def _first_refused_row(report: ParityReport) -> tuple[str, int]:
    for fixture, records in report.javascript_records.items():
        for index, record in enumerate(records):
            if record["margin"] is None:
                return fixture, index
    raise AssertionError("the corpus produced no refused rows at all")


# ---------------------------------------------------------------------------
# Coverage: the gate is only as good as what it looked at
# ---------------------------------------------------------------------------


def test_the_harness_compared_every_row_of_every_fixture() -> None:
    """A harness that silently compared nothing passes exactly like a clean one."""
    report = _report()
    expected = _corpus_row_counts()

    assert not report.structural, "structural disagreement:\n" + "\n".join(report.structural)
    assert len(expected) >= 23, f"only {len(expected)} fixtures in the corpus"
    assert report.rows_compared == sum(expected.values()), (
        f"harness compared {report.rows_compared} rows against "
        f"{sum(expected.values())} in the corpus"
    )
    assert report.rows_compared == report.value_rows + report.refused_rows, (
        "rows were neither compared as values nor agreed as refusals, which is "
        "the shape of a silently skipped row"
    )
    assert report.value_rows >= 289, f"only {report.value_rows} value rows compared"

    tallied = {tally.fixture: tally for tally in report.tallies}
    assert set(tallied) == set(expected), "the per-fixture table does not cover the corpus"
    for name, row_count in expected.items():
        assert tallied[name].rows == row_count, f"{name}: row count disagrees with the fixture"


# ---------------------------------------------------------------------------
# The gate: exactly 0.0 at both measurement points
# ---------------------------------------------------------------------------


def test_margin_parity_is_exactly_zero() -> None:
    """Bit-pattern equality at the margin point, with no tolerance."""
    report = _report()
    assert report.margin_mismatches == [], (
        f"{len(report.margin_mismatches)} margin-point mismatches; first: "
        f"{report.margin_mismatches[0]}"
    )
    assert report.value_rows > 0, "no row reached the margin point"


def test_output_parity_is_exactly_zero() -> None:
    """Bit-pattern equality after the output transform, with no tolerance.

    A margin-only gate passes while a transform mismatch ships, which is this
    project's failure mode relocated one stage downstream.
    """
    report = _report()
    assert report.output_mismatches == [], (
        f"{len(report.output_mismatches)} output-point mismatches; first: "
        f"{report.output_mismatches[0]}"
    )


def test_the_two_sides_were_fed_bit_identical_inputs() -> None:
    """Parity of two answers to two different questions would mean nothing."""
    report = _report()
    assert report.input_disagreements == [], (
        f"the two sides read different inputs; first: {report.input_disagreements[0]}"
    )


def test_both_sides_refuse_the_same_rows_and_refuse_them_the_same_way() -> None:
    """Refusal is a result this gate compares, not a row it drops.

    The status carried through the comparison is the refusal's *name* at both
    measurement points, so a side that refused with a different error -- or
    refused the margin and returned an output -- fails here rather than being
    filed as an agreement.
    """
    report = _report()
    assert report.refusal_disagreements == [], (
        f"the sides disagreed about refusing; first: {report.refusal_disagreements[0]}"
    )
    refusal_fixtures = [tally for tally in report.tallies if tally.refused_rows]
    assert len(refusal_fixtures) == 1, (
        f"expected exactly one refusal fixture, found {[t.fixture for t in refusal_fixtures]}"
    )
    only = refusal_fixtures[0]
    assert only.refused_rows == only.rows, (
        f"{only.fixture}: {only.refused_rows}/{only.rows} rows refused; every row "
        "of that fixture carries an infinity and must be refused by both sides"
    )
    assert only.refused_rows >= 10, f"{only.fixture}: only {only.refused_rows} refused rows"
    assert report.refused_rows == only.refused_rows

    python_rows, javascript_rows = _rows(report, only.fixture)
    for index, (left, right) in enumerate(zip(python_rows, javascript_rows, strict=True)):
        for point in MEASUREMENT_POINTS:
            assert left[f"{point}_refusal"] == "NonFiniteFeatureError", (
                f"{only.fixture}[{index}]: Python refused with {left[f'{point}_refusal']}"
            )
            assert right[f"{point}_refusal"] == "NonFiniteFeatureError", (
                f"{only.fixture}[{index}]: JavaScript refused with {right[f'{point}_refusal']}"
            )


# ---------------------------------------------------------------------------
# The values a `==` comparison would have got wrong
# ---------------------------------------------------------------------------


def test_the_signed_zero_fixture_agrees_at_both_points_as_bit_patterns() -> None:
    """`-0.0` survives the boundary and is not normalized on either side.

    This is the case a value comparison would wrongly pass, so the test asserts
    both halves: the two sides agree on the bit pattern `0x80000000` at the
    margin point, *and* a `==` comparison of that pattern against `0x00000000`
    would not have told them apart.

    The **output** point of the same fixture is `0x3f000000`, exactly one half:
    `sigmoid` maps both signed zeros to the same value, so the sign is not
    observable after the transform. That is not a gap in the fixture, it is the
    reason the gate has two measurement points and the margin one cannot be
    dropped -- a side that normalized `-0.0` would be caught at the margin point
    and by nothing downstream of it. The companion injection test measures
    exactly that.
    """
    report = _report()
    python_rows, javascript_rows = _rows(report, _SIGNED_ZERO_FIXTURE)
    assert python_rows, f"{_SIGNED_ZERO_FIXTURE} has no rows"
    for index, (left, right) in enumerate(zip(python_rows, javascript_rows, strict=True)):
        assert left["margin"] == _NEGATIVE_ZERO, (
            f"{_SIGNED_ZERO_FIXTURE}[{index}]: Python margin is {left['margin']}, "
            f"expected {_NEGATIVE_ZERO}"
        )
        assert right["margin"] == _NEGATIVE_ZERO, (
            f"{_SIGNED_ZERO_FIXTURE}[{index}]: JavaScript margin is {right['margin']}, "
            f"expected {_NEGATIVE_ZERO}"
        )
        for point in MEASUREMENT_POINTS:
            assert left[point] == right[point], f"{_SIGNED_ZERO_FIXTURE}[{index}]: {point}"
        assert left["output"] == _ONE_HALF, (
            f"{_SIGNED_ZERO_FIXTURE}[{index}]: output is {left['output']}, expected "
            f"{_ONE_HALF} -- sigmoid of either signed zero is exactly one half"
        )

    # Why the comparison is on patterns and not on values.
    assert float32_from_bits(_NEGATIVE_ZERO) == float32_from_bits(_POSITIVE_ZERO)
    assert _NEGATIVE_ZERO != _POSITIVE_ZERO
    assert bits32_of(float32_from_bits(_NEGATIVE_ZERO)) == _NEGATIVE_ZERO


def test_positive_infinity_crosses_the_boundary_and_agrees() -> None:
    """`survival:cox` returns `+inf` above margin ~88.72, and JSON has no literal for it."""
    report = _report()
    python_rows, javascript_rows = _rows(report, "survival_cox_overflow_to_infinity")
    infinite = 0
    for index, (left, right) in enumerate(zip(python_rows, javascript_rows, strict=True)):
        if left["output"] != _POSITIVE_INFINITY:
            continue
        assert right["output"] == _POSITIVE_INFINITY, f"row {index}: JavaScript {right['output']}"
        infinite += 1
    assert infinite > 0, "no row reached +inf, so the boundary was never exercised there"
    assert infinite < len(python_rows), "every row is +inf, so the boundary is invisible"


def test_the_logistic_clamp_floor_crosses_as_a_subnormal_and_agrees() -> None:
    """The floor is a subnormal and is never `0.0`; a lossy transport would flatten it."""
    report = _report()
    python_rows, javascript_rows = _rows(report, "adversarial/logistic_clamp_floor_output")
    floored = 0
    for index, (left, right) in enumerate(zip(python_rows, javascript_rows, strict=True)):
        if left["output"] != _CLAMP_FLOOR:
            continue
        assert right["output"] == _CLAMP_FLOOR, f"row {index}: JavaScript {right['output']}"
        assert left["output"] not in (_POSITIVE_ZERO, _NEGATIVE_ZERO)
        floored += 1
    assert floored > 0, "no row reached the clamp floor"


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


def test_the_bit_pattern_transport_is_lossless() -> None:
    """Every probe value comes back as the pattern it was sent as.

    A transport that quietly normalized a value would make this harness report
    perfect parity while hiding a real difference, so the probe covers precisely
    the values that get normalized: `-0.0`, both infinities, `NaN`, the
    subnormal boundary, and the measured clamp floor.
    """
    report = _report()
    transport = report.transport
    assert transport, "the transport probe did not run"
    assert transport["bit_pattern_encoding_lossless"], report.structural
    for probe in transport["probes"]:
        assert probe["echoed"] == probe["sent"], f"transport lost {probe['sent']}"
    sent = {probe["sent"] for probe in transport["probes"]}
    for required in (_NEGATIVE_ZERO, _POSITIVE_INFINITY, "0xff800000", "0x7fc00000", _CLAMP_FLOOR):
        assert required in sent, f"the transport probe does not cover {required}"


def test_a_json_number_transport_would_have_lost_the_values_that_matter_most() -> None:
    """The encoding choice is load-bearing, and this is the measurement that says so.

    `JSON.stringify(-0)` emits `0`; `Infinity` and `NaN` serialize as `null`.
    Each probe value crossed the boundary in both encodings, in the same round
    trip, so this is observed rather than cited.
    """
    report = _report()
    lost = report.transport["values_a_json_number_transport_would_have_lost"]
    for required in (_NEGATIVE_ZERO, _POSITIVE_INFINITY, "0xff800000", "0x7fc00000"):
        assert required in lost, (
            f"a JSON number transport did not lose {required}; the demonstration "
            "that bit patterns are necessary no longer holds"
        )
    by_pattern = {probe["sent"]: probe for probe in report.transport["probes"]}
    assert by_pattern[_NEGATIVE_ZERO]["naive_number"] == 0
    assert bits32_of(by_pattern[_NEGATIVE_ZERO]["naive_number"]) == _POSITIVE_ZERO
    assert by_pattern[_POSITIVE_INFINITY]["naive_number"] is None


# ---------------------------------------------------------------------------
# `objective` is not operative, cross-checked in both languages from one place
# ---------------------------------------------------------------------------


def test_neither_predictor_branches_on_objective() -> None:
    """Overwriting the label after load changes no bit, in either language.

    FORMAT.md section 4 forbids a predictor from branching on `objective`; D028
    records why, and D047 records that a *source-level* check and a
    *behavioural* check catch different things -- an obfuscated branch turns nine
    behavioural tests red while a source scan stays green.

    The two source-level scans stay where they are, one per language
    (`test_predict.py::test_no_prediction_path_function_reads_the_objective_field`
    and `packages/js/test/predict.test.js`, which greps the shipped bundle).
    What lives *here* is the behavioural cross-check for both languages in one
    place, over the whole corpus rather than a hand-built row: each side loads
    every artifact twice, overwrites `objective` on the second predictor, and
    every bit at both measurement points must match the first.
    """
    report = _report()
    assert report.objective_findings == [], (
        f"a prediction path reads `objective`; first: {report.objective_findings[0]}"
    )
    assert OBJECTIVE_OVERWRITE not in {"reg:squarederror", "binary:logistic", "survival:cox"}, (
        "the overwritten label must be outside the enumerated set, or the check "
        "could be satisfied by a branch that happens to pick the same transform"
    )
    # The overwritten run must actually have happened on both sides, or the
    # comparison above compared nothing.
    for fixture, records in report.python_records.items():
        for index, record in enumerate(records):
            for point in MEASUREMENT_POINTS:
                assert f"{point}_relabelled" in record, f"{fixture}[{index}]: Python"
                assert f"{point}_relabelled" in report.javascript_records[fixture][index], (
                    f"{fixture}[{index}]: JavaScript"
                )


# ---------------------------------------------------------------------------
# The harness must be able to fail
# ---------------------------------------------------------------------------


def test_an_injected_one_ulp_difference_fails_at_the_margin_point() -> None:
    """One bit, on one row, at the margin point -- named exactly."""
    report = _report()
    fixture, row = _first_value_row(report)
    perturbed = copy.deepcopy(report.javascript_records)
    original = perturbed[fixture][row]["margin"]
    perturbed[fixture][row]["margin"] = _one_ulp_away(original)

    injected = compare_sides(report.python_records, perturbed)
    assert len(injected.margin_mismatches) == 1, injected.margin_mismatches
    found = injected.margin_mismatches[0]
    assert found.point == "margin"
    assert found.fixture == fixture
    assert found.row == row
    assert found.python == original
    assert found.javascript == _one_ulp_away(original)
    assert not injected.clean
    assert found.python != found.javascript
    # The output point is untouched, so a margin defect must not be reported as
    # an output defect: the two have completely different causes.
    assert injected.output_mismatches == []
    assert str(found).startswith("margin point,")
    assert f"{fixture}[{row}]" in render(injected)


def test_an_injected_one_ulp_difference_fails_at_the_output_point() -> None:
    """The second measurement point fails independently of the first."""
    report = _report()
    fixture, row = _first_value_row(report)
    perturbed = copy.deepcopy(report.javascript_records)
    original = perturbed[fixture][row]["output"]
    perturbed[fixture][row]["output"] = _one_ulp_away(original)

    injected = compare_sides(report.python_records, perturbed)
    assert len(injected.output_mismatches) == 1, injected.output_mismatches
    found = injected.output_mismatches[0]
    assert (found.point, found.fixture, found.row) == ("output", fixture, row)
    assert found.python == original
    assert injected.margin_mismatches == []
    assert not injected.clean


def test_an_injected_negative_zero_normalization_fails_at_the_margin_point() -> None:
    """The defect a value comparison could not see, and the point that sees it.

    Replacing `-0.0` with `+0.0` on one side is invisible to `==` and to any
    tolerance, at any width. It is caught here, at the margin point, on every row
    of the fixture that exists for it -- and it is invisible at the output point,
    because `sigmoid` maps both signed zeros to exactly one half. The gate needs
    both points, and this is the direction that proves the first one is not
    redundant.
    """
    report = _report()
    perturbed = copy.deepcopy(report.javascript_records)
    for record in perturbed[_SIGNED_ZERO_FIXTURE]:
        assert record["margin"] == _NEGATIVE_ZERO
        record["margin"] = _POSITIVE_ZERO

    injected = compare_sides(report.python_records, perturbed)
    rows = len(perturbed[_SIGNED_ZERO_FIXTURE])
    assert len(injected.margin_mismatches) == rows, injected.margin_mismatches
    assert injected.output_mismatches == [], (
        "the output point reported a signed-zero defect it cannot see; the "
        "fixture or the transform has changed"
    )
    for divergence in injected.margin_mismatches:
        assert divergence.fixture == _SIGNED_ZERO_FIXTURE
        assert (divergence.python, divergence.javascript) == (_NEGATIVE_ZERO, _POSITIVE_ZERO)
    assert not injected.clean


def test_a_side_that_stops_refusing_a_refused_row_fails() -> None:
    """A refused row that quietly turns into a number is a disagreement, not a skip."""
    report = _report()
    fixture, row = _first_refused_row(report)
    perturbed = copy.deepcopy(report.javascript_records)
    for point in MEASUREMENT_POINTS:
        perturbed[fixture][row][point] = "0x3f800000"
        perturbed[fixture][row][f"{point}_refusal"] = None

    injected = compare_sides(report.python_records, perturbed)
    assert len(injected.refusal_disagreements) == 1, injected.refusal_disagreements
    found = injected.refusal_disagreements[0]
    assert (found.point, found.fixture, found.row) == ("refusal", fixture, row)
    assert found.python.startswith("refused:")
    assert found.javascript == "value"
    assert not injected.clean


def test_a_side_that_refuses_with_a_different_error_fails() -> None:
    """Agreeing to refuse is not enough; the two sides must refuse the same way."""
    report = _report()
    fixture, row = _first_refused_row(report)
    perturbed = copy.deepcopy(report.javascript_records)
    for point in MEASUREMENT_POINTS:
        perturbed[fixture][row][f"{point}_refusal"] = "TypeError"

    injected = compare_sides(report.python_records, perturbed)
    assert len(injected.refusal_disagreements) == 1, injected.refusal_disagreements
    assert injected.refusal_disagreements[0].javascript == "refused:TypeError/TypeError"


def test_an_injected_objective_branch_is_reported_per_language() -> None:
    """The behavioural `objective` check can fail, and it names the language."""
    report = _report()
    fixture, row = _first_value_row(report)
    perturbed = copy.deepcopy(report.javascript_records)
    original = perturbed[fixture][row]["margin_relabelled"]
    perturbed[fixture][row]["margin_relabelled"] = _one_ulp_away(original)

    injected = compare_sides(report.python_records, perturbed)
    assert len(injected.objective_findings) == 1, injected.objective_findings
    found = injected.objective_findings[0]
    assert found.point == "objective/JavaScript"
    assert (found.fixture, found.row) == (fixture, row)
    assert injected.margin_mismatches == []
    assert not injected.clean


def test_a_side_that_read_different_inputs_fails() -> None:
    report = _report()
    fixture, row = _first_value_row(report)
    perturbed = copy.deepcopy(report.javascript_records)
    perturbed[fixture][row]["input_bits"] = ["0x0000000000000000"]

    injected = compare_sides(report.python_records, perturbed)
    assert len(injected.input_disagreements) == 1, injected.input_disagreements
    assert (injected.input_disagreements[0].fixture, injected.input_disagreements[0].row) == (
        fixture,
        row,
    )
    assert not injected.clean


def test_a_malformed_bit_pattern_token_is_refused_rather_than_compared() -> None:
    """A token that is not a uint32 pattern is a structural failure, not a mismatch.

    Found by measurement, not foresight: a one-ULP injection written with
    JavaScript's `& 0xffffffff` emitted `0x-40f572f5`, and an earlier version of
    the comparison happily string-compared it and printed it as a parity defect.
    Two malformed tokens that happen to agree would have been worse -- they would
    have been credited as parity.
    """
    report = _report()
    fixture, row = _first_value_row(report)
    for bad in ("0x-40f572f5", "0x3F0A8D0A", "0x3f0a8d0", "3f0a8d0a", 1056964608):
        perturbed = copy.deepcopy(report.javascript_records)
        perturbed[fixture][row]["margin"] = bad
        injected = compare_sides(report.python_records, perturbed)
        assert injected.structural, f"{bad!r} was accepted as a bit pattern"
        assert "not a uint32 hex bit pattern" in injected.structural[0], injected.structural
        assert injected.margin_mismatches == [], "a malformed token was read as a mismatch"
        assert not injected.clean

    perturbed = copy.deepcopy(report.javascript_records)
    perturbed[fixture][row]["input_bits"] = ["0x3f0a8d0a"]
    injected = compare_sides(report.python_records, perturbed)
    assert injected.structural, "a float32-width input digest was accepted as float64"
    assert not injected.clean


def test_a_side_that_silently_dropped_a_fixture_fails_structurally() -> None:
    """A shrinking corpus on one side must not read as agreement."""
    report = _report()
    perturbed = copy.deepcopy(report.javascript_records)
    dropped = next(iter(perturbed))
    del perturbed[dropped]
    injected = compare_sides(report.python_records, perturbed)
    assert injected.structural, "dropping a whole fixture went unnoticed"
    assert dropped in injected.structural[0]
    assert not injected.clean

    shortened = copy.deepcopy(report.javascript_records)
    fixture, _ = _first_value_row(report)
    shortened[fixture] = shortened[fixture][:-1]
    injected = compare_sides(report.python_records, shortened)
    assert injected.structural, "dropping a row went unnoticed"
    assert not injected.clean


# ---------------------------------------------------------------------------
# Properties of the harness itself
# ---------------------------------------------------------------------------


def test_the_javascript_side_ran_against_the_built_bundle() -> None:
    """`dist/`, never `src/` (D011). Checked three ways, none of them a comment."""
    report = _report()
    url = report.environment["module_url"]
    assert url.endswith("/packages/js/dist/index.js"), url
    assert "/src/" not in url, url
    assert report.environment["module_specifier"] == "../packages/js/dist/index.js"
    assert JS_BUNDLE.is_file(), JS_BUNDLE

    source = EMITTER_PATH.read_text(encoding="utf-8")
    specifiers = re.findall(r"""from\s+["']([^"']+)["']""", source)
    assert specifiers, "the emitter imports nothing, so it cannot be measuring the package"
    for specifier in specifiers:
        assert "js/src" not in specifier, f"the emitter imports from src/: {specifier}"
    assert "../packages/js/dist/index.js" in specifiers


def test_the_harness_carries_no_tolerance_of_any_kind() -> None:
    """Parity is exact equality, and this is the tripwire against that eroding.

    Cross-language parity and upstream accuracy are different gates. The
    accuracy gate's relative bound is not imported here, not restated here, and
    not available here as a fallback -- so a future change that reaches for one
    fails this test rather than quietly widening the headline gate.
    """
    forbidden = (
        r"1e-0?6",
        r"\bisclose\b",
        r"\ballclose\b",
        r"\bapprox\b",
        r"\batol\b",
        r"\brtol\b",
        r"\babs\(",
        r"\bround\(",
    )
    for path in (_RUN_PARITY_SOURCE, EMITTER_PATH):
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            assert not re.search(pattern, text), (
                f"{path.name} contains {pattern!r}: parity is bit-pattern equality "
                "and admits no tolerance, at either measurement point"
            )


def test_the_report_states_counts_and_never_a_mean() -> None:
    """A mean hides one catastrophic row in a large corpus."""
    report = _report()
    rendered = render(report)
    assert "mean" not in rendered.lower()
    assert f"rows compared                 {report.rows_compared}" in rendered
    assert "margin-point mismatches       0" in rendered
    assert "output-point mismatches       0" in rendered
    for tally in report.tallies:
        assert tally.fixture in rendered, f"{tally.fixture} is absent from the report"
    assert "PARITY: 0.0 at both measurement points" in rendered


def test_the_report_of_a_failure_names_the_measurement_point() -> None:
    """A report that says only "parity failed" is not usable."""
    report = _report()
    fixture, row = _first_value_row(report)
    perturbed = copy.deepcopy(report.javascript_records)
    perturbed[fixture][row]["output"] = _one_ulp_away(perturbed[fixture][row]["output"])
    injected = compare_sides(report.python_records, perturbed)
    rendered = render(injected)
    assert "PARITY FAILED" in rendered
    assert "output point" in rendered
    assert f"{fixture}[{row}]" in rendered
    assert injected.first("output") is not None
    assert injected.first("margin") is None
