"""Cross-language parity: do the two implementations produce identical bits?

That is the *only* question this harness answers, and stating the limit is as
important as stating the result. Cross-language agreement is **not evidence of
correctness**: two implementations that agree perfectly prove only that the
same code was written twice, and two sides that are equally wrong are exactly
invisible here. Correctness lives in two other gates -- Python against
XGBoost's own recorded `predict()` output, and each transform against an
`mpmath` reference at 50 digits, per side independently. Neither of those is
touched by, imported into, or referenced from this file.

The consequence for this file: **no tolerance appears anywhere in it, at any
point, for any reason.** The accuracy gate against XGBoost is relative and has
a bound; parity is exact equality on bit patterns and has none. Conflating the
two is how a tolerance leaks into the parity gate, so the bound is not imported
here, not restated here, and not used as a fallback here.

FORMAT.md section 5.3 fixes **two** measurement points, and both are checked:

1. the **margin** -- the accumulator of the normative walk;
2. the **final output** -- after the output transform.

A margin-only check passes while a transform mismatch ships, which is the
failure this project exists to prevent, relocated one stage downstream.

Four properties of the design are load-bearing rather than incidental:

* **Comparison is on bit patterns, never `==`.** `-0.0 == 0.0` is `True` in
  Python and `-0 === 0` is `true` in JavaScript, and they are different
  artifacts reachable through an ordinary default. The corpus carries a fixture
  whose margin is `-0.0` on every row precisely to catch a harness that gets
  this wrong.
* **Transport across the language boundary is uint32 hex bit-pattern strings**
  -- the same encoding the corpus already uses. `JSON.stringify(-0)` emits `0`,
  so exchanging values as JSON *numbers* silently destroys signed zero, and
  `+Infinity` has no JSON number representation at all (it serializes as
  `null`). Both losses are demonstrated by the transport probe below rather than
  asserted from documentation: the probe returns each value in both encodings,
  and the harness checks that the bit-pattern encoding survives and that the
  naive numeric one does not.
* **Rows both sides are expected to refuse are part of the gate.** A row
  silently skipped on both sides is indistinguishable from a row that passed, so
  refusal is recorded as a status and the two sides must *agree* on it, not
  merely fail to produce a number.
* **The two sides must have been fed identical inputs**, or a parity result of
  `0.0` would be a statement about two different questions. Each side emits the
  float64 bit pattern of every value it handed its own walk, and the harness
  requires those to agree before it credits any parity number.

The JavaScript side runs as a subprocess against the **built bundle** under
`packages/js/dist/`, never `packages/js/src/`. The emitter's import specifier is
a static literal, the resolved module URL is reported back, and both are checked.

Run directly for the full report::

    uv run python parity/run_parity.py

`packages/python/tests/test_parity.py` drives the same code over the same corpus
as part of the ordinary suite; nothing here is optional or sampled.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from xgboost_bridge.predict import Predictor

__all__ = [
    "MEASUREMENT_POINTS",
    "Divergence",
    "FixtureTally",
    "ParityReport",
    "bits32_of",
    "bits64_of",
    "compare_sides",
    "corpus_fixture_paths",
    "float32_from_bits",
    "fixture_display_name",
    "javascript_side",
    "main",
    "python_side",
    "render",
    "run_parity",
]

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
PARITY_DIR: Final[Path] = Path(__file__).resolve().parent
CORPUS_DIR: Final[Path] = REPO_ROOT / "fixtures" / "corpus"
ADVERSARIAL_DIR: Final[Path] = CORPUS_DIR / "adversarial"
EMITTER_PATH: Final[Path] = PARITY_DIR / "emit_js_predictions.mjs"

#: The built bundle the JavaScript side must be measured through, and the
#: sources it is built from. Tests import from `dist/`, never `src/`, so a
#: parity number measured against `src/` would be about code that does not ship.
JS_BUNDLE: Final[Path] = REPO_ROOT / "packages" / "js" / "dist" / "index.js"
JS_SOURCE_DIR: Final[Path] = REPO_ROOT / "packages" / "js" / "src"

#: The two measurement points of FORMAT.md section 5.3, in the order a failure
#: should be read in: a margin-point failure and an output-point failure have
#: completely different causes, so the point is named in every report line.
MEASUREMENT_POINTS: Final[tuple[str, ...]] = ("margin", "output")

#: The label written over `objective` on an already-loaded predictor, on both
#: sides, to confirm behaviourally that no prediction path reads that field
#: (FORMAT.md section 4, D028). It is deliberately not a member of the
#: enumerated set: the point is that nothing consults it.
OBJECTIVE_OVERWRITE: Final[str] = "not:an:objective"

#: Values whose survival across the language boundary is the whole reason the
#: transport is bit patterns rather than JSON numbers. Every one of them is
#: reachable in this project: `-0.0` is the logistic intercept at
#: `base_score = 0.5`, `+inf` is a genuine Cox output above margin ~88.72, and
#: `0x0020bd47` is the measured logistic clamp floor, a subnormal that a
#: reduced-precision encoding would flatten to zero.
TRANSPORT_PROBE: Final[tuple[str, ...]] = (
    "0x00000000",  # +0.0
    "0x80000000",  # -0.0
    "0x7f800000",  # +inf
    "0xff800000",  # -inf
    "0x7fc00000",  # NaN
    "0x00000001",  # smallest positive subnormal
    "0x80000001",  # smallest negative subnormal
    "0x0020bd47",  # the logistic clamp floor
    "0x3f800000",  # 1.0
    "0x7f7fffff",  # largest finite float32
    "0x3f11d541",  # the worked example's output
)

_HEX32_RE: Final[re.Pattern[str]] = re.compile(r"^0x[0-9a-f]{8}$")
_HEX64_RE: Final[re.Pattern[str]] = re.compile(r"^0x[0-9a-f]{16}$")

#: Row-encoding sentinels, from the corpus's own convention (D044 addendum):
#: `null` is the missing value and becomes NaN; standard JSON has no literal for
#: an infinity, so the refusal fixture spells them as strings. Both sides
#: decode them identically, and the input digest proves they did.
_MISSING_VALUE_ENCODING: Final[None] = None
_INFINITY_ENCODINGS: Final[dict[str, float]] = {
    "inf": float("inf"),
    "-inf": float("-inf"),
}


# ---------------------------------------------------------------------------
# Bit patterns
# ---------------------------------------------------------------------------


def bits32_of(value: Any) -> str:
    """Render one float32 as its uint32 bit pattern, `"0x3f800000"`.

    Bit patterns handle `+inf`, `-inf`, `NaN`, and `-0.0` with no special case
    at all, which is what makes the correct comparison the only convenient one.
    """
    return f"0x{int(np.float32(value).view(np.uint32)):08x}"


def bits64_of(value: Any) -> str:
    """Render one float64 as its uint64 bit pattern, 16 hex digits.

    Used only for the input digest: a prediction input is a float64 on both
    sides (both walks narrow at the comparison, and a pre-narrowed row would
    make the sample-side cast unobservable), so proving the two sides saw the
    same input is a float64-width question.
    """
    return f"0x{int(np.float64(value).view(np.uint64)):016x}"


def float32_from_bits(token: str) -> np.float32:
    """Parse a uint32 hex bit pattern back to the float32 it names."""
    if not _HEX32_RE.match(token):
        raise ValueError(f"not a uint32 hex bit pattern: {token!r}")
    return np.uint32(int(token, 16)).view(np.float32)[()]


# ---------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------


def corpus_fixture_paths() -> tuple[Path, ...]:
    """Every fixture file, ordinary then adversarial, in sorted order.

    Listing is deliberately non-recursive per directory:
    `fixtures/corpus/reference/` holds the `mpmath` reference table used by the
    transform's accuracy gate, not fixtures, and walking it would drag a
    different gate's data into this one.
    """
    if not CORPUS_DIR.is_dir():
        raise RuntimeError(f"fixture corpus directory is absent: {CORPUS_DIR}")
    paths = tuple(sorted(CORPUS_DIR.glob("*.json"))) + tuple(
        sorted(ADVERSARIAL_DIR.glob("*.json"))
    )
    if not paths:
        raise RuntimeError(f"fixture corpus is empty: {CORPUS_DIR}")
    return paths


def fixture_display_name(path: Path) -> str:
    """`"single_row_model"`, or `"adversarial/equality_boundary_routing"`."""
    if path.parent == ADVERSARIAL_DIR:
        return f"adversarial/{path.stem}"
    return path.stem


def _row_to_input(feature_names: list[str], row: list[Any]) -> dict[str, float]:
    """Decode one fixture row into a feature mapping, per the corpus encoding."""
    if len(row) != len(feature_names):
        raise RuntimeError(
            f"row has {len(row)} values against {len(feature_names)} feature names"
        )
    mapping: dict[str, float] = {}
    for name, value in zip(feature_names, row, strict=True):
        if value is _MISSING_VALUE_ENCODING:
            mapping[name] = float("nan")
        elif isinstance(value, str):
            if value not in _INFINITY_ENCODINGS:
                raise RuntimeError(f"unrecognized row encoding: {value!r}")
            mapping[name] = _INFINITY_ENCODINGS[value]
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"unrecognized row encoding: {value!r}")
        else:
            mapping[name] = float(value)
    return mapping


# ---------------------------------------------------------------------------
# The Python side
# ---------------------------------------------------------------------------


def _record(
    predictor: Predictor,
    relabelled: Predictor,
    feature_names: list[str],
    row_input: dict[str, float],
) -> dict[str, Any]:
    """Compute one row's record: both measurement points, or a refusal.

    Every exception is caught and its class name recorded rather than allowed
    to abort the run. That is not leniency: a refusal is a *result* this gate
    compares between the two sides, and a harness that let one side's raise
    become a crash could not report which row disagreed. A row where the two
    points disagree about refusing is recorded too, and always fails.

    The `relabelled` predictor is the same artifact with `objective` overwritten
    after load. Its bits must equal the first predictor's, on every row.
    """
    record: dict[str, Any] = {
        "input_bits": [bits64_of(row_input[name]) for name in feature_names],
    }
    for label, source in (("", predictor), ("_relabelled", relabelled)):
        for point in MEASUREMENT_POINTS:
            try:
                value = getattr(source, point)(row_input)
            except Exception as failure:  # noqa: BLE001 -- a refusal is a result
                record[f"{point}{label}"] = None
                record[f"{point}{label}_refusal"] = type(failure).__name__
            else:
                record[f"{point}{label}"] = bits32_of(value)
                record[f"{point}{label}_refusal"] = None
    return record


def python_side(paths: tuple[Path, ...]) -> dict[str, list[dict[str, Any]]]:
    """Every row of every fixture, through the Python predictor."""
    side: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        fixture = json.loads(path.read_text(encoding="utf-8"))
        artifact = fixture["artifact"]
        predictor = Predictor.from_json(artifact)
        relabelled = Predictor.from_json(artifact)
        relabelled._objective = OBJECTIVE_OVERWRITE  # noqa: SLF001 -- the point
        feature_names = list(artifact["feature_names"])
        side[fixture_display_name(path)] = [
            _record(predictor, relabelled, feature_names, _row_to_input(feature_names, row))
            for row in fixture["rows"]
        ]
    return side


# ---------------------------------------------------------------------------
# The JavaScript side
# ---------------------------------------------------------------------------


def _require_built_bundle() -> None:
    """Refuse to measure parity against a bundle that is absent or stale.

    A stale bundle is the silent case: the harness would run, report exactly
    `0.0`, and be describing code that no longer matches its source. So the
    freshness of `dist/` against `src/` is a refusal with an instruction, not a
    warning.
    """
    if not JS_BUNDLE.is_file():
        raise RuntimeError(
            f"the built JavaScript bundle is absent: {JS_BUNDLE}\n"
            "Build it first: npm --prefix packages/js run build"
        )
    sources = sorted(JS_SOURCE_DIR.rglob("*.ts"))
    if not sources:
        raise RuntimeError(f"no TypeScript sources found under {JS_SOURCE_DIR}")
    newest_source = max(path.stat().st_mtime for path in sources)
    bundle_built = JS_BUNDLE.stat().st_mtime
    if newest_source > bundle_built:
        raise RuntimeError(
            f"the built bundle {JS_BUNDLE.name} is older than "
            f"{JS_SOURCE_DIR.relative_to(REPO_ROOT)}; a parity number measured "
            "against it would describe code that is not the source.\n"
            "Rebuild it: npm --prefix packages/js run build"
        )


def javascript_side(paths: tuple[Path, ...]) -> dict[str, Any]:
    """Run the emitter as a subprocess and return its parsed response.

    One subprocess for the whole corpus. The request carries the fixture paths,
    so both sides read the same files in the same order and a fixture that one
    side silently failed to see becomes a structural mismatch rather than a
    smaller row count nobody notices.
    """
    node = shutil.which("node")
    if node is None:
        raise RuntimeError(
            "node was not found on PATH. This gate compares two "
            "implementations; it cannot be satisfied by running one of them."
        )
    _require_built_bundle()
    if not EMITTER_PATH.is_file():
        raise RuntimeError(f"the emitter is absent: {EMITTER_PATH}")

    request = {
        "fixtures": [
            {"name": fixture_display_name(path), "path": str(path)} for path in paths
        ],
        "transport_probe": list(TRANSPORT_PROBE),
        "objective_overwrite": OBJECTIVE_OVERWRITE,
    }
    completed = subprocess.run(
        [node, str(EMITTER_PATH)],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"the JavaScript emitter exited {completed.returncode}\n"
            f"--- stderr ---\n{completed.stderr}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as failure:
        raise RuntimeError(
            f"the JavaScript emitter did not produce JSON: {failure}\n"
            f"--- stdout ---\n{completed.stdout[:2000]}\n"
            f"--- stderr ---\n{completed.stderr[:2000]}"
        ) from failure


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Divergence:
    """One disagreement, naming the measurement point that produced it.

    A report that says only "parity failed" is not usable: the margin point and
    the output point have different causes -- a missing narrowing site in the
    walk against a transform evaluated in the wrong space -- so the point,
    fixture, row, and both bit patterns are all carried.
    """

    point: str
    fixture: str
    row: int
    python: str
    javascript: str

    def __str__(self) -> str:
        return (
            f"{self.point} point, {self.fixture}[{self.row}]: "
            f"Python {self.python}, JavaScript {self.javascript}"
        )


@dataclass(frozen=True)
class FixtureTally:
    """Per-fixture counts. An aggregate alone hides which fixture moved."""

    fixture: str
    rows: int
    value_rows: int
    refused_rows: int
    margin_mismatches: int
    output_mismatches: int
    other_disagreements: int


@dataclass
class ParityReport:
    """The result of one full-corpus comparison.

    Counts and first divergences only. **No mean is computed anywhere**: a mean
    of a bit-pattern comparison is meaningless, and a mean of anything else
    hides one catastrophic row in a large corpus.
    """

    rows_compared: int = 0
    value_rows: int = 0
    refused_rows: int = 0
    margin_mismatches: list[Divergence] = field(default_factory=list)
    output_mismatches: list[Divergence] = field(default_factory=list)
    refusal_disagreements: list[Divergence] = field(default_factory=list)
    input_disagreements: list[Divergence] = field(default_factory=list)
    objective_findings: list[Divergence] = field(default_factory=list)
    structural: list[str] = field(default_factory=list)
    tallies: list[FixtureTally] = field(default_factory=list)
    transport: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    #: Both sides' raw records, retained rather than discarded. They are what
    #: lets the suite re-run :func:`compare_sides` against a deliberately
    #: perturbed copy and confirm the comparison reports the perturbation at the
    #: right measurement point, on the right fixture and row -- without paying
    #: for a second subprocess and a second walk of the corpus.
    python_records: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    javascript_records: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    @property
    def failures(self) -> list[Divergence]:
        """Every disagreement, in the order a reader should triage them."""
        return [
            *self.margin_mismatches,
            *self.output_mismatches,
            *self.refusal_disagreements,
            *self.input_disagreements,
            *self.objective_findings,
        ]

    @property
    def clean(self) -> bool:
        return not self.failures and not self.structural

    def first(self, point: str) -> Divergence | None:
        for divergence in self.failures:
            if divergence.point == point:
                return divergence
        return None


#: Every value key a record must carry, at both measurement points, for both the
#: artifact as loaded and the artifact with `objective` overwritten.
_VALUE_KEYS: Final[tuple[str, ...]] = tuple(
    f"{point}{suffix}" for suffix in ("", "_relabelled") for point in MEASUREMENT_POINTS
)


def _validate_record(record: Any, side: str, fixture: str, row: int) -> list[str]:
    """Refuse a record whose tokens are not bit patterns, rather than comparing it.

    This is not defensive decoration. Two hex strings that are not both valid
    uint32 patterns compare *unequal* as strings, so a malformed token would be
    reported as an ordinary parity mismatch and read as a defect in the
    predictor -- and, worse, a value that a broken emitter rendered
    identically-wrong on both sides would compare **equal** and be credited as
    parity. Measured: a one-ULP injection written with a JavaScript `&
    0xffffffff` emitted `0x-40f572f5`, which this comparison accepted and
    printed. The tokens are therefore checked before any of them is compared.
    """
    notes: list[str] = []
    where = f"{side} on {fixture}[{row}]"
    if not isinstance(record, dict):
        return [f"{where}: record is {type(record).__name__}, not an object"]

    for key in _VALUE_KEYS:
        if key not in record or f"{key}_refusal" not in record:
            notes.append(f"{where}: record has no {key!r}/{key + '_refusal'!r} pair")
            continue
        token = record[key]
        refusal = record[f"{key}_refusal"]
        if token is None:
            if not isinstance(refusal, str) or not refusal:
                notes.append(
                    f"{where}: {key} is absent with no refusal name, so the row "
                    "was neither measured nor refused"
                )
        elif not isinstance(token, str) or not _HEX32_RE.match(token):
            notes.append(f"{where}: {key} is {token!r}, not a uint32 hex bit pattern")
        elif refusal is not None:
            notes.append(f"{where}: {key} carries both a value and a refusal {refusal!r}")

    input_bits = record.get("input_bits")
    if not isinstance(input_bits, list) or not input_bits:
        notes.append(f"{where}: input_bits is {input_bits!r}, not a non-empty array")
    else:
        for token in input_bits:
            if not isinstance(token, str) or not _HEX64_RE.match(token):
                notes.append(
                    f"{where}: input_bits carries {token!r}, not a uint64 hex bit pattern"
                )
    return notes


def _status(record: dict[str, Any], side: str, fixture: str, row: int) -> str:
    """Classify a record as a value or a refusal, refusing to guess.

    A record that produced a margin but refused the output -- or the reverse --
    is neither, and is reported as its own inconsistency rather than being
    filed under whichever branch happened to come first.
    """
    refusals = tuple(record[f"{point}_refusal"] for point in MEASUREMENT_POINTS)
    values = tuple(record[point] for point in MEASUREMENT_POINTS)
    if all(refusal is None for refusal in refusals) and all(
        value is not None for value in values
    ):
        return "value"
    if all(refusal is not None for refusal in refusals) and all(
        value is None for value in values
    ):
        return f"refused:{refusals[0]}/{refusals[1]}"
    raise RuntimeError(
        f"{side} produced a record that neither refuses nor returns at both "
        f"points, on {fixture}[{row}]: {record!r}"
    )


def compare_sides(
    python_records: dict[str, list[dict[str, Any]]],
    javascript_records: dict[str, list[dict[str, Any]]],
) -> ParityReport:
    """Compare two already-collected sides. Pure, so it can be self-tested.

    Kept free of subprocesses and file reads so that the suite can hand it a
    deliberately perturbed side and confirm that this comparison actually
    reports the perturbation, at the right measurement point, on the right
    fixture and row. A parity harness that has never been observed to fail is
    not evidence of anything.
    """
    report = ParityReport()

    python_names = list(python_records)
    javascript_names = list(javascript_records)
    if python_names != javascript_names:
        report.structural.append(
            f"the two sides covered different fixtures: only in Python "
            f"{sorted(set(python_names) - set(javascript_names))}, only in "
            f"JavaScript {sorted(set(javascript_names) - set(python_names))}"
        )

    for fixture in python_names:
        left = python_records[fixture]
        right = javascript_records.get(fixture)
        if right is None:
            continue
        if len(left) != len(right):
            report.structural.append(
                f"{fixture}: Python produced {len(left)} rows, JavaScript {len(right)}"
            )
            continue

        margin_count = 0
        output_count = 0
        other_count = 0
        value_rows = 0
        refused_rows = 0

        malformed = [
            note
            for row, (python_row, javascript_row) in enumerate(zip(left, right, strict=True))
            for note in (
                *_validate_record(python_row, "Python", fixture, row),
                *_validate_record(javascript_row, "JavaScript", fixture, row),
            )
        ]
        if malformed:
            report.structural.extend(malformed)
            continue

        for row, (python_row, javascript_row) in enumerate(zip(left, right, strict=True)):
            report.rows_compared += 1

            if python_row["input_bits"] != javascript_row["input_bits"]:
                report.input_disagreements.append(
                    Divergence(
                        "input",
                        fixture,
                        row,
                        ",".join(python_row["input_bits"]),
                        ",".join(javascript_row["input_bits"]),
                    )
                )
                other_count += 1

            for label, side_name in (("Python", "python"), ("JavaScript", "javascript")):
                source = python_row if side_name == "python" else javascript_row
                for point in MEASUREMENT_POINTS:
                    own = (source[point], source[f"{point}_refusal"])
                    relabelled = (
                        source[f"{point}_relabelled"],
                        source[f"{point}_relabelled_refusal"],
                    )
                    if own != relabelled:
                        report.objective_findings.append(
                            Divergence(
                                f"objective/{label}",
                                fixture,
                                row,
                                f"{point}={own[0]!s} refusal={own[1]!s}",
                                f"{point}={relabelled[0]!s} refusal={relabelled[1]!s}",
                            )
                        )
                        other_count += 1

            python_status = _status(python_row, "Python", fixture, row)
            javascript_status = _status(javascript_row, "JavaScript", fixture, row)
            if python_status != javascript_status:
                report.refusal_disagreements.append(
                    Divergence("refusal", fixture, row, python_status, javascript_status)
                )
                other_count += 1
                continue

            if python_status == "value":
                value_rows += 1
                report.value_rows += 1
                for point, bucket in (
                    ("margin", report.margin_mismatches),
                    ("output", report.output_mismatches),
                ):
                    if python_row[point] != javascript_row[point]:
                        bucket.append(
                            Divergence(
                                point,
                                fixture,
                                row,
                                python_row[point],
                                javascript_row[point],
                            )
                        )
                        if point == "margin":
                            margin_count += 1
                        else:
                            output_count += 1
            else:
                refused_rows += 1
                report.refused_rows += 1

        report.tallies.append(
            FixtureTally(
                fixture=fixture,
                rows=len(left),
                value_rows=value_rows,
                refused_rows=refused_rows,
                margin_mismatches=margin_count,
                output_mismatches=output_count,
                other_disagreements=other_count,
            )
        )

    return report


def _check_transport(report: ParityReport, response: dict[str, Any]) -> None:
    """Verify the boundary encoding lost nothing, and that a naive one would.

    Two halves, and the second is what makes the first meaningful. A transport
    that quietly normalizes a value would let this harness report perfect parity
    while hiding a real difference, so it is not enough to observe that the
    chosen encoding survives: the encoding that would *not* have survived is
    exercised on the same values, in the same round trip, and its losses are
    required to appear.
    """
    probes = response["transport"]
    if [probe["sent"] for probe in probes] != list(TRANSPORT_PROBE):
        report.structural.append("the transport probe came back with different values")
        return

    lossless = True
    lossy_numeric: list[str] = []
    for probe in probes:
        sent = probe["sent"]
        if probe["echoed"] != sent:
            report.structural.append(
                f"transport lost bits: sent {sent}, echoed {probe['echoed']}"
            )
            lossless = False
        # The same value, re-encoded by the harness from the float32 it names,
        # must reproduce the pattern too -- otherwise the agreement above could
        # be two sides sharing one decoding defect.
        if bits32_of(float32_from_bits(sent)) != sent:
            report.structural.append(f"Python could not round-trip {sent}")
            lossless = False
        naive = probe["naive_number"]
        if naive is None or bits32_of(naive) != sent:
            lossy_numeric.append(sent)

    report.transport = {
        "probes": probes,
        "bit_pattern_encoding_lossless": lossless,
        "values_a_json_number_transport_would_have_lost": lossy_numeric,
    }

    for required in ("0x80000000", "0x7f800000", "0xff800000", "0x7fc00000"):
        if required not in lossy_numeric:
            report.structural.append(
                f"a JSON number transport did not lose {required}, so this "
                "harness cannot demonstrate why the encoding is bit patterns"
            )


def run_parity() -> ParityReport:
    """Collect both sides over the whole corpus and compare them."""
    paths = corpus_fixture_paths()
    response = javascript_side(paths)
    python_records = python_side(paths)
    javascript_records = response["fixtures"]
    report = compare_sides(python_records, javascript_records)
    report.python_records = python_records
    report.javascript_records = javascript_records
    _check_transport(report, response)
    report.environment = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "node": response["node_version"],
        "module_specifier": response["module_specifier"],
        "module_url": response["module_url"],
        "fixtures": len(paths),
    }
    if "/dist/" not in report.environment["module_url"]:
        report.structural.append(
            "the JavaScript side did not run against the built bundle: "
            f"{report.environment['module_url']}"
        )
    return report


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render(report: ParityReport) -> str:
    """A report a reader can act on: counts, per fixture, first divergence."""
    lines: list[str] = []
    lines.append("cross-language parity, Python vs JavaScript")
    lines.append("=" * 78)
    for key in ("python", "numpy", "node", "fixtures", "module_specifier", "module_url"):
        if key in report.environment:
            lines.append(f"{key:<18} {report.environment[key]}")
    lines.append("")
    lines.append(
        f"{'fixture':<48}{'rows':>6}{'refused':>9}{'margin':>8}{'output':>8}"
    )
    lines.append("-" * 78)
    for tally in report.tallies:
        lines.append(
            f"{tally.fixture:<48}{tally.rows:>6}{tally.refused_rows:>9}"
            f"{tally.margin_mismatches:>8}{tally.output_mismatches:>8}"
        )
    lines.append("-" * 78)
    lines.append(
        f"{'TOTAL':<48}{report.rows_compared:>6}{report.refused_rows:>9}"
        f"{len(report.margin_mismatches):>8}{len(report.output_mismatches):>8}"
    )
    lines.append("")
    lines.append(f"rows compared                 {report.rows_compared}")
    lines.append(f"  value rows                  {report.value_rows}")
    lines.append(f"  rows refused by both sides  {report.refused_rows}")
    lines.append(f"margin-point mismatches       {len(report.margin_mismatches)}")
    lines.append(f"output-point mismatches       {len(report.output_mismatches)}")
    lines.append(f"refusal disagreements         {len(report.refusal_disagreements)}")
    lines.append(f"input-bit disagreements       {len(report.input_disagreements)}")
    lines.append(f"objective-branch findings     {len(report.objective_findings)}")
    transport = report.transport
    if transport:
        lines.append(
            f"bit-pattern transport         "
            f"{'lossless' if transport['bit_pattern_encoding_lossless'] else 'LOSSY'}"
            f" over {len(transport['probes'])} probe values"
        )
        lines.append(
            "  a JSON number transport would have lost "
            f"{len(transport['values_a_json_number_transport_would_have_lost'])}"
            " of them: "
            + ", ".join(transport["values_a_json_number_transport_would_have_lost"])
        )
    lines.append("")
    if report.clean:
        lines.append("PARITY: 0.0 at both measurement points, on bit patterns.")
    else:
        lines.append("PARITY FAILED. First divergence at each point:")
        for point in ("margin", "output", "refusal", "input"):
            first = report.first(point)
            if first is not None:
                lines.append(f"  {first}")
        for divergence in report.objective_findings[:1]:
            lines.append(f"  {divergence}")
        for note in report.structural:
            lines.append(f"  structural: {note}")
        lines.append("")
        lines.append(f"total disagreements: {len(report.failures)}")
        for divergence in report.failures[:20]:
            lines.append(f"  {divergence}")
        if len(report.failures) > 20:
            lines.append(f"  ... and {len(report.failures) - 20} more")
    return "\n".join(lines)


def main() -> int:
    report = run_parity()
    print(render(report))
    return 0 if report.clean else 1


if __name__ == "__main__":
    sys.exit(main())
