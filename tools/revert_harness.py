#!/usr/bin/env python3
"""Apply each recorded revert and confirm the expected tests actually go red.

`CLAUDE.md` requires that every protection be verified by reverting it **in
isolation** and confirming specific tests fail. Until this file existed that was a
practice: something done once, by hand, at the time the protection was added, and
never again. A practice cannot tell you when it has stopped working.

The distinction matters because of how these protections decay. Nobody deletes a
narrowing site. What happens is that a *different* change makes the site
redundant -- narrowing after every addition partially absorbs leaf narrowing, so
one site can start covering for another -- and from then on the suite passes with
the protection removed. Nothing goes red. Nothing announces it. The protection is
still in the source, still commented, and no longer pinned by anything.

**So a revert that turns nothing red is a finding, not a pass.** That inverts the
usual polarity and is the reason this exists: the harness fails when the suite
*succeeds* under a revert.

What it does per entry: apply one textual substitution, run one narrowly-selected
set of tests, require a specific test to fail, restore the file, and confirm the
restoration is byte-exact. The tree is left as it was found, including on
`KeyboardInterrupt` -- every mutation is inside `try/finally`, and the harness
refuses to start if any target file already has uncommitted changes, because the
restore path writes back what it read and would otherwise be indistinguishable
from a revert of the user's own work.

Run:
    uv run python tools/revert_harness.py                 # Python-side reverts
    uv run python tools/revert_harness.py --list
    uv run python tools/revert_harness.py --only float32-sample-side
    uv run python tools/revert_harness.py --include-js    # rebuilds dist/, slow
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Revert:
    """One protection, the edit that removes it, and what must notice.

    `expect_failing` names a test that must fail, not merely "the suite fails".
    A revert verified only by "something went red" is satisfied by an unrelated
    flake, and would keep passing after the test that genuinely pinned it was
    renamed away.
    """

    name: str
    protects: str
    decision: str
    file: str
    old: str
    new: str
    pytest_args: tuple[str, ...] = ()
    node_test: str | None = None
    expect_failing: tuple[str, ...] = ()
    note: str = ""


REVERTS: tuple[Revert, ...] = (
    Revert(
        name="float32-sample-side",
        protects="the cast on the *sample* side of the split comparison",
        decision="D004",
        file="packages/python/src/xgboost_bridge/trees.py",
        old="elif np.float32(feature_value) < np.float32(threshold):",
        new="elif feature_value < np.float32(threshold):",
        pytest_args=("packages/python/tests/test_predict.py", "fixtures/tests"),
        expect_failing=("test_",),
        note=(
            "The highest-value invariant in the repository: casting only the "
            "threshold cost 6.6 percentage points of probability on a real row. "
            "Note it is *only* detectable for a strongly-typed float64 sample -- "
            "NEP 50 narrows a bare Python float at the comparison anyway."
        ),
    ),
    Revert(
        name="float32-threshold-side",
        protects="the cast on the *threshold* side of the split comparison",
        decision="D004",
        file="packages/python/src/xgboost_bridge/trees.py",
        old="elif np.float32(feature_value) < np.float32(threshold):",
        new="elif np.float32(feature_value) < threshold:",
        pytest_args=("packages/python/tests/test_predict.py", "fixtures/tests"),
        expect_failing=("test_",),
        note=(
            "Tested separately from the sample side rather than as a pair. "
            "CLAUDE.md is explicit that reverting both at once pins neither, "
            "because each can absorb the other's failure."
        ),
    ),
    Revert(
        name="parse-time-narrowing",
        protects="narrowing node_values to float32 at parse time",
        decision="D004",
        file="packages/python/src/xgboost_bridge/predict.py",
        old="values = np.asarray(raw, dtype=np.float32)",
        new="values = np.asarray(raw, dtype=np.float64)",
        pytest_args=("packages/python/tests/test_predict.py",),
        expect_failing=("test_",),
        note=(
            "The structural half of the invariant. `JSON.parse` and `json.load` "
            "both return float64 unconditionally, and on 104/104 measured "
            "thresholds that float64 is a different number from the engine's "
            "float32. Narrowing here makes it a property of the data structure "
            "rather than a discipline every future reader has to remember."
        ),
    ),
    Revert(
        name="accumulator-narrowing",
        protects="narrowing the accumulator after every single addition",
        decision="D004",
        file="packages/python/src/xgboost_bridge/trees.py",
        old="accumulator = np.float32(accumulator + np.float32(node_values[node]))",
        new="accumulator = accumulator + np.float32(node_values[node])",
        pytest_args=("packages/python/tests/test_predict.py",),
        expect_failing=("test_",),
        note=(
            "A float64 sum narrowed once at the end scored 318-2541/5000 "
            "bit-exact. This is the deviation that looks most harmless."
        ),
    ),
    Revert(
        name="in-degree-python",
        protects="refusing a DAG that is structurally not a tree",
        decision="D058",
        file="packages/python/src/xgboost_bridge/trees.py",
        old="""            parents[child] += 1
            if parents[child] > 1:""",
        new="""            parents[child] += 1
            if False:""",
        pytest_args=("packages/python/tests/test_validator_gaps.py",),
        expect_failing=("test_",),
        note=(
            "Two parents pointing at one child was accepted by *both* readers, "
            "which agreed on the margin -- so cross-language parity could never "
            "have caught it. In `extract_trees` the consequence was worse than a "
            "wrong number: export's path enumeration made `export_model` hang, "
            "34 nodes in 14.18 s and 60 nodes never returning."
        ),
    ),
    Revert(
        name="in-degree-javascript",
        protects="refusing a DAG in the JavaScript reader",
        decision="D058",
        file="packages/js/src/artifact.ts",
        old="      if ((parents[child] as number) < 2) {",
        new="      if (true) {",
        node_test="npm --prefix packages/js test",
        expect_failing=("not ok", "failing"),
        note=(
            "Rebuilds `dist/`, because the JavaScript suite imports from `dist/` "
            "and never from `src/`. Slow, and off by default for that reason."
        ),
    ),
)


def _run(command: list[str], cwd: Path = ROOT) -> tuple[int, str]:
    finished = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, timeout=1800
    )
    return finished.returncode, finished.stdout + finished.stderr


def _refuse_dirty(reverts: tuple[Revert, ...]) -> list[str]:
    """Uncommitted changes in a target file make restore indistinguishable from loss.

    The harness writes a file back from a string it read into memory. If the file
    was already modified, that is fine -- the same bytes return -- but a crash
    between the two writes would leave the user's work replaced by a deliberate
    defect. Refusing up front is cheaper than being careful.
    """
    code, out = _run(["git", "status", "--porcelain"])
    if code != 0:
        return []
    dirty = {line[3:].strip() for line in out.splitlines() if line.strip()}
    return sorted({revert.file for revert in reverts if revert.file in dirty})


def _apply(revert: Revert) -> str:
    path = ROOT / revert.file
    original = path.read_text()
    count = original.count(revert.old)
    if count != 1:
        raise SystemExit(
            f"FAIL [{revert.name}]: the target text occurs {count} times in "
            f"{revert.file}, expected exactly 1.\n"
            f"The source moved and this revert no longer describes it. That is a "
            f"finding: the protection may still be there, but this harness is no "
            f"longer testing it.\n  looking for: {revert.old!r}"
        )
    path.write_text(original.replace(revert.old, revert.new, 1))
    return original


def _check(revert: Revert) -> tuple[bool, str]:
    """Run the selected tests and report whether the revert was caught."""
    if revert.node_test:
        code, out = _run(revert.node_test.split())
    else:
        code, out = _run(
            [sys.executable, "-m", "pytest", "-x", "-q", *revert.pytest_args]
        )
    caught = code != 0 and any(marker in out for marker in revert.expect_failing)
    return caught, out


def _first_failure(output: str) -> str:
    for line in output.splitlines():
        if line.startswith("FAILED") or line.startswith("not ok"):
            return line.strip()[:110]
    for line in output.splitlines():
        if "assert" in line.lower() and "Error" in line:
            return line.strip()[:110]
    return "(failed; no FAILED line parsed)"


def verify(revert: Revert) -> bool:
    print(f"\n[{revert.name}] {revert.protects}  ({revert.decision})")
    original = _apply(revert)
    try:
        caught, output = _check(revert)
    finally:
        path = ROOT / revert.file
        path.write_text(original)
        # Byte-exact restoration, asserted rather than assumed. A harness that
        # leaves a deliberate defect in the tree is worse than no harness.
        assert path.read_text() == original, f"RESTORE FAILED for {revert.file}"

    if caught:
        print(f"  red, as required: {_first_failure(output)}")
        return True

    print(
        "  *** GREEN UNDER REVERT -- this protection is no longer pinned. ***\n"
        "  The suite passed with the protection removed. Either a test that used "
        "to cover it\n  was renamed, narrowed or deleted, or another change now "
        "absorbs its failure.\n  Do not treat this as a passing harness run."
    )
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="describe the reverts and exit")
    parser.add_argument("--only", metavar="NAME", help="run a single revert by name")
    parser.add_argument(
        "--include-js",
        action="store_true",
        help="include reverts that rebuild dist/ and run the Node suite",
    )
    arguments = parser.parse_args()

    selected = REVERTS
    if not arguments.include_js:
        selected = tuple(r for r in selected if r.node_test is None)
    if arguments.only:
        selected = tuple(r for r in REVERTS if r.name == arguments.only)
        if not selected:
            print(f"no revert named {arguments.only!r}", file=sys.stderr)
            return 2

    if arguments.list:
        for revert in REVERTS:
            print(f"{revert.name:24} {revert.decision}  {revert.protects}")
            if revert.note:
                print(f"{'':24} {revert.note}")
        return 0

    dirty = _refuse_dirty(selected)
    if dirty:
        print(
            "refusing to run: these target files have uncommitted changes, and "
            "this harness rewrites them:\n  " + "\n  ".join(dirty),
            file=sys.stderr,
        )
        return 2

    results = {revert.name: verify(revert) for revert in selected}
    unpinned = [name for name, pinned in results.items() if not pinned]

    print(f"\n{'-' * 70}")
    print(f"{len(results) - len(unpinned)}/{len(results)} protections confirmed pinned")
    if unpinned:
        print(f"NOT PINNED: {', '.join(unpinned)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
