#!/usr/bin/env python3
"""Generate `docs/DECISION_INDEX.md` -- code to the decisions that govern it.

`docs/DECISIONS.md` answers "why is this the way it is" for someone who already
knows which decision to read. It does not answer the question a maintainer
actually arrives with, which is "I am about to edit `walk_margin`; what do I need
to know first?" With 63 entries across 2000 lines, the honest answer today is
"read all of it", and nobody does. This file inverts the direction: keyed by
file, then by function, pointing outward at the entries.

`docs/DECISION_COVERAGE.md` is a third thing again -- decisions to their pinning
tests -- and does not serve this purpose either.

**Generated, not written.** The join is:

* the **symbol** side comes from the source, by parsing it: `ast` for Python,
  the TypeScript declaration grammar for `.ts`;
* the **decision** side comes from `DECISIONS.md`, by reading the backticked
  identifiers each entry already mentions.

Neither side is a hand-maintained list, which is the property that matters: a
hand-written index is correct on the day it is written and silently wrong
afterwards, and being silently wrong about where the governing constraint lives
is the same failure this project exists to prevent, one level up. A new entry
mentioning `walk_margin` lands in the index on the next generation, and the
pinning test fails until it is regenerated.

The matcher is deliberately conservative -- an exact identifier match, plus the
`module.function` form the entries use. It under-reports rather than guessing:
a missing row sends someone to `DECISIONS.md`, which is where they were going
anyway, while a wrong row sends them somewhere confidently useless.

Run:
    uv run python tools/build_decision_index.py          # write the file
    uv run python tools/build_decision_index.py --check   # exit 1 if stale
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECISIONS = ROOT / "docs" / "DECISIONS.md"
INDEX = ROOT / "docs" / "DECISION_INDEX.md"

# The files a maintainer edits when they change behaviour. Fixture generators and
# tests are deliberately out: an entry mentioning a test is describing where a
# constraint is *pinned*, which DECISION_COVERAGE.md already maps.
SOURCE_GLOBS = (
    "packages/python/src/xgboost_bridge/*.py",
    "packages/js/src/*.ts",
)

# Symbols too generic to attribute. `predict` appears in prose about XGBoost's own
# `predict()` far more often than about ours, and a row pointing at every entry is
# indistinguishable from no row at all.
AMBIGUOUS = frozenset({"predict", "main", "value", "index", "load", "get", "check"})


@dataclass(frozen=True)
class Symbol:
    """A definition a maintainer could put their cursor in.

    `end_line` exists so a `D0nn` reference written in a comment *inside* a
    function can be attributed to that function. Python gets exact ranges from
    `ast`; TypeScript gets "until the next declaration", which is coarser and
    stated as such rather than presented as precise.
    """

    name: str
    file: str
    line: int
    kind: str
    end_line: int = 0


@dataclass
class Decision:
    """One `## Dnnn` entry, with the identifiers its body mentions."""

    id: str
    title: str
    mentions: set[str] = field(default_factory=set)


def collect_python_symbols(path: Path) -> list[Symbol]:
    """Top-level functions and classes, plus methods, from a real parse.

    `ast` rather than a regex, because a regex over Python cannot tell a
    definition from the same words inside a docstring -- and this repository's
    docstrings quote its own function names constantly.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    relative = str(path.relative_to(ROOT))
    symbols: list[Symbol] = []

    for node in tree.body:
        end = getattr(node, "end_lineno", node.lineno) or node.lineno
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(Symbol(node.name, relative, node.lineno, "function", end))
        elif isinstance(node, ast.ClassDef):
            symbols.append(Symbol(node.name, relative, node.lineno, "class", end))
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    child_end = getattr(child, "end_lineno", child.lineno) or child.lineno
                    symbols.append(
                        Symbol(
                            f"{node.name}.{child.name}",
                            relative,
                            child.lineno,
                            "method",
                            child_end,
                        )
                    )
    return symbols


# TypeScript declarations, anchored at the start of a line so a mention inside a
# comment or a string cannot match. Covers the four shapes `packages/js/src` uses;
# a shape it does not cover is absent from the index rather than misattributed.
_TS_PATTERNS = (
    (re.compile(r"^export function (\w+)"), "function"),
    (re.compile(r"^export class (\w+)"), "class"),
    (re.compile(r"^export (?:const|type|interface|enum) (\w+)"), "declaration"),
    (re.compile(r"^  (?:public |private |readonly )?(\w+)\("), "method"),
)


def collect_typescript_symbols(path: Path) -> list[Symbol]:
    relative = str(path.relative_to(ROOT))
    symbols: list[Symbol] = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        for pattern, kind in _TS_PATTERNS:
            match = pattern.match(line)
            if match:
                symbols.append(Symbol(match.group(1), relative, number, kind))
                break

    # Each declaration runs until the next one begins. Methods nest inside a class
    # and would otherwise swallow their siblings' references, so the ranges are
    # assigned in source order over the flat list, which is what "nearest
    # preceding declaration" means in practice.
    total = len(path.read_text().splitlines())
    ranged: list[Symbol] = []
    for position, symbol in enumerate(symbols):
        following = symbols[position + 1].line - 1 if position + 1 < len(symbols) else total
        ranged.append(
            Symbol(symbol.name, symbol.file, symbol.line, symbol.kind, max(following, symbol.line))
        )
    return ranged


def collect_symbols() -> list[Symbol]:
    symbols: list[Symbol] = []
    for glob in SOURCE_GLOBS:
        for path in sorted(ROOT.glob(glob)):
            if path.suffix == ".py":
                symbols.append(Symbol(path.name, str(path.relative_to(ROOT)), 0, "module"))
                symbols.extend(collect_python_symbols(path))
            else:
                symbols.append(Symbol(path.name, str(path.relative_to(ROOT)), 0, "module"))
                symbols.extend(collect_typescript_symbols(path))
    return symbols


_D_REFERENCE = re.compile(r"\bD(\d{3})\b")


def collect_code_citations() -> dict[tuple[str, int], set[str]]:
    """`D0nn` references the source already carries, keyed by (file, line).

    The second direction of the join, and the one that recovers the entries a
    maintainer most needs. The mention-based direction only finds entries that
    name a symbol, and the oldest entries state a rule -- "cast both sides" --
    without naming the function that implements it, because the function did not
    exist yet. The code does cite them: 135 references across 33 distinct entries.

    Neither direction is a hand-kept list, which is the point. A comment citing
    `D004` puts a row in this index automatically, and a citation of an entry that
    does not exist fails the test rather than rendering a dead link.
    """
    citations: dict[tuple[str, int], set[str]] = {}
    for glob in SOURCE_GLOBS:
        for path in sorted(ROOT.glob(glob)):
            relative = str(path.relative_to(ROOT))
            for number, line in enumerate(path.read_text().splitlines(), start=1):
                for found in _D_REFERENCE.findall(line):
                    citations.setdefault((relative, number), set()).add(f"D{found}")
    return citations


_BACKTICKED = re.compile(r"`([^`\n]+)`")
_ENTRY = re.compile(r"^## (D\d{3})\s*[—-]\s*(.+)$", re.MULTILINE)


def parse_decisions(text: str) -> list[Decision]:
    """Split into entries and harvest each one's backticked identifiers.

    A mention is taken from the entry *body*, not the title: titles are prose
    summaries and rarely name the code. Trailing `()` and a leading module
    qualifier are both stripped, because the entries use `walk_margin`,
    `walk_margin()` and `trees.walk_margin` interchangeably and they are one
    referent.
    """
    matches = list(_ENTRY.finditer(text))
    decisions: list[Decision] = []
    for position, match in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        body = text[match.end() : end]
        decision = Decision(id=match.group(1), title=match.group(2).strip())
        for token in _BACKTICKED.findall(body):
            token = token.strip().removesuffix("()")
            if not token or " " in token:
                continue
            decision.mentions.add(token)
            # `trees.extract_trees` also credits `extract_trees`; `predict.ts`
            # keeps its suffix because that is the symbol's own name.
            if "." in token and not token.endswith((".py", ".ts", ".mjs", ".md", ".json")):
                decision.mentions.add(token.rsplit(".", 1)[-1])
        decisions.append(decision)
    return decisions


def build_rows(
    symbols: list[Symbol], decisions: list[Decision]
) -> dict[str, list[tuple[Symbol, list[str]]]]:
    """Join symbols to decisions in both directions, grouped by file.

    A symbol earns a citation if an entry names it *or* if the symbol's own body
    cites the entry. Modules collect what their file cites outside any symbol, so
    a reference in a module docstring is not lost.
    """
    citations = collect_code_citations()
    by_file: dict[str, list[tuple[Symbol, list[str]]]] = {}
    claimed: dict[str, set[tuple[str, int]]] = {}

    for symbol in symbols:
        bare = symbol.name.rsplit(".", 1)[-1]
        if symbol.name in AMBIGUOUS or bare in AMBIGUOUS:
            continue

        governing = {
            decision.id
            for decision in decisions
            if symbol.name in decision.mentions or bare in decision.mentions
        }
        if symbol.kind != "module":
            for (file, line), ids in citations.items():
                if file == symbol.file and symbol.line <= line <= symbol.end_line:
                    governing |= ids
                    claimed.setdefault(symbol.file, set()).add((file, line))

        if governing:
            by_file.setdefault(symbol.file, []).append((symbol, sorted(governing)))

    # File-level rows pick up citations no symbol enclosed -- module docstrings,
    # imports, module-level constants. Attributing these to the nearest function
    # would be a guess, and a confident wrong pointer is worse than a file-level one.
    for file, rows in list(by_file.items()):
        loose = {
            identifier
            for (candidate, line), ids in citations.items()
            if candidate == file and (candidate, line) not in claimed.get(file, set())
            for identifier in ids
        }
        for position, (symbol, ids) in enumerate(rows):
            if symbol.kind == "module":
                rows[position] = (symbol, sorted(set(ids) | loose))
                break
        else:
            if loose:
                rows.append((Symbol(Path(file).name, file, 0, "module"), sorted(loose)))

    for rows in by_file.values():
        rows.sort(key=lambda pair: (pair[0].line, pair[0].name))
    return by_file


_PREAMBLE = """<!-- GENERATED by tools/build_decision_index.py. Do not edit by hand. -->
<!-- Regenerate: uv run python tools/build_decision_index.py -->

# Decision index — code to the decisions that govern it

**About to change something? Find it here first.**

`docs/DECISIONS.md` is ordered by when a decision was made, which is the wrong
order for the question a maintainer actually has. This table is keyed by file and
symbol: look up what you are about to edit, read those entries, *then* edit.

Several of these decisions exist because an earlier belief that looked sound was
measured false. If a change here seems obviously correct and an entry disagrees
with it, the entry has evidence attached and the intuition does not.

This file is **generated** — from the symbols in the source and the identifiers
the entries mention, joined mechanically. A test fails if it drifts from either
side. It under-reports by design: a missing row means no entry named that symbol,
not that no decision governs it. `docs/DECISION_COVERAGE.md` runs the other
direction, decisions to their pinning tests.

"""


def render(by_file: dict[str, list[tuple[Symbol, list[str]]]], decisions: list[Decision]) -> str:
    titles = {decision.id: decision.title for decision in decisions}
    out = [_PREAMBLE]

    for file in sorted(by_file):
        out.append(f"## `{file}`\n\n")
        out.append("| Symbol | Kind | Line | Decisions |\n|---|---|---:|---|\n")
        for symbol, governing in by_file[file]:
            if symbol.kind == "module":
                name, line = "*(whole file)*", ""
            else:
                name, line = f"`{symbol.name}`", str(symbol.line)
            ids = " ".join(f"[{d}](DECISIONS.md#{d.lower()})" for d in sorted(governing))
            out.append(f"| {name} | {symbol.kind} | {line} | {ids} |\n")
        out.append("\n")

    out.append("## Entry titles\n\n")
    out.append("| Decision | Title |\n|---|---|\n")
    cited = sorted({d for rows in by_file.values() for _, ids in rows for d in ids})
    for decision_id in cited:
        out.append(f"| [{decision_id}](DECISIONS.md#{decision_id.lower()}) | {titles[decision_id]} |\n")

    total = sum(len(rows) for rows in by_file.values())
    out.append(
        f"\n---\n\n{total} symbols across {len(by_file)} files, citing {len(cited)} of "
        f"{len(decisions)} entries. An entry absent here is one that governs the "
        f"project rather than a named symbol — release mechanics, format design, "
        f"scope. Those are not skippable; they are simply not reachable from a "
        f"cursor position.\n"
    )
    return "".join(out)


def generate() -> str:
    decisions = parse_decisions(DECISIONS.read_text())
    return render(build_rows(collect_symbols(), decisions), decisions)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 if the file is stale")
    arguments = parser.parse_args()

    rendered = generate()
    if arguments.check:
        if not INDEX.exists():
            print(f"FAIL: {INDEX} does not exist", file=sys.stderr)
            return 1
        if INDEX.read_text() != rendered:
            print(
                "FAIL: docs/DECISION_INDEX.md is stale. Run:\n"
                "  uv run python tools/build_decision_index.py",
                file=sys.stderr,
            )
            return 1
        print("OK: docs/DECISION_INDEX.md matches the sources")
        return 0

    INDEX.write_text(rendered)
    print(f"wrote {INDEX.relative_to(ROOT)} ({len(rendered.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
