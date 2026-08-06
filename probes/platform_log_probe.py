"""Probe: is XGBoost's float32 logarithm platform-dependent, and which of the
candidate derivation routes reproduces it?

Run this on two platforms and diff the output. That is the entire method, and it
is why the inputs below are embedded as float32 **bit patterns** rather than
decimal literals: a decimal literal makes the comparison partly a test of two
parsers, and the question here is about the logarithm alone.

The inputs are not a uniform sample. They were selected on darwin/arm64 as
values where ``np.log`` of a float32 and a correctly-rounded (50-digit) float32
logarithm **disagree**. Targeting the disagreeing inputs is the point: a sample
that does not deliberately include them cannot distinguish the routes, and its
silence is not evidence of equivalence. An earlier version of this probe used 13
hand-picked "interesting" values, found all routes in agreement, and concluded
the wrong thing.

Routes compared, all of them narrowing the input to float32 first:

    R1  np.float32(np.log(np.float32(x)))                 -- what the exporter does
    R2  np.float32(math.log(float(np.float32(x))))         -- float64 log, then narrow
    R4  Decimal(...).ln() at 40 digits, then narrow        -- stdlib, correctly rounded
    CR  mpmath at 60 digits, then narrow                   -- correctly rounded reference

The oracle is XGBoost's own observed zero-tree margin, which is independent of
every route above: it comes out of XGBoost's C++ predictor, not out of any
recipe in this repository.
"""

from __future__ import annotations

import math
import platform
import sys
from decimal import Decimal, localcontext

import numpy as np
import xgboost as xgb
from mpmath import mp

mp.dps = 60

# Selected on darwin/arm64 as inputs where R1 and CR disagree. For survival:cox
# the logarithm is taken of base_score itself; for binary:logistic it is taken
# of the derived odds, so these are base_score values whose *odds* are hard.
COX_HARD = [
    0x3F7F9A17, 0x3F7FC1F4, 0x3F799D53, 0x3F7F48F1, 0x3F7FA449, 0x3F7FAE82,
    0x3F7F9104, 0x3F6DE198, 0x3F7C952C, 0x3ED942E3, 0x3F4B3CC9, 0x3F5D543A,
    0x3F7A65B3, 0x3F7B63A3, 0x3F7F78BD, 0x3F733B18, 0x3F7F84BC, 0x3F746149,
    0x3F79AAB9, 0x3F7FC504, 0x3EC2A3B3, 0x3F7E6514, 0x3F7DDDB8, 0x3F74A808,
]
LOGISTIC_HARD = [
    0x3F7915BD, 0x3F078994, 0x3EFE4BC3, 0x3EFF80C6, 0x3F03718D, 0x3EFF5AA6,
    0x3EFE436B, 0x3EFF89A7, 0x3F01A174, 0x3F0019CB, 0x3F005A98, 0x3EFFA063,
    0x3EFAC211, 0x3F3C6DBF, 0x3EFFC1EE, 0x3EFE5EFF, 0x3E9F9606, 0x3F01CAD5,
    0x3F3108D7, 0x3EF95524, 0x3EFFC352, 0x3F006CB4, 0x3F0017BC, 0x3F062960,
]
# Ordinary values, included as a control. All routes agree on these on
# darwin/arm64, so a disagreement here is a strictly larger finding.
CONTROL = [0x3F19999A, 0x3E99999A, 0x3F0A3D71, 0x3B03126F, 0x3F733333]


def bits(x) -> int:
    return int(np.float32(x).view(np.uint32))


def as_f32(pattern: int) -> np.float32:
    return np.uint32(pattern).view(np.float32)


def route_np(x) -> np.float32:
    return np.float32(np.log(np.float32(x)))


def route_float64(x) -> np.float32:
    return np.float32(math.log(float(np.float32(x))))


def route_decimal(x) -> np.float32:
    with localcontext() as ctx:
        ctx.prec = 40
        return np.float32(float(Decimal(float(np.float32(x))).ln()))


def route_exact(x) -> np.float32:
    return np.float32(float(mp.log(mp.mpf(float(np.float32(x))))))


ROUTES = (
    ("R1_np_log", route_np),
    ("R2_float64_log", route_float64),
    ("R4_decimal_ln", route_decimal),
    ("CR_mpmath", route_exact),
)


def logistic_odds(base_score: np.float32) -> np.float32:
    """The exporter's derivation up to but not including the logarithm."""
    lo = np.float32(1e-6)
    hi = np.float32(1.0) - np.float32(1e-6)
    clamped = np.float32(min(max(float(base_score), float(lo)), float(hi)))
    reciprocal = np.float32(np.float32(1.0) / clamped)
    return np.float32(reciprocal - np.float32(1.0))


def xgboost_zero_tree_margin(base_score: float, objective: str) -> np.float32:
    """XGBoost's own intercept, observed rather than derived.

    A zero-round booster's margin *is* the intercept: there are no trees to add.
    """
    matrix = xgb.DMatrix(
        np.zeros((1, 1), dtype=np.float64),
        label=np.array([1.0]),
        feature_names=["f0"],
    )
    booster = xgb.train(
        {"objective": objective, "base_score": base_score, "nthread": 1},
        matrix,
        num_boost_round=0,
        verbose_eval=False,
    )
    return np.float32(booster.predict(matrix, output_margin=True)[0])


def run(objective: str, patterns: list[int], label: str) -> dict[str, int]:
    print(f"--- {objective} :: {label} ({len(patterns)} inputs) ---")
    print(
        f"{'base_score_bits':<17} {'XGBoost':<11} "
        + " ".join(f"{name:<15}" for name, _ in ROUTES)
    )
    agree = {name: 0 for name, _ in ROUTES}
    for pattern in patterns:
        base_score = as_f32(pattern)
        observed = bits(xgboost_zero_tree_margin(float(base_score), objective))
        if objective == "survival:cox":
            argument = base_score
            derive = lambda route, arg=argument: route(arg)
        else:
            argument = logistic_odds(base_score)
            derive = lambda route, arg=argument: np.float32(-route(arg))
        cells = []
        for name, route in ROUTES:
            got = bits(derive(route))
            if got == observed:
                agree[name] += 1
            cells.append(f"0x{got:08X}{'=' if got == observed else ' '}      ")
        print(f"0x{pattern:08X}        0x{observed:08X}  " + " ".join(cells))
    print()
    print(f"  agreement with XGBoost, {label}:")
    for name, _ in ROUTES:
        print(f"    {name:<16} {agree[name]:>3}/{len(patterns)}")
    print()
    return agree


def main() -> None:
    print("=" * 78)
    print("PLATFORM LOG PROBE")
    print("=" * 78)
    print(f"platform        : {platform.system().lower()}/{platform.machine()}")
    print(f"python          : {sys.version.split()[0]}")
    print(f"numpy           : {np.__version__}")
    print(f"xgboost         : {xgb.__version__}")
    print(f"libc            : {platform.libc_ver()}")
    print()

    totals: dict[str, list[int]] = {name: [0, 0] for name, _ in ROUTES}
    for objective, patterns, label in (
        ("survival:cox", COX_HARD, "hard inputs"),
        ("survival:cox", CONTROL, "control"),
        ("binary:logistic", LOGISTIC_HARD, "hard inputs"),
        ("binary:logistic", CONTROL, "control"),
    ):
        agree = run(objective, patterns, label)
        for name, count in agree.items():
            totals[name][0] += count
            totals[name][1] += len(patterns)

    print("=" * 78)
    print("TOTAL agreement with XGBoost's observed intercept")
    print("=" * 78)
    for name, (count, total) in totals.items():
        print(f"  {name:<16} {count:>3}/{total}")


if __name__ == "__main__":
    main()
