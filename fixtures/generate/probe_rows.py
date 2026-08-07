"""The row class that can detect a missing float32 narrowing on the sample side.

Extracted from the generator so a test can import it without importing xgboost,
and so there is exactly one definition. The generator's probe is the only guard in
this repository that closes the fixture door -- "change a comparison, regenerate
the fixtures, watch everything pass" -- and a guard whose failure mode is vacuity
needs a test of its own. See `fixtures/tests/test_fixture_door.py`.
"""

from __future__ import annotations

import numpy as np


def narrows_onto(threshold: float) -> tuple[float, float] | None:
    """Two float64 values that round to ``threshold`` in float32 without equalling it.

    This is the row class that can detect a missing narrowing on the *sample*
    side of the split comparison, and the corpus rows cannot: every value they
    carry is already float32-exact, so narrowing it is a no-op and an
    implementation that skips it still routes correctly.
    """
    exact = np.float32(threshold)
    if exact == 0 or not np.isfinite(exact):
        return None
    up = np.nextafter(exact, np.float32(np.inf))
    down = np.nextafter(exact, np.float32(-np.inf))
    if not (np.isfinite(up) and np.isfinite(down)):
        return None
    wide = np.float64(exact)
    above = float((wide + (wide + np.float64(up)) / 2) / 2)
    below = float((wide + (wide + np.float64(down)) / 2) / 2)
    if np.float32(above) != exact or np.float32(below) != exact:
        return None
    if not (below < float(wide) < above):
        return None
    return below, above
