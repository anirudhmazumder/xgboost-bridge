"""Fixture corpus generation.

:mod:`fixtures.generate.corpus` fits real models with `xgboost`, exports them
through `xgboost_bridge.export.export_model`, and writes one JSON file per
fixture under ``fixtures/corpus/``. Every fixture carries XGBoost's own
`predict()` output as ground truth, so nothing on the JavaScript side ever
needs XGBoost installed to verify itself.

Not a published package -- this directory exists only to be run from within
the workspace, the same way ``fixtures/tests`` does.
"""

from __future__ import annotations
