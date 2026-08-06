#!/usr/bin/env bash
#
# Build the wheel and sdist, verify their CONTENTS, then install the wheel into
# an environment containing nothing else and predict from it.
#
# Why this exists as its own check: every other test in this repository runs
# against the source tree through the uv workspace. A module that imports only
# because packages/python/src happens to be on the path, a data file that never
# ships, or a dependency that is really required but declared optional all pass
# the entire suite and fail on `pip install`. That class of defect is invisible
# from inside the workspace, and one of them shipped -- the export extra
# resolved an XGBoost version the exporter then refused.
#
# Runs identically on a laptop and in CI. No arguments.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "== building distributions =="
uv build "$REPO/packages/python" --out-dir "$WORK/dist" >/dev/null
WHEEL="$(find "$WORK/dist" -name '*.whl' | head -1)"
SDIST="$(find "$WORK/dist" -name '*.tar.gz' | head -1)"
echo "   wheel: $(basename "$WHEEL")"
echo "   sdist: $(basename "$SDIST")"

echo
echo "== verifying wheel CONTENTS =="
python3 - "$WHEEL" <<'PY'
import sys, zipfile

wheel = sys.argv[1]
names = zipfile.ZipFile(wheel).namelist()

# Every module the runtime imports, at import time or predict time. Derived
# from the package's own import graph, not guessed: __init__ imports errors,
# export, objectives, predict, transform and _version; export imports trees and
# validate. A missing one here is an ImportError on a user's machine and a
# green suite on ours.
required = [
    "xgboost_bridge/__init__.py",
    "xgboost_bridge/_version.py",
    "xgboost_bridge/errors.py",
    "xgboost_bridge/export.py",
    "xgboost_bridge/objectives.py",
    "xgboost_bridge/predict.py",
    "xgboost_bridge/transform.py",
    "xgboost_bridge/trees.py",
    "xgboost_bridge/validate.py",
]
missing = [r for r in required if r not in names]
if missing:
    sys.exit(f"FAIL: wheel is missing required modules: {missing}")
print(f"   all {len(required)} runtime modules present")

# The licence must actually ship, not merely be declared. Both manifests said
# MIT while neither distribution contained the text.
if not any("licenses/LICENSE" in n or n.endswith("/LICENSE") for n in names):
    sys.exit("FAIL: wheel ships no LICENSE file")
metadata = next(n for n in names if n.endswith(".dist-info/METADATA"))
meta = zipfile.ZipFile(wheel).read(metadata).decode()
if "License-File:" not in meta:
    sys.exit("FAIL: wheel METADATA carries no License-File")
print("   LICENSE present, and METADATA declares License-File")

# Nothing that should not ship.
for n in names:
    if "/tests/" in n or n.endswith("_test.py"):
        sys.exit(f"FAIL: wheel ships test code: {n}")
print("   no test code in the wheel")

# The declared dependency surface, asserted rather than assumed.
requires = sorted(line.split(":", 1)[1].strip() for line in meta.splitlines()
                  if line.startswith("Requires-Dist:"))
print("   Requires-Dist:", requires)
PY

echo
echo "== installing the wheel into an empty environment =="
uv venv --python 3.12 "$WORK/venv" >/dev/null 2>&1
uv pip install --python "$WORK/venv/bin/python" --quiet "$WHEEL"

# The smoke test lives outside the repository and runs with the repository
# nowhere on sys.path, so it exercises the installed package rather than the
# working tree.
cat > "$WORK/smoke.py" <<'PY'
import json, sys, pathlib
import numpy as np

# Prove we are testing the INSTALLED package, not a source tree.
import xgboost_bridge
loaded_from = pathlib.Path(xgboost_bridge.__file__).resolve()
assert "site-packages" in loaded_from.parts, f"not the installed package: {loaded_from}"
assert "src" not in loaded_from.parts, f"resolved to a source tree: {loaded_from}"

# The base install must not require XGBoost (D010).
import importlib.util
assert importlib.util.find_spec("xgboost") is None, "xgboost present in a base install"

from xgboost_bridge import Predictor

fixture = json.loads(pathlib.Path(sys.argv[1]).read_text())
artifact, rows = fixture["artifact"], fixture["rows"]
predictor = Predictor.from_json(artifact)

ok = 0
for i, row in enumerate(rows):
    values = {n: (float("nan") if v is None else float(v))
              for n, v in zip(artifact["feature_names"], row)}
    got = predictor.margin(values)
    want_bits = int(fixture["expected_margin"][i], 16)
    if int(np.float32(got).view(np.uint32)) == want_bits:
        ok += 1

print(f"   loaded from : {loaded_from}")
print(f"   margin bit-exact vs XGBoost ground truth: {ok}/{len(rows)}")
if ok != len(rows):
    sys.exit("FAIL: installed package does not reproduce the recorded margins")
PY

cp "$REPO/fixtures/corpus/binary_logistic_signed_zero.json" "$WORK/fixture.json"
(cd "$WORK" && "$WORK/venv/bin/python" smoke.py fixture.json)

echo
echo "== installing with the export extra and round-tripping a real model =="
uv venv --python 3.12 "$WORK/venv-export" >/dev/null 2>&1
uv pip install --python "$WORK/venv-export/bin/python" --quiet "${WHEEL}[export]"
"$WORK/venv-export/bin/python" -c "import xgboost; print('   resolved xgboost:', xgboost.__version__)"

cat > "$WORK/roundtrip.py" <<'PY'
import json, pathlib
import numpy as np, xgboost as xgb
from xgboost_bridge import export_model, to_json, Predictor

assert "site-packages" in pathlib.Path(__import__("xgboost_bridge").__file__).resolve().parts

names = ["feature_0", "feature_1", "feature_2"]
rng = np.random.default_rng(0)
X = rng.uniform(-2.0, 2.0, size=(200, 3))
y = (X[:, 0] + 0.5 * X[:, 1] - X[:, 2] > 0.0).astype(float)
matrix = xgb.DMatrix(X, label=y, feature_names=names, nthread=1)
booster = xgb.train(
    {"objective": "binary:logistic", "max_depth": 3, "eta": 0.3, "nthread": 1, "seed": 0},
    matrix, num_boost_round=10,
)

artifact = export_model(booster, feature_names=names)
predictor = Predictor.from_json(json.loads(to_json(artifact)))
row = {"feature_0": 0.70, "feature_1": -0.20, "feature_2": 1.00}

got = predictor.margin(row)
want = np.float32(booster.predict(
    xgb.DMatrix([[0.70, -0.20, 1.00]], feature_names=names), output_margin=True)[0])
print(f"   export -> predict margin {got!r}")
if got.view(np.int32) != want.view(np.int32):
    raise SystemExit(f"FAIL: installed export path disagrees with XGBoost: {got!r} vs {want!r}")
print("   margin bit-exact against XGBoost from the installed package")
PY
(cd "$WORK" && "$WORK/venv-export/bin/python" roundtrip.py)

echo
echo "PYTHON CLEAN INSTALL: OK"
