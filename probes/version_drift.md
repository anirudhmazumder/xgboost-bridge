# Probe — version drift against the newest XGBoost available

Drift detection against the D001 reference pin. Every 3.3.0 finding recorded in
`probes/float32_thresholds.md`, `probes/base_score.md`, `probes/boosters.md`, and
`probes/tree_structure.md` was re-measured on the newest XGBoost build obtainable on
2026-08-01 and compared.

Every claim below is backed by a pasted command and its real output. Anything not directly
measured is labelled **INFERRED**. Ambiguity is presented, not resolved.

The word "latest" appears nowhere as a version. Resolved numbers only.

---

## Environment — resolved version numbers

### The newest *released* XGBoost is 3.3.0 — the pinned reference version itself

There is no newer release on PyPI. A bare `uv pip install xgboost` on 2026-08-01 resolves
to the D001 pin:

```
$ uv pip install --python drift-env/bin/python xgboost numpy
Resolved 3 packages in 152ms
Installed 3 packages in 90ms
 + numpy==2.5.1
 + scipy==1.18.0
 + xgboost==3.3.0
```

Confirmed against the index metadata rather than the resolver alone:

```
$ curl -s https://pypi.org/pypi/xgboost/json > pypi_xgboost.json
$ python -c "...print last 8 releases with upload dates..."
=== last 8 xgboost releases on PyPI, with upload dates ===
3.0.5        first-file 2025-09-05T09:18:59.300752Z
3.1.0        first-file 2025-10-17T23:27:48.468055Z
3.1.0rc1     first-file 2025-09-26T06:06:52.532022Z
3.1.1        first-file 2025-10-21T23:08:33.851318Z
3.1.2        first-file 2025-11-20T18:06:21.217283Z
3.1.3        first-file 2026-01-10T00:17:42.980013Z
3.2.0        first-file 2026-02-10T10:50:57.440307Z
3.3.0        first-file 2026-06-17T21:20:53.707692Z

PyPI info.version = 3.3.0
yanked for 3.3.0 files: {False}
```

No prerelease is newer either:

```
$ uv pip install --python drift-env/bin/python --prerelease=allow --dry-run --upgrade xgboost
Resolved 3 packages in 162ms
Would make no changes
```

GitHub agrees — `v3.3.0` is the newest tag, released 2026-06-17, and no `3.4.0` prerelease
tag exists:

```
$ curl -s "https://api.github.com/repos/dmlc/xgboost/releases?per_page=10"
v3.3.0         prerelease=False published=2026-06-17T20:56:43Z
v3.2.0         prerelease=False published=2026-02-10T08:14:33Z
v3.1.3         prerelease=False published=2026-01-09T09:57:40Z
v3.1.2         prerelease=False published=2025-11-20T13:33:54Z
v3.1.1         prerelease=False published=2025-10-21T21:09:34Z
v3.1.0         prerelease=False published=2025-10-17T23:32:14Z
v3.1.0rc1      prerelease=True  published=2025-09-26T17:58:03Z
v3.0.5         prerelease=False published=2025-09-05T08:47:40Z
v3.0.4         prerelease=False published=2025-08-11T10:58:32Z
v3.0.3         prerelease=False published=2025-07-30T15:56:34Z
```

**So a drift probe restricted to released versions would find nothing and would be
worthless.** The upstream `master` nightly channel *does* carry something newer, and that
is where the drift is.

### The newest *available build* is 3.4.0.dev0, resolved to an exact commit

Newest nightly wheel for this platform, selected by S3 `LastModified` across all 5
listing pages (223 candidate wheels):

```
$ python -c "...paginate xgboost-nightly-builds, filter macosx_12_0_arm64, sort by LastModified..."
pages fetched: 5
macosx_12_0_arm64 xgboost wheels found: 223
=== 6 most recent by LastModified ===
2026-07-30T04:41:10.000Z master/2c58cf87ea570f127a7a90518704760a45a140e9/xgboost-3.4.0.dev0-py3-none-macosx_12_0_arm64.whl
2026-07-31T07:01:06.000Z master/182c32c7a115d8596f19881c476d1d11528aebdb/xgboost-3.4.0.dev0-py3-none-macosx_12_0_arm64.whl
2026-07-31T08:56:05.000Z master/8145d39f7cf64fa41fc7fcd99698ba7c68ec24bd/xgboost-3.4.0.dev0-py3-none-macosx_12_0_arm64.whl
2026-07-31T14:40:25.000Z master/894367746460d5efff789b39164fa8d1043d28c5/xgboost-3.4.0.dev0-py3-none-macosx_12_0_arm64.whl
2026-07-31T17:42:11.000Z master/e44223b6b81a799883c44f4daff352dcdeba9ebd/xgboost-3.4.0.dev0-py3-none-macosx_12_0_arm64.whl
2026-07-31T19:06:20.000Z master/e787a447de12c15bdf06f65ddbf79b056743113d/xgboost-3.4.0.dev0-py3-none-macosx_12_0_arm64.whl
```

The exact artifact used for every "newer version" number in this report:

| Field | Value |
|---|---|
| Wheel version string | `3.4.0.dev0` |
| `xgboost.__version__` | `3.4.0-dev` |
| Upstream commit | `e787a447de12c15bdf06f65ddbf79b056743113d` |
| Branch | `master` |
| Wheel built | `2026-07-31T19:06:20.000Z` |
| Probe run | `2026-08-01` |
| Wheel URL | `https://s3-us-west-2.amazonaws.com/xgboost-nightly-builds/master/e787a447de12c15bdf06f65ddbf79b056743113d/xgboost-3.4.0.dev0-py3-none-macosx_12_0_arm64.whl` |

```
$ uv pip install --python new-env/bin/python numpy "$WHL"
 + numpy==2.5.1
 + scipy==1.18.0
 + xgboost==3.4.0.dev0 (from https://s3-us-west-2.amazonaws.com/xgboost-nightly-builds/master/e787a447de12c15bdf06f65ddbf79b056743113d/xgboost-3.4.0.dev0-py3-none-macosx_12_0_arm64.whl)

$ new-env/bin/python -c "import sys, numpy, xgboost; print(...)"
xgboost 3.4.0-dev
numpy 2.5.1
python 3.12.8 (main, Dec  3 2024, 18:42:41) [Clang 16.0.0 (clang-1600.0.26.4)]
build_info: {'BUILTIN_PREFETCH_PRESENT': True, 'CLANG_VERSION': [15, 0, 0], 'DEBUG': False,
             'MM_PREFETCH_PRESENT': False, 'USE_CUDA': False, 'USE_DLOPEN_NCCL': False,
             'USE_FEDERATED': False, 'USE_NCCL': False, 'USE_OPENMP': True, 'USE_RMM': False}
```

Python 3.12 was sufficient; no newer Python was required. Both sides ran on the identical
`numpy==2.5.1` and `python 3.12.8`, so numpy and Python are held constant and every
difference below is attributable to XGBoost.

Reference side, in a throwaway venv independent of the workspace `.venv`:

```
$ drift-env/bin/python -c "import sys, numpy, xgboost; print(...)"
xgboost 3.3.0
numpy 2.5.1
python 3.12.8 (main, Dec  3 2024, 18:42:41) [Clang 16.0.0 (clang-1600.0.26.4)]
```

Workspace pin left untouched and verified unchanged:

```
$ ./.venv/bin/python -c "import xgboost; print('workspace xgboost', xgboost.__version__)"
workspace xgboost 3.3.0
$ grep xgboost fixtures/pyproject.toml
    "xgboost==3.3.0",
```

### Method note — why every number below is a paired measurement

One battery script (`battery.py`) was run **unmodified** under both interpreters, same
seed (`20260801`), same synthetic data, same generic feature names `f0..f5`. Drift is
reported as a `diff` of the two logs, not as a comparison against the prose of the earlier
probes. This means a bug in the battery cannot masquerade as drift — it appears identically
on both sides and cancels.

```
$ drift-env/bin/python battery.py out330 > out330/battery.log 2>&1 ; echo exit=$?
exit=0
$ new-env/bin/python battery.py out340 > out340/battery.log 2>&1 ; echo exit=$?
exit=0
```

### Artifact version marker

```
3.3.0     : doc['version']      : [3, 3, 0]
3.4.0-dev : doc['version']      : [3, 4, 0]
```

**CHANGED.** The newer build writes `[3, 4, 0]`. Note it writes the *release* triple
`[3, 4, 0]`, **not** anything carrying `dev` — the marker gives a reader no way to tell a
`3.4.0.dev0` nightly artifact from a future `3.4.0` release artifact. Relevant to any
out-of-range version-marker check under D007: a range test written as "reject > [3, 3, 0]"
rejects this file, and a range test written as "accept [3, 4, 0]" accepts a nightly it has
never been verified against.

---

## Verdicts, up front

**The entire numerical core is unchanged. One structural field moved, and that one move
silently corrupts dart predictions across the version boundary.**

The complete content diff of the two battery logs, with only the deprecation-warning noise
lines filtered out, is three items long:

```
$ diff <(grep -v "UserWarning\|bst.update\|b2.save_model\|b3.save_model\|Parameters: {\|^$" out330/battery.log) \
       <(grep -v "UserWarning\|bst.update\|b2.save_model\|b3.save_model\|Parameters: {\|^$" out340/battery.log)
4c4
< xgboost.__version__ : 3.3.0
---
> xgboost.__version__ : 3.4.0-dev
8c8
< doc['version']      : [3, 3, 0]
---
> doc['version']      : [3, 4, 0]
16,18c16,18
<     gradient_booster keys           = ['model', 'name', 'weight_drop']
<     model keys                      = ['cats', 'gbtree_model_param', 'iteration_indptr', 'tree_info', 'trees']
<     weight_drop len=8 values=[0.7905139, 0.90909094, 0.6993007, 0.7905139, 0.63572794, 0.3030303, 0.43478262, 0.3030303]
---
>     gradient_booster keys           = ['model', 'name']
>     model keys                      = ['cats', 'gbtree_model_param', 'iteration_indptr', 'tree_info', 'trees', 'weight_drop']
>     weight_drop                     = <ABSENT>
46,48c46,48
<     (the same three lines again, for the booster=gbtree + rate_drop variant)
243,244c243,244
< wrote out330/xversion_3_3_0.json and out330/xversion_3_3_0.ubj
< version marker in that file: [3, 3, 0]
---
> wrote out340/xversion_3_4_0-dev.json and out340/xversion_3_4_0-dev.ubj
> version marker in that file: [3, 4, 0]
246c246
< BATTERY COMPLETE for xgboost 3.3.0
---
> BATTERY COMPLETE for xgboost 3.4.0-dev
```

Threshold grammar, comparison operator, `base_score` form and transform residuals, tree
field inventory, and gamma-pruning behavior produced **byte-identical log output** on both
versions.

| Item | Verdict |
|---|---|
| **`weight_drop` relocated** | **CHANGED — and it is a silent wrong-number bug across the boundary.** `$.learner.gradient_booster.weight_drop` → `$.learner.gradient_booster.model.weight_drop`. Reading the 3.3.0 path on a 3.4.0-dev artifact yields `1.261264324e+00` margin error, `0.124498432` probability error, 359/400 rows off by >0.05, **0/400 rows correct**, no exception, no warning. |
| **Artifact version marker** | **CHANGED.** `[3, 3, 0]` → `[3, 4, 0]`, with no `dev` discriminator. |
| **`loss_changes` last-digit drift** | **CHANGED, harmless.** Differs in the last 1–2 significant digits on every tree model. Not a prediction-path field. Matters only for cross-version byte-identity (D008). |
| dart survival | **SAME.** Still accepted, still deprecated, same warning text. |
| gblinear survival | **SAME.** Still accepted, still deprecated, same warning text. Still not removed. |
| `weight_drop` is the only in-artifact dart signal | **SAME** — still exactly one signal, merely at a different path. |
| Threshold token grammar | **SAME**, all clauses. |
| Comparison operator, strict `<` / equality RIGHT | **SAME.** `LRR` on 216/216 internal nodes across 4 models. |
| `base_score` bracketed JSON string | **SAME.** |
| logistic float32 `1/p - 1` intermediate | **SAME.** Bit-exact 15/15; textbook form still wrong. |
| cox `ln`, reg identity | **SAME.** |
| Tree structure, field inventory, `default_left`, pruning | **SAME.** No new field, no removed field. |
| gblinear determinism per updater | **SAME.** `shotgun` @ `nthread=4` still 12/12 distinct. |

---

## 1. BOOSTER SURVIVAL — both survive

### dart — still accepted, still deprecated

```
$ new-env/bin/python warn2.py dart
### xgboost 3.4.0-dev  booster=dart
### RESULT: ACCEPTED — trained without error
### UserWarning, verbatim:
      [21:22:57] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:343: `booster=dart` is deprecated. Use the tree booster directly with dropout parameters like `rate_drop`, `skip_drop`, or `one_drop`.
```

3.3.0 control, same script:

```
$ drift-env/bin/python warn2.py dart
### xgboost 3.3.0  booster=dart
### RESULT: ACCEPTED — trained without error
### UserWarning, verbatim:
      [21:22:54] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:341: `booster=dart` is deprecated. Use the tree booster directly with dropout parameters like `rate_drop`, `skip_drop`, or `one_drop`.
```

**SAME.** Character-for-character identical warning text. The only difference is the C++
source line number, `learner.cc:341` → `learner.cc:343` — an artifact of surrounding edits,
not a behavior change. Still a warning, not an error. **dart is not gone.**

### gblinear — still accepted, still deprecated, removal still only announced

```
$ new-env/bin/python warn2.py gblinear
### xgboost 3.4.0-dev  booster=gblinear
### RESULT: ACCEPTED — trained without error
### UserWarning, verbatim:
      [21:22:59] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:825: `booster=gblinear` is deprecated and support will be removed in a future release.
```

3.3.0 control:

```
$ drift-env/bin/python warn2.py gblinear
### xgboost 3.3.0  booster=gblinear
### RESULT: ACCEPTED — trained without error
### UserWarning, verbatim:
      [21:22:59] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:824: `booster=gblinear` is deprecated and support will be removed in a future release.
```

**SAME.** Identical text; `learner.cc:824` → `learner.cc:825`. **gblinear has not been
removed in `3.4.0.dev0` at commit `e787a447de12c15bdf06f65ddbf79b056743113d`.** The
announced removal has not landed on `master` as of that commit.

The escalation the brief anticipated — a booster disappearing — **did not happen**. Neither
warning was upgraded to an error, and neither booster's serialized structure was withdrawn.

### gblinear serialized structure and determinism — unchanged

```
3.4.0-dev:
    ACCEPTED. gradient_booster.name = "gblinear"
    gradient_booster keys           = ['model', 'name']
    model keys                      = ['boosted_rounds', 'weights']
```

Weights bit-identical across versions on the same seed and data:

```
out330: {'boosted_rounds': 8, 'weights': [1.4101176, 0.77869374, -0.086313866, 0.030298369, -0.024975093, 0.020622177, -0.03970256]}
out340: {'boosted_rounds': 8, 'weights': [1.4101176, 0.77869374, -0.086313866, 0.030298369, -0.024975093, 0.020622177, -0.03970256]}
```

Determinism per updater, 12 trials each, both thread counts — the two tables are identical:

```
### out330 (xgboost 3.3.0)                          ### out340 (xgboost 3.4.0-dev)
  updater=shotgun        nthread=4  12 / 12  False    updater=shotgun        nthread=4  12 / 12  False
  updater=shotgun        nthread=1   1 / 12  True     updater=shotgun        nthread=1   1 / 12  True
  updater=coord_descent  nthread=4   1 / 12  True     updater=coord_descent  nthread=4   1 / 12  True
  updater=coord_descent  nthread=1   1 / 12  True     updater=coord_descent  nthread=1   1 / 12  True
```

**SAME.** `shotgun` at `nthread=4` still produces 12 distinct weight vectors in 12 trials at
a fixed seed; the non-determinism is still thread-parallel rather than seed-related, since
`nthread=1` is still reproducible. The `boosters.md` conclusion stands: pin `coord_descent`,
do not rely on `nthread=1`.

---

## 2. THE ONE REAL CHANGE — `weight_drop` moved one level deeper

### The relocation

Same fitted dart model, same seed, both versions:

```
$ drift-env/bin/python weightdrop.py out330wd        $ new-env/bin/python weightdrop.py out340wd
xgboost 3.3.0                                        xgboost 3.4.0-dev

=== WHERE IS weight_drop? ===                        === WHERE IS weight_drop? ===
$.learner.gradient_booster keys                      $.learner.gradient_booster keys
  : ['model', 'name', 'weight_drop']                   : ['model', 'name']
$.learner.gradient_booster.model keys                $.learner.gradient_booster.model keys
  : ['cats', 'gbtree_model_param',                     : ['cats', 'gbtree_model_param',
     'iteration_indptr', 'tree_info', 'trees']            'iteration_indptr', 'tree_info', 'trees',
                                                          'weight_drop']
'weight_drop' in gradient_booster      : True        'weight_drop' in gradient_booster      : False
'weight_drop' in gradient_booster.model: False       'weight_drop' in gradient_booster.model: True
```

The values are **unchanged** — this is a pure relocation:

```
3.3.0     weight_drop len=8 values=[0.7905139, 0.90909094, 0.6993007, 0.7905139, 0.63572794, 0.3030303, 0.43478262, 0.3030303]
3.4.0-dev weight_drop len=8 values=[0.7905139, 0.90909094, 0.6993007, 0.7905139, 0.63572794, 0.3030303, 0.43478262, 0.3030303]
weight_drop VALUES identical         : True
```

Verbatim on-disk context, showing the nesting difference — in 3.3.0 the field sits *after*
the `model` object closes, as a sibling of `name`; in 3.4.0-dev it sits *inside* `model`,
directly after the `trees` array:

```
3.3.0:      ...ture":"6","num_nodes":"15","size_leaf_vector":"1"}}]},"name":"gbtree","weight_drop":[7.905139E-1,9.0909094E-1,...]
                                                                 ^^^ model closes here, then name, then weight_drop

3.4.0-dev:  ...ted":"0","num_feature":"6","num_nodes":"15","size_leaf_vector":"1"}}],"weight_drop":[7.905139E-1,9.0909094E-1,...]
                                                                                  ^^^ still inside model
```

Exhaustive JSON path census confirms the move is the *only* structural difference — one path
removed, one path added, nothing else:

```
3.3.0 census                                          3.4.0-dev census
  $.learner.gradient_booster.model.cats.enc[]           $.learner.gradient_booster.model.cats.enc[]
  ... (identical) ...                                   ... (identical) ...
  $.learner.gradient_booster.model.trees[]  len=8       $.learner.gradient_booster.model.trees[]  len=8
  $.learner.gradient_booster.name                       $.learner.gradient_booster.model.weight_drop[]  len=8   <-- ADDED
  $.learner.gradient_booster.weight_drop[]  len=8       $.learner.gradient_booster.name
      ^^^ REMOVED
```

The relocation applies to **UBJSON as well as JSON**, so it is not a JSON-writer quirk:

```
$ new-env/bin/python -c "...save_model('d340.ubj')..."
   ubj token dart             present: False
   ubj token weight_drop      present: True
   ubj token rate_drop        present: False
   ubj token normalize_type   present: False
   bytes before weight_drop in ubj: b'\x00\x10size_leaf_vectorSL\x00\x00\x00\x00\x00\x00\x00\x011}}L\x00\x00\x00\x00\x00\x00\x00\x0bweight_drop'
```

The trailing `}}` closes `tree_param` and the tree, and `weight_drop` follows inside the
enclosing `model` map — same nesting as the JSON.

### The cost: reading the 3.3.0 path on a 3.4.0-dev artifact

A reader that looks only at `$.learner.gradient_booster.weight_drop` finds **nothing** on a
3.4.0-dev dart artifact. Under `boosters.md`'s Signal A, absence of `weight_drop` means
"plain gbtree, no per-tree weights" — a legitimate, correctly-shaped reading. So the reader
does not fail; it applies weights of `1.0`:

```
=== COST OF READING THE 3.3.0 PATH ON THIS ARTIFACT ===   (identical output under both versions)
reconstruction APPLYING weight_drop (found at the 3.4.0 path)
   max|recon - xgb margin| = 1.503371974e-07
reconstruction IGNORING weight_drop (what a 3.3.0-path reader does: field looks ABSENT)
   max|recon - xgb margin| = 1.261264145e+00
   mean|recon - xgb margin| = 6.829166641e-01
   max probability error from ignoring = 0.124498424
   rows where the ignoring reader is off by >0.05 in probability: 359 / 400
```

`1.5e-07` versus `1.26e+00` — seven orders of magnitude, and a `0.124` probability error on
a binary classifier. This is `boosters.md`'s measured `1.269166159e+00` failure mode,
reached not by a coding mistake but **by a correct 3.3.0 reader encountering a 3.4.0-dev
file.**

### XGBoost itself exhibits the failure

This is not hypothetical about our reader. XGBoost 3.3.0 reads a 3.4.0-dev dart artifact and
produces wrong numbers:

```
$ drift-env/bin/python dartcross.py          # loader = 3.3.0
LOADING xgboost is 3.3.0
  artifact out330wd/dart.json   marker=[3, 3, 0]  weight_drop at gradient_booster.weight_drop
    margin[:4] = [-1.3397263  -1.8095889   1.5204504   0.53535175]
    margin sum = 116.9387037884444
  artifact out340wd/dart.json   marker=[3, 4, 0]  weight_drop at gradient_booster.model.weight_drop
    margin[:4] = [-2.3540397 -3.0708532  2.297151   0.7616343]
    margin sum = 135.7469861060381

$ new-env/bin/python dartcross.py            # loader = 3.4.0-dev
LOADING xgboost is 3.4.0-dev
  artifact out330wd/dart.json   marker=[3, 3, 0]  weight_drop at gradient_booster.weight_drop
    margin[:4] = [-1.3397263  -1.8095889   1.5204504   0.53535175]
    margin sum = 116.9387037884444
  artifact out340wd/dart.json   marker=[3, 4, 0]  weight_drop at gradient_booster.model.weight_drop
    margin[:4] = [-1.3397263  -1.8095889   1.5204504   0.53535175]
    margin sum = 116.9387037884444
```

Read the four rows together:

- **3.4.0-dev reads both artifacts correctly** and gives the same margins for both —
  `[-1.3397263, ...]`. It accepts `weight_drop` at either path. Backward-compatible.
- **3.3.0 reads its own artifact correctly** — same `[-1.3397263, ...]`.
- **3.3.0 reads the 3.4.0-dev artifact wrongly** — `[-2.3540397, ...]`. Not
  backward-compatible in the forward direction.

Quantified, against 3.4.0-dev's reading of the same file as reference:

```
3.4.0-dev on 3.4.0-dev artifact (reference) margin[:4]: [-1.3397263  -1.8095889   1.5204504   0.53535175]
3.3.0     on 3.4.0-dev artifact             margin[:4]: [-2.3540397 -3.0708532  2.297151   0.7616343]
3.3.0     on 3.3.0     artifact             margin[:4]: [-1.3397263  -1.8095889   1.5204504   0.53535175]
reference == 3.3.0-native (same model)     : True

max |3.3.0-on-new  - reference| = 1.261264324e+00
mean|3.3.0-on-new  - reference| = 6.829166412e-01
max probability error           = 0.124498432
rows off by >0.05 in probability= 359 / 400
rows bit-identical              = 0 / 400
```

**`0 / 400` rows correct.** Not "wrong on a few" — wrong on every row.

And it is completely silent. Zero warnings, zero exceptions, exit code 0:

```
$ drift-env/bin/python silent.py out340wd/dart.json
loader xgboost: 3.3.0 | artifact: out340wd/dart.json | marker: [3, 4, 0]
python warnings raised during load+predict: 0
margin[:4]: [-2.3540397 -3.0708532  2.297151   0.7616343]

=== exit code: 0 ===
```

Control, the other direction:

```
$ new-env/bin/python silent.py out330wd/dart.json
loader xgboost: 3.4.0-dev | artifact: out330wd/dart.json | marker: [3, 3, 0]
python warnings raised during load+predict: 0
margin[:4]: [-1.3397263  -1.8095889   1.5204504   0.53535175]
```

(The only warning either run emitted was a `ResourceWarning` from the probe script's own
`json.load(open(...))`, not from XGBoost.)

3.3.0 does not merely mis-predict — it **discards** the field. Re-saving after load shows
`weight_drop` gone from both candidate paths, so a load/save round trip through 3.3.0
destroys the dropout weights permanently:

```
--- load dart.json (marker [3, 4, 0]) under 3.3.0
    LOADED OK. margin[:4] = [-2.3540397 -3.0708532  2.297151   0.7616343]
    re-saved marker: [3, 3, 0]
    re-saved gradient_booster keys: ['model', 'name']
    re-saved model keys           : ['cats', 'gbtree_model_param', 'iteration_indptr', 'tree_info', 'trees']
```

**This is exactly the silent-failure signature `xgboost-bridge` exists to prevent, occurring
inside XGBoost itself, between two adjacent versions, on a supported code path.**

### `weight_drop` is still the only in-artifact dart signal

Re-checked on 3.4.0-dev. The H2 problem from `boosters.md` is unchanged — one signal, now at
a new address:

```
3.4.0-dev, booster=dart rate_drop=0.3 skip_drop=0.1:
    ACCEPTED. gradient_booster.name = "gbtree"
    token dart             present in file: False
    token rate_drop        present in file: False
    token skip_drop        present in file: False
    token one_drop         present in file: False
    token normalize_type   present in file: False
    token weight_drop      present in file: True
    token sample_type      present in file: False
    save_config learner_train_param.booster = "gbtree"
    save_config gradient_booster.name       = "gbtree"
    save_config dart_train_param            = {"normalize_type": "tree", "one_drop": "0", "rate_drop": "0.300000012", "sample_type": "uniform", "skip_drop": "0.100000001"}
```

**SAME.** `gradient_booster.name` is still `"gbtree"` for dart; the string `dart` is still
absent from the file; `dart_train_param` is still config-only and still absent from the
artifact. And `booster=gbtree` with `rate_drop=0.3` still produces `weight_drop` — the
"dropout-activity-based, not booster-name-based" finding holds, at the new path.

---

## 3. THRESHOLD GRAMMAR — SAME, every clause

3.4.0-dev output. The corresponding 3.3.0 block is byte-identical.

```
--- binary_exact : 8 split_conditions arrays, 112 tokens
    tree 0 split_conditions verbatim from disk:
    2.3784617E-1,8.684404E-2,-2.9674232E-1,-5.971699E-1,-1.4647689E0,7.030622E-1,4.8033127E-1,-6.2545604E-1,-2.5546628E-1,-4.7742146E-1,5.7444755E-2,-2.3038948E-1,2.90544E-1

--- extreme : 6 split_conditions arrays, 128 tokens
    tree 0 split_conditions verbatim from disk:
    2.5E0,1.5E0,1.7893053E-30,1.5637359E30,-6.80501E-31,3.5E0,9.390298E-2,1.854E-42,-2.6485479E-2,-3.11E-43,7.936725E-31,-4.208944E29,5.3055686E-1,-5.4227185E-1,-1.4687394E-1,-9.138415E-2,2.0105623E-1,-3.3927825E-1,5.4495025E-2,1.9932026E-1,4.2261946E-1

--- TOKEN GRAMMAR CHECK over 240 tokens (both models) ---
total tokens                            : 240
tokens NOT matching -?D(.D+)?E-?D+      : 0 []
any quoted                              : False
any Infinity/NaN/inf/nan                : []
any '+' in exponent                     : []
any lowercase 'e'                       : []
any token with no 'E' at all            : []
exponent range                          : -44 .. 30
mantissa significant-digit range        : 1 .. 9
tokens with decimal point DROPPED       : 1, examples ['5E-1']
negative zero present ('-0E0')          : False
```

Clause by clause against `float32_thresholds.md` §2:

| Grammar clause | 3.3.0 | 3.4.0-dev | Verdict |
|---|---|---|---|
| JSON numbers, never strings | not quoted | `any quoted: False` | SAME |
| Always exponent notation | always | `any token with no 'E': []` | SAME |
| Uppercase `E` | uppercase | `any lowercase 'e': []` | SAME |
| No `+` in exponent | none | `any '+' in exponent: []` | SAME |
| Decimal point dropped for single-digit mantissa | `5E-1` | `['5E-1']` | SAME |
| At most 9 significant digits | max 9 | `1 .. 9` | SAME |
| No `Infinity`/`NaN` token | none | `[]` | SAME |

Shortest-float32 round trip, re-checked over a larger pooled corpus (5 models, 277
internal-node thresholds), with the exponent normalised so the comparison is digit-for-digit:

```
$ drift-env/bin/python rt.py out330/m_*.json          $ new-env/bin/python rt.py out340/m_*.json
internal-node threshold tokens : 277                   internal-node threshold tokens : 277
tokens whose digits != numpy shortest-float32 : 0/277  tokens whose digits != numpy shortest-float32 : 0/277
tokens where float64(token) != float32(token) : 258/277 tokens where float64(token) != float32(token): 258/277
tokens exactly representable in float32 (dyadic): 19/277 tokens exactly representable in float32 (dyadic): 19/277
```

**SAME.** `0 / 277` digit mismatches on both — thresholds are still the shortest decimal
round-tripping in float32. `258 / 277` still have `float64(token) != float32(token)`, so the
float32 hazard is still the overwhelming default. (The 19 dyadic tokens are all from the
extreme-scale model's constructed columns — `2.5E0`, `1.5E0`, `3.5E0`, `5E-1`.)

Nine significant digits is still the emission floor, and 8 still corrupts:

```
3.4.0-dev (3.3.0 identical):
    rounded to  5 significant digits: 209 / 240 land on a DIFFERENT float32
    rounded to  6 significant digits: 199 / 240 land on a DIFFERENT float32
    rounded to  7 significant digits: 142 / 240 land on a DIFFERENT float32
    rounded to  8 significant digits:   5 / 240 land on a DIFFERENT float32
    rounded to  9 significant digits:   0 / 240 land on a DIFFERENT float32
    rounded to 10 significant digits:   0 / 240 land on a DIFFERENT float32
    rounded to 17 significant digits:   0 / 240 land on a DIFFERENT float32
```

**SAME.** No grammar change, so **no format-design input from this section.**

---

## 4. COMPARISON OPERATOR — SAME, strict `<` with equality routing RIGHT

Method as in `float32_thresholds.md` §4: at every internal node, feed one float32 ULP below
`float32(threshold)`, exactly `float32(threshold)`, and one ULP above, and read the branch
from XGBoost's own `predict(pred_leaf=True)`. The root→node path is derived structurally
from `left_children`/`right_children`, so no comparison semantics of ours enter the
decision. A probe row whose new leaf falls outside the node's subtree is counted as
"left-the-path" and excluded rather than scored.

```
3.4.0-dev:
  binary_exact  nodes tested= 52  skipped=0  left-the-path=0  pattern histogram={'LRR': 52}
  binary_hist   nodes tested= 54  skipped=0  left-the-path=0  pattern histogram={'LRR': 54}
  reg_exact     nodes tested= 56  skipped=0  left-the-path=0  pattern histogram={'LRR': 56}
  cox_exact     nodes tested= 54  skipped=0  left-the-path=0  pattern histogram={'LRR': 54}
  pattern key: 'LRR' = strict '<' with equality routing RIGHT
```

The 3.3.0 block is identical, node counts included.

**Node count checked: 216 internal nodes** (52 + 54 + 56 + 54), spanning
`binary:logistic` (`exact` and `hist`), `reg:squarederror`, and `survival:cox`.
`LRR` on **216/216** on each version, `0` skipped, `0` off-path.

**CONFIRMED, not refuted.** The specification is unchanged:

```
go_left  iff  float32(value) < float32(split_condition)      # STRICT less-than
go_right otherwise, INCLUDING exact equality
NaN      -> left if default_left[node] else right
```

Because `nextafter_below(float32(token))` routes LEFT and `float32(token)` routes RIGHT on
216/216 nodes, the engine's effective threshold in 3.4.0-dev is still exactly
`float32(parse(token))` bit for bit — the same conclusion, and the same licence for
"parse as float64, then narrow."

---

## 5. `base_score` — SAME form, SAME transforms, SAME residuals

### Still a JSON string containing a bracketed array

3.4.0-dev, verbatim, all three in-scope objectives:

```
--- reg:squarederror
    learner_model_param      = {"base_score": "[-1.6863173E-1]", "boost_from_average": "1", "num_class": "0", "num_feature": "4", "num_target": "1"}
    base_score raw           = '[-1.6863173E-1]'   python type = str
    file substring verbatim  = "[-1.6863173E-1]"
    is JSON string           = True ; is bracketed = True
    inner token              = '-1.6863173E-1'

--- binary:logistic
    learner_model_param      = {"base_score": "[4.5E-1]", "boost_from_average": "1", "num_class": "0", "num_feature": "4", "num_target": "1"}
    base_score raw           = '[4.5E-1]'   python type = str
    file substring verbatim  = "[4.5E-1]"
    is JSON string           = True ; is bracketed = True
    inner token              = '4.5E-1'

--- survival:cox
    learner_model_param      = {"base_score": "[1E0]", "boost_from_average": "1", "num_class": "0", "num_feature": "4", "num_target": "1"}
    base_score raw           = '[1E0]'   python type = str
    file substring verbatim  = "[1E0]"
    is JSON string           = True ; is bracketed = True
    inner token              = '1E0'
```

**SAME.** Still `str`, still bracketed, still uppercase-`E` exponent notation including for
integral values (`"[1E0]"`), still the same key at the same path
`learner.learner_model_param.base_score` for every objective. The two-parse requirement is
unchanged.

### Per-objective transform, measured leaf-free

Method as in `base_score.md`: fit *N*>0 rounds, read the serialized `base_score`, then fit a
**0-round** model with `base_score` pinned to that exact value so the margin is a constant
equal to `transform(base_score)`, with no leaf value in the chain. Verdicts rest on float32
bit patterns.

```
3.4.0-dev:

--- reg:squarederror   (stored token '-1.6863173E-1')
    MEASURED intercept       = -0.16863173246383667  bits=3190599116
      identity                         = -0.16863173246383667     bits=3190599116  bit-exact=True  residual=0.0

--- binary:logistic    (stored token '4.5E-1')
    MEASURED intercept       = -0.2006707787513733  bits=3192749220
      identity                         = 0.44999998807907104      bits=1055286886  bit-exact=False residual=0.6506707668304443
      ln (float32-snapped input)       = -0.7985077500343323      bits=3209456385  bit-exact=False residual=0.597836971282959
      float32 (1/p - 1) then -log      = -0.2006707787513733      bits=3192749220  bit-exact=True  residual=0.0
      textbook float64 log(p/(1-p))    = -0.2006707489490509      bits=3192749218  bit-exact=False residual=2.9802322387695312e-08

--- survival:cox       (stored token '1E0')
    MEASURED intercept       = 0.0  bits=0
      identity                         = 1.0                      bits=1065353216  bit-exact=False residual=1.0
      ln (float32-snapped input)       = 0.0                      bits=0           bit-exact=True  residual=0.0
      float32 (1/p - 1) then -log      = nan                      bits=2143289344  bit-exact=False residual=nan
```

**SAME on all three.** `reg:squarederror` → identity. `binary:logistic` → the float32
`1/p - 1` form, bit-exact; the textbook float64 logit is 2 ULP wrong even here.
`survival:cox` → `ln`. The 3.3.0 block is identical. No objective's storage space moved, and
none needed to be inferred by analogy.

### The float32 `1/p - 1` intermediate — still required, residuals per form

15-value sweep, 3.4.0-dev:

```
  p          measured                 f32(1/p-1) then -log     resid        textbook f64 logit       resid
  0.05       -2.944438934326172       -2.944438934326172       0.000e+00    -2.944438934326172       0.000e+00     bitexact a=True b=True
  0.1        -2.1972246170043945      -2.1972246170043945      0.000e+00    -2.1972246170043945      0.000e+00     bitexact a=True b=True
  0.2        -1.3862943649291992      -1.3862943649291992      0.000e+00    -1.3862943649291992      0.000e+00     bitexact a=True b=True
  0.25       -1.0986123085021973      -1.0986123085021973      0.000e+00    -1.0986123085021973      0.000e+00     bitexact a=True b=True
  0.3        -0.8472978472709656      -0.8472978472709656      0.000e+00    -0.8472977876663208      5.960e-08     bitexact a=True b=False
  0.4        -0.40546509623527527     -0.40546509623527527     0.000e+00    -0.40546509623527527     0.000e+00     bitexact a=True b=True
  0.48       -0.08004285395145416     -0.08004285395145416     0.000e+00    -0.0800427496433258      1.043e-07     bitexact a=True b=False
  0.5        -0.0                     -0.0                     0.000e+00    0.0                      0.000e+00     bitexact a=True b=False
  0.52       0.08004263788461685      0.08004263788461685      0.000e+00    0.08004263043403625      7.451e-09     bitexact a=True b=False
  0.6        0.40546515583992004      0.40546515583992004      0.000e+00    0.4054652154445648       5.960e-08     bitexact a=True b=False
  0.7        0.8472977876663208       0.8472977876663208       0.000e+00    0.8472977876663208       0.000e+00     bitexact a=True b=True
  0.75       1.0986121892929077       1.0986121892929077       0.000e+00    1.0986123085021973       1.192e-07     bitexact a=True b=False
  0.8        1.3862943649291992       1.3862943649291992       0.000e+00    1.3862944841384888       1.192e-07     bitexact a=True b=False
  0.9        2.1972241401672363       2.1972241401672363       0.000e+00    2.1972243785858154       2.384e-07     bitexact a=True b=False
  0.95       2.9444382190704346       2.9444382190704346       0.000e+00    2.9444386959075928       4.768e-07     bitexact a=True b=False
  float32 (1/p-1) form  : bit-exact 15/15   worst residual 0.0
  textbook f64 logit    : bit-exact 6/15   worst residual 4.76837158203125e-07
  textbook values breaching the 1e-6 gate: 0
  signed zero at p=0.5  : intercept=-0.0 bits=2147483648 is_negative_zero=True
  cox at base_score=1.0 : intercept=0.0 bits=0
```

The 3.3.0 block is identical, every digit.

Residuals as requested, per form:

| Form | Bit-exact | Worst residual | Verdict vs 3.3.0 |
|---|---|---|---|
| float32 `1/p - 1` then `-log` | **15 / 15** | **`0.0`** | SAME — still the only exact form |
| textbook float64 `log(p/(1-p))` | 6 / 15 | `4.76837158203125e-07` | SAME — still wrong on the majority |
| identity (logistic) | 0 / 15 | `0.6506707668304443` | SAME — wrong space |
| `ln` (logistic) | 0 / 15 | `0.597836971282959` | SAME — wrong space |
| identity (reg) | bit-exact | `0.0` | SAME |
| `ln` (cox) | bit-exact | `0.0` | SAME |

Signed zero holds in both directions: `binary:logistic` at `p=0.5` still yields **negative**
zero (`bits=2147483648`), which only `-log(1/0.5 - 1)` produces, and `survival:cox` at
`base_score=1.0` still yields **positive** zero (`bits=0`), matching `ln(1.0)`. The two
objectives still differ in the sign of zero.

**One coverage caveat, stated rather than glossed.** `base_score.md` reports 2 of 27 textbook
values breaching the `1e-6` gate, worst `6.198883056640625e-06`. My 15-value sweep found
`0` breaches, worst `4.77e-07`. This is **sweep coverage, not drift** — my `p` grid differs
from theirs, and my two runs are digit-identical, so nothing moved between versions. The
`base_score.md` conclusion is not weakened: the textbook form is still bit-wrong on 9/15
here, and this probe cannot and does not claim the breach disappeared.

---

## 6. TREE STRUCTURE — SAME. No new field, no removed field

3.4.0-dev. The 3.3.0 block is byte-identical.

```
trees[0] keys                  : ['base_weights', 'categories', 'categories_nodes', 'categories_segments', 'categories_sizes', 'default_left', 'id', 'left_children', 'loss_changes', 'parents', 'right_children', 'split_conditions', 'split_indices', 'split_type', 'sum_hessian', 'tree_param']
tree_param keys                : ['num_deleted', 'num_feature', 'num_nodes', 'size_leaf_vector']
tree_param verbatim            : {"num_deleted": "0", "num_feature": "6", "num_nodes": "13", "size_leaf_vector": "1"}
tree_param value python types  : ['str']
model keys under gradient_booster: ['cats', 'gbtree_model_param', 'iteration_indptr', 'tree_info', 'trees']
EXHAUSTIVE key census, every key at any depth (46):
   ['attributes', 'base_score', 'base_weights', 'boost_from_average', 'categories', 'categories_nodes', 'categories_segments', 'categories_sizes', 'cats', 'default_left', 'enc', 'feature_names', 'feature_segments', 'feature_types', 'gbtree_model_param', 'gradient_booster', 'id', 'iteration_indptr', 'learner', 'learner_model_param', 'left_children', 'loss_changes', 'model', 'name', 'num_class', 'num_deleted', 'num_feature', 'num_nodes', 'num_parallel_tree', 'num_target', 'num_trees', 'objective', 'parents', 'reg_loss_param', 'right_children', 'scale_pos_weight', 'size_leaf_vector', 'sorted_idx', 'split_conditions', 'split_indices', 'split_type', 'sum_hessian', 'tree_info', 'tree_param', 'trees', 'version']
leaf test agreement left_children[i]==-1 vs right_children[i]==-1: 112 / 112
distinct default_left values observed : [0, 1]
arrays whose length == num_nodes for every tree: {'base_weights': True, 'default_left': True, 'left_children': True, 'loss_changes': True, 'parents': True, 'right_children': True, 'split_conditions': True, 'split_indices': True, 'split_type': True, 'sum_hessian': True}
parents[0] root sentinel              : 2147483647 == 2**31-1 ? True
split_type distinct values            : [0]
'id' field present on tree 0          : True value = 0
any 'depth' field anywhere            : False
```

| `tree_structure.md` claim | 3.4.0-dev | Verdict |
|---|---|---|
| Parallel arrays, all length `num_nodes` | all `True` | SAME |
| Leaf iff `left_children[i] == -1` | `112 / 112` agreement with the `right_children` test | SAME |
| `default_left` is 0/1 | `[0, 1]` | SAME |
| 16 tree fields, exactly | same 16 names | SAME |
| `tree_param` values are JSON **strings** | `['str']` | SAME |
| `parents` root sentinel `2147483647` | `True` | SAME |
| No depth field anywhere | `False` | SAME |
| `split_type` 0 for numeric | `[0]` | SAME |
| gbtree `model` keys | `cats`, `gbtree_model_param`, `iteration_indptr`, `tree_info`, `trees` | SAME |

**Key census: 46 keys on both versions, identical sets.** No field appeared, none vanished,
within the gbtree/plain path.

### New and removed fields — the strict-reader hazards, called out explicitly

Only one field changed address anywhere in the artifact format, and under D007 it is
**both** a new-field hazard and a removed-field hazard simultaneously:

| Path | 3.3.0 | 3.4.0-dev | Strict-reader consequence |
|---|---|---|---|
| `$.learner.gradient_booster.weight_drop` | present (dart) | **REMOVED** | A reader keyed to this path finds nothing and, per `boosters.md` Signal A, concludes "no per-tree weights." That conclusion is *well-formed and wrong*: `1.261264324e+00` margin error, `0.124498432` probability error, 0/400 rows correct, no exception. **Absence is indistinguishable from plain gbtree, which is why this is silent rather than loud.** |
| `$.learner.gradient_booster.model.weight_drop` | absent | **NEW** | Under D007 ("unknown artifact field → raise"), a strict reader enumerating keys of `gradient_booster.model` encounters an unrecognized `weight_drop` and **must raise**. That is a hard failure on every 3.4.0-dev dart artifact — the intended, safe outcome, but it means dart artifacts from the newer version are unreadable rather than degraded. |

The asymmetry is worth stating plainly: **the removed field fails silently, the added field
fails loudly.** D007's fail-loudly rule catches the added field and cannot catch the removed
one, because a missing optional field is not an unknown field. Any dart support must
therefore treat "no `weight_drop` at any known path" as a condition to verify, not to
assume — for example by cross-checking numerically at export time (`boosters.md` Signal B),
which separates the two cases by six orders of magnitude.

No other field was added or removed anywhere in the artifact, on any objective or booster
tested.

### `loss_changes` drifts in the last digits — a new observation, harmless to inference

Field-level comparison of all 19 paired artifacts:

```
$ drift-env/bin/python (field-level diff across every paired artifact)
artifact                           bytes==   differing tree fields (3.3.0 vs 3.4.0-dev)
----------------------------------------------------------------------------------------
b_booster_dart_ALL_DEFAULTS.json   False     ['loss_changes']       pred-path fields differing: NONE
b_booster_dart_rate_drop_0_3_...   False     ['loss_changes']       pred-path fields differing: NONE
b_booster_gblinear_..._coord_des.. False     (none)                 pred-path fields differing: NONE
b_booster_gblinear_..._shotgun...  False     (none)                 pred-path fields differing: NONE
b_booster_gbtree_rate_drop_0_3...  False     ['loss_changes']       pred-path fields differing: NONE
bs_binary_logistic.json            False     ['loss_changes']       pred-path fields differing: NONE
bs_reg_squarederror.json           False     ['loss_changes']       pred-path fields differing: NONE
bs_survival_cox.json               False     ['loss_changes']       pred-path fields differing: NONE
gl.json                            False     (none)                 pred-path fields differing: NONE
m_binary_exact.json                False     ['loss_changes']       pred-path fields differing: NONE
m_binary_hist.json                 False     ['loss_changes']       pred-path fields differing: NONE
m_cox_exact.json                   False     ['loss_changes']       pred-path fields differing: NONE
m_extreme.json                     False     ['loss_changes']       pred-path fields differing: NONE
m_reg_exact.json                   False     ['loss_changes']       pred-path fields differing: NONE
primary.json                       False     ['loss_changes']       pred-path fields differing: NONE
prune_0.0.json                     False     ['loss_changes']       pred-path fields differing: NONE
prune_1000000000.0.json            False     ['loss_changes']       pred-path fields differing: NONE
prune_5.0.json                     False     ['loss_changes']       pred-path fields differing: NONE
prune_50.0.json                    False     ['loss_changes']       pred-path fields differing: NONE
```

`loss_changes` is the **only** field that differs, on every tree model, and the difference is
in the last one or two significant digits:

```
--- tree 1 differs. per-field:
   field loss_changes
     3.3.0    : [55.111233, 8.123938, 12.042358, 5.4954376, 3.471344, 8.410732, 5.581661, 0.0, ...]
     3.4.0-dev: [55.111233, 8.123934, 12.042358, 5.4954376, 3.471344, 8.410731, 5.581661, 0.0, ...]
--- tree 2 differs. per-field:
   field loss_changes
     3.3.0    : [32.970257, 10.8916, 8.786297, 3.4592018, 5.4189386, 1.877764, 2.2893734, 0.0, ...]
     3.4.0-dev: [32.970257, 10.8916, 8.786297, 3.4592018, 5.4189386, 1.877764, 2.2893715, 0.0, ...]
```

Critically, **`split_conditions`, `split_indices`, `left_children`, `right_children`,
`default_left`, `split_type`, `base_weights`, `sum_hessian`, and `tree_param` are identical
on every tree of every model.** `pred-path fields differing: NONE`, 19 for 19. `loss_changes`
is the split gain and is not read during inference, which is why margins are bit-identical
despite it (§8).

**INFERRED**, not separately measured: the mechanism is a small change to gain accumulation
order or associativity in the split evaluator. The measurement establishes only that the
field's last digits moved and that nothing on the prediction path did.

Relevance is confined to **D008 (byte-identical export)**: the same model fitted under two
XGBoost versions does not produce byte-identical XGBoost artifacts. This does not threaten
D008 as written — D008 governs *our* exporter's output for a *given* input model — but it
does mean a fixture corpus regenerated under a different XGBoost version will not be
byte-identical if the exporter ever carries `loss_changes`. Since `loss_changes` is not on
the prediction path, omitting it from the artifact format removes the exposure entirely.
That is a Phase 3 format-design observation, not a finding about correctness.

---

## 7. GAMMA PRUNING — SAME. Dead nodes still interleaved, parents still stale

3.4.0-dev. The 3.3.0 block is byte-identical.

```
  gamma=0.0      num_nodes=45 num_deleted=0 len(arrays)=45 reachable=45 unreachable=0
     unreachable indices            : []
     unreachable == INT32_MAX set   : True
     num_deleted == len(unreachable): True
     dead nodes with STALE parents (parent is now a leaf): 0 []

  gamma=5.0      num_nodes=45 num_deleted=6 len(arrays)=45 reachable=39 unreachable=6
     unreachable indices            : [27, 28, 35, 36, 37, 38]
     split_indices==INT32_MAX at    : [27, 28, 35, 36, 37, 38]
     unreachable == INT32_MAX set   : True
     num_deleted == len(unreachable): True
     unreachable is contiguous suffix: False
     dead nodes with STALE parents (parent is now a leaf): 6 [(27, 14), (28, 14), (35, 21), (36, 21), (37, 23), (38, 23)]

  gamma=50.0     num_nodes=45 num_deleted=26 len(arrays)=45 reachable=19 unreachable=26
     unreachable indices            : [15, 16, 19, 20, 21, 22, 23, 24, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
     split_indices==INT32_MAX at    : [15, 16, 19, 20, 21, 22, 23, 24, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
     unreachable == INT32_MAX set   : True
     num_deleted == len(unreachable): True
     unreachable is contiguous suffix: False
     dead nodes with STALE parents (parent is now a leaf): 26 [(15, 7), (16, 7), (19, 9), (20, 9), (21, 10), (22, 10)]

  gamma=1000000000.0 num_nodes=45 num_deleted=44 len(arrays)=45 reachable=1 unreachable=44
     unreachable indices            : [1, 2, 3, 4, ... 24]
     unreachable == INT32_MAX set   : True
     num_deleted == len(unreachable): True
     unreachable is contiguous suffix: True
     dead nodes with STALE parents (parent is now a leaf): 44 [(1, 0), (2, 0), (3, 1), (4, 1), (5, 2), (6, 2)]
```

**SAME, every clause of `tree_structure.md` §7:**

- Arrays are **not** truncated on pruning — `len(arrays) == num_nodes == 45` at every gamma.
- Dead nodes are **interleaved, not a contiguous suffix** — `unreachable is contiguous
  suffix: False` at `gamma=5.0` and `gamma=50.0` (`[27, 28, 35, 36, 37, 38]` skips 29–34).
- `parents` links are **stale** — e.g. `(27, 14)` means `parents[27] == 14` while node 14 is
  now a leaf. 6, 26, and 44 stale-parent dead nodes at the three nonzero gammas.
- Deleted nodes still carry `split_indices == 2147483647`, and the INT32_MAX set exactly
  equals the unreachable set (`True` at every gamma).
- `num_deleted` still equals the unreachable count exactly (`True` at every gamma).

The `gamma=1e9` case still collapses to a leaf-only root with 44 dead nodes retained in the
arrays — the sharpest format hazard from `tree_structure.md` is unchanged.

---

## 8. CROSS-VERSION READABILITY, BOTH DIRECTIONS

### Direction A — XGBoost 3.3.0 loading artifacts written by 3.4.0-dev

```
$ drift-env/bin/python xload.py out340/xversion_X.npy out340/xversion_3_4_0-dev.json out340/xversion_3_4_0-dev.ubj out340wd/dart.json
LOADING xgboost is 3.3.0

--- load xversion_3_4_0-dev.json                    [gbtree, binary:logistic]
    version marker in file: [3, 4, 0]
    LOADED OK. margin[:4] = [-1.9108418 -2.3413422  1.6714829  0.6555339]
    max|margin - producing-version margin| = 0.0
    bit-identical to producing version     = True
    re-saved marker: [3, 3, 0]

--- load xversion_3_4_0-dev.ubj                     [gbtree, UBJSON]
    LOADED OK. margin[:4] = [-1.9108418 -2.3413422  1.6714829  0.6555339]
    max|margin - producing-version margin| = 0.0
    bit-identical to producing version     = True
    re-saved marker: [3, 3, 0]

--- load dart.json                                  [dart]
    version marker in file: [3, 4, 0]
    LOADED OK. margin[:4] = [-2.3540397 -3.0708532  2.297151   0.7616343]     <-- WRONG, see §2
    re-saved marker: [3, 3, 0]
    re-saved gradient_booster keys: ['model', 'name']
    re-saved model keys           : ['cats', 'gbtree_model_param', 'iteration_indptr', 'tree_info', 'trees']
                                     ^^^ weight_drop silently DROPPED on re-save
```

**No error text to paste — there was no error.** The out-of-range version marker `[3, 4, 0]`
did not trip anything: 3.3.0 accepted it without comment. For gbtree the load is exact,
`max|margin - producing-version margin| = 0.0` and **bit-identical**. For dart it is
silently wrong (§2) and destructive on re-save.

### Direction B — XGBoost 3.4.0-dev loading artifacts written by 3.3.0

```
$ new-env/bin/python xload.py out330/xversion_X.npy out330/xversion_3_3_0.json out330/xversion_3_3_0.ubj out330wd/dart.json
LOADING xgboost is 3.4.0-dev

--- load xversion_3_3_0.json                        [gbtree, binary:logistic]
    version marker in file: [3, 3, 0]
    LOADED OK. margin[:4] = [-1.9108418 -2.3413422  1.6714829  0.6555339]
    max|margin - producing-version margin| = 0.0
    bit-identical to producing version     = True
    re-saved marker: [3, 4, 0]

--- load xversion_3_3_0.ubj                         [gbtree, UBJSON]
    LOADED OK. margin[:4] = [-1.9108418 -2.3413422  1.6714829  0.6555339]
    max|margin - producing-version margin| = 0.0
    bit-identical to producing version     = True
    re-saved marker: [3, 4, 0]

--- load dart.json                                  [dart]
    version marker in file: [3, 3, 0]
    LOADED OK. margin[:4] = [-1.3397263  -1.8095889   1.5204504   0.53535175]   <-- CORRECT
    re-saved marker: [3, 4, 0]
    re-saved model keys: ['cats', 'gbtree_model_param', 'iteration_indptr', 'tree_info', 'trees', 'weight_drop']
                                                                                                  ^^^ preserved, at the NEW path
```

**No error text to paste — there was no error.** 3.4.0-dev reads the old `weight_drop` path
correctly and migrates it to the new path on re-save. It accepts **either** path.

### gblinear, both directions

```
$ drift-env/bin/python glcross.py                    $ new-env/bin/python glcross.py
loader: 3.3.0                                        loader: 3.4.0-dev
  artifact out330/...coord_descent.json marker=[3,3,0]  artifact out330/...coord_descent.json marker=[3,3,0]
     LOADED OK  margin[:3]=[-2.1481488 -2.0172522        LOADED OK  margin[:3]=[-2.1481488 -2.0172522
                             2.8114982]                                          2.8114982]
       sum=139.06885221687844                                sum=139.06885221687844
  artifact out340/...coord_descent.json marker=[3,4,0]  artifact out340/...coord_descent.json marker=[3,4,0]
     LOADED OK  margin[:3]=[-2.1481488 -2.0172522        LOADED OK  margin[:3]=[-2.1481488 -2.0172522
                             2.8114982]                                          2.8114982]
       sum=139.06885221687844                                sum=139.06885221687844
```

**Fully bidirectional and numerically identical** — all four combinations give the same
margin sum to the last digit.

### Is the boundary a floor, a ceiling, or a window?

| Booster / format | 3.3.0 reads 3.4.0-dev | 3.4.0-dev reads 3.3.0 | Boundary |
|---|---|---|---|
| gbtree, JSON | OK, bit-identical (`0.0`) | OK, bit-identical (`0.0`) | **None — fully bidirectional** |
| gbtree, UBJSON | OK, bit-identical (`0.0`) | OK, bit-identical (`0.0`) | **None — fully bidirectional** |
| gblinear, JSON | OK, identical | OK, identical | **None — fully bidirectional** |
| **dart, JSON** | **loads, SILENTLY WRONG** (`1.26e+00`, 0/400 rows correct) | OK, correct | **CEILING at 3.3.0** |

**The answer is booster-dependent, and this is the load-bearing conclusion for D001.**

- For the three in-scope 1.0 objectives on **gbtree** — binary classification, regression,
  Cox survival — there is **no version boundary at all** between 3.3.0 and
  `3.4.0.dev0@e787a447`. Both directions, both serialization formats, bit-identical margins.
  D001's pin remains a convenience for reproducibility, not a correctness requirement.
- For **dart**, the boundary is a **ceiling**: 3.4.0-dev is a valid *reader* of 3.3.0 files,
  but 3.3.0 is **not** a valid reader of 3.4.0-dev files, and it fails silently rather than
  loudly. An artifact produced by a newer XGBoost cannot be re-verified against the pinned
  3.3.0 reference — and the mismatch will not announce itself.

**Not a window.** Nothing in either direction refused to load; the only true incompatibility
is one-directional and undetectable from the load result alone.

---

## 9. DRIFT TABLE

One row per 3.3.0 finding. `3.4.0.dev0` throughout means the wheel at commit
`e787a447de12c15bdf06f65ddbf79b056743113d`.

| # | Finding (3.3.0) | 3.3.0 behavior | 3.4.0.dev0 behavior | Same / Changed |
|---|---|---|---|---|
| 1 | Newest released version | `3.3.0` (2026-06-17) | `3.3.0` is still the newest *release*; newest *build* is `3.4.0.dev0` on `master` | **CHANGED** (new build channel only) |
| 2 | Artifact version marker | `[3, 3, 0]` | `[3, 4, 0]`, no `dev` discriminator | **CHANGED** |
| 3 | `weight_drop` JSON path | `$.learner.gradient_booster.weight_drop` | `$.learner.gradient_booster.model.weight_drop` | **CHANGED — silent wrong-number hazard** |
| 4 | `weight_drop` values | `[0.7905139, 0.90909094, ...]` | identical | SAME |
| 5 | `weight_drop` relocation in UBJSON | sibling of `name` | inside `model` | **CHANGED** (same as JSON) |
| 6 | `loss_changes` values | e.g. `8.123938` | e.g. `8.123934` | **CHANGED, harmless** (not on prediction path) |
| 7 | dart accepted | yes, `UserWarning` | yes, `UserWarning`, identical text (`learner.cc:341`→`343`) | SAME |
| 8 | gblinear accepted | yes, `UserWarning` | yes, `UserWarning`, identical text (`learner.cc:824`→`825`) | SAME |
| 9 | gblinear removal | announced, not done | **still announced, still not done** | SAME |
| 10 | dart `gradient_booster.name` | `"gbtree"` | `"gbtree"` | SAME |
| 11 | String `dart` in artifact | absent | absent | SAME |
| 12 | `weight_drop` is the only in-artifact dart signal | yes | yes (at the new path) | SAME |
| 13 | `dart_train_param` serialized | no, config-only | no, config-only | SAME |
| 14 | dart via `booster=gbtree`+`rate_drop` | produces `weight_drop` | produces `weight_drop` | SAME |
| 15 | Threshold tokens are JSON numbers | not quoted | `any quoted: False` | SAME |
| 16 | Uppercase `E`, always exponent form | yes | `any lowercase 'e': []`, `no 'E' at all: []` | SAME |
| 17 | No `+` in exponent | yes | `any '+' in exponent: []` | SAME |
| 18 | Decimal point dropped for single-digit mantissa | `5E-1` | `['5E-1']` | SAME |
| 19 | At most 9 significant digits | max 9 | `1 .. 9` | SAME |
| 20 | No `Infinity`/`NaN` threshold token | none | `[]` | SAME |
| 21 | Thresholds are shortest float32 decimal | 0 mismatches | `0 / 277` mismatches | SAME |
| 22 | `float64(token) != float32(token)` typical | 195/195 | `258 / 277` (19 dyadic, all constructed) | SAME |
| 23 | 9 significant digits is the emission floor | 8 corrupts | `8 → 5/240`, `9 → 0/240` | SAME |
| 24 | Comparison is strict `<`, equality routes RIGHT | `LRR` 104/104 + 195 more | `LRR` **216/216** across 4 models, 0 skipped | SAME |
| 25 | Engine threshold == `float32(parse(token))` | measured | re-measured, 216/216 | SAME |
| 26 | `base_score` is a JSON **string** | `str` | `str` | SAME |
| 27 | `base_score` is a **bracketed** array | `"[4.8E-1]"` | `"[4.5E-1]"` | SAME |
| 28 | `base_score` path, all objectives | `learner.learner_model_param.base_score` | identical | SAME |
| 29 | logistic intercept = float32 `1/p - 1` then `-log` | bit-exact | **bit-exact 15/15, worst residual `0.0`** | SAME |
| 30 | textbook `log(p/(1-p))` is wrong | 16/27 bit-wrong | **6/15 bit-exact, worst residual `4.768e-07`** | SAME |
| 31 | cox intercept = `ln(base_score)` | bit-exact | bit-exact, residual `0.0` | SAME |
| 32 | reg intercept = identity | bit-exact | bit-exact, residual `0.0` | SAME |
| 33 | logistic `p=0.5` → **negative** zero | `bits=2147483648` | `bits=2147483648` | SAME |
| 34 | cox `base_score=1.0` → **positive** zero | `bits=0` | `bits=0` | SAME |
| 35 | Tree is parallel arrays of length `num_nodes` | yes | all `True` | SAME |
| 36 | Leaf iff `left_children[i] == -1` | yes | `112 / 112` agreement | SAME |
| 37 | `default_left` is 0/1 | `[0, 1]` | `[0, 1]` | SAME |
| 38 | 16 tree fields | 16 | same 16 names | SAME |
| 39 | `tree_param` values are strings | yes | `['str']` | SAME |
| 40 | `parents` root sentinel `2147483647` | yes | `True` | SAME |
| 41 | No depth field | none | `False` | SAME |
| 42 | Whole-artifact key census | 46 keys | **46 keys, identical set** | SAME |
| 43 | Pruning leaves arrays untruncated | yes | `len(arrays) == num_nodes` at every gamma | SAME |
| 44 | Dead nodes **interleaved**, not a suffix | yes | `contiguous suffix: False` at gamma 5 and 50 | SAME |
| 45 | Dead nodes have **stale parents** | yes | 6 / 26 / 44 stale-parent dead nodes | SAME |
| 46 | Deleted nodes carry `split_indices == INT32_MAX` | yes | set equality `True` at every gamma | SAME |
| 47 | `num_deleted` == unreachable count | yes | `True` at every gamma | SAME |
| 48 | `shotgun` non-deterministic @ `nthread=4` | 12/12 distinct | **12/12 distinct** | SAME |
| 49 | `shotgun` reproducible @ `nthread=1` | 1/12 | **1/12** | SAME |
| 50 | `coord_descent` deterministic, both thread counts | 1/12 | **1/12** both | SAME |
| 51 | gblinear structure `['boosted_rounds','weights']` | yes | identical | SAME |
| 52 | gblinear `len(weights) == num_feature + 1` | yes | 7 for 6 features | SAME |
| 53 | Cross-version load, gbtree, both directions | n/a | both OK, **bit-identical, `0.0`** | new measurement |
| 54 | Cross-version load, gblinear, both directions | n/a | both OK, identical | new measurement |
| 55 | Cross-version load, **dart, 3.3.0 reads 3.4.0.dev0** | n/a | **loads, SILENTLY WRONG, `1.26e+00`, 0/400 rows correct** | new measurement |
| 56 | Cross-version load, dart, 3.4.0.dev0 reads 3.3.0 | n/a | OK, correct, migrates path on re-save | new measurement |

**Rows changed: 1, 2, 3, 5, 6.** Of these, only **row 3** has numerical consequences, and
**row 55** is its realization inside XGBoost itself. Rows 24, 29–34 — the whole numerical
core — are unchanged.

---

## Ambiguity, presented rather than resolved

1. **"Newest XGBoost available" has two defensible readings, and they give different
   answers.** (a) Newest *release*: `3.3.0` — identical to the D001 pin, so the drift
   surface is empty and the probe finds nothing. (b) Newest *available build*:
   `3.4.0.dev0` at commit `e787a447de12c15bdf06f65ddbf79b056743113d` — which is where the
   `weight_drop` relocation lives. **I measured both and report both** rather than picking
   one. Reading (a) is what a user gets from `pip install`; reading (b) is what will ship
   next. The `weight_drop` finding exists only under (b).

2. **A nightly is not a release, and `master` can change before 3.4.0 ships.** The
   `weight_drop` relocation is real at that commit and reproducible from that URL, but it is
   **not a committed upstream API**. It could be reverted, or it could ship with a
   compatibility shim that reads both paths — 3.4.0-dev already reads both paths on *load*,
   which is consistent with either an intentional migration or an incomplete one. Two
   readings: (i) the relocation is deliberate and will ship, so a dart reader must accept
   both paths; (ii) it is incidental churn on `master` and may not survive to release. **I
   cannot distinguish these from the artifacts** and did not try to settle it from
   documentation or commit messages, per the "only from output you produced" constraint.
   Re-confirming against the eventual 3.4.0 release is the only way to close this.

3. **The version marker writes `[3, 4, 0]`, not a dev-distinguishable value.** A nightly
   artifact is therefore indistinguishable from a future 3.4.0-release artifact by marker
   alone. Whether an out-of-range check should reject `[3, 4, 0]` outright, or accept it and
   rely on structural validation, is a format/policy question I am not resolving.

4. **`loss_changes` drift mechanism is inferred, not measured.** I established that the field
   changed in its last digits and that no prediction-path field changed. I did **not**
   establish *why*. Attributing it to gain-accumulation order is a guess, labelled as such.

5. **`base_score` textbook-logit gate breaches: sweep coverage differs from
   `base_score.md`.** That probe found 2/27 values breaching `1e-6`; my 15-value grid found
   0/15, worst `4.77e-07`. My two runs are digit-identical, so this is **not** drift. I am
   explicitly **not** claiming the breach went away — my grid simply does not include the
   values that breach. The `base_score.md` conclusion stands unchanged.

6. **Single platform, single build.** All numbers are `darwin` / `arm64`, one nightly wheel,
   one machine. The `shotgun` non-determinism result in particular is 12 trials on one
   platform, exactly as `boosters.md` cautioned.

---

## Out of scope, things that looked wrong

1. **XGBoost 3.3.0 silently mis-predicts a 3.4.0-dev dart artifact.** This is upstream
   behavior, not ours, and it is the precise failure mode this project was created to guard
   against — a load that succeeds, a prediction that runs, `0 / 400` rows correct, no
   exception and no warning. Recorded here because it is the strongest available evidence
   that D007's fail-loudly rule and D001's version discipline are load-bearing rather than
   cosmetic. **It also demonstrates that the failure class is not hypothetical and not
   confined to ONNX conversion.**

2. **A version marker bump with no corresponding compatibility signal.** `[3, 4, 0]` tells a
   reader the writer was newer; it does not tell the reader *what moved*. 3.3.0 had every
   opportunity to refuse the file on marker grounds and did not. **Inferred**, not measured:
   the marker is informational upstream, not enforced. Any strictness our reader wants from
   the marker, our reader must implement.

3. **`booster=gblinear` still silently ignores tree parameters**, unchanged from
   `boosters.md` §8.2, with `tree_method` now also named:

   ```
   WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:794:
   Parameters: { "max_depth", "tree_method" } are not used.
   ```

   Same silent-acceptance pattern; a warning, not an error.

4. **`base_weights[0]` was not `-0E0` in my model set.** `float32_thresholds.md` §2 notes
   `base_weights[0] == -0E0` as evidence that XGBoost's writer emits signed zero. My primary
   model gave `-2.8670437E-8` on *both* versions, so this is a difference in model/data
   choice, **not drift**. The signed-zero property itself is independently confirmed in §5
   above (`binary:logistic` at `p=0.5` → `bits=2147483648`), so the underlying finding is
   intact.

---

## Reproducing this probe

Scripts and both venvs live in the session scratch directory, never in the repository:

```
scratchpad/probe-drift/
  drift-env/           throwaway venv, xgboost==3.3.0        (reference side)
  new-env/             throwaway venv, xgboost==3.4.0.dev0   (newer side)
  battery.py           the paired battery: versions, booster survival, grammar,
                       operator, base_score, tree structure, pruning
  rt.py                shortest-float32 round-trip, exponent-normalised
  weightdrop.py        weight_drop path census + cost of reading the 3.3.0 path
  dartcross.py         native vs cross-version dart margins
  xload.py             cross-version load, both directions, JSON + UBJSON
  glcross.py           gblinear cross-version load, both directions
  warn2.py             verbatim deprecation warning capture
  silent.py            proves the cross-version dart misread emits zero warnings
  out330/  out340/     paired artifacts and logs, one directory per version
```

Newer-side wheel, pinned by commit so the run is reproducible:

```
https://s3-us-west-2.amazonaws.com/xgboost-nightly-builds/master/e787a447de12c15bdf06f65ddbf79b056743113d/xgboost-3.4.0.dev0-py3-none-macosx_12_0_arm64.whl
```

Everything is seeded (`seed = 20260801`, `nthread = 1`, `tree_method = exact` unless a
variant is named). Data is `numpy` synthetic; feature names are `f0..f6`. No named datasets,
no domain vocabulary. The workspace `.venv`, `uv.lock`, and every `pyproject.toml` were left
untouched, and no fitted model or wheel was written into the repository.
