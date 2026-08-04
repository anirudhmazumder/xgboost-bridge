# Probe — early stopping: which trees does `predict()` actually use?

Resolves the measurement D023 defers. Every number below was produced by a script in this
probe's scratch directory and is pasted verbatim from its stdout.

**Headline:** the question has **two different correct answers on the same model file**,
depending on which `predict()` the caller uses. `Booster.predict()` uses **all trees**;
`XGBRegressor.predict()` / `XGBClassifier.predict()` use **only the first
`best_iteration + 1` iterations**. Bit-exact agreement between the two paths is
`0/2500`. This is not version drift — it reproduces identically on 3.3.0 and 3.4.0.

---

## 0. Environment

```
$ uv run python e01_setup.py
python  : 3.12.8 (main, Dec  3 2024, 18:42:41) [Clang 16.0.0 (clang-1600.0.26.4)]
platform: macOS-26.5.2-arm64-arm-64bit
xgboost : 3.3.0
numpy   : 2.5.1
xgboost file: .../xgboost-bridge/.venv/lib/python3.12/site-packages/xgboost/__init__.py
```

`scikit-learn` is **not** in the workspace venv, so the sklearn-wrapper measurements were
run in two throwaway scratch venvs (nothing installed into the workspace; `uv.lock` and
every `pyproject.toml` untouched):

```
$ ./venv_330/bin/python e05_version.py
python       : 3.12.8
xgboost      : 3.3.0
numpy        : 2.5.1
scikit-learn : 1.9.0

$ ./venv_new/bin/python e05_version.py
python       : 3.12.8
xgboost      : 3.4.0
numpy        : 2.5.1
scikit-learn : 1.9.0
```

The newest XGBoost that `uv pip install --upgrade xgboost` resolved on this platform is
**`3.4.0`** (a release, not `3.4.0-dev`).

### Setup

`reg:squarederror` primary, `binary:logistic` confirmation. Six generic features
`c0..c5`, synthetic normal data, fixed seed `20260804`, `tree_method="exact"`,
`eta=0.3`, `max_depth=5`, `base_score=0.7` (deliberately **not** `0.5` — FORMAT.md §10
records that `0.5` makes every wrong accumulation variant pass). 600 training rows, 400
validation rows, 2500 held-out prediction rows.
`num_boost_round=400`, `early_stopping_rounds=8`. Early stopping fires hard: 16 of 400
requested rounds for regression, 19 for binary. The walk is the normative recipe from
`probes/accumulation.md` §6 / FORMAT.md §10, reimplemented in `walk.py`.

---

## 1. What is serialized, and where

```
$ uv run python e01_setup.py
=== A. python-level booster attributes on the in-memory booster ===
  bst.best_iteration      : 7
  bst.best_score          : 1.9654302687006682
  bst.num_boosted_rounds(): 16
  len(evals_result['valid']['rmse']) : 16
  bst.attributes()        : {'best_iteration': '7', 'best_score': '1.9654302687006682'}

=== B. serialized fields and their exact locations ===
  top-level keys                                    : ['learner', 'version']
  version                                           : [3, 3, 0]
  learner keys                                      : ['attributes', 'feature_names', 'feature_types', 'gradient_booster', 'learner_model_param', 'objective']
  learner.attributes                                : {"best_iteration": "7", "best_score": "1.9654302687006682"}
  learner.gradient_booster.name                     : gbtree
  gradient_booster.model keys                       : ['cats', 'gbtree_model_param', 'iteration_indptr', 'tree_info', 'trees']
  gradient_booster.model.gbtree_model_param         : {'num_parallel_tree': '1', 'num_trees': '16'}
  len(gradient_booster.model.trees)                 : 16
  len(gradient_booster.model.iteration_indptr)      : 17
  gradient_booster.model.iteration_indptr[:12]      : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
  gradient_booster.model.iteration_indptr[-6:]      : [11, 12, 13, 14, 15, 16]
  len(gradient_booster.model.tree_info)             : 16
  set(gradient_booster.model.tree_info)             : [0]
  learner.learner_model_param                       : {"base_score": "[7E-1]", "boost_from_average": "0", "num_class": "0", "num_feature": "6", "num_target": "1"}

  'best_iteration' appears in learner_model_param?  : False
  'best_iteration' appears in learner.attributes?   : True
  'best_iteration' substring count in whole raw JSON: 1
  'best_score'     substring count in whole raw JSON: 1
```

Raw bytes, so the string typing is not in doubt:

```
=== C. raw byte excerpt: the learner prefix ===
{"learner":{"attributes":{"best_iteration":"7","best_score":"1.9654302687006682"},"feature_names":["c0","c1","c2","c3","c4","c5"],"feature_types":[],"gradient_booster":{"model":{"cats":{"enc":[],"feature_segments":[],"sorted_idx":[]},"gbtree_model_param":{"num_parallel_tree":"1","num_trees":"16"},"iteration_indptr":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],"tree_info":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"trees":[{"bas
```

| Item | Exact location | Observed value | Type |
|---|---|---|---|
| `best_iteration` | `learner.attributes.best_iteration` | `"7"` | **JSON string** |
| `best_score` | `learner.attributes.best_score` | `"1.9654302687006682"` | **JSON string**, full precision |
| Tree count | `learner.gradient_booster.model.gbtree_model_param.num_trees` | `"16"` | JSON string |
| `len(trees)` | `learner.gradient_booster.model.trees` | `16` | array |
| Round boundaries | `learner.gradient_booster.model.iteration_indptr` | `[0,1,…,16]`, len 17 | int array |
| Per-round tree count | `gbtree_model_param.num_parallel_tree` | `"1"` | JSON string |

**All 16 trees are present. XGBoost did not truncate anything.** 400 rounds were
requested, 16 were run, the best was round 7, and rounds 8–15 are still in `trees[]` with
nothing marking them:

```
$ uv run python e07_slice.py
=== C. do the trees past best_iteration carry nonzero leaves? ===
  tree  8 : num_nodes=45  n_leaves=23  leaf min/max = -0.2802963 / 0.18953893
  tree 47 : num_nodes=35  n_leaves=18  leaf min/max = -0.10375438 / 0.16653302
```

The post-`best_iteration` trees carry real leaf values, so carrying or dropping them is a
numerically consequential choice, not a cosmetic one. Confirms the D023 premise verbatim.

---

## 2. THE CORE QUESTION — `Booster.predict()` uses ALL trees

Ground truth is `bst.predict(dte, output_margin=True)` with **no** `iteration_range`
argument — the default.

```
$ uv run python e02_core.py
rows = 2500
objective          : reg:squarederror
base_score stored  : [7E-1]
intercept (f32)    : np.float32(0.7)
len(trees)         : 16
num_trees param    : 16
num_parallel_tree  : 1
best_iteration attr: 7
bst.best_iteration : 7

predict(output_margin=True)  NO iteration_range given (the DEFAULT)
  first 4 : array([ 1.5421633 , -0.64820105, -2.549025  , -3.6205177 ], dtype=float32)

bit-exact match counts of walk vs predict() DEFAULT, over 2500 rows
  (a) ALL trees in trees[]            [0:16] -> bit-exact 2500/2500  max abs err 0.0
  (b) first best_iteration+1 iters    [0:8] -> bit-exact 0/2500  max abs err 1.5515437126159668
  (c) first best_iteration iters      [0:7] -> bit-exact 0/2500  max abs err 1.8696037530899048
```

Clean separation: `2500/2500` for (a), `0/2500` for (b) and (c), with margin errors of
`1.55` and `1.87`. No ambiguity from an under-separated model.

The walk itself is validated independently — it reproduces `predict()` bit-exactly on
**every** explicitly requested span, not just the full one:

```
cross-check: predict() with EXPLICIT iteration_range, vs walk on the same span
  iteration_range=(0,16) trees[0:16] -> walk bit-exact 2500/2500 ; equals DEFAULT pred 2500/2500
  iteration_range=(0,8) trees[0:8] -> walk bit-exact 2500/2500 ; equals DEFAULT pred 0/2500
  iteration_range=(0,7) trees[0:7] -> walk bit-exact 2500/2500 ; equals DEFAULT pred 0/2500
  iteration_range=(0,1) trees[0:1] -> walk bit-exact 2500/2500 ; equals DEFAULT pred 0/2500
  iteration_range=(0,3) trees[0:3] -> walk bit-exact 2500/2500 ; equals DEFAULT pred 0/2500

does predict() default equal iteration_range=(0, num_boosted_rounds)?
  bit-exact default vs (0,16): 2500/2500
  bit-exact default vs (0,8): 0/2500

inplace_predict (a different call path) default behaviour
  bit-exact inplace vs predict-default : 2500/2500
  bit-exact inplace vs (0,16)          : 2500/2500
  bit-exact inplace vs (0,8)           : 0/2500
```

`Booster.inplace_predict` agrees with `Booster.predict`: all trees.

### 2.1 The same three counts at `num_parallel_tree=1` and `3`, both objectives

```
$ uv run python e03_matrix.py
================================================================================================
reg:squarederror  num_parallel_tree=1  base_score=[7E-1]
  best_iteration=7  best_score=1.9654302687006682  num_boosted_rounds=16  num_trees=16  len(trees)=16
  iteration_indptr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  (len 17)
  tree_info uniq = [0] ; len(tree_info)=16
  intercept(f32) = np.float32(0.7)
  MAPPING CHECK: indptr[i+1]-indptr[i] uniq = [1] ; indptr[-1]==len(trees)? True ; len(indptr)-1==rounds? True
    (a) ALL trees                      trees[0:16] -> bit-exact 2500/2500  max abs err 0.0
    (b) first best_iteration+1 iters   trees[0:8] -> bit-exact 0/2500  max abs err 1.5515437126159668
    (c) first best_iteration iters     trees[0:7] -> bit-exact 0/2500  max abs err 1.8696037530899048
    (d) first best_iteration+1 TREES   trees[0:8] -> bit-exact 0/2500  max abs err 1.5515437126159668
    (e) first best_iteration TREES     trees[0:7] -> bit-exact 0/2500  max abs err 1.8696037530899048

================================================================================================
reg:squarederror  num_parallel_tree=3  base_score=[7E-1]
  best_iteration=7  best_score=1.9654302442171467  num_boosted_rounds=16  num_trees=48  len(trees)=48
  iteration_indptr = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48]  (len 17)
  tree_info uniq = [0] ; len(tree_info)=48
  intercept(f32) = np.float32(0.7)
  MAPPING CHECK: indptr[i+1]-indptr[i] uniq = [3] ; indptr[-1]==len(trees)? True ; len(indptr)-1==rounds? True
    (a) ALL trees                      trees[0:48] -> bit-exact 2500/2500  max abs err 0.0
    (b) first best_iteration+1 iters   trees[0:24] -> bit-exact 0/2500  max abs err 1.5515451431274414
    (c) first best_iteration iters     trees[0:21] -> bit-exact 0/2500  max abs err 1.869603917002678
    (d) first best_iteration+1 TREES   trees[0:8] -> bit-exact 0/2500  max abs err 2.8082711696624756
    (e) first best_iteration TREES     trees[0:7] -> bit-exact 0/2500  max abs err 3.0885674953460693

================================================================================================
binary:logistic  num_parallel_tree=1  base_score=[7E-1]
  best_iteration=10  best_score=0.5345686957845465  num_boosted_rounds=19  num_trees=19  len(trees)=19
  iteration_indptr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  (len 20)
  tree_info uniq = [0] ; len(tree_info)=19
  intercept(f32) = np.float32(0.8472978)
  MAPPING CHECK: indptr[i+1]-indptr[i] uniq = [1] ; indptr[-1]==len(trees)? True ; len(indptr)-1==rounds? True
    (a) ALL trees                      trees[0:19] -> bit-exact 2500/2500  max abs err 0.0
    (b) first best_iteration+1 iters   trees[0:11] -> bit-exact 0/2500  max abs err 1.3622877597808838
    (c) first best_iteration iters     trees[0:10] -> bit-exact 0/2500  max abs err 1.5868394374847412
    (d) first best_iteration+1 TREES   trees[0:11] -> bit-exact 0/2500  max abs err 1.3622877597808838
    (e) first best_iteration TREES     trees[0:10] -> bit-exact 0/2500  max abs err 1.5868394374847412

================================================================================================
binary:logistic  num_parallel_tree=3  base_score=[7E-1]
  best_iteration=10  best_score=0.5345686831558123  num_boosted_rounds=19  num_trees=57  len(trees)=57
  iteration_indptr = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57]  (len 20)
  tree_info uniq = [0] ; len(tree_info)=57
  intercept(f32) = np.float32(0.8472978)
  MAPPING CHECK: indptr[i+1]-indptr[i] uniq = [3] ; indptr[-1]==len(trees)? True ; len(indptr)-1==rounds? True
    (a) ALL trees                      trees[0:57] -> bit-exact 2500/2500  max abs err 0.0
    (b) first best_iteration+1 iters   trees[0:33] -> bit-exact 0/2500  max abs err 1.3622868061065674
    (c) first best_iteration iters     trees[0:30] -> bit-exact 0/2500  max abs err 1.5868382453918457
    (d) first best_iteration+1 TREES   trees[0:11] -> bit-exact 0/2500  max abs err 3.23114013671875
```

Summary table — ground truth `Booster.predict()` default, 2500 rows:

| Objective | `num_parallel_tree` | (a) all trees | (b) `best_iteration+1` iters | (c) `best_iteration` iters |
|---|---|---|---|---|
| `reg:squarederror` | 1 | **2500/2500** | 0/2500 | 0/2500 |
| `reg:squarederror` | 3 | **2500/2500** | 0/2500 | 0/2500 |
| `binary:logistic` | 1 | **2500/2500** | 0/2500 | 0/2500 |
| `binary:logistic` | 3 | **2500/2500** | 0/2500 | 0/2500 |

Exactly one variant matches in every cell. `survival:cox` was **not measured** in this
probe.

---

## 3. Iterations versus trees, and the authoritative mapping

### 3.1 `iteration_indptr` is the authoritative mapping, and it was confirmed, not assumed

```
  MAPPING CHECK: indptr[i+1]-indptr[i] uniq = [1] ; indptr[-1]==len(trees)? True ; len(indptr)-1==rounds? True   # npt=1
  MAPPING CHECK: indptr[i+1]-indptr[i] uniq = [3] ; indptr[-1]==len(trees)? True ; len(indptr)-1==rounds? True   # npt=3
```

Measured on all four models: `len(iteration_indptr) - 1 == num_boosted_rounds()`,
`iteration_indptr[-1] == len(trees)`, and every stride equals `num_parallel_tree`. So at
`num_parallel_tree=1` a round is one tree — **confirmed, not assumed** — and at
`num_parallel_tree=3` a round is three trees.

`iteration_indptr` is the only field that carries this directly. `num_parallel_tree ×
rounds` reproduced it in every model measured, but that is a derived quantity;
`iteration_indptr` is explicit and per-round, so it remains authoritative. Its stride was
uniform in all models here (`uniq = [1]` and `uniq = [3]`), which is consistent with —
but does not prove — that a non-uniform stride is impossible.

### 3.2 The answer is expressed in ITERATIONS, and this is where a tree-based truncation goes silently wrong

Rows (d) and (e) above truncate at `best_iteration + 1` **trees** instead of
`best_iteration + 1` **iterations**. At `num_parallel_tree=1` the two coincide
(`trees[0:8]` in both). At `num_parallel_tree=3` they diverge: `trees[0:24]` versus
`trees[0:8]`.

The decisive measurement is against the *sklearn* path, since that is the path that
truncates at all (§5). `reg:squarederror`, `num_parallel_tree=3`, `best_iteration=7`:

```
$ ./venv_330/bin/python e06_sk_roundtrip.py
=== B. sklearn estimator, num_parallel_tree=3, best_iteration=7, len(trees)=48 ===
  est.predict(output_margin=True) vs walk ALL       : 0/2500
  est.predict(output_margin=True) vs walk bi+1 iter : 2500/2500
```

and from the full matrix:

```
$ ./venv_330/bin/python e05_version.py
SKLEARN XGBRegressor  reg:squarederror  num_parallel_tree=3
  est.best_iteration = 7 ; est.best_score = 1.9654302442171467 ; est.n_estimators = 400
  --- ground truth = est.predict(..., output_margin=True) i.e. the SKLEARN default path
    (a) ALL trees           trees[0:48] -> bit-exact 0/2500  max abs err 1.5515451431274414
    (b) best_iteration+1 it trees[0:24] -> bit-exact 2500/2500  max abs err 0.0
    (c) best_iteration  it  trees[0:21] -> bit-exact 0/2500  max abs err 0.5856728553771973
    (d) best_iteration+1 TR trees[0:8] -> bit-exact 0/2500  max abs err 2.185603380203247
```

`trees[0:24]` → `2500/2500`. `trees[0:8]` → `0/2500`, max abs error `2.19`.

**A truncation implemented in trees rather than iterations is bit-exact at
`num_parallel_tree=1` and silently wrong at `num_parallel_tree=3`** — correct on the
common case, wrong on the uncommon one, with no error. This is the exact signature this
project exists to prevent. Any truncation must be
`trees[0 : iteration_indptr[best_iteration + 1]]`, never `trees[0 : best_iteration + 1]`.

---

## 4. `predict()`'s parameter surface and its default

```
$ uv run python e04_params.py
=== A. Booster.predict signature, verbatim ===
predict (self, data: xgboost.core.DMatrix, *, output_margin: bool = False, pred_leaf: bool = False, pred_contribs: bool = False, approx_contribs: bool = False, pred_interactions: bool = False, validate_features: bool = True, training: bool = False, iteration_range: Tuple[Union[int, numpy.integer], Union[int, numpy.integer]] = (0, 0), strict_shape: bool = False) -> numpy.ndarray

=== B. Booster.inplace_predict signature, verbatim ===
inplace_predict (self, data: Any, *, iteration_range: Tuple[Union[int, numpy.integer], Union[int, numpy.integer]] = (0, 0), predict_type: str = 'value', missing: float = nan, validate_features: bool = True, base_margin: Any = None, strict_shape: bool = False) -> numpy.ndarray
```

`iteration_range` is the one parameter that changes which trees are used. Its default on
`Booster.predict` and `Booster.inplace_predict` is the literal `(0, 0)`, which
`probes/base_score.md` §8 already established means "all iterations". Measured here:

```
=== D. iteration_range values, bit-exact vs the no-argument default ===
    iteration_range=None      -> 2500/2500 bit-exact vs default ; first value 1.5421632528305054
    iteration_range=(0, 0)    -> 2500/2500 bit-exact vs default ; first value 1.5421632528305054
    iteration_range=(0, 16)   -> 2500/2500 bit-exact vs default ; first value 1.5421632528305054
    iteration_range=(0, 8)    -> 0/2500 bit-exact vs default ; first value 1.3440992832183838
    iteration_range=(0, 7)    -> 0/2500 bit-exact vs default ; first value 1.2665417194366455
    iteration_range=(1, 16)   -> 0/2500 bit-exact vs default ; first value 1.4849848747253418
    iteration_range=(8, 16)   -> 0/2500 bit-exact vs default ; first value 0.8980639576911926
```

`(0, 0)` was measured only as a demonstration that it means "all", not used as a control;
the separating controls are `(0, 8)` and `(0, 7)`.

### 4.1 `Booster.predict()` does not consult `best_iteration` at all

Direct sensitivity test — mutate the attribute and re-predict:

```
=== E. does predict() consult the best_iteration ATTRIBUTE at all? ===
    attributes before : {'best_iteration': '7', 'best_score': '1.9654302687006682'}
    set_attr(best_iteration=0  ) -> bst.best_iteration=0 ; pred==original-default 2500/2500
    set_attr(best_iteration=1  ) -> bst.best_iteration=1 ; pred==original-default 2500/2500
    set_attr(best_iteration=3  ) -> bst.best_iteration=3 ; pred==original-default 2500/2500
    set_attr(best_iteration=99 ) -> bst.best_iteration=99 ; pred==original-default 2500/2500

=== F. remove the attribute entirely ===
    attributes after removal: {}
    b3.best_iteration raised: AttributeError: `best_iteration` is only defined when early stopping is used.
    pred with attribute removed == original default: 2500/2500
    num_boosted_rounds: 16
    re-serialized attributes: "attributes":{},"feature_names":["c0","c
```

Setting `best_iteration` to `0`, `1`, `3`, `99`, or deleting it outright leaves the
prediction bit-identical at `2500/2500`. `Booster.predict()` is completely insensitive to
it. The library's own docstring states the same thing:

```
$ uv run python -c "import xgboost as xgb; print(xgb.Booster.predict.__doc__[:300])"
Predict with data.  The full model will be used unless `iteration_range` is
        specified, meaning users have to either slice the model or use the
        ``best_iteration`` attribute to get prediction from best model returned from
        early stopping.
```

`best_iteration` is absent from a model trained without early stopping:

```
=== G. best_iteration on a model trained with NO early stopping ===
    attributes(): {}
    b4.best_iteration raised: AttributeError: `best_iteration` is only defined when early stopping is used.
    serialized attributes: "attributes":{},"feature_names
```

---

## 5. The sklearn wrapper gives the OPPOSITE answer on the same model

This is the finding that makes the question ambiguous rather than settled.

```
$ ./venv_330/bin/python e05_version.py
SKLEARN XGBRegressor  reg:squarederror  num_parallel_tree=1
  est.best_iteration = 7 ; est.best_score = 1.9654302687006682 ; est.n_estimators = 400
  --- ground truth = est.predict(..., output_margin=True) i.e. the SKLEARN default path
  best_iteration=7  num_boosted_rounds=16  num_trees=16  len(trees)=16  npt=1
    (a) ALL trees           trees[0:16] -> bit-exact 0/2500  max abs err 1.5515437126159668
    (b) best_iteration+1 it trees[0:8] -> bit-exact 2500/2500  max abs err 0.0
    (c) best_iteration  it  trees[0:7] -> bit-exact 0/2500  max abs err 0.5856728553771973
    (d) best_iteration+1 TR trees[0:8] -> bit-exact 2500/2500  max abs err 0.0
  --- ground truth = booster.predict(DMatrix) i.e. the NATIVE default path
    (a) ALL trees           trees[0:16] -> bit-exact 2500/2500  max abs err 0.0
    (b) best_iteration+1 it trees[0:8] -> bit-exact 0/2500  max abs err 1.5515437126159668
    (c) best_iteration  it  trees[0:7] -> bit-exact 0/2500  max abs err 1.8696037530899048
    (d) best_iteration+1 TR trees[0:8] -> bit-exact 0/2500  max abs err 1.5515437126159668
  SKLEARN default vs NATIVE default: bit-exact 0/2500
```

The two default paths on the **same booster object** agree on `0/2500` rows. Reproduced
for `XGBRegressor` at `num_parallel_tree=1` and `3` and for `XGBClassifier` at
`num_parallel_tree=1`; `SKLEARN default vs NATIVE default: bit-exact 0/2500` in all three.

The mechanism, read out of the installed package:

```
$ uv run python e07_slice.py
=== A. xgboost/sklearn.py, verbatim lines 1380-1400 ===
   1386|     def _can_use_inplace_predict(self) -> bool:
   1387|         return self.booster != "gblinear"
   1388|
   1389|     def _get_iteration_range(
   1390|         self, iteration_range: Optional[IterationRange]
   1391|     ) -> IterationRange:
   1392|         if iteration_range is None or iteration_range[1] == 0:
   1393|             # Use best_iteration if defined.
   1394|             try:
   1395|                 iteration_range = (0, self.best_iteration + 1)
   1396|             except AttributeError:
   1397|                 iteration_range = (0, 0)
   1398|         if self.booster == "gblinear":
   1399|             iteration_range = (0, 0)
   1400|         return iteration_range
```

Identical source on 3.4.0 (`inspect.getsource(xgb.XGBModel._get_iteration_range)`, printed
verbatim in §7). The estimator substitutes `(0, best_iteration + 1)` for the default; the
bare `Booster` does not.

### 5.1 The same file, two answers

The most direct statement of the hazard: one file on disk, loaded two ways.

```
$ ./venv_330/bin/python e06_sk_roundtrip.py
=== D. SAVE/LOAD round trip through the SKLEARN estimator ===
  est2.best_iteration : 7
  b2.attributes()     : {"best_iteration": "7", "best_score": "1.9654302442171467"}
  b2.num_boosted_rounds(): 16
  est2.predict vs walk ALL       : 0/2500
  est2.predict vs walk bi+1 iter : 2500/2500
  est2.predict vs est.predict    : 2500/2500

=== E. load the SAME file into a bare Booster ===
  b3.attributes() : {"best_iteration": "7", "best_score": "1.9654302442171467", "scikit_learn": "{\"_estimator_type\": \"regressor\"}"}
  b3.predict vs walk ALL       : 2500/2500
  b3.predict vs walk bi+1 iter : 0/2500
```

`sk_est.json` → `XGBRegressor.load_model` → truncates at 24 of 48 trees.
`sk_est.json` → `Booster.load_model` → uses all 48. **The artifact file does not determine
which trees are used; the caller's API does.** No field in the file distinguishes the two.

An explicit `iteration_range` overrides the sklearn default in both directions:

```
=== C. sklearn predict with EXPLICIT iteration_range overrides the default ===
  iteration_range=None      -> vs walk ALL 0/2500 ; vs walk bi+1 2500/2500 ; first 1.3440991640090942
  iteration_range=(0, 16)   -> vs walk ALL 2500/2500 ; vs walk bi+1 0/2500 ; first 1.5421627759933472
  iteration_range=(0, 8)    -> vs walk ALL 0/2500 ; vs walk bi+1 2500/2500 ; first 1.3440991640090942
```

### 5.2 A carried `best_iteration` does not imply truncation is needed

When `early_stopping_rounds` is set but never fires, `best_iteration` is still written and
happens to equal the last round, so both readings coincide:

```
=== F. early_stopping_rounds set but NEVER TRIGGERED: is best_iteration still written? ===
  attributes            : {"best_iteration": "4", "best_score": "1.9858323361111818"}
  best_iteration=4  num_boosted_rounds=5  len(trees)=5  indptr=[0, 1, 2, 3, 4, 5]
  best_iteration+1 == num_boosted_rounds ? True
  sklearn predict vs walk ALL : 2500/2500
  native  predict vs walk ALL : 2500/2500
  sklearn predict vs native predict : 2500/2500
```

So `best_iteration` present is **not** the same predicate as "trees past the best
iteration exist". The discriminating predicate is
`iteration_indptr[best_iteration + 1] != len(trees)`.

### 5.3 `learner.attributes` gains a third key on the sklearn path

`{"best_iteration": "7", "best_score": "…", "scikit_learn": "{\"_estimator_type\": \"regressor\"}"}`
— see §E above. D020 whitelists nothing today; a whitelist for `best_iteration` must not
accidentally admit `scikit_learn` or `best_score`.

---

## 6. Save/load round trip

The truncation point survives serialization in every format and every load path measured.

```
$ uv run python e03_matrix.py     # reg:squarederror, num_parallel_tree=3
    round-trip m_reg_npt3_sm.json best_iteration attr='7'  bst.best_iteration=7  num_boosted_rounds=16  pred==orig 2500/2500
                                   vs walk ALL 2500/2500 ; vs walk best_iter+1 0/2500
    round-trip m_reg_npt3_sm.ubj  best_iteration attr='7'  bst.best_iteration=7  num_boosted_rounds=16  pred==orig 2500/2500
                                   vs walk ALL 2500/2500 ; vs walk best_iter+1 0/2500
```

Same for all four models in the matrix (`reg`×{1,3}, `binary`×{1,3}) and for
`save_raw(raw_format="json")`, `save_model(.json)`, and `save_model(.ubj)`:

```
$ ./venv_330/bin/python e08_census.py
native save/load round trip, best_iteration=7 len(trees)=48
  nat_v330.json        attrs={"best_iteration": "7", "best_score": "1.9654302442171467"}  rounds=16  vs walk ALL 2500/2500  vs walk bi+1 0/2500
  nat_v330.ubj         attrs={"best_iteration": "7", "best_score": "1.9654302442171467"}  rounds=16  vs walk ALL 2500/2500  vs walk bi+1 0/2500
```

`best_iteration` **does** survive serialization, as a string, in `learner.attributes`, in
both JSON and UBJSON. The artifact *can* record the truncation point. The sklearn path
also round-trips its truncating behaviour (§5.1, `est2.predict vs est.predict 2500/2500`).

### 6.1 `bst[0:best_iteration+1]` — the other route the docstring names

```
$ uv run python e07_slice.py
=== B. bst[0:best_iteration+1] -- the 'slice the model' route the docstring names ===
  before slice: len(trees)=48  num_trees=48  indptr len=17  attributes={"best_iteration": "7", "best_score": "1.9654302442171467"}
  after  slice: len(trees)=24  num_trees=24  indptr=[0, 3, 6, 9, 12, 15, 18, 21, 24]  attributes={}
  sliced num_parallel_tree=3 ; intercept=np.float32(0.7) ; base_score=[7E-1]
  sliced.predict vs walk-on-full-model ALL      : 0/2500
  sliced.predict vs walk-on-full-model bi+1 it  : 2500/2500
  sliced.predict vs walk-on-SLICED artifact ALL : 2500/2500
  walk-on-SLICED == walk-on-full-prefix         : 2500/2500
```

Three facts worth noting:

1. Slicing is expressed in **iterations**: `bst[0:8]` at `num_parallel_tree=3` yields 24
   trees, not 8, and rebuilds `iteration_indptr` to `[0,3,…,24]`.
2. **Slicing drops `learner.attributes` to `{}`** — `best_iteration` and `best_score` are
   gone. A sliced booster is therefore an ordinary, unambiguous model with no early-stopping
   marker at all, and `Booster.predict()` on it agrees bit-exactly with the sklearn
   truncated answer (`2500/2500`).
3. `base_score` and the derived intercept are unchanged by slicing
   (`[7E-1]` → `np.float32(0.7)`).

So `walk(all trees of bst[0:bi+1]) == walk(trees[0:indptr[bi+1]] of the full model)` at
`2500/2500`: pre-slicing and prefix-truncation are the same arithmetic.

---

## 7. Version sensitivity — 3.3.0 and 3.4.0 are identical

Resolved version: **`xgboost 3.4.0`** (numpy `2.5.1`, scikit-learn `1.9.0`, Python
`3.12.8`). Not a dev build.

```
$ ./venv_new/bin/python e05_version.py
python       : 3.12.8
xgboost      : 3.4.0
numpy        : 2.5.1
scikit-learn : 1.9.0

============================================================================================
NATIVE xgb.train  reg:squarederror  num_parallel_tree=1
  best_iteration=7  num_boosted_rounds=16  num_trees=16  len(trees)=16  npt=1
  learner.attributes = {"best_iteration": "7", "best_score": "1.9654302687006682"}
  iteration_indptr len=17  [:6]=[0, 1, 2, 3, 4, 5]  [-3:]=[14, 15, 16]
  learner_model_param = {"base_score": "[7E-1]", "boost_from_average": "0", "num_class": "0", "num_feature": "6", "num_target": "1"}
  gb.model keys = ['cats', 'gbtree_model_param', 'iteration_indptr', 'tree_info', 'trees']
  version = [3, 4, 0]
    (a) ALL trees           trees[0:16] -> bit-exact 2500/2500  max abs err 0.0
    (b) best_iteration+1 it trees[0:8] -> bit-exact 0/2500  max abs err 1.5515437126159668
    (c) best_iteration  it  trees[0:7] -> bit-exact 0/2500  max abs err 1.8696037530899048
    (d) best_iteration+1 TR trees[0:8] -> bit-exact 0/2500  max abs err 1.5515437126159668

============================================================================================
NATIVE xgb.train  reg:squarederror  num_parallel_tree=3
  best_iteration=7  num_boosted_rounds=16  num_trees=48  len(trees)=48  npt=3
  iteration_indptr len=17  [:6]=[0, 3, 6, 9, 12, 15]  [-3:]=[42, 45, 48]
  version = [3, 4, 0]
    (a) ALL trees           trees[0:48] -> bit-exact 2500/2500  max abs err 0.0
    (b) best_iteration+1 it trees[0:24] -> bit-exact 0/2500  max abs err 1.5515451431274414
    (c) best_iteration  it  trees[0:21] -> bit-exact 0/2500  max abs err 1.869603917002678
    (d) best_iteration+1 TR trees[0:8] -> bit-exact 0/2500  max abs err 2.8082711696624756

============================================================================================
SKLEARN XGBRegressor  reg:squarederror  num_parallel_tree=3
  est.best_iteration = 7 ; est.best_score = 1.9654302442171467 ; est.n_estimators = 400
  --- ground truth = est.predict(..., output_margin=True) i.e. the SKLEARN default path
    (a) ALL trees           trees[0:48] -> bit-exact 0/2500  max abs err 1.5515451431274414
    (b) best_iteration+1 it trees[0:24] -> bit-exact 2500/2500  max abs err 0.0
    (c) best_iteration  it  trees[0:21] -> bit-exact 0/2500  max abs err 0.5856728553771973
    (d) best_iteration+1 TR trees[0:8] -> bit-exact 0/2500  max abs err 2.185603380203247
  SKLEARN default vs NATIVE default: bit-exact 0/2500
```

Every number in the 3.4.0 run is byte-identical to the 3.3.0 run of the same script,
including the trailing digits of `best_score` and the `max abs err` values.

Structural key census over the early-stopped artifact, run in both venvs:

```
$ ./venv_330/bin/python e08_census.py
xgboost: 3.3.0  version marker: [3, 3, 0]  keypaths: 50
keypaths containing 'best' or 'iteration':
    /learner/attributes/best_iteration
    /learner/attributes/best_score
    /learner/gradient_booster/model/iteration_indptr

$ ./venv_new/bin/python e08_census.py
xgboost: 3.4.0  version marker: [3, 4, 0]  keypaths: 50
keypaths containing 'best' or 'iteration':
    /learner/attributes/best_iteration
    /learner/attributes/best_score
    /learner/gradient_booster/model/iteration_indptr

$ diff keys_v330.txt keys_v340.txt
(no key differences)
```

50 key paths on both, zero diff. Unlike the `weight_drop` relocation in
`probes/version_drift.md` §3, **nothing early-stopping-related moved between 3.3.0 and
3.4.0.** Only the `version` marker changes, `[3,3,0]` → `[3,4,0]`.

The sklearn mechanism is also unchanged:

```
$ ./venv_new/bin/python -c "import inspect, xgboost as xgb; print(xgb.__version__); print(inspect.getsource(xgb.XGBModel._get_iteration_range))"
3.4.0
    def _get_iteration_range(
        self, iteration_range: Optional[IterationRange]
    ) -> IterationRange:
        if iteration_range is None or iteration_range[1] == 0:
            # Use best_iteration if defined.
            try:
                iteration_range = (0, self.best_iteration + 1)
            except AttributeError:
                iteration_range = (0, 0)
        if self.booster == "gblinear":
            iteration_range = (0, 0)
        return iteration_range
```

A 3.3.0-produced early-stopped artifact also loads and slices identically under 3.4.0
(`e07_slice.py` run under `venv_new` against `m_reg_npt3.json`, produced by 3.3.0, gave
byte-identical output to the 3.3.0 run).

**Version-dependence result: not version-dependent across 3.3.0 → 3.4.0.** It is
*API-path*-dependent instead, and that dependence is present in both versions.

---

## 8. What must the exporter do — one line

**Carry all trees and refuse to export, because the correct tree count is not a property
of the model: `Booster.predict()` uses all trees and `XGBRegressor.predict()` uses only
`iteration_indptr[best_iteration+1]` of them, on the same file, at `0/2500` agreement.**

Evidence, compressed:

| Ground truth | all trees | `best_iteration+1` iters | `best_iteration` iters | `best_iteration+1` **trees** |
|---|---|---|---|---|
| `Booster.predict()` default, npt=1 | **2500/2500** | 0/2500 | 0/2500 | 0/2500 |
| `Booster.predict()` default, npt=3 | **2500/2500** | 0/2500 | 0/2500 | 0/2500 |
| `XGBRegressor.predict()` default, npt=1 | 0/2500 | **2500/2500** | 0/2500 | **2500/2500** |
| `XGBRegressor.predict()` default, npt=3 | 0/2500 | **2500/2500** | 0/2500 | 0/2500 |
| `XGBClassifier.predict()` default, npt=1 | 0/2500 | **2500/2500** | 0/2500 | **2500/2500** |

D023's own tie-breaker — "If it is ambiguous, export raises" — is the clause that fires.
FORMAT.md §11 already raises on any model carrying `best_iteration`; this probe supplies
the measurement that justifies keeping that rule rather than replacing it, and supplies the
exact predicate and slice arithmetic for whichever escape hatch is chosen.

```
DECISION NEEDED: An early-stopped model has two correct tree counts; which is the artifact's
Context:  Booster.predict() default = ALL trees (2500/2500, max abs err 0.0).
          XGBRegressor/XGBClassifier.predict() default = trees[0:iteration_indptr[bi+1]]
          (2500/2500). The two disagree on 2500/2500 rows, max abs err 1.55 (regression)
          and 1.36 (binary). The SAME model file gives either answer depending only on
          whether it is loaded into a Booster or into an estimator (probes/early_stopping.md
          section 5.1). No field in the artifact distinguishes them. Reproduces identically
          on xgboost 3.3.0 and 3.4.0.
Options:  A) Keep FORMAT.md section 11 as written: raise on any model carrying
             best_iteration. Zero wrong numbers; refuses a common real workflow, and
             refuses even the harmless case in section 5.2 where best_iteration+1 already
             equals the round count.
          B) Raise only when it actually matters -- when
             iteration_indptr[best_iteration+1] != len(trees). Admits the section 5.2
             model (measured: both readings agree, 2500/2500 either way). Still refuses
             every genuinely truncated model.
          C) Require an explicit caller choice, e.g. an export-time argument naming which
             predict() the artifact must reproduce, with no default. Exports both
             workflows; the wrong choice is a silently wrong artifact, which is what this
             library exists to prevent -- though the choice is at least recorded.
          D) Require the caller to pre-slice: bst[0:bst.best_iteration+1], then export
             normally. Measured: slicing yields a model with attributes == {} and
             len(trees) == iteration_indptr[bi+1], on which Booster.predict() and the
             sklearn truncated answer AGREE at 2500/2500 (section 6.1). The ambiguity is
             resolved upstream of the exporter by an XGBoost operation, not by a guess
             inside it.
Lean:     B for the gate plus D as the documented escape hatch. B removes a false
          rejection that is provably harmless; D is the only option under which the
          exporter never has to choose between the two answers, because a sliced model has
          exactly one -- and it is XGBoost's own slicing that does the truncation, in
          iterations, so the exporter never implements the iteration-versus-tree
          arithmetic that section 3.2 shows is the silent-failure path.
Blocks:   FORMAT.md section 11 (the early-stopping raise), the D020 attributes whitelist
          (see below), and any fixture covering an early-stopped model.
```

```
DECISION NEEDED: Does best_iteration go on the D020 learner.attributes whitelist?
Context:  best_iteration survives every save/load path measured, as a JSON string, only at
          learner.attributes.best_iteration -- it is in no model param (section 1, key
          census: 3 matching key paths, none in learner_model_param, on both 3.3.0 and
          3.4.0). So it is readable for a gate. But under option D above the exporter
          never needs to STORE it, since a sliced model has attributes == {}. Note also
          that the sklearn path writes a THIRD key, scikit_learn, alongside
          best_iteration and best_score (section 5.3), and best_score is the
          full-precision nondeterministic string D020 was written to exclude.
Options:  A) Whitelist nothing; read best_iteration at export time for the gate only,
             never emit it. Preserves D008 byte-determinism trivially.
          B) Whitelist best_iteration and emit it as an integer. Artifact records the
             truncation point; the reader must then decide what to do with it, which
             re-imports the section 8 ambiguity into the JS predictor.
Lean:     A. The gate needs to READ it, which D020 does not restrict. Emitting it would
          push a two-answer field into the artifact and therefore into the JS reader.
Blocks:   The artifact schema, and the exporter's gate implementation.
```

---

## 9. Ambiguity, and how it was handled

- **Two readings of "what `predict()` uses", both measured, neither resolved here.** §2
  and §5. Presented as a `DECISION NEEDED` rather than a finding, per instruction. The
  probe does not pick one.
- **`iteration_indptr` stride uniformity.** Every model measured had a constant stride
  equal to `num_parallel_tree`. That is consistent with the stride always being constant,
  but the probe cannot exclude a configuration where it varies, so `iteration_indptr` is
  reported as authoritative and `num_parallel_tree × rounds` as a derived quantity that
  happened to agree.

## 10. Not measured — stated so nothing here is mistaken for a finding

- **`survival:cox`.** Not exercised. Nothing in this probe should be extended to it by
  analogy; the per-objective rule in CLAUDE.md applies.
- **`dart` and `gblinear`.** Not exercised. Note the pasted
  `_get_iteration_range` source contains a `gblinear` branch forcing `(0, 0)`; that is
  **read source, not measured behaviour**, and `gblinear` is refused by FORMAT.md §11
  anyway.
- **Custom `early_stopping` callbacks, `maximize=True`, `save_best=True`.** Only the
  `early_stopping_rounds=` parameter on `xgb.train` and on the estimator constructor was
  measured. A `save_best=True` callback is documented to trim the model and was **not**
  measured here; if it does trim, an early-stopped artifact could legitimately have
  `iteration_indptr[best_iteration+1] == len(trees)`, which is exactly the predicate
  option B keys on. **Worth a follow-up before the gate ships.** INFERRED that this is a
  possibility; not observed.
- **XGBoost versions other than 3.3.0 and 3.4.0.**
- **The JavaScript side.** No Node was run in this probe.

## 11. Out of scope but looked wrong

- The `scikit_learn` key that appears in `learner.attributes` when a model is saved through
  an estimator (§5.3) is a serialized Python type marker
  (`{"_estimator_type": "regressor"}`). A whitelist that pattern-matches loosely on
  `learner.attributes` will pick it up.
- `_get_iteration_range` treats `iteration_range[1] == 0` as "unset" and substitutes
  `best_iteration + 1`. Combined with the `(0, 0)`-means-all overload in the C API
  (`probes/base_score.md` §8), an explicit `iteration_range=(0, 0)` means **all trees**
  through `Booster.predict` and **`best_iteration + 1` iterations** through the estimator.
  Same argument, same model, two answers. Measured both: §4 line
  `iteration_range=(0, 0) -> 2500/2500 bit-exact vs default` on the Booster, and §5.1
  where the estimator's `None`/unset path truncates.
