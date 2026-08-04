# Probe: the export arity gate (evidence gap G3)

Closes `FORMAT.md` §14 gap **G3** — whether `num_class` can be `"1"` on a genuine
single-output model — and validates every field the arity gate of `FORMAT.md` §11 / D017
depends on, across the full range of in-scope model shapes.

**Every claim below is backed by a pasted command and its real output.** Anything not
directly measured is labelled `INFERRED`. Anything that admits two readings is presented as
both readings under *Ambiguities*, not resolved.

No library code was written. All fitted models and scripts live in scratch; nothing was
written into the repository except this file.

---

## 0. Environment

```
$ export PATH="$HOME/.local/bin:$PATH"
$ uv run python -c "
import sys, numpy, xgboost
print('python', sys.version)
print('xgboost', xgboost.__version__)
print('numpy', numpy.__version__)
import importlib
for m in ['sklearn','scipy','pandas','pyarrow']:
    try:
        mod=importlib.import_module(m); print(m, mod.__version__)
    except Exception as e: print(m,'ABSENT', type(e).__name__)
"
python 3.12.8 (main, Dec  3 2024, 18:42:41) [Clang 16.0.0 (clang-1600.0.26.4)]
xgboost 3.3.0
numpy 2.5.1
sklearn ABSENT ModuleNotFoundError
scipy 1.18.0
pandas ABSENT ModuleNotFoundError
pyarrow ABSENT ModuleNotFoundError
```

**`scikit-learn` is NOT installed and was NOT installed.** The sklearn-API route
(`XGBClassifier`) is therefore **NOT MEASURED**; see route R16 in §1. `xgboost.XGBClassifier`
is not merely untested here, it is unconstructible in this environment:

```
  xgb.XGBClassifier() itself raises: ImportError: sklearn needs to be installed in order to use this module
```

Unless a case required otherwise, every model: 400 rows × 4 generic columns `c0..c3`,
`numpy` `default_rng(20260802)`, generic synthetic normal features, `max_depth=2`, `eta=0.3`,
`seed=20260802`, `nthread=1`, `tree_method="exact"`, 3 boosting rounds. Generic feature
names, generic synthetic data, no domain vocabulary.

**Precision discipline.** Every threshold comparison in this probe casts *both* sides:
`np.float32(value) < np.float32(threshold)`. Leaf values and intercepts are `np.float32` and
accumulation is narrowed after every add, per `FORMAT.md` §10.

---

## 1. THE CORE QUESTION — answered: YES

> **`num_class` CAN be `"1"` on a genuine single-output model, for all three in-scope
> objectives, produced by ordinary training with no artifact editing, with no warning of any
> kind. `FORMAT.md` §11's `num_class == "0"` requirement therefore falsely rejects a valid
> model.**

The single sharpest piece of evidence — `binary:logistic`, `num_class=1` passed as an
ordinary training parameter:

```
$ uv run python g1_core_question.py
==============================================================================
ROUTE: R2 binary:logistic, num_class=1 passed explicitly
  params passed: {'tree_method': 'exact', 'max_depth': 2, 'eta': 0.3, 'seed': 20260802, 'nthread': 1, 'objective': 'binary:logistic', 'num_class': 1}  num_boost_round=3
  RESULT: trained and serialized OK
    objective.name           = 'binary:logistic'
    num_class                = '1'
    num_target               = '1'
    size_leaf_vector (trees) = ['1']
    base_score raw           = '[4.775E-1]'  len=1
    predict().shape          = (400,)
    predict(margin).shape    = (400,)
    len(trees)               = 3
    save_config num_class    = '<<ABSENT>>'
```

`num_target == "1"`, `size_leaf_vector == "1"`, one output per row, a one-element
`base_score` — every arity signal says single-output — and `num_class == "1"`.

### 1.1 Every route attempted, and its outcome

Auditable, not asserted. 16 routes.

| # | Route | Outcome | Serialized `num_class` |
|---|---|---|---|
| R1 | `binary:logistic`, `num_class` not passed | trained | `'0'` |
| **R2** | **`binary:logistic`, `num_class=1`** | **trained, single output** | **`'1'`** |
| R3 | `binary:logistic`, `num_class=2` | **RAISED** at train time | — |
| R4 | `binary:logistic`, `num_class=0` | trained | `'0'` |
| **R5** | **`binary:logistic`, `num_class="1"` (string)** | **trained, single output** | **`'1'`** |
| R6 | `multi:softprob`, `num_class=1`, 2-valued labels | **RAISED** (label range) | — |
| R7 | `multi:softmax`, `num_class=1`, 2-valued labels | **RAISED** (label range) | — |
| **R8** | **`reg:squarederror`, `num_class=1`** | **trained, single output** | **`'1'`** |
| **R9** | **`survival:cox`, `num_class=1`** | **trained, single output** | **`'1'`** |
| **R10** | **`binary:logistic`, `num_class=1` + `num_target=1`** | **trained, single output** | **`'1'`** |
| R11 | `binary:logitraw`, `num_class=1` (out of scope, probed anyway) | trained, single output | `'1'` |
| R12 | `binary:hinge`, `num_class=1` (out of scope, probed anyway) | trained, single output | `'1'` |
| R13 | `multi:softprob`, `num_class=2` | trained, **2 outputs** | `'2'` |
| **R14** | **`set_param("num_class", 1)` on a trained booster, then `save_model`** | **no exception, single output** | **`'1'`** |
| **R15** | **hand-edit artifact to `num_class:"1"`, `load_model`, predict, re-save** | **loads, predicts identically, re-serializes `"1"`** | **`'1'`** |
| R16 | sklearn API `XGBClassifier` on 2-class labels | **NOT MEASURED — `sklearn` absent, not installed** | — |
| R17 | `multi:softprob`, `num_class=1`, all labels `0` (§4 truth table) | trained, single output | `'1'` |
| R18 | `.ubj` save → load → `.json` re-save of the R2 model | round-trips | `'1'` |

Raw output for the routes that raised, so the negative results are checkable:

```
ROUTE: R3 binary:logistic, num_class=2 passed explicitly
  RESULT: RAISED XGBoostError: [00:23:29] .../src/objective/regression_obj.cu:51: Check failed: info.labels.Size() == preds.Size() (400 vs. 800) : Invalid shape of labels.

ROUTE: R6 multi:softprob, num_class=1
  RESULT: RAISED XGBoostError: [00:23:29] .../src/objective/multiclass_obj.cu:64: Check failed: valid: SoftmaxMultiClassObj: label must be discrete values in the range of [0, num_class).

ROUTE: R7 multi:softmax, num_class=1
  RESULT: RAISED XGBoostError: [00:23:29] .../src/objective/multiclass_obj.cu:64: Check failed: valid: SoftmaxMultiClassObj: label must be discrete values in the range of [0, num_class).
```

R6/R7 raise only because the labels were 2-valued. With all labels `0` the same
configuration trains and serializes `num_class == "1"` — see the truth-table row
`multi:softprob num_class=1 (all labels 0)` in §4, which reproduces the `softprob k=1`
finding of `probes/multiclass_extensibility.md` §7.

R14 and R15 verbatim:

```
ROUTE: R14 train binary:logistic, then bst.set_param('num_class', 1), then save_model
  set_param('num_class', 1) -> no exception
  serialized gate fields: {'objective_name': 'binary:logistic', 'num_class': '1', 'num_target': '1', 'size_leaf_vector_per_tree': ['1'], 'base_score_raw': '[4.775E-1]', 'n_trees': 3}
  predict().shape = (400,)
ROUTE: R15 hand-edit a binary:logistic artifact to num_class="1", load_model, predict, re-save
  load_model: no exception
  predict().shape = (400,)
  max|predict(edited) - predict(original)| = 0.000000000e+00
  re-saved gate fields: {'objective_name': 'binary:logistic', 'num_class': '1', 'num_target': '1', 'size_leaf_vector_per_tree': ['1'], 'base_score_raw': '[4.775E-1]', 'n_trees': 3}
```

```
E. UBJSON round-trip: does num_class=1 survive .ubj -> .json?
   after .ubj save + load + json re-save: num_class = '1'  num_target = '1'  objective = 'binary:logistic'
   predict shape (400,)
```

### 1.2 The `num_class="1"` model is the SAME model, not a different one

R1 (`num_class` absent) and R2 (`num_class=1`) differ in **exactly one leaf path in the whole
artifact**, and the trees are byte-identical:

```
$ uv run python g2_equivalence.py
A. R1 (num_class absent) vs R2 (num_class=1): full artifact diff
  sha256 R1 = c2f0ae02d4af9a26307dbcf21b1388a25786db55d7e05102f66e6dd3310b1a02
  sha256 R2 = 39d86f101ef9eda8c05d9872f7c484e343d87fd2b04082b7c085433d426c3520
  bytes identical: False
  total leaf paths: R1=300 R2=300
  paths that DIFFER: 1
    $.learner.learner_model_param.num_class
      R1 = '0'
      R2 = '1'

B. trees array byte-identical between R1 and R2?
  identical: True
  tree_info R1=[0, 0, 0]  R2=[0, 0, 0]

C. predictions from the two RELOADED artifacts
  R1 num_class absent    margin.shape=(400,) predict.shape=(400,) margin[:3]=[-1.3047063 -1.3047063  1.0285069]
  R2 num_class=1         margin.shape=(400,) predict.shape=(400,) margin[:3]=[-1.3047063 -1.3047063  1.0285069]
  max|margin_R1 - margin_R2| = 0.000000000e+00
  bitwise identical margins  : True
```

**The `num_class="1"` model is bit-for-bit the same predictor as the `num_class="0"` model.**
One string in `learner_model_param` differs; nothing else in 300 leaf paths does.

### 1.3 The `num_class="1"` models are exportable at exactly `0.0`

This is what makes the rejection *false* rather than a lucky escape. The `FORMAT.md` §6 + §10
normative walk — float32 both sides, per-objective intercept, narrow after every add —
reproduces `predict(output_margin=True)` bit-exactly on all three `num_class="1"` artifacts:

```
$ uv run python g5_walk_warn_zeroround.py
A. FORMAT.md 10 reference walk on the num_class='1' artifacts vs predict(output_margin=True)
  binary:logistic  num_class absent  num_class='0'  max|walk - predict| = 0.000000000e+00   bit-exact rows = 400/400
  binary:logistic  num_class=1       num_class='1'  max|walk - predict| = 0.000000000e+00   bit-exact rows = 400/400
  reg:squarederror num_class=1       num_class='1'  max|walk - predict| = 0.000000000e+00   bit-exact rows = 400/400
  survival:cox     num_class=1       num_class='1'  max|walk - predict| = 0.000000000e+00   bit-exact rows = 400/400
```

`400/400` bit-exact, max abs error `0.0`, on models the specified gate refuses.

### 1.4 Nothing warns

XGBoost emits no Python warning and no C++ log line when `num_class=1` is passed to
`binary:logistic`. Measured in a subprocess so the C++ `stderr` channel is captured:

```
$ uv run python g5_walk_warn_zeroround.py
B. does XGBoost warn or log anything when num_class=1 is passed to binary:logistic?
  --- stdout ---
  PYTHON WARNINGS CAPTURED: 0
  serialized num_class: 1
  --- stderr (XGBoost C++ logs go here) ---
  <<EMPTY: no C++ warning emitted>>
  returncode = 0
```

This silence is specific, not general — XGBoost 3.3.0 does warn about other things through
the same channel, so the absence is informative:

```
UserWarning: [00:30:57] WARNING: .../src/learner.cc:341: `booster=dart` is deprecated. Use the tree booster directly with dropout parameters like `rate_drop`, `skip_drop`, or `one_drop`.
```

### 1.5 No route was found where XGBoost writes `"1"` on its own

Every producing route required the *caller* to supply `num_class` (as a training parameter,
via `set_param`, or by editing the artifact). Nothing XGBoost does by itself produced `"1"`:

```
F. any route where XGBoost itself writes num_class="1" WITHOUT the caller supplying num_class?
   default binary         num_class='0'
   scale_pos_weight=3     num_class='0'
   eval_metric=auc        num_class='0'
   booster=dart           num_class='0'
   num_parallel_tree=4    num_class='0'
   max_delta_step=1       num_class='0'
   objective=reg:logistic num_class='0'
```

That is a **negative result over 7 parameter variations**, not an exhaustive proof. `INFERRED`,
not measured: whether some untested parameter, wrapper, or third-party tool sets `num_class`
implicitly. What is measured and sufficient for G3 is that a *caller* can trivially do it and
that a hyperparameter dict carrying `num_class` alongside a binary objective — an entirely
ordinary thing in a sweep harness — is enough.

### 1.6 `num_class` IS load-bearing, and `"0"` and `"1"` mean the same thing

Editing only `num_class` on a fixed 3-tree binary artifact changes the output arity XGBoost
itself reports:

```
E. is num_class LOAD-BEARING at predict time? hand-edit num_class on a binary model
  num_class="0" -> loaded OK, margin.shape=(400,), max|m - m(nc=0)|=0.000000000e+00
  num_class="1" -> loaded OK, margin.shape=(400,), max|m - m(nc=0)|=0.000000000e+00
  num_class="2" -> loaded OK, margin.shape=(400, 2), max|m - m(nc=0)|=0.000000000e+00
  num_class="3" -> loaded OK, margin.shape=(400, 3), max|m - m(nc=0)|=0.000000000e+00
```

`"0"` and `"1"` both yield `(400,)`; `"2"` and `"3"` yield `(400, 2)` and `(400, 3)`. So on
this measurement the single-output condition is `num_class in {"0", "1"}`, and the
multi-output condition is `num_class >= 2`.

The asymmetry at the other end: `"0"` is *rejected* on a genuine multi-class artifact, while
`"1"` is accepted and silently collapses it to one output.

```
F. reverse direction: hand-edit a genuine 3-class model down to num_class="1"
  genuine 3-class: num_class='3' num_trees=9 tree_info=[0, 1, 2, 0, 1, 2, 0, 1, 2] margin.shape=(400, 3)
  3-class artifact forced to num_class="0": RAISED XGBoostError: value 0 for Parameter num_class should be greater equal to 1
  3-class artifact forced to num_class="1": loaded OK, margin.shape=(400,)
```

---

## 2. On-disk JSON type of all four gate fields — confirmed: three strings, one string

`probes/tree_structure.md` found these are JSON strings. **Confirmed.** All four gate fields
are JSON **strings**, quoted in the file bytes, and `str` after `json.load`.

```
$ uv run python g3_types.py
==============================================================================
A. binary:logistic, num_class absent (num_class serialized '0')   (m_r1.json)
  --- raw bytes: learner_model_param, verbatim slice of the file ---
  "learner_model_param":{"base_score":"[4.775E-1]","boost_from_average":"1","num_class":"0","num_feature":"4","num_target":"1"}
  --- raw bytes: objective block, verbatim ---
  "objective":{"name":"binary:logistic","reg_loss_param":{"scale_pos_weight":"1"}}
  --- raw bytes: tree_param of tree 0, verbatim ---
  "tree_param":{"num_deleted":"0","num_feature":"4","num_nodes":"7","size_leaf_vector":"1"}
  --- python types after json.load ---
  num_class                          type=str   repr='0'
  num_target                         type=str   repr='1'
  objective.name                     type=str   repr='binary:logistic'
  trees[0].tree_param.size_leaf_vector type=str   repr='1'
  --- string-vs-integer comparison, actually evaluated ---
  num_class == "0"  -> True
  num_class == 0    -> False          <-- integer comparison
  num_target == "1" -> True
  num_target == 1   -> False          <-- integer comparison
  size_leaf_vector == "1" -> True
  size_leaf_vector == 1   -> False          <-- integer comparison
```

| Gate field | JSON path | On-disk form | Type after `json.load` |
|---|---|---|---|
| `objective.name` | `$.learner.objective.name` | `"binary:logistic"` | `str` |
| `num_class` | `$.learner.learner_model_param.num_class` | `"0"` — quoted digit | `str` |
| `num_target` | `$.learner.learner_model_param.num_target` | `"1"` — quoted digit | `str` |
| `size_leaf_vector` | `$.learner.gradient_booster.model.trees[i].tree_param.size_leaf_vector` | `"1"` — quoted digit | `str` |

**A gate comparing these against integers never fires.** Evaluated, not reasoned about:
`num_class == 0` is `False` on a model whose `num_class` is `"0"`. A gate written
`num_class == 0 and num_target == 1 and size_leaf_vector == 1` would reject **every** model,
including all three in-scope objectives — a total false-rejection, which at least fails
loudly. The dangerous inverse is a gate written as an inequality or a truthiness test:
`int(num_class) != 0`-style code is correct, but `if num_class:` is `True` for the string
`"0"`, and `num_class > 0` on a `str` raises `TypeError` in Python 3.

### 2.1 `size_leaf_vector` has no learner-level home — it exists once per tree only

This is a structural fact the gate depends on and it was not previously recorded. Census over
all 20 artifacts on disk at that point:

```
C. WHERE does size_leaf_vector live? exhaustive key census over every fitted artifact
  m_edit_3to0.json           trees=  9  occurrences of "size_leaf_vector" in file =   9  size_leaf_vector in learner_model_param = False
  m_edit_nc1.json            trees=  3  occurrences of "size_leaf_vector" in file =   3  size_leaf_vector in learner_model_param = False
  m_r1.json                  trees=  3  occurrences of "size_leaf_vector" in file =   3  size_leaf_vector in learner_model_param = False
  m_r13.json                 trees=  6  occurrences of "size_leaf_vector" in file =   6  size_leaf_vector in learner_model_param = False
  m_softprob3.json           trees=  9  occurrences of "size_leaf_vector" in file =   9  size_leaf_vector in learner_model_param = False
  ...  (all 20 artifacts: occurrences == len(trees), never in learner_model_param)

  learner_model_param key set (m_r1.json): ['base_score', 'boost_from_average', 'num_class', 'num_feature', 'num_target']
```

Occurrences `==` number of trees in every artifact, and `size_leaf_vector` is never a member
of `learner_model_param`. **Consequence: a zero-tree model contains no `size_leaf_vector`
field anywhere in the document.** `FORMAT.md` §11 requires `size_leaf_vector == "1"` without
saying what to do when the field does not exist. That is the ambiguity A2 in §5 and the reason
§4 scores two readings of the gate.

---

## 3. `num_class` is not the only field the gate needs — the other three, measured

Nothing in §4 relies on inference about what these fields mean. For completeness of the gate
surface:

- `num_target == "2"` is produced by fitting a 2-column label matrix under
  `reg:squarederror` — an objective **on** the allow-list — and gives `(400, 2)` margins.
  Reproduces `probes/multiclass_extensibility.md` §7.
- `binary:logistic` with `num_target=2` is **not** producible: it raises at train time.
  ```
  binary:logistic, num_target=2 (attempt)
      RAISED: [00:26:06] .../src/common/error_msg.cc:109: Invalid `base_score`, it should match the number of outputs for multi-class/target models. `base_score` len: 1, `n_targets`: 2
  ```
- `size_leaf_vector == "2"` / `"3"` is produced only by `multi_strategy="multi_output_tree"`,
  which additionally requires `tree_method="hist"`. Both vector-leaf shapes appear in §4.

---

## 4. Truth table

Gate implemented **exactly as `FORMAT.md` §11 specifies**:

```python
ALLOW = {"reg:squarederror", "binary:logistic", "survival:cox"}
gate = (objective_name in ALLOW) and num_target == "1" \
       and <size_leaf_vector == "1"> and num_class == "0"
```

`size_leaf_vector` is per-tree (§2.1), so the specification's scalar comparison needs a
quantifier the specification does not supply. Both readings are scored:

- **reading A**, vacuous-true: `all(t.size_leaf_vector == "1" for t in trees)` — `True` when
  `trees == []`.
- **reading B**, must-exist: `sorted(set(...)) == ["1"]` — `False` when `trees == []`.

They differ on exactly the two zero-round rows and agree everywhere else.

`outs` is measured, not assumed: `1` if `predict(output_margin=True).ndim == 1`, else
`shape[1]`.

```
$ uv run python g4_truth_table.py
case                                                 objective.name     nc    nt   slv        trees  outs  gateA   gateB   truth        verdict
------------------------------------------------------------------------------------------------------------------------------------------------------
reg:squarederror, 1 target                           reg:squarederror   0     1    ['1']      3      1     ACCEPT  ACCEPT  ACCEPT       CORRECT
binary:logistic                                      binary:logistic    0     1    ['1']      3      1     ACCEPT  ACCEPT  ACCEPT       CORRECT
survival:cox                                         survival:cox       0     1    ['1']      3      1     ACCEPT  ACCEPT  ACCEPT       CORRECT
reg:squarederror, num_target=2                       reg:squarederror   0     2    ['1']      6      2     REJECT  REJECT  REJECT       CORRECT
multi:softprob num_class=3                           multi:softprob     3     1    ['1']      9      3     REJECT  REJECT  REJECT       CORRECT
multi:softmax num_class=3                            multi:softmax      3     1    ['1']      9      3     REJECT  REJECT  REJECT       CORRECT
multi:softprob num_class=1 (all labels 0)            multi:softprob     1     1    ['1']      3      1     REJECT  REJECT  open         OPEN
multi:softprob k=3 + multi_output_tree               multi:softprob     3     1    ['3']      3      3     REJECT  REJECT  REJECT       CORRECT
reg 2-target + multi_output_tree                     reg:squarederror   0     2    ['2']      3      2     REJECT  REJECT  REJECT       CORRECT
reg:squarederror, 0 boosting rounds                  reg:squarederror   0     1    <<no trees 0      1     ACCEPT  REJECT  open         OPEN
binary:logistic, 0 boosting rounds                   binary:logistic    0     1    <<no trees 0      1     ACCEPT  REJECT  open         OPEN
reg:squarederror, 1 feature                          reg:squarederror   0     1    ['1']      3      1     ACCEPT  ACCEPT  ACCEPT       CORRECT
reg:squarederror, 1 training row                     reg:squarederror   0     1    ['1']      3      1     ACCEPT  ACCEPT  ACCEPT       CORRECT
binary:logistic, 1 feature + 1 row                   binary:logistic    0     1    ['1']      3      1     ACCEPT  ACCEPT  ACCEPT       CORRECT
reg:squarederror, gamma=1e9 (leaf-only pruned trees) reg:squarederror   0     1    ['1']      3      1     ACCEPT  ACCEPT  ACCEPT       CORRECT
binary:logistic, gamma=1e9 (leaf-only pruned trees)  binary:logistic    0     1    ['1']      3      1     ACCEPT  ACCEPT  ACCEPT       CORRECT
binary:logistic + num_class=1  <<G3>>                binary:logistic    1     1    ['1']      3      1     REJECT  REJECT  ACCEPT       *** WRONG ***
reg:squarederror + num_class=1  <<G3>>               reg:squarederror   1     1    ['1']      3      1     REJECT  REJECT  ACCEPT       *** WRONG ***
survival:cox + num_class=1  <<G3>>                   survival:cox       1     1    ['1']      3      1     REJECT  REJECT  ACCEPT       *** WRONG ***
binary:logistic, num_parallel_tree=4                 binary:logistic    0     1    ['1']      12     1     ACCEPT  ACCEPT  ACCEPT       CORRECT
reg:squarederror, num_target=2 + num_class=1         reg:squarederror   1     2    ['1']      6      2     REJECT  REJECT  REJECT       CORRECT
binary:logistic, num_target=2 (attempt)              RAISED AT TRAIN TIME: [00:26:06] /Users/runner/work/xgboost/xgboost/src/common/error_msg.cc:
reg:squarederror, hist tree_method                   reg:squarederror   0     1    ['1']      3      1     ACCEPT  ACCEPT  ACCEPT       CORRECT
```

Per-row detail (`tree_info`, `len(base_score)`, `margin.shape`) for the four most load-bearing
rows, from the same run:

```
reg:squarederror, num_target=2
    objective.name='reg:squarederror'  num_class='0'  num_target='2'  size_leaf_vector=['1']
    len(trees)=6  tree_info[:8]=[0, 1, 0, 1, 0, 1]  margin.shape=(400, 2)  outputs=2  len(base_score)=2
multi:softprob num_class=1 (all labels 0)
    objective.name='multi:softprob'  num_class='1'  num_target='1'  size_leaf_vector=['1']
    len(trees)=3  tree_info[:8]=[0, 0, 0]  margin.shape=(400,)  outputs=1  len(base_score)=1
reg 2-target + multi_output_tree
    objective.name='reg:squarederror'  num_class='0'  num_target='2'  size_leaf_vector=['2']
    len(trees)=3  tree_info[:8]=[0, 0, 0]  margin.shape=(400, 2)  outputs=2  len(base_score)=2
binary:logistic + num_class=1  <<G3>>
    objective.name='binary:logistic'  num_class='1'  num_target='1'  size_leaf_vector=['1']
    len(trees)=3  tree_info[:8]=[0, 0, 0]  margin.shape=(400,)  outputs=1  len(base_score)=1
```

### 4.1 Rows where the gate's verdict is WRONG

**Three rows. All three in-scope objectives. All are false rejections.**

| Row | Gate | Truth | Why the truth is what it is |
|---|---|---|---|
| `binary:logistic + num_class=1` | REJECT | must ACCEPT | 1 output, `num_target="1"`, `size_leaf_vector="1"`, 1-element `base_score`, `tree_info=[0,0,0]`, and the §10 walk reproduces `predict` at `0.0` on 400/400 rows (§1.3) |
| `reg:squarederror + num_class=1` | REJECT | must ACCEPT | same, `0.0` on 400/400 |
| `survival:cox + num_class=1` | REJECT | must ACCEPT | same, `0.0` on 400/400 |

No row in the table is a **false acceptance**. Every multi-output shape is rejected by at
least one of the four conditions, and the two vector-leaf shapes are rejected twice over.
`reg:squarederror, num_target=2 + num_class=1` — a shape that defeats a `num_class`-only
check *and* an objective-only check — is still correctly rejected by `num_target`.

### 4.2 The specific hole `num_class in {"0","1"}` would open, and its size

Relaxing `num_class == "0"` to `num_class in {"0", "1"}` fixes all three WRONG rows and, on
this table, admits exactly one additional shape: `multi:softprob num_class=1`, `num_class="1"`,
1 output, structurally indistinguishable from a single-group model. That shape is **still
rejected**, by the objective allow-list — `multi:softprob` is not in it. So on the 23 shapes
measured here the relaxation costs nothing.

That is a measurement over 23 shapes, not a proof. Deciding it is not this probe's call; see
ambiguity A1.

---

## 5. Is any valid in-scope shape falsely rejected? — YES, three of them

Stated as plainly as the brief asks for:

> **Yes. `FORMAT.md` §11 as written falsely rejects `reg:squarederror`, `binary:logistic`, and
> `survival:cox` models carrying `num_class == "1"`. Each is a genuine single-output model.
> Each is exportable and reproduces `predict(output_margin=True)` bit-exactly at `0.0` on
> 400/400 rows under the `FORMAT.md` §10 walk. Each is producible by ordinary training with
> one extra parameter, with no warning from XGBoost.**

Under **reading B** of `size_leaf_vector` (must-exist), two more valid in-scope shapes are
falsely rejected: the zero-boosting-round `reg:squarederror` and `binary:logistic` models,
because a zero-tree artifact contains no `size_leaf_vector` field to compare against (§2.1).
`FORMAT.md` §13 explicitly permits an empty ensemble and §6.1 *requires* a zero-tree fixture
in the corpus, so reading B contradicts the rest of the specification. Under **reading A**
those two rows are accepted. The specification does not say which reading is meant — ambiguity
A2.

Every other in-scope shape measured is correctly accepted, including the single-feature,
single-row, `num_parallel_tree=4`, `hist`, and gamma-pruned-leaf-only cases.

---

## 6. Zero-boosting-round models: `trees` is an EMPTY ARRAY, present, never absent

`trees` is present and is `[]`. It is **not** absent. Measured on all three in-scope
objectives, with the entire file pasted:

```
$ uv run python g5_walk_warn_zeroround.py
C. ZERO BOOSTING ROUNDS: is `trees` an empty array or absent? raw file bytes.
  --- reg:squarederror, num_boost_round=0 ---
  RAW FILE (523 bytes):
  {"learner":{"attributes":{},"feature_names":["c0","c1","c2","c3"],"feature_types":[],"gradient_booster":{"model":{"cats":{"enc":[],"feature_segments":[],"sorted_idx":[]},"gbtree_model_param":{"num_parallel_tree":"1","num_trees":"0"},"iteration_indptr":[0],"tree_info":[],"trees":[]},"name":"gbtree"},"learner_model_param":{"base_score":"[5E-1]","boost_from_average":"1","num_class":"0","num_feature":"4","num_target":"1"},"objective":{"name":"reg:squarederror","reg_loss_param":{"scale_pos_weight":"1"}}},"version":[3,3,0]}
  'trees' key present in gradient_booster.model : True
  type(model['trees'])                          : list
  repr(model['trees'])                          : []
  substring '"trees":[]' present in file bytes  : True
  occurrences of '"size_leaf_vector"' in file    : 0
  gate fields: objective.name='reg:squarederror' num_class='0' num_target='1' size_leaf_vector=<<no trees, field absent from the whole document>>
  tree_info=[]  iteration_indptr=[0] num_trees='0'
  base_score='[5E-1]'  predict[:3]=[0.5 0.5 0.5] margin.shape=(400,)

  --- binary:logistic, num_boost_round=0 ---
  RAW FILE (522 bytes):
  {"learner":{"attributes":{},"feature_names":["c0","c1","c2","c3"],"feature_types":[],"gradient_booster":{"model":{"cats":{"enc":[],"feature_segments":[],"sorted_idx":[]},"gbtree_model_param":{"num_parallel_tree":"1","num_trees":"0"},"iteration_indptr":[0],"tree_info":[],"trees":[]},"name":"gbtree"},"learner_model_param":{"base_score":"[5E-1]","boost_from_average":"1","num_class":"0","num_feature":"4","num_target":"1"},"objective":{"name":"binary:logistic","reg_loss_param":{"scale_pos_weight":"1"}}},"version":[3,3,0]}
  'trees' key present in gradient_booster.model : True
  type(model['trees'])                          : list
  repr(model['trees'])                          : []
  substring '"trees":[]' present in file bytes  : True
  occurrences of '"size_leaf_vector"' in file    : 0
  tree_info=[]  iteration_indptr=[0] num_trees='0'
  base_score='[5E-1]'  predict[:3]=[0.62245935 0.62245935 0.62245935] margin.shape=(400,)

  --- survival:cox, num_boost_round=0 ---
  RAW FILE (477 bytes):
  {"learner":{"attributes":{},"feature_names":["c0","c1","c2","c3"],"feature_types":[],"gradient_booster":{"model":{"cats":{"enc":[],"feature_segments":[],"sorted_idx":[]},"gbtree_model_param":{"num_parallel_tree":"1","num_trees":"0"},"iteration_indptr":[0],"tree_info":[],"trees":[]},"name":"gbtree"},"learner_model_param":{"base_score":"[5E-1]","boost_from_average":"1","num_class":"0","num_feature":"4","num_target":"1"},"objective":{"name":"survival:cox"}},"version":[3,3,0]}
  'trees' key present in gradient_booster.model : True
  type(model['trees'])                          : list
  repr(model['trees'])                          : []
  substring '"trees":[]' present in file bytes  : True
  occurrences of '"size_leaf_vector"' in file    : 0
  tree_info=[]  iteration_indptr=[0] num_trees='0'
  base_score='[5E-1]'  predict[:3]=[1.6487212 1.6487212 1.6487212] margin.shape=(400,)
```

**Which a reader must handle: empty, not absent.** `"trees":[]` on 3/3 objectives, plus
`tree_info: []`, `iteration_indptr: [0]` (not `[]`), `num_trees: "0"`. This reproduces
`probes/tree_structure.md` §7(c) and extends it from `reg:squarederror` to all three in-scope
objectives. `INFERRED`, not measured: whether some other code path can omit the `trees` key
entirely. A reader following D007 should raise on an absent `trees` key rather than treat it
as empty, since absence was never observed.

Gate fields for a zero-round model: `objective.name` in the allow-list, `num_class == "0"`,
`num_target == "1"`, and **`size_leaf_vector` does not exist** — `occurrences ... = 0`. So
three of the four gate conditions are satisfiable and the fourth is unevaluable. Whether the
gate accepts it depends entirely on how the missing field is quantified (§4, readings A/B).

Zero rounds and `num_class=1` compose, giving a shape that trips both open questions at once:

```
D. zero rounds WITH num_class=1 (both open questions at once)
  RAW FILE:
  {"learner":{"attributes":{},...,"trees":[]},"name":"gbtree"},"learner_model_param":{"base_score":"[5E-1]","boost_from_average":"1","num_class":"1","num_feature":"4","num_target":"1"},"objective":{"name":"binary:logistic",...}},"version":[3,3,0]}
```

---

## 7. Out of scope, and it looks wrong: the zero-tree intercept is NOT in link space

Found while measuring the zero-round row and reported loudly because it breaches the
`1e-6` margin gate on a fixture `FORMAT.md` §6.1 **requires** the corpus to contain. This is
G1 / D015 territory, not mine; I measured it and I am not resolving it.

**On a zero-tree model with `boost_from_average == "1"`, XGBoost's margin is the RAW
`base_score` value, not the per-objective link transform.** For `binary:logistic` the
`FORMAT.md` §6 rule gives `-0.0` while the actual margin is `0.5`:

```
$ uv run python g8_zero_round_diff.py
B. FORMAT.md 6+10 walk on ZERO-TREE models (no trees, so margin == intercept)
objective         base_score passed?   serialized bs   FORMAT 6 intercept   xgboost margin   abs error      1e-6 gate
---------------------------------------------------------------------------------------------------------------------
reg:squarederror  NOT passed           [5E-1]          0.5                  0.5              0.000000000e+00 PASS
reg:squarederror  passed 0.5           [5E-1]          0.5                  0.5              0.000000000e+00 PASS
reg:squarederror  passed 0.8           [8E-1]          0.800000012          0.800000012      0.000000000e+00 PASS
binary:logistic   NOT passed           [5E-1]          -0                   0.5              5.000000000e-01 *** BREACH ***
binary:logistic   passed 0.5           [5E-1]          -0                   -0               0.000000000e+00 PASS
binary:logistic   passed 0.8           [8E-1]          1.38629436           1.38629436       0.000000000e+00 PASS
survival:cox      NOT passed           [5E-1]          -0.693147182         0.5              1.193147182e+00 *** BREACH ***
survival:cox      passed 0.5           [5E-1]          -0.693147182         -0.693147182     0.000000000e+00 PASS
survival:cox      passed 0.8           [8E-1]          -0.223143533         -0.223143533     0.000000000e+00 PASS
```

`0.5` and `1.19` in margin space, with no error raised — this project's exact failure
signature.

**The artifact does determine the prediction; nothing unserialized is involved.** In-memory
and reloaded boosters agree in every case, and the two variants are not byte-identical:

```
$ uv run python g7_zero_round_intercept.py
OBJECTIVE binary:logistic, num_boost_round=0
  --- base_score NOT passed ---
    serialized base_score      = '[5E-1]'
    margin from IN-MEMORY bst  = 0.5   bits=0x3f000000   all rows equal: True
    margin from RELOADED bst   = 0.5   bits=0x3f000000   all rows equal: True
    in-memory == reloaded      : True
  --- base_score=0.5 passed ---
    serialized base_score      = '[5E-1]'
    margin from IN-MEMORY bst  = -0   bits=0x80000000   all rows equal: True
    margin from RELOADED bst   = -0   bits=0x80000000   all rows equal: True
    in-memory == reloaded      : True
  --- comparison between the two variants ---
    artifacts byte-identical                : False
    max|margin(no bs) - margin(bs=0.5)|     : 5.000000000e-01
```

**The discriminating byte is `boost_from_average`.** One leaf path differs between the two
zero-round variants, on all three objectives:

```
$ uv run python g8_zero_round_diff.py
A. artifact diff: zero-round, base_score NOT passed  vs  base_score=0.5 passed
  binary:logistic: 1 differing leaf path(s)
    $.learner.learner_model_param.boost_from_average:  no-bs = '1'   bs=0.5 = '0'
  survival:cox: 1 differing leaf path(s)
    $.learner.learner_model_param.boost_from_average:  no-bs = '1'   bs=0.5 = '0'
  reg:squarederror: 1 differing leaf path(s)
    $.learner.learner_model_param.boost_from_average:  no-bs = '1'   bs=0.5 = '0'

  raw learner_model_param bytes, binary:logistic, both variants:
    nobs: "learner_model_param":{"base_score":"[5E-1]","boost_from_average":"1","num_class":"0","num_feature":"4","num_target":"1"}
    bs05: "learner_model_param":{"base_score":"[5E-1]","boost_from_average":"0","num_class":"0","num_feature":"4","num_target":"1"}
```

`boost_from_average` is **load-bearing at predict time** on a zero-tree model — flipping that
one string changes the margin:

```
$ uv run python g9_bfa.py
B. is boost_from_average load-bearing at predict time? hand-edit it on a 0-round artifact
  0-round artifact with boost_from_average="0", base_score='[5E-1]' -> margin[0] = -0  bits=0x80000000
  0-round artifact with boost_from_average="1", base_score='[5E-1]' -> margin[0] = 0.5  bits=0x3f000000
```

And it is genuinely "raw versus link", not a `0.5` coincidence — swept across four
`base_score` values on both non-identity objectives:

```
D. is the zero-tree bfa=1 margin RAW base_score, or a 0.5 coincidence? sweep base_score by hand-edit
objective         bfa  base_score   margin[0]        raw f32(bs)    logit/ln       match
binary:logistic   1    [5E-1]       0.5              0.5            -0             ['raw']
binary:logistic   0    [5E-1]       -0               0.5            -0             ['link']
binary:logistic   1    [8E-1]       0.800000012      0.800000012    1.38629436     ['raw']
binary:logistic   0    [8E-1]       1.38629436       0.800000012    1.38629436     ['link']
binary:logistic   1    [2.5E-1]     0.25             0.25           -1.09861231    ['raw']
binary:logistic   0    [2.5E-1]     -1.09861231      0.25           -1.09861231    ['link']
binary:logistic   1    [1E0]        1                1              inf            ['raw']
binary:logistic   0    [1E0]        13.7451601       1              inf            []
survival:cox      1    [5E-1]       0.5              0.5            -0.693147182   ['raw']
survival:cox      0    [5E-1]       -0.693147182     0.5            -0.693147182   ['link']
survival:cox      1    [8E-1]       0.800000012      0.800000012    -0.223143533   ['raw']
survival:cox      0    [8E-1]       -0.223143533     0.800000012    -0.223143533   ['link']
survival:cox      1    [2.5E-1]     0.25             0.25           -1.38629436    ['raw']
survival:cox      0    [2.5E-1]     -1.38629436      0.25           -1.38629436    ['link']
survival:cox      1    [1E0]        1                1              0              ['raw']
survival:cox      0    [1E0]        0                1              0              ['link']
```

**`boost_from_average == "1"` alone is NOT the trigger.** With trees present it is `"1"` and
the link transform is correct — verified by full `FORMAT.md` §10 walk, not by subtraction:

```
$ uv run python g9_bfa.py
A. boost_from_average / base_score / implied-intercept space, binary:logistic, rounds 0..3
rounds  bfa   base_score     num_trees  margin(0 trees only)   space that matches
0 no bs        1     [5E-1]         0          0.5                    raw:err=0.000e+00  logit:err=5.000e-01
0 bs=0.5       0     [5E-1]         0          -0                     raw:err=5.000e-01  logit:err=0.000e+00
1 no bs        1     [4.775E-1]     1          -0.504692912           raw:err=5.676e-01  logit:err=0.000e+00
1 bs=0.5       0     [5E-1]         1          -0.44034335            raw:err=5.000e-01  logit:err=0.000e+00
3 no bs        1     [4.775E-1]     3          -1.30470634            raw:err=5.676e-01  logit:err=0.000e+00
3 bs=0.5       0     [5E-1]         3          -1.25993228            raw:err=5.000e-01  logit:err=0.000e+00
```

**Measured rule (four cells, all four observed):**

| `trees` | `boost_from_average` | margin intercept |
|---|---|---|
| `[]` | `"1"` | **raw `f32(base_score)`**, no transform, all three objectives |
| `[]` | `"0"` | per-objective link transform |
| non-empty | `"1"` | per-objective link transform |
| non-empty | `"0"` | per-objective link transform |

The anomalous cell is exactly one: zero trees **and** `boost_from_average == "1"`. In this
probe's artifact inventory that cell is populated by 12 artifacts, and the other cells by 46
and 4:

```
C. boost_from_average across every artifact fitted in this probe
  boost_from_average='0'  zero_trees=True  -> 4 artifact(s), e.g. ['bfa_0.json', 'iso_binary_logistic_bs05.json', ...]
  boost_from_average='1'  zero_trees=False  -> 46 artifact(s), e.g. ['flip_1.json', 'flip_2.json', 'flip_3.json']
  boost_from_average='1'  zero_trees=True  -> 12 artifact(s), e.g. ['bfa_1.json', 'flip_0.json', 'flipcox_0.json']
```

Bearing on `FORMAT.md`, stated as facts and not as recommendations:

1. §6.1's required fixture — "a zero-tree model ... at `base_score = 0.5` for
   `binary:logistic`" — has an intercept of `0.5`, **not** `-0.0`, if `base_score` is left at
   its default. It has `-0.0` only if `base_score=0.5` is passed **explicitly**. The two are
   distinguishable in the artifact solely by `boost_from_average`. A corpus generator that
   omits the explicit `base_score` builds the fixture in the wrong space, and it will not
   exercise the signed-zero path §6.1 exists to test.
2. §11's "Intercept agreement (D015)" check — "raise if the derived `intercept` disagrees with
   the transform of `provenance.base_score`" — **cannot catch this.** Both sides of that
   comparison are the same transform of the same value; it is self-consistent and wrong
   together.
3. §15 lists `boost_from_average` nowhere. It is currently neither carried nor read, yet on a
   zero-tree model it is the only field that determines the intercept space.

Also visible in that sweep and worth one line: `binary:logistic` with `base_score = 1.0` and
`boost_from_average = "0"` gives margin `13.7451601`, not `inf`. `FORMAT.md` §6's expression
yields `inf`, which §9.3 requires a reader to raise on. XGBoost clamps. Not investigated
further; it belongs to G1.

---

## Ambiguities — presented, not resolved

**A1. Is refusing `num_class == "1"` a false rejection or correct D007 strictness?**
*Reading 1:* it is a false rejection. The model is single-output on every measured signal, is
bit-identical to its `num_class="0"` twin (§1.2), and exports at `0.0` (§1.3). Refusing it
rejects a valid model, and per the brief a false rejection is the worse failure direction
because it looks like correct strictness. The fix is `num_class in {"0", "1"}`, which admitted
nothing extra across 23 shapes (§4.2). *Reading 2:* `num_class == "1"` under a non-`multi:*`
objective is a value XGBoost never writes on its own (§1.5) — it only appears because a caller
supplied a parameter that objective does not use. D007 says an unrecognized input raises, and
`"1"` here is arguably exactly that: a model whose metadata does not describe a shape XGBoost
would produce. Refusing it is loud, recoverable, and one `num_class=1` removal away from
working. I cannot choose between these from measurement: both are consistent with every number
in this report. They differ in what a caller experiences, not in any prediction.

**A2. What does `size_leaf_vector == "1"` mean when there are no trees?** The field exists only
per-tree (§2.1), so `FORMAT.md` §11's scalar comparison has no referent on a zero-tree model.
*Reading A:* universally quantified over `trees`, hence vacuously true, hence a zero-round
model is accepted — consistent with §13's "a reader MUST NOT raise on ... an empty ensemble is
legitimate" and with §6.1's required zero-tree fixture. *Reading B:* the field must be present
and equal to `"1"`, hence a zero-round model is rejected — which would make the §6.1 required
fixture unexportable. Reading A is the only one consistent with the rest of the specification,
but §11 as written does not say so, and the two readings produce different verdicts on two rows
of §4. This is a specification-text gap, not an empirical one.

**A3. Whether the zero-tree raw-intercept behaviour is intended or a defect.** *Reading 1:*
`boost_from_average == "1"` means "the value in `base_score` has not yet been converted into
link space"; the conversion happens at the first boosting round, so with zero rounds it never
ran and the stored value is a raw margin intercept. Under this reading the artifact is
self-consistent and a reader must branch on `boost_from_average` for zero-tree models.
*Reading 2:* it is an edge-case defect in XGBoost 3.3.0 — the same `boost_from_average == "1"`
artifact means link-space with trees and raw-space without, which no consumer would guess. Under
this reading a v1 exporter should refuse zero-tree models rather than encode the quirk. Nothing I
measured distinguishes intent. What is measured is the four-cell table above, reproducibly, and
that both cells with `boost_from_average == "1"` are reachable by default training parameters.

**A4. Whether `num_class >= 2` is the true multi-output condition or just the observed one.**
Editing `num_class` to `"2"`/`"3"` on a 3-tree binary artifact produced `(400, 2)`/`(400, 3)`
margins (§1.6), and `"0"`/`"1"` produced `(400,)`. *Reading 1:* `num_class` is a group count
where `0` and `1` are both spellings of "one group", so `num_class <= 1` is the single-output
condition. *Reading 2:* the mapping is an artifact of how the loader sizes its output buffer,
and `"0"` versus `"1"` could diverge in a future version. I measured the behaviour on one
version only, so a gate relying on `{"0","1"}` is relying on a 3.3.0 observation, which the D018
version ceiling already bounds.

---

## Not measured

Stated so the gaps are visible rather than assumed away.

- **The sklearn API.** `scikit-learn` is absent and was not installed. `XGBClassifier`,
  `XGBRegressor`, and `XGBRanker` were not exercised at all. `XGBClassifier` on 2-class labels
  is the single most likely real-world route to whatever `num_class` a wrapper sets, and it is
  **unmeasured**. This is the largest hole in the probe.
- **Any XGBoost version other than 3.3.0** (D001, D018).
- **`num_class` under `booster="gblinear"`.** `booster="dart"` was touched only in §1.5.
- **Whether `num_class > 0` and `num_target > 1` can co-occur from training.** They co-occur here
  only because `num_class=1` was passed alongside a 2-column label
  (`reg:squarederror, num_target=2 + num_class=1`, §4). Bears on M2 of
  `probes/multiclass_extensibility.md`.
- **GPU / `device="cuda"`.**
- **`num_class` values above 4, and negative or non-numeric `num_class` strings.**
- **Categorical or `feature_types`-carrying models** in combination with any of the above.
- **Whether `trees` can be absent rather than `[]`** on some path not exercised here.

---

## Scratch inventory

9 probe scripts and 60+ fitted artifacts, all under `…/scratchpad/probe-g3-rerun/`. Nothing was
written into the repository except this file.

| Script | Covers |
|---|---|
| `g1_core_question.py` | §1 all 16 primary routes to `num_class == "1"` |
| `g2_equivalence.py` | §1.2 R1-vs-R2 full artifact diff, §1.6 load-bearing sweep, reverse 3-class edit |
| `g3_types.py` | §2 on-disk JSON types, raw bytes, string-vs-int comparisons, §2.1 `size_leaf_vector` census |
| `g4_truth_table.py` | §4 truth table, 23 shapes, both `size_leaf_vector` readings |
| `g5_walk_warn_zeroround.py` | §1.3 §10 walk on `num_class="1"`, §1.4 warning capture, §6 zero-round raw bytes |
| `g6_zero_round_margin.py` | §7 first pass, `base_score` sweep at 0 and 3 rounds |
| `g7_zero_round_intercept.py` | §7 in-memory vs reloaded, artifact byte comparison, flip point |
| `g8_zero_round_diff.py` | §7 `boost_from_average` as the discriminating field, `1e-6` gate table |
| `g9_bfa.py` | §7 four-cell rule, `boost_from_average` load-bearing edit, artifact census |
