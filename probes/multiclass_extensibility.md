# Probe: multi-class extensibility check

**Scope.** This probe answers exactly one question, per D003:

> If a future version added per-class grouping, would the tree representation observed
> today need to be **restructured**, or merely **extended alongside**?

Multi-class is out of scope for 1.0. Nothing here proposes a multi-class format, recommends
fields to reserve, or designs a per-class representation. The detection surface for
*refusing* multi-class is in scope and is reported in §7.

**Every claim below is backed by a pasted command and its real output.** Anything not
measured is labelled `INFERRED`. Anything that admits two readings is presented as both
readings under *Ambiguities*, not resolved.

No library code was written. All fitted models and scripts live in scratch; nothing was
written into the repository except this file. Prior findings in `probes/tree_structure.md`
and `probes/boosters.md` were read first and are the starting point; where this probe
touches them it confirms, and it contradicts nothing.

---

## 0. Environment

```
$ export PATH="$HOME/.local/bin:$PATH"
$ uv run python -c "
import sys, numpy, xgboost
print('python', sys.version)
print('xgboost', xgboost.__version__)
print('numpy', numpy.__version__)
"
python 3.12.8 (main, Dec  3 2024, 18:42:41) [Clang 16.0.0 (clang-1600.0.26.4)]
xgboost 3.3.0
numpy 2.5.1
```

Matches the D001 reference pin. Every model: 400 rows × 4 generic columns `c0..c3`,
`numpy` `default_rng(20260801)`, generic synthetic normal features, `max_depth=2`,
`eta=0.3`, `seed=20260801`, `nthread=1`, 3 boosting rounds. `tree_method="exact"` unless a
case required `hist` (§6), in which case the comparison model was refit on `hist` too so
the difference is attributable to the strategy and not to the tree method.

**Precision discipline.** Every threshold comparison in this probe casts *both* sides:
`np.float32(value) < np.float32(threshold)`. Leaf values, `base_score` elements, and
accumulation are all `np.float32`.

---

## 1. Verdict

> **EXTEND ALONGSIDE.** Default multi-class in XGBoost 3.3.0 produces *ordinary scalar-leaf
> trees* that are byte-for-byte the same shape as the binary/regression trees already
> recorded in `probes/tree_structure.md` — same 16 keys, same
> `left_children[i] == -1` leaf rule, same `2147483647` root sentinel,
> `size_leaf_vector == "1"` throughout — and per-class grouping is carried **entirely
> outside the tree objects**, by `learner.gradient_booster.model.tree_info`, a field that
> is *already present and already per-tree* in the single-group case where it is uniformly
> `0`.

The load-bearing evidence: the **same unchanged scalar tree walk**, with no multi-class
special case anywhere in it, reproduced `predict(output_margin=True)` for one-group,
three-class, and two-target models alike:

```
$ uv run python q06_census_and_target.py
F. does the SAME unchanged scalar tree walk reproduce all three shapes?
  binary:logistic (1 group)      num_groups=1 tree_info=[0, 0, 0]... max err = 5.960464478e-08
  multi:softprob k=3             num_groups=3 tree_info=[0, 1, 2, 0, 1, 2]... max err = 0.000000000e+00
  reg:squarederror 2 targets     num_groups=2 tree_info=[0, 1, 0, 1, 0, 1]... max err = 0.000000000e+00
```

(The `5.96e-08` on the binary row is the `logit(base_score)` float32 rounding, not a walk
error; the multi-class rows are exactly `0.0` because their `base_score` elements are added
raw — see §5.)

And an exhaustive path/key census over the whole artifact: multi-class **adds exactly one
key anywhere in the file**, and it is an objective-parameter block, sibling to the one
`binary:logistic` already carries. Nothing is added inside a tree object; nothing is
removed.

```
$ uv run python q06_census_and_target.py
A. PATHS PRESENT IN 3-class softprob BUT NOT binary:logistic
  $.learner.objective.softmax_multiclass_param  =  object{num_class}
  $.learner.objective.softmax_multiclass_param.num_class  =  str='3'

B. PATHS PRESENT IN binary:logistic BUT NOT 3-class softprob
  $.learner.objective.reg_loss_param  =  object{scale_pos_weight}
  $.learner.objective.reg_loss_param.scale_pos_weight  =  str='1'

D. exhaustive key-name census (every key at any depth)
  binary   key count = 46
  softprob key count = 45
  keys in softprob NOT in binary : ['softmax_multiclass_param']
  keys in binary NOT in softprob : ['reg_loss_param', 'scale_pos_weight']
```

Everything else that differs differs only in **length or value**, never in kind:

```
C. PATHS IN BOTH WHERE THE SHAPE/LENGTH DIFFERS (binary vs softprob)
  $.learner.gradient_booster.model.gbtree_model_param.num_trees
    binary  : str='3'
    softprob: str='9'
  $.learner.gradient_booster.model.tree_info
    binary  : array[3]<int>
    softprob: array[9]<int>
  $.learner.gradient_booster.model.trees
    binary  : array[3]<dict>
    softprob: array[9]<dict>
  $.learner.learner_model_param.base_score
    binary  : str='[3.35E-1]'
    softprob: str='[-2.4969578E-3,-2.4969578E-3,4.993677E-3]'
  $.learner.learner_model_param.num_class
    binary  : str='0'
    softprob: str='3'
  $.learner.objective
    binary  : object{name,reg_loss_param}
    softprob: object{name,softmax_multiclass_param}
  $.learner.objective.name
    binary  : str='binary:logistic'
    softprob: str='multi:softprob'
```

**The one caveat, and it is a scope caveat, not a structural one:** the genuinely different
vector-leaf shape (`multi_output_tree`) recorded in `probes/tree_structure.md` §7g is
**not** what `multi:softprob` / `multi:softmax` produce by default. It is an opt-in
`multi_strategy` path, and it *is* a restructure. §6 states this plainly. §8 states what
would falsify the verdict.

---

## 2. Are individual trees the same shape?

Yes — identical key set, identical leaf rule, identical sentinels, identical array
lengths, across 1-group, 3-class, and 4-class models.

```
$ uv run python q01_fit_and_inventory.py
B. per-tree key sets: are multi-class trees the SAME shape?
--- bin_baseline: 3 trees, 1 distinct key set(s)
       ('base_weights', 'categories', 'categories_nodes', 'categories_segments', 'categories_sizes', 'default_left', 'id', 'left_children', 'loss_changes', 'parents', 'right_children', 'split_conditions', 'split_indices', 'split_type', 'sum_hessian', 'tree_param')
    per-tree size_leaf_vector: ['1', '1', '1']
    'leaf_weights' present in any tree: False
--- softprob_3: 9 trees, 1 distinct key set(s)
       ('base_weights', 'categories', 'categories_nodes', 'categories_segments', 'categories_sizes', 'default_left', 'id', 'left_children', 'loss_changes', 'parents', 'right_children', 'split_conditions', 'split_indices', 'split_type', 'sum_hessian', 'tree_param')
    per-tree size_leaf_vector: ['1', '1', '1', '1', '1', '1', '1', '1', '1']
    per-tree num_nodes       : ['7', '7', '7', '7', '7', '7', '7', '7', '7']
    per-tree id              : [0, 1, 2, 3, 4, 5, 6, 7, 8]
    'leaf_weights' present in any tree: False
--- softprob_4: 12 trees, 1 distinct key set(s)
       (... identical 16-key tuple ...)
    per-tree size_leaf_vector: ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']
--- softmax_3: 9 trees, 1 distinct key set(s)
       (... identical 16-key tuple ...)
    per-tree size_leaf_vector: ['1', '1', '1', '1', '1', '1', '1', '1', '1']

    key set identical across reg_baseline / bin_baseline / softprob_3 / softprob_4 / softmax_3 : True
```

Leaf detection and array lengths, all nine trees of the 3-class model:

```
C. leaf detection rule and per-node array lengths, softprob_3
  tree 0: num_nodes=7 lens={'default_left': 7, 'left_children': 7, 'loss_changes': 7, 'parents': 7, 'right_children': 7, 'split_conditions': 7, 'split_indices': 7, 'split_type': 7, 'sum_hessian': 7, 'base_weights': 7}
          leaves_by_left=[3, 4, 5, 6] leaves_by_right=[3, 4, 5, 6] agree=True parents[0]=2147483647
  ... trees 1-8 identical in every one of those quantities ...
```

Point by point against the shape recorded in `probes/tree_structure.md`:

| Property | binary / regression (prior probe) | multi-class, measured here |
|---|---|---|
| Per-tree key set | the 16 keys above | **same 16 keys** |
| Leaf test `left_children[i] == -1` | holds | **holds** |
| `right_children[i] == -1` at leaves | holds (scalar leaves) | **holds** |
| `parents[0]` root sentinel | `2147483647` | **`2147483647`** |
| `split_conditions[leaf]` = leaf output | yes | **yes** (walk reproduces `predict` at `0.0`) |
| `tree_param.size_leaf_vector` | `"1"` | **`"1"`** |
| `len(base_weights)` | `num_nodes * size_leaf_vector` | **`7` for `num_nodes=7`** |
| `leaf_weights` key | absent | **absent** |

A 3-class tree verbatim (`json.dumps(trees[0], indent=1)`, `softprob_3`) — nothing in it
identifies a class:

```json
{
 "base_weights": [-1.8359918e-09, 1.0095309, -0.4071454, 0.30026206, 1.3453114, 0.09657657, -0.59177804],
 "categories": [], "categories_nodes": [], "categories_segments": [], "categories_sizes": [],
 "default_left": [1, 1, 1, 0, 0, 0, 0],
 "id": 0,
 "left_children": [1, 3, 5, -1, -1, -1, -1],
 "loss_changes": [73.80175, 11.885807, 11.957123, 0.0, 0.0, 0.0, 0.0],
 "parents": [2147483647, 0, 0, 1, 1, 2, 2],
 "right_children": [2, 4, 6, -1, -1, -1, -1],
 "split_conditions": [-0.46327198, -0.6678909, -0.06582001, 0.09007862, 0.40359345, 0.028972972, -0.17753342],
 "split_indices": [0, 1, 0, 0, 0, 0, 0],
 "split_type": [0, 0, 0, 0, 0, 0, 0],
 "sum_hessian": [177.55501, 50.60318, 126.951836, 16.867725, 33.73545, 34.17934, 92.77249],
 "tree_param": {"num_deleted": "0", "num_feature": "4", "num_nodes": "7", "size_leaf_vector": "1"}
}
```

There is **no per-class field, no class index, and no length change of any kind inside the
tree object.** A tree in a 3-class model is indistinguishable, in isolation, from a tree in
a binary model.

---

## 3. How class membership is carried

**Field:** `learner.gradient_booster.model.tree_info` — a JSON array of `int`, length
`num_trees`, one entry per tree, positionally aligned with `trees[]`.

**Measured semantics:** `tree_info[i]` is the **index of the output group whose margin
tree `i` contributes to.** For a multi-class model the output group is the class. Trees
whose `tree_info` entry is `k` sum into class `k`'s margin and into no other.

Literal bytes as XGBoost writes them, 3-class `multi:softprob`:

```
$ uv run python - (literal file bytes, softprob_3.json)
"gbtree_model_param":{"num_parallel_tree":"1","num_trees":"9"},"iteration_indptr":[0,3,6,9],"tree_info":[0,1,2,0,1,2,0,1,2],"trees":[{"base_
```

Booster-level view for all measured models:

```
$ uv run python q01_fit_and_inventory.py
A. learner_model_param + booster-level grouping fields, per model
--- bin_baseline
    num_trees              : 3      tree_info : [0, 0, 0]              iteration_indptr : [0, 1, 2, 3]
--- softprob_3
    num_trees              : 9      tree_info : [0, 1, 2, 0, 1, 2, 0, 1, 2]        iteration_indptr : [0, 3, 6, 9]
--- softprob_4
    num_trees              : 12     tree_info : [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]   iteration_indptr : [0, 4, 8, 12]
--- softmax_3
    num_trees              : 9      tree_info : [0, 1, 2, 0, 1, 2, 0, 1, 2]        iteration_indptr : [0, 3, 6, 9]
```

`multi:softmax` and `multi:softprob` produce **identical** grouping structure; they differ
only in `objective.name` and in what `predict()` returns.

### 3.1 The semantics are VERIFIED BY PREDICTION, not read off the field name

Six competing grouping rules were reconstructed from the artifact with the float32 walk and
arbitrated by `predict(output_margin=True)`:

```
$ uv run python q02_verify_grouping.py
==============================================================================
MODEL softprob_3: num_class=3 num_parallel_tree=1 num_trees=9
  tree_info        = [0, 1, 2, 0, 1, 2, 0, 1, 2]
  i % num_class    = [0, 1, 2, 0, 1, 2, 0, 1, 2]
  identical?       = True
  iteration_indptr = [0, 3, 6, 9]
  base_score raw   = '[-2.4969578E-3,-2.4969578E-3,4.993677E-3]'
  base_score parsed= [-0.002496957778930664, -0.002496957778930664, 0.0049936771392822266]  (len=3, num_class=3)
  predict(output_margin=True).shape = (400, 3)
  --- competing grouping rules, arbitrated by predict(output_margin=True) ---
  R1 tree_info[i], intercept = base_score[k]           max|recon - xgb_margin| = 0.000000000e+00
  R1b tree_info[i], intercept = 0.0                    max|recon - xgb_margin| = 4.993677139e-03
  R1c tree_info[i], intercept = base_score[0] for all k max|recon - xgb_margin| = 7.490634918e-03
  R2 i %% num_class, intercept = base_score[k]         max|recon - xgb_margin| = 0.000000000e+00
  R3 contiguous blocks (i // (n_trees/num_class))      max|recon - xgb_margin| = 1.011604622e+00
  R4 (tree_info[i] + 1) %% num_class  [rotated]        max|recon - xgb_margin| = 1.576030493e+00
  R5 grouping IGNORED: all trees -> class 0            max|recon - xgb_margin| (class 0 col) = 1.074002400e+00
```

Same for 4 classes and for `multi:softmax`:

```
MODEL softprob_4: num_class=4 num_parallel_tree=1 num_trees=12
  tree_info        = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
  R1 tree_info[i], intercept = base_score[k]           max|recon - xgb_margin| = 0.000000000e+00
  R3 contiguous blocks (i // (n_trees/num_class))      max|recon - xgb_margin| = 1.233021855e+00
  R4 (tree_info[i] + 1) %% num_class  [rotated]        max|recon - xgb_margin| = 1.721434951e+00
  R5 grouping IGNORED: all trees -> class 0            max|recon - xgb_margin| (class 0 col) = 1.426313266e+00

MODEL softmax_3: num_class=3 num_parallel_tree=1 num_trees=9
  R1 tree_info[i], intercept = base_score[k]           max|recon - xgb_margin| = 0.000000000e+00
  R3 contiguous blocks (i // (n_trees/num_class))      max|recon - xgb_margin| = 1.011604622e+00
  R4 (tree_info[i] + 1) %% num_class  [rotated]        max|recon - xgb_margin| = 1.576030493e+00
```

So the wrong readings are wrong by **1.01 to 1.72 in margin space** — plausible wrong
numbers, no error raised. `tree_info` is not decorative.

### 3.2 `tree_info` is authoritative; positional interleaving is NOT the rule

On the models above, `tree_info == [i % num_class]`, so those two rules cannot be
distinguished. `num_parallel_tree=2` separates them, and only `tree_info` survives:

```
MODEL softprob_3_npt2: num_class=3 num_parallel_tree=2 num_trees=18
  tree_info        = [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2]
  i % num_class    = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
  identical?       = False
  iteration_indptr = [0, 6, 12, 18]
  R1 tree_info[i], intercept = base_score[k]           max|recon - xgb_margin| = 0.000000000e+00
  R2 i %% num_class, intercept = base_score[k]         max|recon - xgb_margin| = 7.629465312e-01
  R3 contiguous blocks (i // (n_trees/num_class))      max|recon - xgb_margin| = 1.011604607e+00
  R4 (tree_info[i] + 1) %% num_class  [rotated]        max|recon - xgb_margin| = 1.576030672e+00
```

**An interleaving convention (`i % num_class`) is a wrong reading that is right on the
common case.** It costs `0.763` in margin space as soon as `num_parallel_tree > 1`. Only
`tree_info` reproduced `predict` in every configuration.

`tree_info` also survives model slicing intact:

```
$ uv run python q05_detection_matrix.py
=== slicing a 3-class model: does tree_info survive? ===
  full       num_trees= 12 tree_info=[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2] iteration_indptr=[0, 3, 6, 9, 12] ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] num_class=3
  bst3[1:3]  num_trees=  6 tree_info=[0, 1, 2, 0, 1, 2] iteration_indptr=[0, 3, 6] ids=[0, 1, 2, 3, 4, 5] num_class=3
```

Slicing is by boosting round: `bst3[1:3]` keeps whole rounds, so every class keeps the same
number of trees and `tree_info` stays consistent.

### 3.3 `tree_info` means "output group", not specifically "class"

The same field, the same rule, groups by **target** in a multi-target regression where
`num_class == "0"`:

```
$ uv run python q06_census_and_target.py
E. tree_info in the multi-TARGET case, verified by prediction
  learner_model_param : {"base_score": "[9.0577826E-2,9.547111E-1]", "boost_from_average": "1", "num_class": "0", "num_feature": "4", "num_target": "2"}
  tree_info           : [0, 1, 0, 1, 0, 1]
  group by tree_info, intercept = base_score[k]: max err = 0.000000000e+00
  group by tree_info ROTATED by 1              : max err = 2.468508720e+00
```

So `tree_info` is a generic output-group index. Whether group `k` is "class `k`" or
"target `k`" is decided by `num_class` / `num_target`, not by `tree_info` itself. Both
readings were verified by prediction, not inferred.

---

## 4. Inside or outside the tree objects — the crux

**OUTSIDE**, for the default multi-class path. Measured, on `hist` so that the vector-leaf
comparison in §6 is at equal `tree_method`:

```
$ uv run python q04_mot_hist.py
C. VERIFY BY PREDICTION: where does class membership live in each shape?
--- hist one_output_per_tree: num_trees=9 tree_info=[0, 1, 2, 0, 1, 2, 0, 1, 2] margin shape=(400, 3)
    OUTSIDE-the-tree rule (group by tree_info, scalar leaf): max err = 0.000000000e+00
```

Restated as three separately measured facts:

1. The tree object gains **no** field and loses none (§2 key-set table, §1 census).
2. No array inside the tree object changes length; `size_leaf_vector` stays `"1"`
   (§2, all nine trees).
3. The grouping field `tree_info` sits at
   `learner.gradient_booster.model.tree_info` — a **sibling of** `trees[]`, not a member of
   any tree — and it is already present, already length `num_trees`, in the single-group
   case (`[0, 0, 0]` for binary, §3).

That third point is what makes this an extension rather than a migration: a reader written
today that groups by `tree_info` is already correct for the single-group case, because
`tree_info` is uniformly `0` there. `probes/tree_structure.md` §12 already lists
`tree_info` among the fields its exact-`0.0` walks read.

---

## 5. `base_score` for a multi-class model

Reported, not designed. `probes/base_score.md` recorded that the value is a **JSON string
containing a bracketed array**, observed with exactly one element in every case, and marked
the multi-element question `INFERRED` / not measured. **It is multi-element, one per class.**
Verbatim:

```
$ uv run python q01_fit_and_inventory.py
E. base_score verbatim, and the raw file substring around it
--- bin_baseline
    python type after json.load : str
    raw value repr              : '[5E-1]'
    file substring              : '"base_score":"[5E-1]","boost_from_average":"1","num_class":"0","num_feature":"4","num_target":"1"},"objective":{"name":"'
--- softprob_3
    python type after json.load : str
    raw value repr              : '[-2.4969578E-3,-2.4969578E-3,4.993677E-3]'
    file substring              : '"base_score":"[-2.4969578E-3,-2.4969578E-3,4.993677E-3]","boost_from_average":"1","num_class":"3","num_feature":"4","num'
--- softprob_4
    python type after json.load : str
    raw value repr              : '[0E0,0E0,0E0,0E0]'
    file substring              : '"base_score":"[0E0,0E0,0E0,0E0]","boost_from_average":"1","num_class":"4","num_feature":"4","num_target":"1"},"objective'
--- softmax_3
    python type after json.load : str
    raw value repr              : '[-2.4969578E-3,-2.4969578E-3,4.993677E-3]'
```

Facts, each measured:

- **One element per class.** `len == num_class`: 3 elements for `num_class=3`, 4 for
  `num_class=4`. Elements are comma-separated with no spaces, same uppercase-`E` exponent
  form, same string-wrapping-an-array double parse.
- The elements are **not all equal** (`[-2.4969578E-3, -2.4969578E-3, 4.993677E-3]`), so a
  reader cannot collapse the vector to a scalar.
- The `0`-round case is `'[5E-1,5E-1,5E-1]'` — the untouched `0.5` default, replicated
  per class (§7 table).
- Multi-**target** regression also produces a multi-element vector
  (`'[9.0577826E-2,9.547111E-1]'`, `num_class="0"`, `num_target="2"`), so element count
  tracks the number of output groups, not the class count specifically.
- On the space question, and reported only because §3.1 had to pick an intercept to
  verify the grouping at all: adding `base_score[k]` **raw**, with no transform,
  reproduced `predict(output_margin=True)` at exactly `0.0`; using `0.0` instead was off by
  `4.99e-03` and using `base_score[0]` for every class by `7.49e-03`. Per D003 and the
  per-objective `base_score` invariant, **that is one observation on one fit, not a rule
  for a `multi:*` objective, and nothing should be inferred from it by analogy.** These
  objectives raise on export.

---

## 6. `multi_output_tree` is a separate opt-in path, not what softprob/softmax produce

**Stated plainly: default multi-class produces ORDINARY scalar-leaf trees.
`multi_output_tree` is opt-in via the `multi_strategy` parameter, and it is the genuinely
different shape.** This is the key distinction, and it is why the verdict is "extend
alongside" rather than "restructure".

The default is `one_output_per_tree`, from both the config and the artifact:

```
$ uv run python q04_mot_hist.py
B. is the DEFAULT strategy one_output_per_tree? (config + artifact shape)
    learner_train_param.multi_strategy = one_output_per_tree
    default artifact identical in shape to explicit one_output_per_tree?  True
    default num_trees / tree_info: 9 [0, 1, 2, 0, 1, 2, 0, 1, 2]
    default per-tree size_leaf_vector: ['1', '1', '1', '1', '1', '1', '1', '1', '1']
```

`multi_strategy` is **not serialized into the model file at all** — it is a training-time
parameter, and the string does not appear:

```
$ uv run python q01_fit_and_inventory.py
F. does the token 'multi' / class count appear anywhere else in the file?
    token 'multi:softprob'     count = 1
    token 'num_class'          count = 2
    token 'multi_strategy'     count = 0
    token 'num_target'         count = 1
    token 'tree_info'          count = 1
    token 'leaf_weights'       count = 0
    token 'size_leaf_vector'   count = 9
```

`multi_output_tree` additionally requires `tree_method` `hist` or `auto`; it raises under
`exact`:

```
$ uv run python q03_mot_and_detection.py
--- softprob3_multi_output_tree: RAISED XGBoostError: [21:14:42] .../src/gbm/gbtree.cc:227: Check failed: tparam_.tree_method == TreeMethod::kHist || tparam_.tree_method == TreeMethod::kAuto: Only the hist tree method is supported for building multi-target trees with vector leaf.
--- softmax3_multi_output_tree: RAISED XGBoostError: [21:14:43] .../src/gbm/gbtree.cc:227: Check failed: ... Only the hist tree method is supported for building multi-target trees with vector leaf.
--- multitarget_mot: RAISED XGBoostError: [21:14:43] .../src/gbm/gbtree.cc:227: Check failed: ... Only the hist tree method is supported for building multi-target trees with vector leaf.
```

Refit on `hist`, both strategies, same data and seed:

```
$ uv run python q04_mot_hist.py
A. tree_method=hist, multi:softprob, both multi_strategy values
--- softprob3 hist one_output_per_tree
    num_trees              : 9
    tree_info              : [0, 1, 2, 0, 1, 2, 0, 1, 2]
    per-tree size_leaf_vector: ['1', '1', '1', '1', '1', '1', '1', '1', '1']
    'leaf_weights' in trees: False
    tree 0 parents[0]      : 2147483647
    tree 0 right_children  : [2, 4, 6, -1, -1, -1, -1]
--- softprob3 hist multi_output_tree
    num_trees              : 3
    tree_info              : [0, 0, 0]
    iteration_indptr       : [0, 1, 2, 3]
    per-tree size_leaf_vector: ['3', '3', '3']
    'leaf_weights' in trees: True
    tree 0 parents[0]      : -1
    tree 0 right_children  : [2, 4, 6, 0, 1, 2, 3]
    tree 0 split_conditions: [0.24028805, -0.67527074, 0.36758664, 1e-45, 1e-45, 1e-45, 1e-45]
    tree 0 len(base_weights): 21  num_nodes: 7
    tree 0 leaf_weights    : [0.36098838, -0.1561645, -0.20407951, 0.0019877488, 0.13092284, -0.13242173, -0.2198631, -0.14206988, 0.3606116, -0.13921805, 0.1255914, 0.013577685]
--- softmax3 hist multi_output_tree
    (identical shape: num_trees 3, tree_info [0,0,0], size_leaf_vector ['3','3','3'], leaf_weights present, parents[0] = -1)
```

Every vector-leaf difference recorded in `probes/tree_structure.md` §7g reproduces here for
`multi:softprob`, with `size_leaf_vector == 3` instead of `2`: `leaf_weights` appears,
`len(base_weights) == num_nodes * size_leaf_vector == 21`, `parents[0] == -1` not
`2147483647`, `right_children` at leaves holds the leaf's `leaf_weights` block index
(`0,1,2,3`) instead of `-1`, and `split_conditions` at leaves is the `1e-45` subnormal
rather than the leaf value.

**And in this shape the grouping moves INSIDE the tree.** Verified by prediction:

```
C. VERIFY BY PREDICTION: where does class membership live in each shape?
--- hist multi_output_tree: num_trees=3 tree_info=[0, 0, 0] margin shape=(400, 3)
    size_leaf_vector = 3
    INSIDE-the-tree rule (all trees, leaf_weights slot k): max err = 0.000000000e+00
    scalar/tree_info rule applied to a vector-leaf tree: max err = 9.296978712e-01
    INSIDE-the-tree rule with slot rotated by 1: max err = 1.523051679e+00
```

Every tree contributes to every class; class `k`'s value is slot `k` of the leaf's
`leaf_weights` block, and `tree_info` degenerates to all-zeros and carries no class
information whatsoever. Applying the scalar/`tree_info` rule to a vector-leaf tree yields a
wrong margin by `0.930` — again, silently.

Literal bytes, `multi:softprob` + `multi_output_tree`:

```
"tree_info":[0,0,0],"trees":[{"base_weights":[-1.8359918E-9,-1.8359918E-9,4.7891245E-8,4.345213E-1,9.835266E-2,-5.309041
"leaf_weights":[3.6098838E-1,-1.561645E-1,-2.0407951E-1,1.9877488E-3,1.3092284E-1,-1.3242173E-1,-2.198631E-1,-1.4206988E-1,3.606116E-1,-1.3921805E-1,1.255914E-1,1.3577685E-2],"left_children":[1,3,5,-1,-1,-1,-1],"loss_changes":[1.2
```

So: two different multi-class *representations* exist in XGBoost 3.3.0.
`one_output_per_tree` (the default) is an extension of the observed shape.
`multi_output_tree` (opt-in, `hist` only) is a restructure. They are distinguished in the
artifact by `size_leaf_vector` / the presence of `leaf_weights`, not by anything
multi-class-specific.

---

## 7. Detection surface: the field to inspect to refuse multi-class loudly

Fifteen model shapes, chosen to break the obvious signals, with every candidate field
printed:

```
$ uv run python q05_detection_matrix.py
case                                 objective.name    sm_param num_class num_tgt len(bs) max(ti) slv    leaf_w margin.shape   trees
------------------------------------------------------------------------------------------------------------------------------------
reg:squarederror 1-target            reg:squarederror  False            0       1       1       0    [1]  False (400,)         3
binary:logistic                      binary:logistic   False            0       1       1       0    [1]  False (400,)         3
survival:cox                         survival:cox      False            0       1       1       0    [1]  False (400,)         3
reg 1-target, 0 rounds               reg:squarederror  False            0       1       1    None     []  False (400,)         0
softprob k=3                         multi:softprob    True             3       1       3       2    [1]  False (400, 3)       9
softmax k=3                          multi:softmax     True             3       1       3       2    [1]  False (400, 3)       9
softprob k=4                         multi:softprob    True             4       1       4       3    [1]  False (400, 4)       12
softprob k=2                         multi:softprob    True             2       1       2       1    [1]  False (400, 2)       6
softprob k=1                         multi:softprob    True             1       1       1       0    [1]  False (400,)         3
softprob k=3, 0 rounds               multi:softprob    True             3       1       3    None     []  False (400, 3)       0
softprob k=3, npt=2                  multi:softprob    True             3       1       3       2    [1]  False (400, 3)       18
softprob k=3 + multi_output_tree     multi:softprob    True             3       1       3       0    [3]   True (400, 3)       3
softmax k=3 + multi_output_tree      multi:softmax     True             3       1       3       0    [3]   True (400, 3)       3
reg 2-target (one_output_per_tree)   reg:squarederror  False            0       2       2       1    [1]  False (400, 2)       6
reg 2-target + multi_output_tree     reg:squarederror  False            0       2       2       0    [2]   True (400, 2)       3
```

Each candidate scored for false positives (would reject an in-scope model) and misses
(would let a multi-class model through):

```
=== which single candidate signal separates multi-class from the 1.0 scope? ===
  objective.name in {multi:softmax, multi:softprob}    false-positives=[] missed=[]
  objective.softmax_multiclass_param present           false-positives=[] missed=[]
  learner_model_param.num_class != "0"                 false-positives=[] missed=[]
  learner_model_param.num_class > 1                    false-positives=[] missed=['softprob k=1']
  max(tree_info) > 0                                   false-positives=['reg 2-target (one_output_per_tree)'] missed=['softprob k=1', 'softprob k=3, 0 rounds', 'softprob k=3 + multi_output_tree', 'softmax k=3 + multi_output_tree']
  len(base_score vector) > 1                           false-positives=['reg 2-target (one_output_per_tree)', 'reg 2-target + multi_output_tree'] missed=['softprob k=1']
  any size_leaf_vector > 1                             false-positives=['reg 2-target + multi_output_tree'] missed=['softprob k=3', 'softmax k=3', 'softprob k=4', 'softprob k=2', 'softprob k=1', 'softprob k=3, 0 rounds', 'softprob k=3, npt=2']
```

### Fields an exporter/reader inspects

**Primary — `$.learner.objective.name`.** A JSON string. Refuse on `"multi:softmax"` and
`"multi:softprob"`. Zero false positives, zero misses over all 15 shapes. This is also
already the D007 dispatch point: any objective not on the 1.0 allow-list raises anyway, so
multi-class is refused by the allow-list without a special case.

**Confirming, independent of the objective string — `$.learner.learner_model_param.num_class`.**
A JSON **string**, not a number. `"0"` for every in-scope model measured
(regression, binary, Cox, and multi-target regression); `"1"`,`"2"`,`"3"`,`"4"` only under a
`multi:*` objective. `!= "0"` scored zero false positives and zero misses.

**Third, structurally redundant — `$.learner.objective.softmax_multiclass_param`.** Present
iff the objective is `multi:*` across all 15 shapes; it is the *only* key multi-class adds
anywhere in the file (§1 census). Its `num_class` duplicates the value as a string.

### Signals that do NOT work, and why each is a silent-failure trap

- **`max(tree_info) > 0`** — misses `multi_output_tree` multi-class (`tree_info` is
  `[0,0,0]`), misses a `0`-round model (`tree_info` is `[]`), misses `num_class=1`, and
  **false-positives on multi-target regression**, where `tree_info` is `[0,1,0,1,0,1]` with
  `num_class == "0"`.
- **`len(base_score) > 1`** — false-positives on multi-target regression, misses
  `num_class=1`.
- **`size_leaf_vector > 1` / `leaf_weights` present** — misses every *default* multi-class
  model, which is all of them unless `multi_strategy` was set. It detects the vector-leaf
  shape, which is a different and also-out-of-scope thing.
- **`num_class > 1`** — misses a real, reachable state: `multi:softprob` with
  `num_class=1` is accepted by XGBoost and produces a single-output model.

```
=== num_class=1 softprob: is it structurally a single-group model? ===
  num_class          : 1
  base_score         : [0E0]
  tree_info          : [0, 0, 0]
  num_trees          : 3
  margin shape       : (400,)
  predict shape      : (400,)
```

Also worth having on record for the exporter: `multi:softprob` with `num_class` **omitted**
never reaches serialization — XGBoost raises at train time.

```
softprob num_class omitted        RAISED XGBoostError: value 0 for Parameter num_class should be greater equal to 1
```

---

## 8. What would falsify the verdict

Each of these is a concrete, checkable condition. Any one of them turning up flips the
answer from "extend alongside" to "restructure".

1. **A default multi-class fit producing vector-leaf trees.** If XGBoost changed the
   default `multi_strategy` from `one_output_per_tree` to `multi_output_tree`, or if a
   future objective produced vector leaves without opt-in, the tree object itself changes
   shape (§6) — `leaf_weights` appears, `parents[0]` flips to `-1`, `right_children` at
   leaves stops being `-1`, `split_conditions` at leaves stops being the leaf value. The
   check: `any(int(t["tree_param"]["size_leaf_vector"]) > 1 for t in trees)` on a
   default-parameter multi-class fit. Measured `False` on XGBoost 3.3.0 for both
   `multi:softprob` and `multi:softmax`.
2. **A per-class field appearing inside the tree object.** Currently the key set is
   identical across 1-group, 3-class, and 4-class models (§2). The check: the exhaustive
   per-tree key-set comparison in §2 returning `False`.
3. **`tree_info` ceasing to be present in the single-group case.** The whole argument rests
   on the grouping field already existing, uniformly `0`, for binary and regression. If a
   single-group artifact omitted `tree_info`, adding grouping later would mean adding the
   field, i.e. a format change rather than a widening of one already carried.
4. **`tree_info` not being authoritative** — e.g. a configuration where grouping is
   positional and `tree_info` disagrees with `predict`. §3.2 tested the case designed to
   break it (`num_parallel_tree=2`) and `tree_info` won at exactly `0.0`; a configuration
   where it does not would mean the grouping semantics are not localized in that one field.
5. **A per-class quantity that cannot be expressed alongside the existing scalar ones.**
   `base_score` goes from 1 element to `k` (§5) — a length change in a field whose
   serialized notation is *already* a bracketed vector, even at `k=1`. If some future
   per-class quantity needed to live inside the tree objects instead, the extension would
   stop being additive.
6. **A different XGBoost version.** Everything here is 3.3.0 per D001. The `multi_strategy`
   default, the `hist`-only restriction, and `tree_info`'s authority are all version-specific
   observations.

---

## Ambiguities — presented, not resolved

**M1. Which detection field is *the* field.** `objective.name in {multi:softmax,
multi:softprob}`, `num_class != "0"`, and the presence of
`objective.softmax_multiclass_param` were **all** perfect discriminators — zero false
positives, zero misses — over the 15 shapes in §7. I therefore cannot distinguish them
empirically. Reading 1: `objective.name` is the signal, since D007's objective allow-list
already refuses anything unrecognized and `num_class` is then a consistency assertion.
Reading 2: `num_class != "0"` is the signal, because it is independent of the objective
string and would also catch a *future* multi-class objective whose name is not on any
list. Reading 3: require all three to agree and raise on disagreement, which is a
validation choice rather than a detection choice. These differ in what happens to a
malformed or third-party artifact where the fields disagree; on a genuine XGBoost artifact
they never disagreed.

**M2. `tree_info` — one field, two vocabularies.** It grouped by class for `multi:*`
(§3.1) and by target for multi-target regression (§3.3), both verified by prediction at
`0.0`. Reading 1: there is one concept, "output group", and class/target is only what the
surrounding metadata calls it — the reading I lean to, since one rule reproduced both.
Reading 2: they are two distinct concepts that happen to share a serialization slot, in
which case a future per-class extension and a future multi-target extension are separate
migrations even though they look identical in the artifact today. Nothing I measured
distinguishes these, because in 3.3.0 `num_class > 0` and `num_target > 1` never co-occurred
in any model I produced.

**M3. Scope boundary of the refusal.** `multi:softmax` / `multi:softprob` raise on export
per D003. But three *other* shapes in §7 also produce grouped output and are not on the 1.0
objective list either: multi-target regression (`num_target="2"` with a `reg:squarederror`
objective that *is* on the list), and both `multi_output_tree` variants. Reading 1: the
objective allow-list is the only gate, so multi-target regression under
`reg:squarederror` would pass it — and produce `(400, 2)` margins against a predictor that
expects `(400,)`. Reading 2: the gate needs `num_target == "1"` and
`size_leaf_vector == "1"` assertions alongside the objective check. This is adjacent to my
brief rather than in it, and I am flagging it rather than deciding it — see *Out of scope
that looked wrong* below.

---

## Not measured

- **Whether `num_class > 0` and `num_target > 1` can co-occur.** Not produced; bears on M2.
- **Any XGBoost version other than 3.3.0** (D001).
- **`multi:*` under `booster="dart"` or `gblinear`.** Out of scope, and `probes/boosters.md`
  owns booster shape.
- **The probability/label output of softprob and softmax.** Only `output_margin=True` was
  reconstructed. The softmax link function was deliberately not investigated — that would
  be designing multi-class support.
- **Categorical splits in a multi-class model.** All models here used numeric features; the
  `categories*` arrays were empty throughout.
- **GPU / `device="cuda"`.**

---

## Out of scope that looked wrong

1. **Multi-target regression passes an objective-name allow-list.** `num_target="2"` with
   `objective="reg:squarederror"` — an in-scope objective name — produces
   `tree_info=[0,1,0,1,0,1]`, a 2-element `base_score`, and `(400, 2)` margins. A gate that
   checks only `objective.name` admits it, and a predictor that then groups everything into
   one output produces a wrong number rather than an error. This is the same silent-failure
   signature as the multi-class case but arrives through an objective that is on the list.
   Raised as M3; it is a decision for the exporter's validation surface, not mine.
2. **`multi:softprob` with `num_class=1` is accepted by XGBoost** and produces a
   single-output model (`margin shape (400,)`, `base_score "[0E0]"`, `tree_info [0,0,0]`).
   Any refusal test written as `num_class > 1` lets it through, and it is structurally
   indistinguishable from a single-group model except by the objective string.
3. **`multi_strategy` is not serialized.** The artifact records the *consequence*
   (`size_leaf_vector`, `leaf_weights`) but never the parameter — the same
   pattern `probes/boosters.md` found for dart's dropout parameters. Structural consequence,
   not configuration, is the only thing a reader can rely on.
4. **`probes/base_score.md` has one `INFERRED` item that this probe resolves.** It recorded
   the bracketed `base_score` as "1-element in every case observed" and marked the
   multi-element question as inferred-from-notation / not measured. It is confirmed
   multi-element, one element per output group (§5). I did not edit that file.

---

## Scratch inventory

6 probe scripts and 15 fitted models, all under `…/scratchpad/probe-multiclass/`. Nothing
was written into the repository except this file.

| Script | Covers |
|---|---|
| `q01_fit_and_inventory.py` | §2 tree shape and key sets, §3 booster-level grouping, §5 `base_score` verbatim, §6 token census |
| `q02_verify_grouping.py` | §3.1 six competing grouping rules arbitrated by `predict`, §3.2 the `num_parallel_tree=2` falsification |
| `q03_mot_and_detection.py` | §6 `multi_strategy` raise under `exact`, multi-target contrast, first detection pass |
| `q04_mot_hist.py` | §4 and §6 `hist` comparison of both strategies, inside-vs-outside verified by prediction |
| `q05_detection_matrix.py` | §7 15-shape detection matrix, `num_class=1`, model slicing |
| `q06_census_and_target.py` | §1 exhaustive path/key census, §3.3 multi-target `tree_info` by prediction, unchanged-walk check |
