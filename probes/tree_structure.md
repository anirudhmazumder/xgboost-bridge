# Probe: tree structure and missing-value routing

Empirical findings about how XGBoost serializes tree structure in its native JSON model
format, how missing values are routed, and what the degenerate cases look like.

**Every claim below is backed by a pasted command and its real output.** Anything not
measured is labelled `INFERRED`. Anything that admits two readings is presented as both
readings under *Ambiguities*, not resolved.

No library code was written for this probe. All fitted models and dumps live in scratch
and are not in the repository.

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

Also present in the environment and used: `scipy 1.18.0`. `pandas` and `pyarrow` are
**absent**, so no DataFrame-shaped input was measured (see *Not measured*).

```
$ uv run python -c "
import importlib
for m in ['scipy','pandas','pyarrow']:
    try:
        mod=importlib.import_module(m); print(m, mod.__version__)
    except Exception as e: print(m,'ABSENT', type(e).__name__)
"
scipy 1.18.0
pandas ABSENT ModuleNotFoundError
pyarrow ABSENT ModuleNotFoundError
```

Unless a case required otherwise, every model below uses `objective="reg:squarederror"`,
`booster="gbtree"`, `tree_method="exact"`, and a fixed `seed`. Generic synthetic normal
data, generic feature names.

**Precision discipline used in this probe.** Every threshold comparison performed by
probe code casts *both* sides: `np.float32(value) < np.float32(threshold)`. Leaf values
and `base_score` are also carried as `np.float32`, and accumulation across trees is done
in `np.float32`. Section 8 shows what happens when that discipline is relaxed.

---

## 1. Where the tree representation lives

```
$ uv run python p01_inventory.py
=== TOP LEVEL KEYS ===
['learner', 'version']

=== raw['version'] ===
[3, 3, 0]
```

Full skeleton of a 3-tree, `max_depth=2`, 4-feature model:

```
learner: dict(6 keys)
  attributes: dict(0 keys)
  feature_names: list[len=4, elem0=str]
  feature_types: list[len=0, elem0=empty]
  gradient_booster: dict(2 keys)
    model: dict(5 keys)
      cats: dict(3 keys)
        enc: list[len=0, elem0=empty]
        feature_segments: list[len=0, elem0=empty]
        sorted_idx: list[len=0, elem0=empty]
      gbtree_model_param: dict(2 keys)
        num_parallel_tree: str = '1'
        num_trees: str = '3'
      iteration_indptr: list[len=4, elem0=int]
      tree_info: list[len=3, elem0=int]
      trees: list[len=3, elem0=dict]
        [0]: <see inventory below>
    name: str = 'gbtree'
  learner_model_param: dict(5 keys)
    base_score: str = '[1.2025498E-1]'
    boost_from_average: str = '1'
    num_class: str = '0'
    num_feature: str = '4'
    num_target: str = '1'
  objective: dict(2 keys)
    name: str = 'reg:squarederror'
    reg_loss_param: dict(1 keys)
      scale_pos_weight: str = '1'
version: list[len=3, elem0=int]
```

Tree array path: `learner.gradient_booster.model.trees` — a JSON array of tree objects.

---

## 2. Layout, in one paragraph

A tree is **not** nested nodes. It is a flat object of **parallel arrays**, one entry per
node, all of length `tree_param.num_nodes`. The **root is always node index 0.** Children
are referenced by integer index into those same arrays: `left_children[i]` and
`right_children[i]`. A node is a **leaf iff `left_children[i] == -1`** (for scalar-leaf
trees `right_children[i] == -1` at the same nodes, but see §6g — that is not true for
vector-leaf trees, so `left_children[i] == -1` is the only leaf test that held in every
measured shape). For an internal node, `split_indices[i]` is the **0-based feature
column index** and `split_conditions[i]` is the **float32 threshold**; for a leaf,
`split_conditions[i]` is the **leaf output value** — the same array is overloaded.
`base_weights[i]` is the *unshrunk* node weight and is **not** the leaf output.
`default_left[i]` encodes missing-value routing. `parents[i]` is the parent index with
`2147483647` (`INT32_MAX`) as the root sentinel. Node indices are dense and contiguous
over `0 .. num_nodes-1`, allocated breadth-first, with `right_children[i] ==
left_children[i] + 1` and `left_children[i] > i` — but **not** the positional `2i+1` rule.
There is **no depth field anywhere** in the tree representation. Pruned nodes remain in
the arrays and are counted by `num_nodes`; `tree_param.num_deleted` says how many
(§6e/§7).

---

## 3. Complete field inventory for one tree

Model: 400 rows, 4 columns, `max_depth=2`, `eta=0.3`, 3 rounds. `num_nodes = 7`.

| Field | Type | Length | Meaning (measured unless noted) |
|---|---|---|---|
| `base_weights` | array of float | `num_nodes * size_leaf_vector` | Unshrunk optimal weight for the node. At a leaf, `split_conditions[i] == base_weights[i] * eta` (measured, §4). **Not** the leaf output — a walk using it is off by 5.10 in margin (§4). Not needed for inference. |
| `categories` | array of int | variable (0 when no categorical split) | Flat concatenation of the category values that route "in-set" at each categorical node. Segmented by `categories_segments` / `categories_sizes`. |
| `categories_nodes` | array of int | = number of categorical split nodes | Node indices that carry a categorical split, ascending. |
| `categories_segments` | array of int | = `len(categories_nodes)` | Start offset into `categories` for the k-th categorical node. |
| `categories_sizes` | array of int | = `len(categories_nodes)` | Number of entries in `categories` for the k-th categorical node. |
| `default_left` | array of int (0/1) | `num_nodes` | **1 = a missing value goes to `left_children[i]`; 0 = it goes to `right_children[i]`.** Demonstrated both ways in §5. Meaningless at leaves (observed `0` at every leaf across 6 seeds × 4 trees × depth 5, §3.1). |
| `id` | int | scalar | Position of this tree inside `trees[]`. Verified `id == index` for all 45 saved models, 0 mismatches. Renumbered to `0..k-1` by model slicing, so it is **positional, not a stable identifier** (§7). |
| `leaf_weights` | array of float | `num_leaves * size_leaf_vector` | **Present only when `size_leaf_vector > 1`** (vector leaves). Absent from every scalar-leaf tree measured. |
| `left_children` | array of int | `num_nodes` | Left child index, or `-1` at a leaf. |
| `loss_changes` | array of float | `num_nodes` | Split gain. `0.0` at every leaf (measured). Not needed for inference. |
| `parents` | array of int | `num_nodes` | Parent index. Root sentinel `2147483647` for scalar-leaf trees, **`-1`** for vector-leaf trees (§6g). **Stale after pruning** — a deleted node's `parents` entry still points at what is now a leaf (§7). Not needed for inference. |
| `right_children` | array of int | `num_nodes` | Right child index. `-1` at a leaf for scalar-leaf trees; **for vector-leaf trees it holds the leaf's block index into `leaf_weights` instead** (§6g). |
| `split_conditions` | array of float | `num_nodes` | Internal numeric node: float32 threshold. Leaf: leaf output value (already shrunk by `eta`). Internal **categorical** node: garbage — the smallest positive subnormal float32, printed as `1e-45` (§9). Vector-leaf tree, at leaves: also `1e-45`. |
| `split_indices` | array of int | `num_nodes` | Feature column index at an internal node. `0` at leaves (measured across all sweeps). **`2147483647` at a deleted node** (§7). |
| `split_type` | array of int | `num_nodes` | `0` = numeric split, `1` = categorical split (measured, §9). Only `0` and `1` observed. |
| `sum_hessian` | array of float | `num_nodes` | Sum of Hessians reaching the node — for `reg:squarederror` this equals the training row count. Not needed for inference. |
| `tree_param` | dict of **strings** | 4 keys | `num_deleted`, `num_feature`, `num_nodes`, `size_leaf_vector`. All values are **JSON strings, not numbers**. |

`tree_param` sub-fields:

| Key | Measured meaning |
|---|---|
| `num_nodes` | Length of every per-node array, **including deleted nodes**. |
| `num_deleted` | Count of pruned/unreachable nodes still present in the arrays. `0` normally; `2`, `14`, `22`, `28`, `48`, `58` observed under `gamma` pruning (§7). |
| `num_feature` | Feature count, duplicated per tree. Matched `learner_model_param.num_feature` in every model measured. |
| `size_leaf_vector` | `1` for scalar leaves; `2` measured for a 2-target `multi_output_tree` model. Values `{1, 2}` observed. |

### 3.1 Fields whose purpose I could NOT determine

Stated explicitly, per the brief:

1. **`learner.gradient_booster.model.cats`** — the `{enc, feature_segments, sorted_idx}`
   block. It was `{"enc": [], "feature_segments": [], "sorted_idx": []}` in **every**
   model I fitted, including two models with a genuine categorical split and `feature_types
   == ['c','float']`. I could not produce a non-empty value, so I could not determine what
   it holds or when it is populated. It is a model-level (not per-tree) field, so it is
   probably a category re-encoding table, but that is a guess and is `INFERRED` at best.
2. **`base_weights` at internal nodes** — I confirmed the leaf relationship
   (`split_conditions == base_weights * eta`) but did not determine whether the internal-node
   values are used by anything other than SHAP/`pred_contribs`. Not needed for margin
   inference.
3. **`loss_changes` at internal nodes** — value is the split gain; I did not determine
   whether any prediction path reads it. Not needed for margin inference.
4. **`sum_hessian` semantics for non-squared-error objectives** — for
   `reg:squarederror` it equals the row count reaching the node. I did not measure it for
   other objectives.
5. **`num_target` vs `num_class`** — both live in `learner_model_param`. `num_class=0`
   with `num_target=1` for regression; `num_class=3, num_target=1` for `multi:softprob`;
   `num_class=0, num_target=2` for multi-target regression. I measured the values but did
   not determine whether `num_class=0` and `num_class=1` are ever both produced for
   binary classification (only `0` was observed).

Leaf-node field values, swept across 6 seeds × 4 trees, `max_depth=5`, 12% NaN injected:

```
$ uv run python p09_categorical_attrs.py
E. leaf-node field values across many fitted trees (what is safe to ignore)
  LEAF nodes    -> {'default_left': [0], 'split_indices': [0], 'loss_changes': [0.0], 'split_type': [0]}
  INTERNAL nodes-> {'split_type': [0], 'default_left': [0, 1]}
```

---

## 4. One complete tree, verbatim

Both the parsed form and the literal bytes as XGBoost writes them.

Parsed (`json.dumps(trees[0], indent=2)`):

```json
{
  "base_weights": [
    -7.0975608e-09,
    -1.5202886,
    1.3222975,
    -3.147526,
    -0.9670939,
    0.5819044,
    2.6416783
  ],
  "categories": [],
  "categories_nodes": [],
  "categories_segments": [],
  "categories_sizes": [],
  "default_left": [
    1,
    1,
    1,
    0,
    0,
    0,
    0
  ],
  "id": 0,
  "left_children": [
    1,
    3,
    5,
    -1,
    -1,
    -1,
    -1
  ],
  "loss_changes": [
    808.13,
    165.28955,
    208.48782,
    0.0,
    0.0,
    0.0,
    0.0
  ],
  "parents": [
    2147483647,
    0,
    0,
    1,
    1,
    2,
    2
  ],
  "right_children": [
    2,
    4,
    6,
    -1,
    -1,
    -1,
    -1
  ],
  "split_conditions": [
    -0.08433227,
    -1.0347615,
    0.9437468,
    -0.94425786,
    -0.29012817,
    0.17457134,
    0.79250354
  ],
  "split_indices": [
    0,
    0,
    0,
    0,
    0,
    0,
    0
  ],
  "split_type": [
    0,
    0,
    0,
    0,
    0,
    0,
    0
  ],
  "sum_hessian": [
    400.0,
    186.0,
    214.0,
    46.0,
    140.0,
    138.0,
    76.0
  ],
  "tree_param": {
    "num_deleted": "0",
    "num_feature": "4",
    "num_nodes": "7",
    "size_leaf_vector": "1"
  }
}
```

The same tree as literal bytes on disk. Note the number format: XGBoost writes floats in
an uppercase-`E` scientific form (`-8.433227E-2`, `4E2`, `0E0`), and `tree_param` values
as quoted strings.

```
"trees":[{"base_weights":[-7.0975608E-9,-1.5202886E0,1.3222975E0,-3.147526E0,-9.670939E-1,5.819044E-1,2.6416783E0],"categories":[],"categories_nodes":[],"categories_segments":[],"categories_sizes":[],"default_left":[1,1,1,0,0,0,0],"id":0,"left_children":[1,3,5,-1,-1,-1,-1],"loss_changes":[8.0813E2,1.6528955E2,2.0848782E2,0E0,0E0,0E0,0E0],"parents":[2147483647,0,0,1,1,2,2],"right_children":[2,4,6,-1,-1,-1,-1],"split_conditions":[-8.433227E-2,-1.0347615E0,9.437468E-1,-9.4425786E-1,-2.9012817E-1,1.7457134E-1,7.9250354E-1],"split_indices":[0,0,0,0,0,0,0],"split_type":[0,0,0,0,0,0,0],"sum_hessian":[4E2,1.86E2,2.14E2,4.6E1,1.4E2,1.38E2,7.6E1],"tree_param":{"num_deleted":"0","num_feature":"4","num_nodes":"7","size_leaf_vector":"1"}}
```

### 4.1 Layout claims, measured

```
$ uv run python p02_layout.py
=== leaf detection: left_children == -1 vs right_children == -1 agreement ===
tree 0: leaves_by_left=[3, 4, 5, 6] leaves_by_right=[3, 4, 5, 6] agree=True
tree 1: leaves_by_left=[3, 4, 5, 6] leaves_by_right=[3, 4, 5, 6] agree=True
tree 2: leaves_by_left=[3, 4, 5, 6] leaves_by_right=[3, 4, 5, 6] agree=True

=== at LEAF nodes: base_weights vs split_conditions, and ratio ===
  node 3: base_weights=-3.147526 split_conditions=-0.94425786 sc/bw=0.3000000190625907
  node 4: base_weights=-0.9670939 split_conditions=-0.29012817 sc/bw=0.30000000000000004
  node 5: base_weights=0.5819044 split_conditions=0.17457134 sc/bw=0.3000000343699068
  node 6: base_weights=2.6416783 split_conditions=0.79250354 sc/bw=0.30000001892736144

=== at INTERNAL nodes: split_indices / split_conditions / default_left ===
  node 0: feat_idx=0 thr=-0.08433227 default_left=1 L=1 R=2
  node 1: feat_idx=0 thr=-1.0347615 default_left=1 L=3 R=4
  node 2: feat_idx=0 thr=0.9437468 default_left=1 L=5 R=6

=== parents sentinel and parent consistency ===
tree 0: parents[0]=2147483647 == 2**31-1 ? True
  every child's parents[] entry points back at its parent: True
tree 1: parents[0]=2147483647 == 2**31-1 ? True
  every child's parents[] entry points back at its parent: True
tree 2: parents[0]=2147483647 == 2**31-1 ? True
  every child's parents[] entry points back at its parent: True

=== node index density: num_nodes vs array lengths, and reachability from 0 ===
tree 0: num_nodes=7 array_lens={'base_weights': 7, 'default_left': 7, 'left_children': 7, 'loss_changes': 7, 'parents': 7, 'right_children': 7, 'split_conditions': 7, 'split_indices': 7, 'split_type': 7, 'sum_hessian': 7} reachable=7 contiguous_0..n-1=True
tree 1: num_nodes=7 ... reachable=7 contiguous_0..n-1=True
tree 2: num_nodes=7 ... reachable=7 contiguous_0..n-1=True

=== reference float32 tree walk (BOTH sides cast) vs predict(output_margin=True) ===
base_score parsed: 0.12025498
max abs margin error (walk using split_conditions as leaf value): 0.0
max abs margin error (walk using base_weights as leaf value): 5.101206541061401
```

`0.0` — exact, over 400 rows — for the walk that reads leaf values from
`split_conditions`, casts both sides of every comparison, and accumulates in float32.
The same walk reading `base_weights` as the leaf value is off by **5.10** in margin. That
is the whole point of the distinction.

Cross-check against `get_dump` for the same tree, showing the correspondence between the
array form and the `yes` / `no` / `missing` form:

```
  { "nodeid": 0, "depth": 0, "split": "f0", "split_condition": -0.0843322724, "yes": 1, "no": 2, "missing": 1 , "gain": 808.130005, "cover": 400, "children": [
    { "nodeid": 1, "depth": 1, "split": "f0", "split_condition": -1.03476155, "yes": 3, "no": 4, "missing": 3 , "gain": 165.289551, "cover": 186, "children": [
      { "nodeid": 3, "leaf": -0.944257855 , "cover": 46 },
      { "nodeid": 4, "leaf": -0.290128171 , "cover": 140 }
    ]},
    { "nodeid": 2, "depth": 1, "split": "f0", "split_condition": 0.943746805, "yes": 5, "no": 6, "missing": 5 , "gain": 208.487823, "cover": 214, "children": [
      { "nodeid": 5, "leaf": 0.174571335 , "cover": 138 },
      { "nodeid": 6, "leaf": 0.792503536 , "cover": 76 }
    ]}
  ]}
```

For a **numeric** split, `yes == left_children[i]`, `no == right_children[i]`,
`missing == left_children[i]` when `default_left[i] == 1`. (For a **categorical** split
`yes` is the *right* child — see §9.)

### 4.2 Node index allocation

```
$ uv run python p07_deleted_and_indexing.py
=== NODE INDEX ALLOCATION RULE: is left_children[i] always 2i+1? ===
num_nodes=101 num_deleted=0
nodes where left_children[i] != 2i+1: 21 (first 10: [(38, 59, 77), (41, 61, 83), (42, 63, 85), (43, 65, 87), (44, 67, 89), (45, 69, 91), (46, 71, 93), (47, 73, 95), (48, 75, 97), (49, 77, 99)])
is right_children[i] always left_children[i]+1 for internal nodes? True
is left_children[i] > i for every internal node? True
index order is nondecreasing in depth (breadth-first): True
depth histogram: {0: 1, 1: 2, 2: 4, 3: 8, 4: 16, 5: 28, 6: 26, 7: 12, 8: 4}
```

So: **indices are allocated sequentially in breadth-first order, not positionally.**
`left_children[i] == 2i+1` holds only while the tree happens to be complete at that level
and is false in general (21 of 101 nodes here). A reader must follow the child arrays.

---

## 5. Missing-value routing: the `default_left` field, both directions demonstrated

**Field name:** `default_left`, an array of `int` with one entry per node.
**Semantics:** at internal node `i`, a missing value goes to `left_children[i]` when
`default_left[i] == 1`, and to `right_children[i]` when `default_left[i] == 0`.

Two single-split models (`max_depth=1`, one boosting round, `base_score=0`,
`eta=1.0`) were fitted on data where 150 of 600 rows have column 0 missing. The label
attached to those rows was flipped between the two fits to pull the learned default in
opposite directions. `predict(output_margin=True)` on a row with `NaN` in the split
feature is the arbiter: since `base_score` is exactly `0`, the returned margin *is* the
leaf value of whichever branch was taken.

```
$ uv run python p03_default_dir.py
==============================================================================
--- model 'left' (NaN rows labelled -10.0) ---
base_score: [0E0]  boost_from_average: 0
tree 0 verbatim:
{
 "base_weights": [-2.7620633, -9.973958, 9.954128],
 "categories": [], "categories_nodes": [], "categories_segments": [], "categories_sizes": [],
 "default_left": [1, 0, 0],
 "id": 0,
 "left_children": [1, -1, -1],
 "loss_changes": [55215.695, 0.0, 0.0],
 "parents": [2147483647, 0, 0],
 "right_children": [2, -1, -1],
 "split_conditions": [0.001264589, -9.973958, 9.954128],
 "split_indices": [0, 0, 0],
 "split_type": [0, 0, 0],
 "sum_hessian": [600.0, 383.0, 217.0],
 "tree_param": {"num_deleted": "0", "num_feature": "2", "num_nodes": "3", "size_leaf_vector": "1"}
}
root: split_indices=0 split_conditions=np.float32(0.001264589) default_left=1
  left child 1 leaf value = np.float32(-9.973958)
  right child 2 leaf value = np.float32(9.954128)
  predict(NaN) margin=np.float32(-9.973958)  margin-base_score=np.float32(-9.973958)
  => branch actually taken by NaN: LEFT
  default_left field says: LEFT  -> agreement: True
  value below thr -> margin np.float32(-9.973958) (left leaf np.float32(-9.973958))
  value above thr -> margin np.float32(9.954128) (right leaf np.float32(9.954128))
  get_dump:   { "nodeid": 0, ... "yes": 1, "no": 2, "missing": 1 , "children": [ { "nodeid": 1, "leaf": -9.97395802 }, { "nodeid": 2, "leaf": 9.95412827 } ]}
==============================================================================
--- model 'right' (NaN rows labelled 10.0) ---
base_score: [0E0]  boost_from_average: 0
tree 0 verbatim:
{
 "base_weights": [2.2296174, -9.957265, 9.972826],
 "categories": [], "categories_nodes": [], "categories_segments": [], "categories_sizes": [],
 "default_left": [0, 0, 0],
 "id": 0,
 "left_children": [1, -1, -1],
 "loss_changes": [56813.016, 0.0, 0.0],
 "parents": [2147483647, 0, 0],
 "right_children": [2, -1, -1],
 "split_conditions": [0.001264589, -9.957265, 9.972826],
 "split_indices": [0, 0, 0],
 "split_type": [0, 0, 0],
 "sum_hessian": [600.0, 233.0, 367.0],
 "tree_param": {"num_deleted": "0", "num_feature": "2", "num_nodes": "3", "size_leaf_vector": "1"}
}
root: split_indices=0 split_conditions=np.float32(0.001264589) default_left=0
  left child 1 leaf value = np.float32(-9.957265)
  right child 2 leaf value = np.float32(9.972826)
  predict(NaN) margin=np.float32(9.972826)  margin-base_score=np.float32(9.972826)
  => branch actually taken by NaN: RIGHT
  default_left field says: RIGHT  -> agreement: True
  value below thr -> margin np.float32(-9.957265) (left leaf np.float32(-9.957265))
  value above thr -> margin np.float32(9.972826) (right leaf np.float32(9.972826))
  get_dump:   { "nodeid": 0, ... "yes": 1, "no": 2, "missing": 2 , "children": [ { "nodeid": 1, "leaf": -9.9572649 }, { "nodeid": 2, "leaf": 9.972826 } ]}
```

Both directions demonstrated with `predict()` as the arbiter, on two models that are
otherwise identical in shape and share the same threshold. Symmetry was **not** assumed.

`default_left` also varies **within** a single tree — node 0 goes left, nodes 1 and 2 go
right:

```
=== a deeper tree with BOTH directions present in one tree ===
default_left per node: [1, 0, 0, 0, 0, 0, 0]
get_dump:
  { "nodeid": 0, "depth": 0, "split": "a", "split_condition": 0.000166309939, "yes": 1, "no": 2, "missing": 1 , "children": [
    { "nodeid": 1, "depth": 1, "split": "b", "split_condition": 0.00201491173, "yes": 3, "no": 4, "missing": 4 , "children": [
      { "nodeid": 3, "leaf": -5.98692799 },
      { "nodeid": 4, "leaf": -4.50234032 }
    ]},
    { "nodeid": 2, "depth": 1, "split": "b", "split_condition": 0.00124640786, "yes": 5, "no": 6, "missing": 6 , "children": [
      { "nodeid": 5, "leaf": 3.9893899 },
      { "nodeid": 6, "leaf": 5.98861504 }
    ]}
  ]}
```

`missing: 4` and `missing: 6` are the right children of nodes 1 and 2, matching
`default_left == 0` at those nodes.

**When no missing values appear in training,** every internal node in the models I fitted
came out `default_left == 1` — see the tree in §4, `[1,1,1,0,0,0,0]`. That is observed
behaviour on these fits, not a guarantee I established.

---

## 6. What counts as missing at predict time

Two discriminator models are used together, because a single model leaves cases
ambiguous. Both have the same root threshold `0.001264589` (which is **greater than
zero**, so a stored `0.0` and a missing entry land on opposite branches):

- `defL` model: `default_left = 1`. Missing → LEFT. A value `>= thr` → RIGHT.
- `defR` model: `default_left = 0`. Missing → RIGHT. A value `< thr` → LEFT.

A case is only called **MISSING** when `defL` routes left *and* `defR` routes right. If
both route the same way, the input was treated as an ordinary value.

```
$ uv run python p05_disambiguate.py
left : thr=np.float32(0.001264589) default_left=1 leafL=np.float32(-9.973958) leafR=np.float32(9.954128)
right: thr=np.float32(0.001264589) default_left=0 leafL=np.float32(-9.957265) leafR=np.float32(9.972826)

case                                      defL-model  defR-model   verdict
--------------------------------------------------------------------------------------------
dense nan, default missing=                        L           R   MISSING (follows default_left)
dense nan, missing=-999.0                          L           R   MISSING (follows default_left)
dense 0.0, missing=0.0                             L           R   MISSING (follows default_left)
dense -999.0, missing=-999.0                       L           R   MISSING (follows default_left)
dense 0.0, default missing=                        L           L   VALUE < thr
csr col0 absent, default missing=                  L           R   MISSING (follows default_left)
csr col0 stored 0.0, default missing=              L           L   VALUE < thr
csr col0 stored 0.0, missing=0.0                   L           R   MISSING (follows default_left)
csr col0 absent, missing=0.0                       L           R   MISSING (follows default_left)
csr col0 absent, missing=-999.0                    L           R   MISSING (follows default_left)
inplace +inf                                       R           R   VALUE >= thr
inplace -inf                                       L           L   VALUE < thr
inplace nan                                        L           R   MISSING (follows default_left)
inplace 0.0                                        L           L   VALUE < thr
inplace 0.0 missing=0.0                            L           R   MISSING (follows default_left)
inplace csr col0 absent                            L           R   MISSING (follows default_left)
dense +inf, missing=inf                            L           R   MISSING (follows default_left)
dense 1.0, missing=inf (control)                   R           R   VALUE >= thr
```

`+inf` and `-inf` through a `DMatrix` with the default `missing=` are not in that table
because **`DMatrix` construction raises**:

```
$ uv run python p04_missing_semantics.py
  [[+inf, 0.0]]  -> RAISED XGBoostError: [20:55:54] .../src/data/data.cc:1194: Check failed: valid: Input data contains `inf` or a value too large, while `missing` is not set to `inf`
  [[-inf, 0.0]]  -> RAISED XGBoostError: [20:55:54] .../src/data/data.cc:1194: Check failed: valid: Input data contains `inf` or a value too large, while `missing` is not set to `inf`
```

### 6.1 Per-candidate answer, measured

| Candidate | Measured result |
|---|---|
| `np.nan`, dense | **Missing.** Follows `default_left`. |
| `None` in a Python list `[[None, 0.0]]` | Routed like missing. `None` is coerced to `nan` before it reaches the engine — `np.array([[None,0.0]], dtype=float)` prints `[[nan 0.]]`. So this is the NaN path, not a separate one. |
| `None` in a numpy `dtype=object` array | Same — routed like missing, via the same coercion. |
| `+inf`, dense via `DMatrix`, default `missing=` | **Raises** `XGBoostError`. Never reaches the tree walk. |
| `-inf`, dense via `DMatrix`, default `missing=` | **Raises** `XGBoostError`. |
| `+inf` via `inplace_predict` | **An ordinary value**, not missing. Routes as `+inf >= thr` → RIGHT on both discriminators. |
| `-inf` via `inplace_predict` | **An ordinary value**, not missing. Routes as `-inf < thr` → LEFT on both discriminators. |
| `+inf` with `DMatrix(..., missing=np.inf)` | **Missing.** Follows `default_left`. Control row `1.0` with the same `missing=inf` is still an ordinary value. |
| Absent entry in a scipy CSR row (no stored entry for that column) | **Missing.** Follows `default_left`. Same via `inplace_predict` on a CSR. |
| **Explicitly stored `0.0`** in a CSR row, default `missing=` | **NOT missing** — treated as the value `0.0`. Routed LEFT on both discriminators. |
| Explicitly stored `0.0` in a CSR row, `missing=0.0` | **Missing.** |
| Dense `0.0`, default `missing=` | **NOT missing** — the value `0.0`. |
| Dense `0.0`, `missing=0.0` | **Missing.** |
| Dense `-999.0`, `missing=-999.0` | **Missing.** Any finite sentinel works. |
| `np.nan` with `missing=-999.0` set | **Still missing.** `NaN` remains missing even when `missing=` names a different sentinel. |

So the answer to "is a sparse zero missing?" is: **it depends on whether the zero is
stored.** An *absent* CSR entry is missing. A *stored* `0.0` is the value zero. Both were
measured, with the CSR internals printed to prove which case was constructed:

```
     csr_absent  indptr [0 1] indices [1] data [7.]
  csr, column 0 ABSENT (not stored)                          -> RIGHT (missing-default, or value >= thr)
     csr_stored0 indptr [0 2] indices [0 1] data [0. 7.]
  csr, column 0 stored as 0.0                                -> LEFT  (treated as value < thr)
     csr_storedNaN indptr [0 2] indices [0 1] data [nan  7.]
  csr, column 0 stored as nan                                -> RIGHT (missing-default, or value >= thr)
```

### 6.2 The `missing=` parameter is not recorded in the artifact

```
=== is any 'missing' sentinel recorded in the saved artifact? ===
  substring 'missing' present in model JSON: False
  substring 'nan' present in model JSON: False
  substring 'NaN' present in model JSON: False
  substring 'inf' present in model JSON: True
```

The one `inf` hit is a false positive: it is the substring inside `"tree_info"`. There is
**no record of `missing=` anywhere in the saved model.** `missing=` is a property of the
`DMatrix`, chosen per call site, not of the model. A model fitted with `missing=0` cannot
be distinguished from one fitted with the default by reading the artifact.

### 6.3 Threshold boundary, and feature-key strictness at predict time

The comparison is strictly `<` toward the left child. A value exactly equal to the
threshold goes RIGHT, and all float64 values within half a float32 ULP of the threshold
collapse to the same branch:

```
=== threshold boundary: value exactly == split_conditions ===
  exactly thr                      v=0.0012645890237763524 -> branch R  (np.float32(v) < np.float32(thr) = False -> predicts R)
  thr nextafter down (float64)     v=np.float64(0.0012645890237763522) -> branch R  (np.float32(v) < np.float32(thr) = False -> predicts R)
  thr nextafter up (float64)       v=np.float64(0.0012645890237763526) -> branch R  (np.float32(v) < np.float32(thr) = False -> predicts R)
  thr as float32 nextafter down    v=0.0012645889073610306 -> branch L  (np.float32(v) < np.float32(thr) = True -> predicts L)
```

XGBoost's own `DMatrix`-based `predict` is strict about feature names — this is
independent evidence for D005:

```
=== column-count / feature-name strictness at predict time ===
  1 column instead of 2            -> RAISED ValueError: feature_names mismatch: ['a', 'b'] ['a']
expected b in input data
  3 columns instead of 2           -> RAISED ValueError: feature_names mismatch: ['a', 'b'] ['a', 'b', 'c']
training data did not have the following fields: c
  right count, wrong names         -> RAISED ValueError: feature_names mismatch: ['a', 'b'] ['x', 'y']
expected a, b in input data
training data did not have the following fields: x, y
  right count, no names at all     -> RAISED ValueError: data did not contain feature names, but the following fields are expected: a, b
  names in swapped order           -> RAISED ValueError: feature_names mismatch: ['a', 'b'] ['b', 'a']
```

Note the last row: XGBoost rejects a **reordering** as well, it does not silently permute.

---

## 7. Degenerate cases

### (a) A single-node tree — a tree that is only a leaf

**Verdict: a degenerate case of the general layout.** Every array has length 1, the root
is a leaf, `left_children == right_children == [-1]`, `parents == [2147483647]`,
`num_nodes == "1"`, `num_deleted == "0"`. Nothing special-cases.

Forced with a constant label:

```
$ uv run python p06_degenerate.py
--- a1: constant label 4.0, boost_from_average on ---
  learner_model_param: {"base_score": "[4E0]", "boost_from_average": "1", "num_class": "0", "num_feature": "3", "num_target": "1"}
  gbtree_model_param : {"num_parallel_tree": "1", "num_trees": "2"}
  tree_info          : [0, 0]
  iteration_indptr   : [0, 1, 2]
  len(trees)         : 2
  tree verbatim: {"base_weights": [-0.0], "categories": [], "categories_nodes": [], "categories_segments": [], "categories_sizes": [], "default_left": [0], "id": 0, "left_children": [-1], "loss_changes": [0.0], "parents": [2147483647], "right_children": [-1], "split_conditions": [-0.0], "split_indices": [0], "split_type": [0], "sum_hessian": [200.0], "tree_param": {"num_deleted": "0", "num_feature": "3", "num_nodes": "1", "size_leaf_vector": "1"}}
  get_dump: ['  { "nodeid": 0, "leaf": -0 }', '  { "nodeid": 0, "leaf": -0 }']
  predict: [4. 4.]
```

The leaf value is **negative zero**. On disk it is written `-0E0`:

```
"trees":[{"base_weights":[-0E0], ... "split_conditions":[-0E0], ...
python json round-trip of that leaf value:
  value=-0.0  repr(float)=-0.0  is_neg_zero=True
  json.dumps(-0.0) = '-0.0'
  Math.fround equivalent np.float32(-0.0) = np.float32(-0.0)
```

Also forced by `min_child_weight=1e9`, which gives `+0.0` rather than `-0.0`:

```
--- a4: min_child_weight=1e9 ---
  tree verbatim: {"base_weights": [0.0], ..., "left_children": [-1], ..., "right_children": [-1], "split_conditions": [0.0], ..., "tree_param": {"num_deleted": "0", "num_feature": "3", "num_nodes": "1", "size_leaf_vector": "1"}}
```

`max_depth=0` is **not** a route to this shape under `tree_method="exact"`:

```
a3: max_depth=0 RAISED XGBoostError [20:57:43] .../src/tree/updater_colmaker.cc:176: Check failed: param_.max_depth > 0 (0 vs. 0) : exact tree method doesn't support unlimited depth.
```

#### (a′) The dangerous variant: a leaf-only root with 14 dead nodes still in the arrays

**Verdict: a genuinely different shape, and the sharpest format hazard found.** Blocking
splits with `gamma=1e9` (`min_split_loss`) produces a tree that grew and was then pruned.
The arrays are **not** truncated:

```
--- a2: gamma=1e9 (min_split_loss blocks every split), base_score=0 ---
  tree verbatim: {"base_weights": [-0.22855233, -2.5557902, 2.055294, -4.241374, -1.3613782, 0.76255953, 3.3330765, -5.6466174, -3.303595, -2.014946, -0.8023278, 0.18507661, 1.3329241, 2.6361854, 4.6927643], "categories": [], "categories_nodes": [], "categories_segments": [], "categories_sizes": [], "default_left": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "id": 0, "left_children": [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], "loss_changes": [1073.5786, 195.55408, 165.94534, 35.375977, 20.305992, 16.880848, 35.954285, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "parents": [2147483647, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6], "right_children": [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], "split_conditions": [-0.11427616, -1.2778951, 1.027647, -2.120687, -0.6806891, 0.38127977, 1.6665382, -2.8233087, -1.6517975, -1.007473, -0.4011639, 0.092538305, 0.66646206, 1.3180927, 2.3463821], "split_indices": [0, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647], "split_type": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "sum_hessian": [200.0, 99.0, 99.0, ...], "tree_param": {"num_deleted": "14", "num_feature": "3", "num_nodes": "15", "size_leaf_vector": "1"}}
  get_dump: ['  { "nodeid": 0, "leaf": -0.114276163 }', '  { "nodeid": 0, "leaf": -0.057422366 }']
```

`num_nodes == "15"` but only node 0 is live; `num_deleted == "14"`. Note that
`parents[1] == 0` while `left_children[0] == -1` — the parent links are **stale**, they
describe the tree before pruning. Deleted nodes carry `split_indices == 2147483647`.

A gamma sweep shows the deleted set is **not** in general a contiguous suffix:

```
$ uv run python p07_deleted_and_indexing.py
gamma=0.0      num_nodes=59   num_deleted=0    len(arrays)=59   reachable=59   unreachable=0
   unreachable indices          : []
   split_indices==INT32_MAX at  : []
   unreachable == INT32_MAX set : True
   num_deleted == len(unreachable): True
   float32 walk (deleted nodes never visited) max abs err vs predict: 0.0

gamma=5.0      num_nodes=59   num_deleted=2    len(arrays)=59   reachable=57   unreachable=2
   unreachable indices          : [31, 32]
   split_indices==INT32_MAX at  : [31, 32]
   unreachable == INT32_MAX set : True
   num_deleted == len(unreachable): True
   unreachable is a contiguous suffix: False
   sample deleted node 31: parents=15 left=-1 right=-1 split_cond=-4.3974967 base_weight=-8.794993 sum_hessian=42.0 default_left=1
   stale parent link check: parents[31]=15, left_children[15]=-1, right_children[15]=-1
   float32 walk (deleted nodes never visited) max abs err vs predict: 0.0

gamma=50.0     num_nodes=59   num_deleted=22   len(arrays)=59   reachable=37   unreachable=22
   unreachable indices          : [31, 32, 33, 34, 37, 38, 41, 42, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]
   split_indices==INT32_MAX at  : [31, 32, 33, 34, 37, 38, 41, 42, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]
   unreachable == INT32_MAX set : True
   num_deleted == len(unreachable): True
   unreachable is a contiguous suffix: False
   float32 walk (deleted nodes never visited) max abs err vs predict: 0.0

gamma=200.0    num_nodes=59   num_deleted=28   ...  unreachable is a contiguous suffix: True
   float32 walk (deleted nodes never visited) max abs err vs predict: 0.0
gamma=2000.0   num_nodes=59   num_deleted=48   ...  unreachable is a contiguous suffix: True
   float32 walk (deleted nodes never visited) max abs err vs predict: 0.0
gamma=1000000000.0 num_nodes=59  num_deleted=58 ...  unreachable is a contiguous suffix: True
   float32 walk (deleted nodes never visited) max abs err vs predict: 0.0
```

At `gamma=50.0` the dead indices are `[31,32,33,34,37,38,41,42,45,...]` — **interleaved
with live indices**. Two independent markers of deadness agreed on every one of these
six sweeps: unreachability from the root, and `split_indices == 2147483647`. A walk that
starts at node 0 and follows the child arrays never touches a dead node, and reproduces
`predict()` with max abs error `0.0` in all six cases.

### (b) A model with exactly one feature

**Verdict: a degenerate case of the general layout.** No shape change at all. Feature
index `0` everywhere, `num_feature == "1"`, everything else identical.

```
--- b: one feature, named 'only' ---
  num_feature (learner_model_param): 1
  feature_names      : ['only']
  tree verbatim: {"base_weights": [2.9505783e-08, -1.7921084, 1.4692061, -3.085871, -1.0381038, 0.6600204, 2.5982723], ..., "default_left": [1, 1, 1, 0, 0, 0, 0], "id": 0, "left_children": [1, 3, 5, -1, -1, -1, -1], ..., "right_children": [2, 4, 6, -1, -1, -1, -1], "split_conditions": [-0.05594106, -0.96645176, 0.8568711, -1.5429355, -0.5190519, 0.3300102, 1.2991362], "split_indices": [0, 0, 0, 0, 0, 0, 0], ..., "tree_param": {"num_deleted": "0", "num_feature": "1", "num_nodes": "7", "size_leaf_vector": "1"}}
```

### (c) Zero boosting rounds

**Verdict: a degenerate case of the general layout** — every container is simply empty.
The full raw file, verbatim:

```
--- c: num_boost_round=0 ---
  gbtree_model_param : {"num_parallel_tree": "1", "num_trees": "0"}
  tree_info          : []
  iteration_indptr   : [0]
  len(trees)         : 0
  get_dump: []
  num_boosted_rounds(): 0
  predict: [0.5 0.5 0.5]
  RAW FILE:
{"learner":{"attributes":{},"feature_names":["p","q","r"],"feature_types":[],"gradient_booster":{"model":{"cats":{"enc":[],"feature_segments":[],"sorted_idx":[]},"gbtree_model_param":{"num_parallel_tree":"1","num_trees":"0"},"iteration_indptr":[0],"tree_info":[],"trees":[]},"name":"gbtree"},"learner_model_param":{"base_score":"[5E-1]","boost_from_average":"1","num_class":"0","num_feature":"3","num_target":"1"},"objective":{"name":"reg:squarederror","reg_loss_param":{"scale_pos_weight":"1"}}},"version":[3,3,0]}
```

Points worth carrying forward: `trees: []`, `tree_info: []`, `iteration_indptr: [0]` (not
`[]`), `num_trees: "0"`. `base_score` is the untouched default `0.5` even though
`boost_from_average` is `"1"` — nothing was fitted, so nothing overwrote it. `predict`
returns `0.5`.

### (d) The deepest tree

**Verdict: a degenerate case of the general layout.** Only the array lengths grow. **There
is no depth field anywhere** — not in `tree_param`, not per node. Depth is implicit in the
child links and must be derived if it is wanted at all. Trees are **ragged**, not complete.

```
--- d: max_depth=8 ---
  num_nodes=467 array_len=467 leaves=234 internal=233
  observed max node depth (walked via parents)=8
  perfect binary tree of that depth would have 511 nodes -> tree is RAGGED (not complete)
  reachable=467 contiguous=True
  any 'depth'/'max_depth' key in tree_param? ['num_deleted', 'num_feature', 'num_nodes', 'size_leaf_vector']
  breadth-first vs depth-first index order? left_children[:16]=[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
  right_children[:16]=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
  parents[:16]=[2147483647, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7]
  default_left values seen at LEAF nodes: [0]
  loss_changes at LEAF nodes all zero: True
  split_indices at LEAF nodes: [0]
  split_type distinct values: [0]

--- d: max_depth=14 ---
  num_nodes=4749 array_len=4749 leaves=2375 internal=2374
  observed max node depth (walked via parents)=14
  perfect binary tree of that depth would have 32767 nodes -> tree is RAGGED (not complete)
  reachable=4749 contiguous=True
  any 'depth'/'max_depth' key in tree_param? ['num_deleted', 'num_feature', 'num_nodes', 'size_leaf_vector']
```

`leaves == internal + 1` in both cases, consistent with a strictly binary tree.

### Array-length audit across all 45 saved models

```
$ uv run python p10_final_checks.py
D. num_nodes vs len(arrays) vs num_deleted across every saved model
  file / num_nodes / num_deleted / size_leaf_vector / lens(lc,rc,bw,sc,dl,si,st,par,loss,hess) / has leaf_weights
   m01.json 7 0 1 (7, 7, 7, 7, 7, 7, 7, 7, 7, 7) False
   m03_left.json 3 0 1 (3, 3, 3, 3, 3, 3, 3, 3, 3, 3) False
   m06_a1_const_label.json 1 0 1 (1, 1, 1, 1, 1, 1, 1, 1, 1, 1) False
   m06_a2_gamma.json 15 14 1 (15, 15, 15, 15, 15, 15, 15, 15, 15, 15) False
   m06_d_depth14.json 4749 0 1 (4749, ... ) False
   m06_d_depth8.json 467 0 1 (467, ...) False
   m07_gamma5.0.json 59 2 1 (59, 59, 59, 59, 59, 59, 59, 59, 59, 59) False
   m07_ragged.json 101 0 1 (101, ...) False
   m08_multitarget.json 7 0 2 (7, 7, 14, 7, 7, 7, 7, 7, 7, 7) True
   m09_cat.json 5 0 1 (5, 5, 5, 5, 5, 5, 5, 5, 5, 5) False
```

Every per-node array is exactly `num_nodes` long **except `base_weights`, which is
`num_nodes * size_leaf_vector`**. That is only visible in the multi-target row.

### (g) Not on the required list, but a genuinely different shape: vector leaves

`multi_strategy="multi_output_tree"` with 2 targets produces a tree that is **not** a
degenerate case — it is a different shape, and several invariants above break:

```
$ uv run python p08_ensemble.py
--- multi_output_tree, 2 targets ---
  num_target                       : 2
  per-tree size_leaf_vector        : ['2', '2']
  tree 0 verbatim: {"base_weights": [1.1986363e-08, -4.598986e-09, -0.61925685, 0.3140636, 1.0255419, -0.5201159, 0.10790828, -0.056119557, -0.38423836, 0.19581689, 0.54268587, -0.27491748, 0.064365745, -0.032962717], "categories": [], "categories_nodes": [], "categories_segments": [], "categories_sizes": [], "default_left": [0, 0, 0, 0, 0, 0, 0], "id": 0, "leaf_weights": [0.10790828, -0.056119557, -0.38423836, 0.19581689, 0.54268587, -0.27491748, 0.064365745, -0.032962717], "left_children": [1, 3, 5, -1, -1, -1, -1], "loss_changes": [400.80853, 256.40552, 150.31548, 0.0, 0.0, 0.0, 0.0], "parents": [-1, 0, 0, 1, 1, 2, 2], "right_children": [2, 4, 6, 0, 1, 2, 3], "split_conditions": [0.43510655, -0.3109272, -0.01921234, 1e-45, 1e-45, 1e-45, 1e-45], "split_indices": [0, 1, 1, 0, 0, 0, 0], "split_type": [0, 0, 0, 0, 0, 0, 0], "sum_hessian": [1000.0, 624.0, 376.0, 252.0, 372.0, 190.0, 186.0], "tree_param": {"num_deleted": "0", "num_feature": "3", "num_nodes": "7", "size_leaf_vector": "2"}}
```

Differences, all measured:

- A new field appears: **`leaf_weights`**, length `num_leaves * size_leaf_vector` = 8.
- `base_weights` is length 14 = `num_nodes * size_leaf_vector`, while `default_left`,
  `left_children`, etc. stay at 7.
- **`parents[0] == -1`**, not `2147483647`.
- **`right_children` at leaf nodes is `0, 1, 2, 3`** — the leaf's block index into
  `leaf_weights` — **not `-1`**. So `right_children[i] == -1` is not a valid leaf test.
- `split_conditions` at leaves is `1e-45` (garbage), not the leaf value.

Confirmed across all saved models that these two shapes are the only two, and that the two
sentinels split exactly along `size_leaf_vector`:

```
distinct per-tree key sets:
   ('base_weights', 'categories', 'categories_nodes', 'categories_segments', 'categories_sizes', 'default_left', 'id', 'leaf_weights', 'left_children', 'loss_changes', 'parents', 'right_children', 'split_conditions', 'split_indices', 'split_type', 'sum_hessian', 'tree_param')
   ('base_weights', 'categories', 'categories_nodes', 'categories_segments', 'categories_sizes', 'default_left', 'id', 'left_children', 'loss_changes', 'parents', 'right_children', 'split_conditions', 'split_indices', 'split_type', 'sum_hessian', 'tree_param')
parents[0] sentinel by size_leaf_vector: [(1, 2147483647), (2, -1)]
right_children at leaf nodes, by size_leaf_vector: [(1, -1), (2, 0), (2, 1), (2, 2), (2, 3)]
distinct size_leaf_vector: [1, 2]
distinct learner.feature_types tuples: [(), ('c', 'float'), ('float', 'float', 'int')]
```

---

## 8. Float32 discipline: the three readings of a serialized threshold, arbitrated by `predict()`

A serialized threshold is the shortest decimal that round-trips in **float32**. Parsing it
with `json.load` (or `JSON.parse`) gives a **float64** that is *not* bit-equal to the
float32 the engine uses:

```
$ uv run python p10_final_checks.py
  verbatim split_conditions in file: "split_conditions":[-3.078444E-2,-3.2618387E0,3.1117811E0]
  threshold as float64 : -0.03078444  exact=-0x1.f85f4cc1a4a59p-6
  threshold as float32 : np.float32(-0.03078444)  exact_as_f64=-0x1.f85f4c0000000p-6
  float64(thr) == float64(float32(thr)) ? False
```

Three candidate readings, arbitrated by `predict()`:

1. `f64 both` — `v < thr_parsed_as_float64`
2. `cast sample only` — `float(np.float32(v)) < thr_parsed_as_float64`
3. `f32 both` — `np.float32(v) < np.float32(thr)`

```
$ uv run python p11_float32_boundary.py
thr as parsed float64 : -0.03078444  hex=-0x1.f85f4cc1a4a59p-6
thr as float32        : np.float32(-0.03078444)  hex=-0x1.f85f4c0000000p-6
float32 ULP near thr  : -1.862645149230957e-09
thr64 - float(thr32)  : -7.044696805069695e-10
leafL=np.float32(-3.2618387)  leafR=np.float32(3.1117811)

sample value                f64both  cast-sample-only  f32both  predict
-----------------------------------------------------------------------
-0.03078444115817547           True              True     True     LEFT
-0.03078444069251418           True              True     True     LEFT
-0.030784440226852894          True             False    False    RIGHT  <-- f64-both DISAGREES with f32-both
-0.030784440000000003          True             False    False    RIGHT  <-- f64-both DISAGREES with f32-both
-0.03078444                   False             False    False    RIGHT
-0.030784439999999996         False             False    False    RIGHT
-0.030784439761191607         False             False    False    RIGHT
-0.03078443929553032          False             False    False    RIGHT
-0.030784438829869032         False             False    False    RIGHT
-0.030784438364207745         False             False    False    RIGHT
-0.030784437898546457         False             False    False    RIGHT
-0.03078443743288517          False             False    False    RIGHT

rows where 'f64 both' disagrees with predict()          : 2
rows where 'cast sample only' disagrees with predict()  : 0
rows where 'f32 both' disagrees with predict()          : 0
```

On that threshold, `cast sample only` happens to agree. It does **not** agree in general.
Searching for a threshold whose decimal parses to a float64 *greater* than the float32
value produces a case where casting only the sample is wrong on every probe row:

```
$ uv run python p12_one_sided_cast.py
found in m01.json tree 0 node 0
  thr64 = -0.08433227  hex=-0x1.596ccb5a5bec1p-4
  thr32 = np.float32(-0.08433227)  hex=-0x1.596ccc0000000p-4
  float(thr32) < thr64 : True

sample value                f32(v)==thr32  f64both  cast-sample-only  f32both
-0.08433227241039276                 True     True              True    False
-0.08433227                          True    False              True    False
-0.08433227000000001                 True     True              True    False

=== predict() as arbiter on a single-split model with this property ===
  seed=1 thr64=0.045511134 thr32=np.float32(0.045511134) leafL=np.float32(-2.9646049) leafR=np.float32(3.3863893)
  sample value                f64both  cast-only  f32both  predict
  0.04551113396883011            True       True    False    RIGHT
  0.045511134                   False       True    False    RIGHT
  0.045511133999999995           True       True    False    RIGHT
  0.0455111339688301             True       True    False    RIGHT
  disagreements vs predict(): f64both=3 cast-sample-only=4 f32both=0
```

**`cast sample only` is wrong on 4 of 4. `f64 both` is wrong on 3 of 4. `f32 both` is wrong
on 0 of 4.** This is a directly usable adversarial boundary construction: take a
threshold's serialized decimal, parse to float64, and probe at
`float(np.float32(thr))`, `thr64`, `nextafter(thr64, ±inf)`.

---

## 9. Categorical splits (`split_type == 1`)

Present in the field inventory because the `categories*` arrays exist in **every** tree,
including trees with no categorical split (as four empty arrays). A reader that silently
ignores them will produce wrong numbers on a categorical model rather than raising.

```
$ uv run python p09_categorical_attrs.py
A. CATEGORICAL SPLIT: what split_type / categories_* look like
  feature_types: ['c', 'float']
  model-level cats: {"enc": [], "feature_segments": [], "sorted_idx": []}
  tree 0 verbatim:
{
 "base_weights": [-0.01549797, 5.0032234, -4.994254, -5.1510167, -4.8282037],
 "categories": [0, 2, 5],
 "categories_nodes": [0],
 "categories_segments": [0],
 "categories_sizes": [3],
 "default_left": [1, 0, 0, 0, 0],
 "id": 0,
 "left_children": [1, -1, 3, -1, -1],
 "loss_changes": [50023.95, 0.0, 1.2910156, 0.0, 0.0],
 "parents": [2147483647, 0, 0, 2, 2],
 "right_children": [2, -1, 4, -1, -1],
 "split_conditions": [1e-45, 5.0032234, 0.051638976, -5.1510167, -4.8282037],
 "split_indices": [0, 0, 1, 0, 0],
 "split_type": [1, 0, 0, 0, 0],
 "sum_hessian": [2000.0, 996.0, 1004.0, 501.0, 503.0],
 "tree_param": {"num_deleted": "0", "num_feature": "2", "num_nodes": "5", "size_leaf_vector": "1"}
}
  get_dump:
  { "nodeid": 0, "depth": 0, "split": "k", "split_condition": [0, 2, 5], "yes": 2, "no": 1, "missing": 1 , "children": [ ... ]}
```

Routing verified with `predict()`, not read off the dump. `base_score` is exactly `0` and
the left child of the root is a leaf, so the margin identifies the branch:

```
$ uv run python p10_final_checks.py
A. CATEGORICAL ROUTING verified with predict()
  root: split_type=1  split_indices= 0  default_left= 1  left= 1  right= 2
  categories = [0, 2, 5]  categories_nodes = [0]  categories_segments = [0]  categories_sizes = [3]
   k=0 (in categories set: True) margin=np.float32(-5.1510167) -> RIGHT child
   k=1 (in categories set: False) margin=np.float32(5.0032234) -> LEFT child
   k=2 (in categories set: True) margin=np.float32(-5.1510167) -> RIGHT child
   k=3 (in categories set: False) margin=np.float32(5.0032234) -> LEFT child
   k=4 (in categories set: False) margin=np.float32(5.0032234) -> LEFT child
   k=5 (in categories set: True) margin=np.float32(-5.1510167) -> RIGHT child
   k=NaN margin=np.float32(5.0032234) -> LEFT child  (default_left=1)
```

Measured semantics: at a node with `split_type[i] == 1`, **the value being IN the category
set routes to `right_children[i]`; not being in it routes to `left_children[i]`.** This is
the *opposite* nesting from a numeric split, where the "yes" branch is the left child.
`default_left` still governs missing. `split_conditions[i]` is the smallest positive
subnormal float32 and must not be read as a threshold:

```
B. the 1e-45 value sitting in split_conditions at categorical/vector-leaf nodes
  float32(1e-45) bits: 00000001  == smallest positive subnormal float32: True
  float32(1e-45) as float: 1.401298464324817e-45
```

With `max_cat_to_onehot` large, XGBoost emits one-category-per-node splits — same
encoding, `categories_sizes == [1, 1]`, two entries in `categories_nodes`:

```
B. one-hot style categorical (max_cat_to_onehot large)
  tree 0 verbatim: {..., "categories": [0, 2], "categories_nodes": [0, 1], "categories_segments": [0, 1], "categories_sizes": [1, 1], "default_left": [1, 1, 0, 0, 0], "id": 0, "left_children": [1, 3, -1, -1, -1], ..., "right_children": [2, 4, -1, -1, -1], "split_conditions": [1e-45, 1e-45, -4.986205, 2.5966096, -4.9858475], "split_indices": [0, 0, 0, 0, 0], "split_type": [1, 1, 0, 0, 0], ...}
```

Three independent signals for "this model contains a categorical split", all measured
present together: `split_type` contains `1`; `categories_nodes` is non-empty;
`learner.feature_types` contains `'c'`.

---

## 10. The ensemble above the tree level

```
$ uv run python p08_ensemble.py
A. named features (DMatrix feature_names given)
  num_feature (learner_model_param): 3
  num_target                       : 1
  num_class                        : 0
  feature_names                    : ['alpha', 'beta', 'gamma_']
  feature_types                    : []
  num_trees / num_parallel_tree    : {'num_parallel_tree': '1', 'num_trees': '3'}
  tree_info                        : [0, 0, 0]
  iteration_indptr                 : [0, 1, 2, 3]
  per-tree 'id' values             : [0, 1, 2]
  per-tree num_feature             : ['3', '3', '3']
  per-tree size_leaf_vector        : ['1', '1', '1']
  per-tree num_nodes               : ['7', '7', '7']
  per-tree key sets identical      : True
  learner.attributes               : {}
  top-level learner keys           : ['attributes', 'feature_names', 'feature_types', 'gradient_booster', 'learner_model_param', 'objective']
```

| Item | Where it lives | Measured |
|---|---|---|
| Tree count | `gradient_booster.model.gbtree_model_param.num_trees` (a **string**) | Always equalled `len(trees)` in every model measured. |
| Tree ordering | position in `trees[]` | Sequential; `id == position` for all 45 models, 0 mismatches. |
| Tree identifier | `trees[i].id` | Positional only. Model slicing renumbers it (§10.2). Not a durable identity. |
| Per-tree grouping | `gradient_booster.model.tree_info` — one int per tree | Output-group index. `[0,0,0]` for regression and binary; `[0,1,2,0,1,2,0,1,2]` for `multi:softprob` with 3 classes. |
| Boosting-round boundaries | `gradient_booster.model.iteration_indptr` | Length `rounds + 1`, CSR-style offsets into `trees[]`. `[0,1,2,3]` for 3 rounds × 1 tree; `[0,4,8,12]` for `num_parallel_tree=4`; `[0,3,6,9]` for 3 classes; `[0]` for zero rounds. |
| Feature count | `learner.learner_model_param.num_feature` (a **string**), duplicated in every `trees[i].tree_param.num_feature` | Matched in every model measured. |
| Feature names | `learner.feature_names` — array of string | Present when the `DMatrix` was given names. |
| Feature types | `learner.feature_types` — array of string | **Empty `[]` when not supplied**, even when names *are* supplied. Populated as e.g. `['float','float','int']` or `['c','float']` when given. |
| Booster kind | `learner.gradient_booster.name` | `"gbtree"`. |
| Objective | `learner.objective.name` plus an objective-specific param block | e.g. `{"name": "binary:logistic", "reg_loss_param": {"scale_pos_weight": "1"}}`. |
| Format version | top-level `version` | `[3, 3, 0]` — an array of three ints, not a string. |

Selected raw outputs:

```
D. num_parallel_tree = 4 (forest per round)
  num_trees / num_parallel_tree    : {'num_parallel_tree': '4', 'num_trees': '12'}
  tree_info                        : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  iteration_indptr                 : [0, 4, 8, 12]
  per-tree 'id' values             : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

E. binary:logistic (tree_info / base_score space)
  tree_info                        : [0, 0, 0]
  iteration_indptr                 : [0, 1, 2, 3]
  objective block: {"name": "binary:logistic", "reg_loss_param": {"scale_pos_weight": "1"}}
  base_score: [5.06E-1]

F. multi:softprob (grouping only; out of 1.0 scope, measured for tree_info shape)
  num_class                        : 3
  num_trees / num_parallel_tree    : {'num_parallel_tree': '1', 'num_trees': '9'}
  tree_info                        : [0, 1, 2, 0, 1, 2, 0, 1, 2]
  iteration_indptr                 : [0, 3, 6, 9]
```

The `multi:softprob` row is the D003 extensibility observation and nothing more: per-class
grouping is expressed entirely by `tree_info`, with no change to the tree object itself.
`tree_info` is already a per-tree field in the single-group case, where it is uniformly
`0`.

### 10.1 Feature names when the model was fit from a plain array with no names

**`learner.feature_names` is the empty array `[]`.** It is not omitted, and no synthetic
names are stored. `num_feature` still records the count, and `get_dump` synthesizes
`f0, f1, ...` at dump time only — those names are **not** in the artifact.

```
B. NO feature names at all (plain numpy array)
  num_feature (learner_model_param): 3
  feature_names                    : []
  feature_types                    : []
  bst.feature_names -> None
  bst.num_features() -> 3
  get_dump() tree 0  ->   { "nodeid": 0, "depth": 0, "split": "f0", "split_condition": 0.431312025, "yes": 1, "no": 2, "missing": 1 , "children": [ ... ]}
  RAW learner prefix:
    {"learner":{"attributes":{},"feature_names":[],"feature_types":[],
```

So a strict-feature-key predictor has **nothing to be strict about** for an unnamed model.
`len(feature_names) == 0` while `num_feature == 3` is a real, reachable state — see
*Ambiguities*.

### 10.2 `trees[i].id` is renumbered by slicing

```
D. tree 'id' under model slicing
  full  ids: [0, 1, 2, 3, 4, 5]
  slice ids: [0, 1, 2]
  slice num_trees: 3  tree_info: [0, 0, 0]  iteration_indptr: [0, 1, 2, 3]
```

`bst[2:5]` renumbers `id` to `0,1,2` and rebuilds `iteration_indptr`. `id` therefore
carries no information beyond array position.

---

## 11. What can vary between two fits of the same data

Export must be byte-deterministic, so this was measured directly.

```
$ uv run python p08_ensemble.py
H. DETERMINISM: same data + params, refit in this process and in a fresh process
  in-process refit hashes: ['957f32a5beaa69b5', '957f32a5beaa69b5', '957f32a5beaa69b5']
  all identical: True
  incl. fresh-process hashes: ['957f32a5beaa69b5', '957f32a5beaa69b5', '957f32a5beaa69b5', '957f32a5beaa69b5', '957f32a5beaa69b5']
  ALL identical across processes: True

  save_model twice from the SAME booster object:
   identical: True
```

Five fits — three in-process, two in freshly spawned interpreters — produced
**byte-identical** SHA-256. `gbtree` + `tree_method="exact"` + fixed `seed` is
reproducible, and nothing timestamp-like or address-like leaks into the model JSON. Key
order inside each tree object is stable and alphabetical:

```
=== KEY ORDER as it appears in the file (tree 0) ===
['base_weights', 'categories', 'categories_nodes', 'categories_segments', 'categories_sizes', 'default_left', 'id', 'left_children', 'loss_changes', 'parents', 'right_children', 'split_conditions', 'split_indices', 'split_type', 'sum_hessian', 'tree_param']
```

### 11.1 The one thing that does vary: `learner.attributes`

`attributes` was `{}` in every plain fit. **Early stopping writes into it**, and so does
any user call to `set_attr`:

```
$ uv run python p09_categorical_attrs.py
C. learner.attributes — does anything land there?
  attributes after early stopping: {"best_iteration": "10", "best_score": "0.21466829613403393"}
  num_trees: 16
  iteration_indptr len: 17
  best_iteration attr: 10
  attributes after set_attr(note=...): {"best_iteration": "10", "best_score": "0.21466829613403393", "note": "freeform"}
```

Two consequences worth flagging:

- `best_score` is a full-precision float rendered as a **string**. It is data-dependent
  and validation-split-dependent, so two fits that differ only in evaluation data will
  differ here even when every tree is identical.
- `num_trees == 16` while `best_iteration == 10`: **the trees past the best iteration are
  still serialized.** Nothing in the tree representation marks them.

### 11.2 `-0.0` is a real serialized value and a round-trip hazard

From §7(a): a leaf value can be `-0E0`. `json.dumps(-0.0)` in Python gives `'-0.0'`, but
`JSON.stringify(-0)` in JavaScript gives `'0'`. A round-trip that passes a leaf value
through JavaScript serialization can therefore change the bytes without changing the
number. Flagged for the format spec; the JavaScript side of that claim is `INFERRED` from
the language spec and was not measured here (no Node was run in this probe).

---

## 12. Minimal set of fields required for margin inference

Measured, not assumed: the walk in §4.1 and the six walks in §7(a′) reproduce
`predict(output_margin=True)` with max abs error **exactly `0.0`** while reading only:

```
left_children, right_children, split_indices, split_conditions, default_left, split_type
```

plus `tree_param.num_nodes` for bounds, `tree_info` for grouping, and
`learner_model_param.base_score`. `base_weights`, `loss_changes`, `sum_hessian`,
`parents`, `id`, and `num_deleted` were never read by those walks. That does **not** mean
the format should drop them silently — `num_deleted` in particular is the only cheap
cross-check on the dead-node count, and D007 says an unrecognized field must raise rather
than be skipped.

---

## Ambiguities — presented, not resolved

**A1. Two valid tests for a deleted node.** Reachability from node 0, and
`split_indices[i] == 2147483647`. Both agreed on all six gamma sweeps (§7a′), so I cannot
distinguish them empirically. Reading 1: reachability is the definition and
`split_indices` is incidental. Reading 2: `split_indices == INT32_MAX` is the deliberate
marker and reachability merely follows. This matters for whether an exporter can
*validate* (compute both and raise on disagreement) or must pick one. I measured only
`gbtree` + `exact` + `gamma` pruning; other pruning paths were not measured.

**A2. `inf` behaves differently on the two prediction paths.** `DMatrix` **raises** on
`±inf` with the default `missing=`; `inplace_predict` **accepts** it as an ordinary value
and compares it against the threshold. Reading 1: the reference predictor should raise on
`±inf`, matching `DMatrix`, the path most users take, and consistent with "a crash beats
a wrong number". Reading 2: the reference predictor should compare it, matching
`inplace_predict`, which is the path that is actually semantically defined. These give
different predictions for the same input, so the choice is a decision, not a detail.

**A3. `missing=` is not in the artifact (§6.2).** A model fitted with `missing=0` routes
stored zeros to the default branch at fit time, but nothing in the artifact records that.
Reading 1: the artifact is faithful and `missing` is a caller concern; the predictor
treats only `NaN`/absent as missing. Reading 2: the exporter should refuse to export, or
record the value, when the training `DMatrix` used a non-default `missing`. I have no
evidence for which XGBoost intends.

**A4. `feature_names == []` with `num_feature == 3` (§10.1).** Reading 1: strict feature
keys (D005) cannot apply, so export should raise on an unnamed model. Reading 2: export
should accept it and the predictor should take positional input for that case. Reading 3:
export should synthesize `f0..fN-1` to match `get_dump`. I lean away from reading 3
because those names are not in the artifact and synthesizing them invents data, but this
is a D005-adjacent public-API decision.

**A5. Categorical routing inverts the child convention (§9).** In-set → right child, while
numeric "yes" → left child. Measured, and unambiguous as a fact — the ambiguity is scope.
Categorical features are not excluded by the 1.0 objective list (binary / regression /
Cox), and the `categories*` arrays are present in every tree. Reading 1: `split_type == 1`
raises on export (D007), and the four `categories*` arrays must be verified empty rather
than ignored. Reading 2: categorical is in scope because it is orthogonal to objective.

**A6. `best_iteration` in `attributes` with untruncated trees (§11.1).** Reading 1: export
all `num_trees` trees and ignore `best_iteration` — the artifact mirrors the model.
Reading 2: honour `best_iteration` and export only the first `iteration_indptr[best+1]`
trees, matching what `predict()` does by default in the sklearn wrapper. These produce
different predictions. Not mine to resolve.

**A7. `learner.gradient_booster.model.cats` (§3.1).** Empty in every model I could
produce, including categorical ones. I could not determine what populates it. An exporter
following D007 must decide whether a non-empty `cats` raises or is understood, and I have
no data to inform that.

---

## Not measured

Stated so the gap is visible rather than assumed away:

- **DataFrame / Arrow input.** `pandas` and `pyarrow` are absent from this environment, so
  pandas-nullable dtypes, `pd.NA`, and Arrow nulls were not measured. The task's
  "absent when predicting from a dict-like input" case was covered via scipy CSR instead.
- **GPU / `device="cuda"`** paths.
- **`sum_hessian` for objectives other than `reg:squarederror`.**
- **`booster="dart"` and `booster="gblinear"`** tree/model shapes — a separate probe owns
  those.
- **Node-level JavaScript behaviour** for the `-0.0` round-trip claim in §11.2.
- **`split_type` values other than `0` and `1`.** Only those two were produced.
- **Pruning paths other than `gamma`** for the deleted-node layout.

---

## Scratch inventory

45 fitted models and 12 probe scripts, all under
`…/scratchpad/probe-tree-structure/`. Nothing was written into the repository except this
file.

| Script | Covers |
|---|---|
| `p01_inventory.py` | §1, §3, §4 field inventory and verbatim tree |
| `p02_layout.py` | §4.1 layout semantics, float32 walk parity |
| `p03_default_dir.py` | §5 both-directions demonstration |
| `p04_missing_semantics.py` | §6 first pass, `inf` rejection, name strictness |
| `p05_disambiguate.py` | §6 two-discriminator disambiguation, boundary |
| `p06_degenerate.py` | §7 cases (a)–(d) |
| `p07_deleted_and_indexing.py` | §7a′ gamma sweep, §4.2 index allocation, `-0.0` |
| `p08_ensemble.py` | §10 ensemble, §7g vector leaves, §11 determinism |
| `p09_categorical_attrs.py` | §9 categorical, §11.1 attributes, §10.2 slicing |
| `p10_final_checks.py` | §9 categorical routing via `predict()`, array-length audit |
| `p11_float32_boundary.py` | §8 three readings |
| `p12_one_sided_cast.py` | §8 one-sided cast failure |
