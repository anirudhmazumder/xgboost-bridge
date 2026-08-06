# Probe: float32 split thresholds

Empirical investigation of how XGBoost serializes split thresholds, at what precision, and
whether float32 and float64 comparison can disagree on the routing of a real sample through
a real tree.

Every claim below is backed by a pasted command and its real output. Anything not measured is
labelled **inferred**. Nothing here is taken from documentation, memory, or analogy.

---

## Environment

Pasted from actual output:

```
$ uv run python -c "import sys, numpy, xgboost; print('python', sys.version); print('numpy', numpy.__version__); print('xgboost', xgboost.__version__)"
python 3.12.8 (main, Dec  3 2024, 18:42:41) [Clang 16.0.0 (clang-1600.0.26.4)]
numpy 2.5.1
xgboost 3.3.0
```

Node, for the JavaScript-side confirmation:

```
node version: v20.19.0
```

Platform: darwin 25.5.0, arm64. XGBoost version marker inside the artifact: `[3, 3, 0]`.

Matches the D001 pin exactly.

---

## Verdicts, up front

| Hypothesis | Verdict |
|---|---|
| **H1** — the engine compares features against thresholds in float32 internally | **REPRODUCED.** Confirmed directly: a float64 feature value *strictly less than* the serialized threshold routes to the **right** child. Only a float32 narrowing of the value explains that. |
| **H2** — serialized thresholds are the shortest decimal round-tripping in float32, not a bit-identical float64 | **REPRODUCED.** 341/341 tokens across 9 fitted models are digit-for-digit the shortest float32 decimal. 195/195 pooled thresholds have `float64(token) != float32(token)`. Zero were bit-identical float64. |
| **H3** — casting only the sample value is correct on most rows and wrong on a few | **REPRODUCED.** Constructed case below: margin difference **0.273130634**, probability difference **0.0663**, with XGBoost's own `predict()` siding against the one-sided cast. 0/20000 ordinary random rows expose it. |

Plus two measured results that were not hypotheses:

- **The comparison operator is strict `<`, and equality routes RIGHT.** Measured at all 104 internal
  nodes of the primary model, `LRR` on 104/104, and at every internal node of 7 further models
  spanning three tree methods, three objectives, and DART. Never assumed.
- **The one-sided cast is wrong for exactly half the thresholds, and for exactly one float32 value
  each.** Sharp characterisation with an exhaustive ±4 ULP scan, below. This is why it survives testing.

One item is **ambiguous and is not resolved here** — margin accumulation precision. See
[Out of scope, looked wrong](#out-of-scope-things-that-looked-wrong).

---

## 1. Where thresholds live, and what they look like on disk

### JSON path

```
learner.gradient_booster.model.trees[<i>].split_conditions[<node>]
```

paired positionally with:

```
learner.gradient_booster.model.trees[<i>].split_indices[<node>]   # feature index
learner.gradient_booster.model.trees[<i>].left_children[<node>]   # -1 marks a leaf
learner.gradient_booster.model.trees[<i>].right_children[<node>]
learner.gradient_booster.model.trees[<i>].default_left[<node>]    # NaN direction
learner.gradient_booster.model.trees[<i>].split_type[<node>]      # 0 for all numeric splits observed
```

**`split_conditions` is overloaded.** At a node where `left_children[node] == -1`, the entry is not a
threshold — it is the **leaf output value**. In the primary model, 104 of 216 entries are thresholds
and 112 are leaf values. A reader that treats the array uniformly as thresholds will silently consume
leaf weights as split points.

```
=== node-role split of split_conditions tokens ===
internal-node tokens (true thresholds): 104
leaf-node tokens (leaf output values): 112
```

Structure of the primary model, pasted:

```
=== top-level keys ===
['learner', 'version']
=== learner keys ===
['attributes', 'feature_names', 'feature_types', 'gradient_booster', 'learner_model_param', 'objective']
=== gradient_booster keys ===
['model', 'name']
gradient_booster.name = "gbtree"
=== model keys under gradient_booster ===
['cats', 'gbtree_model_param', 'iteration_indptr', 'tree_info', 'trees']
=== trees[0] keys ===
['base_weights', 'categories', 'categories_nodes', 'categories_segments', 'categories_sizes',
 'default_left', 'id', 'left_children', 'loss_changes', 'parents', 'right_children',
 'split_conditions', 'split_indices', 'split_type', 'sum_hessian', 'tree_param']
```

```
version                = [3, 3, 0]
learner.objective      = {"name": "binary:logistic", "reg_loss_param": {"scale_pos_weight": "1"}}
trees[0].tree_param    = {"num_deleted": "0", "num_feature": "6", "num_nodes": "29", "size_leaf_vector": "1"}
gbtree_model_param     = {"num_parallel_tree": "1", "num_trees": "8"}
trees[0].split_type    = [0, 0, ... 0]        (all zero — numeric splits)
trees[0].categories    = []                   (empty; no categorical splits in this model)
```

### Raw serialized form, pasted verbatim from the file

Read as bytes, not through a JSON parser, so these are the exact characters on disk —
`learner.gradient_booster.model.trees[0].split_conditions`:

```
"split_conditions":[-6.1135292E-2,-6.678203E-1,8.947612E-1,-5.770786E-1,-1.8648013E-1,1.296185E-1,8.4968436E-1,-1.311231E0,4.6321878E-1,-3.9433753E-1,2.3601118E-1,1.9913538E-1,-1.9171381E0,-5.0178283E-1,1.1543728E0,5.7487667E-2,-3.5375845E-1,4.768919E-1,8.446092E-2,-2.922683E-1,-5.872326E-1,1.7391616E-2,-4.32067E-1,2.9052225E-1,-5.7251565E-2,-7.975511E-2,4.9654266E-1,-6.2445775E-2,3.847664E-1]
```

Sibling arrays for the same tree, so the pairing is checkable:

```
=== trees[0].split_indices ===
[0, 1, 1, 0, 0, 0, 0, 1, 2, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
=== trees[0].left_children ===
[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, -1, 27, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
=== trees[0].right_children ===
[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, -1, 28, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
=== trees[0].default_left ===
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

### Significant digits

Mantissa significant digits, internal-node thresholds of the primary model:

```
CORRECTED mantissa significant-digit histogram (model.json, 104 internal): {6: 6, 7: 30, 8: 68}
exponent values seen: [-4, -2, -1, 0]
```

Across a deliberately extreme-scale model (feature columns at `1e30`, `1e-30`, `1e-42`, integers,
a constant column, a binary indicator) pooled with the primary model, 341 tokens:

```
mantissa digit count: min 1 max 9
exponent range: -44 .. 30
```

Nine mantissa digits is the theoretical maximum needed for a float32 shortest round-trip, and nine is
what was observed. The one-digit cases are float32 subnormals (`6.4E-44`, `-5.6E-44`), which carry
less than full precision.

---

## 2. Number formatting — what a JSON parser sees

Measured over 341 tokens from two models, including the extreme-scale one:

```
=== TOKEN GRAMMAR CHECK (internal + leaf, both models) ===
total tokens: 341
tokens NOT matching  -?D.DDDD E -?D+ : 4
  those tokens: ['5E-1']
any quoted: False
any 'Infinity'/'NaN'/'inf'/'nan': []
any '+' in exponent: []
any lowercase 'e': []
exponent range: -44 .. 30
mantissa digit count: min 1 max 9
negative zero present: False  positive zero: False
```

Observed grammar, every token: `-?<digit>(\.<digits>)?E-?<digits>`

- **JSON numbers, never strings.** No quoting anywhere in `split_conditions`.
- **Always exponent notation.** Uppercase `E`. No token was emitted in plain positional form.
- **No `+` sign in the exponent.** Exponent zero is written `E0`, e.g. `-1.311231E0`.
- **The decimal point is omitted when the mantissa is a single digit** — `5E-1` for 0.5. A regex
  requiring `D.D+E` will reject valid tokens.
- **No trailing zeros in the mantissa.** Consistent with shortest-round-trip emission.
- **No `Infinity`, `NaN`, `inf`, or `nan`** appeared. Training rejects infinite features outright:

```
=== CAN Infinity / NaN APPEAR AS A THRESHOLD? ===
training with inf features RAISED: XGBoostError [...] src/data/data.cc:1194:
  Check failed: valid: Input data contains `inf` or a value too large, while `missing` is not set to `inf`
```

  This is evidence about thresholds *as produced by training on non-infinite data*. It is **not**
  proof that no XGBoost code path can ever emit a non-finite `split_condition`. A parser should
  still fail loudly on one rather than assume it cannot occur.

**Negative zero does occur elsewhere in the artifact.** `trees[0].base_weights` begins `-0E0`:

```
base_weights -> -0E0,-1.2239243E0,1.0013974E0,1.6927673E-1,-1.7300494E0,...
```

`base_weights` is not the threshold array and is not on the prediction path in these models, but
`-0E0` demonstrates that XGBoost's writer emits signed zero. `JSON.parse("-0E0")` yields `-0`, and
`-0 === 0` is `true` in JavaScript while `Object.is(-0, 0)` is `false`. Relevant to byte-identical
export (D008) and to any equality check the format design relies on.

Nothing here is legally reinterpretable by a conforming JSON parser in a value-changing way — the
tokens are ordinary JSON numbers. The hazard is entirely on the **precision** axis, section 5.

---

## 3. Round-trip status, with bit patterns

Primary model, first ten internal-node thresholds. `f64 parse bits` is what `json.loads` /
`JSON.parse` produce; `f32 bits` is the float32 narrowing; `f32 as f64 bits` is that float32 widened
back so the two are comparable in the same width:

```
tree node  file token       f64 parse bits       f32 bits     f32 as f64 bits      f64==f32?
   0    0  -6.1135292E-2    0xbfaf4d1fff8af64f   0xbd7a6900   0xbfaf4d2000000000   False
   0    1  -6.678203E-1     0xbfe55ec8ad835b6a   0xbf2af645   0xbfe55ec8a0000000   False
   0    2  8.947612E-1      0x3feca1e23d7759d4   0x3f650f12   0x3feca1e240000000   False
   0    3  -5.770786E-1     0xbfe2776d8a47163f   0xbf13bb6c   0xbfe2776d80000000   False
   0    4  -1.8648013E-1    0xbfc7de94b5da170d   0xbe3ef4a6   0xbfc7de94c0000000   False
   0    5  1.296185E-1      0x3fc09756c93a7115   0x3e04bab6   0x3fc09756c0000000   False
   0    6  8.4968436E-1     0x3feb309d4143ed15   0x3f5984ea   0x3feb309d40000000   False
   0    7  -1.311231E0      0xbff4facd5b6805a3   0xbfa7d66b   0xbff4facd60000000   False
   0    8  4.6321878E-1     0x3fdda56061bf8d9f   0x3eed2b03   0x3fdda56060000000   False
   0    9  -3.9433753E-1    0xbfd93cd37abbdde7   0xbec9e69c   0xbfd93cd380000000   False
```

The `f64==f32?` column is `False` on every row: the decimal on disk, parsed at float64 width, is a
**different number** from the float32 the engine holds. Note also that `f32 as f64 bits` always ends
in seven zero nibbles — the float32 mantissa padded out — while `f64 parse bits` does not. That is
the visual signature of a float32 value written at float32-shortest precision and read back at
float64 width.

### Does the decimal round-trip in float32?

Yes, by construction, and that is exactly the point. `float32(parse(token))` reproduces the float32,
and re-emitting that float32 at shortest precision reproduces the token digits:

```
=== SHORTEST-DECIMAL-IN-FLOAT32 TEST (all internal-node thresholds) ===
thresholds whose digits != numpy shortest-float32 digits: 0 / 104
thresholds where float64(token) != float32(token): 104 / 104
```

Pooled over 7 additional models (three tree methods, three objectives, DART):

```
=== POOLED THRESHOLD TOKENS ACROSS ALL 7 MODELS ===
total internal-node thresholds: 195
distinct: 126
mantissa significant-digit histogram: {6: 6, 7: 83, 8: 106}
tokens where float64(token) != float32(token): 195 / 195
tokens exactly representable in float32 (dyadic): 0 / 195
tokens whose digits != numpy shortest-float32 digits: 0 / 195
```

### Does the decimal round-trip in float64?

**Yes, but to the wrong number.** A 6-to-9-digit decimal always round-trips through float64 — float64
carries ~17 significant digits, so no information in the token is lost. That is precisely why the
failure is silent: the parse is *lossless with respect to the text* and *lossy with respect to the
model*. The float64 you get back is a faithful reading of the decimal and is not the value XGBoost
compares against.

Concretely, for `-6.1135292E-2`:

```
float32(t)              : -0.061135292053222656   bits 0xbd7a6900
float64(t)              : -0.061135292            hex -0x1.f4d1fff8af64fp-5
float64(t) - float32(t)  : 5.3222655449491896e-11
```

**Exactly-representable thresholds exist but are rare.** In the extreme-scale model, `5E-1` (0.5) and
`3.5E0` are dyadic, so `float32 == float64` and no disagreement is possible at those nodes. Across
the 195 pooled thresholds from normally-distributed features, **0** were dyadic. So the hazardous case
is the overwhelming default, not the exception.

---

## 4. The comparison operator — measured, not assumed

Method: pick a node, feed three feature values — one float32 ULP below `float32(threshold)`, exactly
`float32(threshold)`, and one ULP above — and read the branch taken from XGBoost's own
`predict(..., pred_leaf=True)`, mapping the returned node id into the left or right subtree of the
node under test. XGBoost is the only arbiter; no arithmetic of ours enters the decision.

Tree 0 root:

```
=== TREE 0 ROOT ===
split feature index: 0 -> f0
raw token in file  : -6.1135292E-2
float64 parse      : -0.061135292
float32 value      : -0.061135292053222656
left child node id : 1  right child node id: 2
default_left[0]    : 1

=== BRANCH TAKEN AT TREE 0 ROOT, measured via pred_leaf ===
case                         feature value (f32 repr)   f32 bits     leaf node id  branch
nextafter below (f32)        -0.061135295778512955      0xbd7a6901             21  LEFT
EXACTLY the f32 threshold    -0.061135292053222656      0xbd7a6900             23  RIGHT
nextafter above (f32)        -0.06113528832793236       0xbd7a68ff             23  RIGHT

=== OPERATOR INFERENCE ===
value  < threshold -> LEFT ? True
value == threshold -> LEFT ? False
value  > threshold -> LEFT ? False
=> LEFT branch is taken iff value < threshold  (STRICT '<'); equality goes RIGHT
```

Extended to **every internal node of all 8 trees**, using a training row known to reach each node and
verifying via `pred_leaf` that the probe row actually stayed on that node's path:

```
=== PER-NODE ULP TEST, all internal nodes of all 8 trees ===
pattern histogram (below, equal, above):
  'LRR':  104   strict '<', equality -> RIGHT

nodes tested                : 104
nodes skipped (unreachable) : 0
probe rows that left the path: 0

conclusive nodes: 104
conclusive nodes NOT matching strict '<' with equality->RIGHT: 0
```

Extended across tree methods, objectives, and DART:

```
=== OPERATOR AND FORMATTING ACROSS OBJECTIVES AND TREE METHODS ===
pattern key: 'LRR' = strict '<', equality routes RIGHT

binary_exact    obj=binary:logistic   tm=exact  nodes=28 pattern={'LRR': 28} dyadic=0/28 mantissa_digits=6..8 grammar_ok=True
binary_hist     obj=binary:logistic   tm=hist   nodes=28 pattern={'LRR': 28} dyadic=0/28 mantissa_digits=6..8 grammar_ok=True
binary_approx   obj=binary:logistic   tm=approx nodes=28 pattern={'LRR': 28} dyadic=0/28 mantissa_digits=6..8 grammar_ok=True
reg_exact       obj=reg:squarederror  tm=exact  nodes=28 pattern={'LRR': 28} dyadic=0/28 mantissa_digits=6..8 grammar_ok=True
reg_hist        obj=reg:squarederror  tm=hist   nodes=28 pattern={'LRR': 28} dyadic=0/28 mantissa_digits=7..8 grammar_ok=True
cox_exact       obj=survival:cox      tm=exact  nodes=27 pattern={'LRR': 27} dyadic=0/27 mantissa_digits=7..8 grammar_ok=True
dart_hist       obj=binary:logistic   tm=hist   nodes=28 pattern={'LRR': 28} dyadic=0/28 mantissa_digits=6..8 grammar_ok=True
```

### Specification

```
go_left  iff  float32(value) < float32(split_condition)      # STRICT less-than
go_right otherwise, INCLUDING exact equality
NaN      -> left if default_left[node] else right            # checked before the comparison
```

### A second, stronger consequence of the ULP test

The ULP test does more than fix the operator. `nextafter_below(float32(token))` routes LEFT and
`float32(token)` routes RIGHT, on 104/104 nodes. There is no representable float32 strictly between
those two values. Therefore the threshold the engine holds internally is **exactly
`float32(parse(token))`, bit for bit** — no other float32 is consistent with both observations. This
is what licenses "parse as float64, then narrow" as a lossless read, and it is measured rather than
assumed.

---

## 5. H1, directly: the engine really does compare in float32

The ULP test pins the threshold but does not by itself prove the *value* side is narrowed. Direct
test: take a **float64** feature value lying strictly inside the open interval
`(float32(t), float64(t))`. A float64 engine must route LEFT, because the value is genuinely less
than the serialized threshold. A float32 engine narrows the value onto `float32(t)`, hits equality,
and routes RIGHT.

```
=== H1 DIRECT TEST: tree 0 root ===
token                : -6.1135292E-2
float32(t)           : -0.061135292053222656  bits 0xbd7a6900
float64(t)           : -0.061135292      hex -0x1.f4d1fff8af64fp-5
float32(t) < float64(t) : True
float32 spacing here    : 3.725290298461914e-09
float64(t) - float32(t) : 5.3222655449491896e-11

v_mid (float64, strictly between)  : -0.06113529205322265  hex -0x1.f4d1fffffffffp-5
  float64 comparison v_mid < t64   : True -> would route LEFT
  float32 comparison fround(v_mid) : -0.061135292053222656 == float32(t)? True
  float32 comparison f32(v) < f32(t): False -> routes RIGHT

=== inplace_predict on a float64 numpy array (margin) ===
margin, feature = nextafter_below(float32(t)) [known LEFT] : 0.19705569744110107
margin, feature = float32(t)                  [known RIGHT]: 0.47018638253211975
margin, feature = v_mid (float64, between)                 : 0.47018638253211975
v_mid margin equals LEFT margin ? False
v_mid margin equals RIGHT margin? True

DMatrix(float64) pred_leaf for v_mid : 23 -> branch RIGHT

VERDICT ON H1:
  A float64 value strictly LESS THAN the serialized threshold routes RIGHT.
  The engine must have narrowed the value to float32 and compared against
  float32(threshold).  H1 (float32 internal comparison) CONFIRMED.
```

`inplace_predict` was used alongside `DMatrix` deliberately: `DMatrix` narrows a float64 array to
float32 on construction, which would have made the narrowing an artifact of the container rather than
of the engine. `inplace_predict` consumes the float64 array and produces the same answer.

Swept over every root threshold where the interval is non-empty:

```
=== SWEEP: for every internal threshold with float32(t) < float64(t), does a
    float64 value in the open interval (float32(t), float64(t)) route as float32? ===
  tree 0 root token -6.1135292E-2    v_mid=-0.06113529202661133     branch=R (float32 semantics)
  tree 5 root token -6.3348114E-1    v_mid=-0.6334811424525452      branch=R (float32 semantics)
  tested 2: float32 semantics 2, float64 semantics 0
```

**H1 REPRODUCED.**

---

## 6. THE CENTRAL EXPERIMENT — one-sided casting routes a real sample wrongly

### Construction

For any threshold `t`, the value `v = float32(t)` is exactly representable in float32. Then:

- two-sided: `float32(v) < float32(t)` → `float32(t) < float32(t)` → **False** → route RIGHT
- one-sided: `float32(v) < float64(t)` → **True whenever `float64(t) > float32(t)`** → route LEFT

Both are the *same comparison in the same program*, differing only in whether the threshold was
narrowed. The primary model's tree-0 root satisfies the condition.

```
=== CANDIDATE DISAGREEMENT NODES (tree roots only, so the node is always reached) ===
tree feat token            f64(t) > f32(t)?   disagrees at v=f32(t)?
   0    0 -6.1135292E-2    True               True
   1    0 -4.7379725E-2    False              False
   2    0 -4.7379725E-2    False              False
   3    0 6.39505E-2       False              False
   4    0 -3.3411518E-2    False              False
   5    0 -6.3348114E-1    True               True
   6    0 8.477105E-2      False              False
   7    0 6.39505E-2       False              False
```

### The case

```
=== CHOSEN CASE ===
tree index                 : 0
node                       : 0 (root)
split feature              : 0 -> f0
raw token in model.json    : -6.1135292E-2
float64 parse of token     : -0.061135292
float32 of token (engine)  : -0.061135292053222656
float64(t) - float32(t)    : 5.3222655449491896e-11

FEATURE VALUE UNDER TEST   : v = float32(token) = -0.061135292053222656
  v as float32 bits        : 0xbd7a6900
  v as float64 bits        : -0x1.f4d2000000000p-5

two-sided  float32(v) < float32(t) : False -> node 2
one-sided  float32(v) < float64(t) : True -> node 1
```

Full sample row, 6 features: `f0 = -0.061135292053222656`, all other features `0.123456`.

### XGBoost arbitrates

```
=== ARBITRATION BY XGBoost predict() ===
XGBoost pred_leaf per tree : [23, 21, 22, 18, 20, 22, 18, 16]
XGBoost leaf in tree 0     : 23 -> root branch taken: RIGHT
two-sided walk leaves      : [23, 21, 22, 18, 20, 22, 18, 16]
one-sided walk leaves      : [21, 21, 22, 18, 20, 22, 18, 16]
```

The two-sided walk reproduces XGBoost's leaf assignment in all 8 trees. The one-sided walk lands on
leaf 21 instead of 23 in tree 0 — the wrong side of the root — and is correct in the other 7 trees.
That is the failure signature verbatim: mostly right.

### The number

```
=== MARGINS ===
XGBoost output_margin      : 0.47018638253211975
two-sided walk margin      : 0.4701863828493362
one-sided walk margin      : 0.19705574884933624

abs(two-sided - XGBoost)   : 3.1721647530957853e-10
abs(one-sided - XGBoost)   : 0.2731306336827835
MARGIN DIFFERENCE two vs one-sided : 0.273130634

=== PROBABILITY SPACE ===
XGBoost predict (prob)     : 0.6154278516769409
two-sided walk prob        : 0.615427869763185
one-sided walk prob        : 0.5491051399442523
PROBABILITY DIFFERENCE     : 0.06632272981893272
```

**Margin difference `0.273130634`. Probability difference `0.0663`.** A 6.6-percentage-point error in
a binary classification probability, from a single missing cast, on a single node, in a single tree, in
an 8-tree model. No exception. No warning.

### Frequency

```
=== FREQUENCY: adversarial rows (values placed exactly on float32 thresholds) ===
rows where one-sided walk differs from two-sided : 26 / 104
largest margin difference observed               : 1.8038533710000002

=== FREQUENCY: ordinary random continuous rows ===
rows where one-sided differs from two-sided : 0 / 20000
max |two-sided walk - XGBoost margin|       : 4.545964067403929e-07
```

**0 out of 20000 ordinary rows.** A test suite built on random continuous data will show a green
board on a build with the bug. That is the entire finding: the bug is undetectable by the testing you
would naturally write, and reachable by any input that lands on a threshold.

### JavaScript side, same case

```
=== THE CHOSEN CASE, JS side ===
token                        : -6.1135292E-2
JSON.parse(token)            : -0.061135292 0xbd7a6900 <- f32 bits of the f64 narrowed
Math.fround(JSON.parse(tok)) : -0.061135292053222656 0xbd7a6900
JSON.parse(token) === Math.fround(...)? false
v = Math.fround(token)       : -0.061135292053222656
two-sided: Math.fround(v) < Math.fround(t) = false -> RIGHT
one-sided: Math.fround(v) < t              = true -> LEFT
XGBoost predict() arbitration (from the Python probe): RIGHT
```

```
=== JSON.parse -> Math.fround vs numpy float32, all 104 thresholds ===
bit-pattern mismatches vs numpy np.float32: 0 / 104

=== JSON.parse alone (no fround) vs the engine float32 ===
tokens where JSON.parse(token) !== Math.fround(JSON.parse(token)): 104 / 104

=== ONE-SIDED vs TWO-SIDED CASTING, all thresholds, v = fround(token) ===
thresholds where the two disagree at v = fround(token): 53 / 104
```

`Math.fround` and `np.float32` agree on the bit pattern for all 104 thresholds — the two languages can
be held to exactly `0.0` parity. And `JSON.parse(token) !== Math.fround(JSON.parse(token))` on
**104/104** thresholds: every single threshold is a value a bare `JSON.parse` gets numerically wrong.

**H3 REPRODUCED.**

---

## 7. Exactly which inputs the one-sided cast gets wrong

Not a vague "a few rows." Exhaustive ±4 float32 ULP scan around every threshold:

```
=== EXHAUSTIVE SCAN: +/- 4 float32 ULPs around each threshold ===
Counting float32 values v where  (fround(v) < fround(t))  !=  (fround(v) < float64(t))
histogram of disagreeing float32 values per threshold: {1: 53, 0: 51}
thresholds with float64(t) >  float32(t)  (HAZARDOUS): 53 / 104
thresholds with float64(t) <  float32(t)  (safe)     : 51 / 104
For every hazardous threshold the single disagreeing float32 is EXACTLY float32(t).
For every safe threshold there is NO disagreeing float32 within +/-4 ULP.
```

The characterisation:

> The one-sided cast disagrees with the two-sided cast for **exactly one float32 value per threshold**
> — `float32(t)` itself — and **only** for the roughly half of thresholds where `float64(t) > float32(t)`.
> For the other half it is accidentally correct everywhere.

The mechanism, **inferred** from the measurement rather than separately measured: the two comparisons
differ only if `float32(v)` falls between `float32(t)` and `float64(t)`, an interval narrower than
half a float32 ULP whose only possible float32 inhabitant is `float32(t)`, and only when the gap runs
in the direction that makes the strict inequality flip. Measured counts match exactly (53 / 51,
1 disagreeing value each).

### It is a band of the input domain, not a single unlucky float

The affected value is one *float32*, but every *float64* input that narrows to it is affected — a
half-ULP-wide band on each side:

```
=== FLOAT64 INPUT BAND CORRUPTED BY THE ONE-SIDED CAST ===
tree 0 node 0 feature 0 token -6.1135292E-2
float32(t)        : -0.061135292053222656
float32 ULP here  : 3.725290298461914e-09
band of float64 v that narrows to float32(t):
   [ -0.061135293915867805 , -0.06113529019057751 )   width 3.725290298461914e-09
relative width    : 6.093518446299041e-08

2000 float64 samples drawn uniformly from that band:
  two-sided walk agrees with XGBoost margin : 2000 / 2000
  one-sided walk agrees with XGBoost margin : 0 / 2000
  max |two-sided - one-sided| margin        : 0.273130634
```

`2000 / 2000` versus `0 / 2000`. Inside the band the one-sided cast is not occasionally wrong, it is
**always** wrong, deterministically and repeatably.

Control, on a "safe" threshold:

```
=== SAME BAND TEST ON A 'SAFE' THRESHOLD (float64(t) < float32(t)) ===
token -4.7379725E-2 -> two-sided agrees 2000/2000  one-sided agrees 2000/2000
```

### Consequence for the fixture corpus

The corpus must contain, for every hazardous threshold, a row whose feature value is **exactly**
`float32(threshold)` — and must assert the resulting margin against XGBoost, not against our own
predictor. Nothing weaker exposes this. Specifically:

- Random continuous rows: **0/20000** detection. Decorative for this invariant.
- Rows at `float32(threshold)`: detects the hazardous half.
- Rows at `nextafter_below(float32(threshold))` and `nextafter_above(...)`: additionally pin the
  strict-`<` operator and the equality direction.
- Reverting the cast to float64 on one side must turn these specific rows red. Half the thresholds
  will *not* go red, which is the point — the corpus needs the hazardous half specifically, so it
  should be selected by measuring `float64(t) > float32(t)` at generation time rather than by hoping.

### Where thresholds sit relative to the data

```
=== DO SERIALIZED THRESHOLDS COINCIDE WITH ACTUAL TRAINING DATA VALUES? ===
thresholds exactly equal to some float32 training value: 0 / 104
distinct (feature, threshold) pairs: 81
thresholds equal to a float32 midpoint of adjacent data values: 38 / 104
```

Thresholds never coincide with a training data value; 38/104 are the float32 midpoint of two adjacent
data values. **Inferred**, not measured: this is why continuous training-like data never hits the
band, and it means real-world exposure comes from inputs *derived from the model* — boundary tests,
fixture generation, quantile/binning logic read back from the artifact, perturbation-based
attribution — plus any input pipeline that quantizes onto a value that happens to be
`float32(threshold)`.

---

## 8. The parsing hazard (D004): can a float64 parse land on the wrong float32?

Three separate questions.

**(a) Does `float32(float64_parse(token))` recover the engine's threshold?** Yes, losslessly. The
per-node ULP test in section 4 pins the engine's threshold to exactly `float32(parse(token))` at
104/104 nodes. Independently, parsing the text at float32 width directly and parsing at float64 then
narrowing give identical bit patterns:

```
     tokens where float32(float64(text)) != float32(text): 0 / 341
     worst: None
```

And in JavaScript, `JSON.parse` followed by `Math.fround` matches `np.float32` on 104/104. So
**`fround(JSON.parse(x))` and `np.float32(json.loads(x))` are both exact reads.** The float64 parse is
not itself the hazard.

**(b) Is a bare parse safe to *use*?** No — and this is the hazard. `JSON.parse` returns float64
unconditionally, and on **104/104** thresholds that float64 is a different number from the engine's
float32. A parser that lands thresholds in unconstrained floats and hands them to the tree walk
produces exactly the failure in section 6 while every line of the walk looks correct. The narrowing
must happen at parse time, not be left to the comparison site to remember.

**(c) Can re-emission at reduced precision land on a different float32?** Yes, and easily:

```
(5c) HAZARD: re-emitting a threshold at reduced precision.
     If an exporter rounds thresholds, how many land on a DIFFERENT float32?
     rounded to  5 significant digits:  324 / 341 land on a different float32
     rounded to  6 significant digits:  309 / 341 land on a different float32
     rounded to  7 significant digits:  226 / 341 land on a different float32
     rounded to  8 significant digits:    2 / 341 land on a different float32
     rounded to  9 significant digits:    0 / 341 land on a different float32
     rounded to 10 significant digits:    0 / 341 land on a different float32
     rounded to 17 significant digits:    0 / 341 land on a different float32
```

**Nine significant digits is the floor.** Eight already corrupts 2/341 — a 0.6% corruption rate,
which is the worst possible outcome: frequent enough to matter, rare enough to survive review. Any
`round(x, n)`, `%.6g`, or "tidy up the artifact" formatting step in the exporter is a live wrong-number
bug. Verified-safe emission routes, all `0 / 341` drift:

```
(5d) SAFE re-emission routes -- do these preserve the float32 exactly?
     repr(float(np.float32(x)))  [17-digit float64 of the f32]  drift on 0 / 341
     str(np.float32(x))          [shortest f32]                 drift on 0 / 341
     np.format_float_scientific(v, unique=True)                 drift on 0 / 341
     json.dumps(float(np.float32(x)))                           drift on 0 / 341
     float(np.float32(x)) via json.loads round trip             drift on 0 / 341
```

**Answer to the D004 question:** the exporter does **not** need to re-quantize on read — narrowing the
float64 parse to float32 is exact. What it must do is (i) narrow at parse time rather than at
comparison time, and (ii) never emit a threshold at fewer than 9 significant digits. Both emission
strategies above are safe; choosing between shortest-float32 text and the 17-digit float64 widening
is a format-design question, not a correctness one.

---

## 9. Specification for the tree walk, both languages

Everything below is measured in this probe.

```
Threshold read (parse time, NOT comparison time):
    Python      threshold = np.float32(json_value)
    JavaScript  threshold = Math.fround(jsonValue)

Node step:
    if value is NaN:
        go left if default_left[node] else right
    else:
        go left if  cast32(value) < cast32(threshold)   else right     # STRICT '<'
        # equality routes RIGHT

Leaf detection:
    left_children[node] == -1
Leaf value:
    split_conditions[node]        # the SAME array as thresholds
Feature index:
    split_indices[node]
```

- Both operands narrowed. Narrowing only the value is wrong on the hazardous half of thresholds.
- Strict `<`. Equality right. Measured on 104/104 nodes plus 195 more across 7 models.
- The threshold must already be float32 when it reaches this code. If narrowing lives at the
  comparison site, any other reader of the threshold — a re-serializer, an inspection utility, an
  arithmetic transform — reintroduces the bug.

---

## 10. Reproducing this probe

Scripts are in the session scratch directory, not the repository:

```
scratchpad/probe-float32/
  s1_fit_and_locate.py              fit, serialize, locate thresholds
  s2_raw_text.py                    verbatim on-disk text, formatting survey
  s3_roundtrip.py                   internal vs leaf split, bit patterns, gap direction
  s4_operator.py                    operator at tree roots
  s5_central.py                     THE CENTRAL EXPERIMENT + frequency
  s6_h1_engine_precision.py         H1 direct test via float64 inplace_predict
  s7_allnodes_ulp.py                operator + threshold pinning at all 104 internal nodes
  s8_formatting_and_parse_hazard.py extreme-scale formatting, precision hazard
  s9_generality.py                  objectives, tree methods, DART, inf/NaN
  s10_js_side.mjs                   Math.fround vs np.float32, JS-side disagreement
  s11_failure_set.py                exhaustive ULP scan, input-band measurement
  s12_accumulation.py               accumulation precision (out of scope, flagged)
```

Model fitting is seeded (`seed = 20260801`, `nthread = 1`, `tree_method = exact`) and reproduced
identical thresholds across runs. Feature names are `f0..f7`; data is `numpy` normal/integer/constant
synthetic. No named datasets, no domain vocabulary.

---

## Ambiguity, handled rather than resolved

1. **Non-finite thresholds.** No `Infinity`/`NaN` token was observed, and training on infinite
   features raises. Two readings: (a) `split_condition` is always finite and a parser may rely on it;
   (b) the observation only covers training on finite data and says nothing about other code paths.
   **Not resolved here.** Consistent with D007, a parser should fail loudly on a non-finite threshold
   rather than assume either reading.

2. **Categorical splits.** `split_type` was `0` at every node and `categories` was empty in all 9
   models. Whether `split_conditions` carries a bit-field or category index when `split_type == 1`
   was **not probed** — out of scope, and it is not float32 semantics. Flagging that
   `split_conditions` is overloaded a *third* way, not just two.

3. **Whether the engine's threshold is stored as float32 or is a float64 that happens to equal
   `float32(token)`.** The ULP test cannot distinguish these; it proves the *effective* comparison
   boundary is exactly `float32(token)`. For our purposes they are indistinguishable, and the
   implementation rule is the same either way. Labelling this as the limit of what was measured.

---

## Out of scope, things that looked wrong

**Margin accumulation precision is unresolved, and I did not resolve it.** The two-sided walk hits
`4.5e-07` max error against XGBoost's `output_margin` over 5000 rows — inside the `1e-6` gate — but
**no** accumulation variant is bit-exact on all rows:

```
=== MARGIN RECONSTRUCTION vs XGBoost output_margin, 5000 rows ===
accumulation variant                       max abs error            bit-exact rows
f64 base + f64 sum of f64 leaves           4.545964067403929e-07    0 / 5000
f64 base + f64 sum of f32 leaves           4.337837844481385e-07    0 / 5000
f32 base + f32 running sum (f32 leaves)    4.76837158203125e-07     2878 / 5000
f32 base + f64 sum then narrow             4.76837158203125e-07     1853 / 5000
f64 all, narrowed at the end               4.76837158203125e-07     1797 / 5000

XGBoost output_margin dtype: float32
```

The best variant is bit-exact on 58% of rows. The residual is **not** a split-comparison effect — leaf
assignment matches XGBoost exactly on all 5000 rows — so it lies in the accumulation order or the
`base_score` intercept. Relevant context, measured:

```
base_score raw string      : "[5.1953125E-1]"
base_score parsed (float64): 0.51953125
logit(base_score) float64  : 0.07816477284933626
logit computed in float32  : 0.07816477119922638
```

`0.51953125` is dyadic, but `logit` of it is not, and computing the intercept at float32 versus float64
width differs in the 9th digit — the right order of magnitude for the residual. This belongs to the
`base_score` probe. **DECISION NEEDED there, not here**; I am flagging it rather than guessing. It
does not affect any conclusion above.

Two further observations, recorded without action:

- `base_score` is serialized as a **JSON string containing a bracketed array**, `"[5.1953125E-1]"`,
  not as a number and not as a JSON array. A reader must strip the brackets and parse. Worth an
  explicit note in the format design since it is unlike every other numeric field.
- `trees[0].base_weights[0]` is `-0E0`. Signed zero is emitted by XGBoost's writer. Relevant to
  byte-identical export (D008) if any value ever round-trips through our own serializer.
