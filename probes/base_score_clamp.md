# Probe — pinning the exact `base_score` clamp bounds for `binary:logistic`

Closes **ambiguity A3** of `probes/output_transform.md` §10. That probe established that a
clamp exists and that clamping to approximately `[f32(1e-6), f32(1 - 1e-6)]` reproduced
XGBoost on all 14 values it tested, but recorded the bounds as **bracketed, not pinned**.
D035 currently states those approximate bounds.

**Result: both bounds are now pinned to an adjacent float32 pair, and the D035
approximation is exact. D035 needs no numeric change.**

Every claim is backed by a pasted command and its real output. Anything not directly
measured is labelled **INFERRED**. Ambiguity is presented, not resolved. Fitted models and
scripts lived entirely outside the repository; nothing was written into the tree except
this file.

---

## Environment

```
$ uv run python p01_env_instrument.py
python 3.12.8 (main, Dec  3 2024, 18:42:41) [Clang 16.0.0 (clang-1600.0.26.4)]
xgboost 3.3.0
numpy 2.5.1
platform macOS-26.5.2-arm64-arm-64bit
machine arm64
```

`mpmath` was used as an independent high-precision oracle in §8:

```
mpmath dps=50 prec=169 bits, version 1.4.1
```

All findings are for `xgboost 3.3.0` exactly. No version-drift claim is made.

---

## Verdicts up front

| Question | Answer, measured |
|---|---|
| Lower bound, pinned? | **Yes.** Adjacent float32 pair `0x358637BE` / `0x358637BF` |
| Upper bound, pinned? | **Yes.** Adjacent float32 pair `0x3F7FFFED` / `0x3F7FFFEE` |
| Was the D035 approximation exact? | **Yes.** `f32(1e-6)` and `f32(1)−f32(1e-6)` both fall inside the observational equivalence class of the true constants. `226/226` vs `226/226` on the sweep |
| Pinned clamp, sweep | **226/226** bit-exact |
| D035 approximate clamp, sweep | **226/226** bit-exact — identical, value for value |
| No clamp, sweep | **52/226** |
| Clamp on `p`, or on the derived intercept? | **Not decidable.** Both reproduce every input XGBoost accepts. §5 |
| Clamp on float32 or float64? | Result depends **only** on the float32 snap of the stored decimal, measured. §5 |
| `base_score` outside `[0, 1]` | **XGBoost raises**, at fit time *and* at `load_model` time. §7 |
| `base_score = 0.0`, `-0.0`, `1.0` | **Accepted**, no warning. Intercept saturates. §7 |
| `nan`, `inf` | **Raise** — but in the JSON parser, not the validity check. §7 |
| `-inf` | **Accepted and silently stored as `[0E0]`.** §9.2 |
| Cox clamped? | **No.** `33/34` unclamped over `1e-45` … `3.4e38` |
| Regression clamped? | **No.** `25/25` unclamped over `-3.4e38` … `3.4e38` |

**Out of scope but wrong, reported loudly (§8):** for `survival:cox`, `FORMAT.md` §6.1's
`log(f32(base_score))` implemented as a *float64* log narrowed once is **1 ULP off
XGBoost on 35 of 6947 measured values**, all with `base_score` in roughly `[0.5, 1.1]`.
A **float32** log matches **6947/6947**. This contradicts nothing in
`probes/base_score.md` §3, but its 17-value sweep never entered the failing region, so
the conclusion drawn there is narrower than it reads.

---

## 1. The instrument, and `boost_from_average == "0"` everywhere

`num_boost_round=0` with `base_score` passed **explicitly**. With zero trees the margin
*is* the intercept. Passing `base_score` explicitly is mandatory: on a zero-tree model
left at the default, `boost_from_average` is `"1"` and XGBoost emits the **raw**
`base_score` with no link transform (`probes/arity_gate.md` §7, D036), which would make
every measurement meaningless.

Synthetic data only: 8 rows × 2 columns from `numpy.random.default_rng(20260804)`,
generic column names `c0`, `c1`, `nthread=1`, `seed=20260804`, `tree_method=exact`.

```
$ uv run python p01_env_instrument.py
==============================================================================
STEP 1 -- establish the instrument on mid-range base_score values
num_boost_round=0, base_score passed EXPLICITLY

         arg           stored  bfa  trees  spread      measured intercept   meas bits        recipe(f32(arg))    rec bits match
         0.5           [5E-1]    0      0     0.0                    -0.0  2147483648                    -0.0  2147483648   YES
        0.25         [2.5E-1]    0      0     0.0     -1.0986123085021973  3213664084     -1.0986123085021973  3213664084   YES
        0.48         [4.8E-1]    0      0     0.0    -0.08004285395145416  3181636994    -0.08004285395145416  3181636994   YES
        0.75         [7.5E-1]    0      0     0.0      1.0986121892929077  1066180435      1.0986121892929077  1066180435   YES
        0.05           [5E-2]    0      0     0.0      -2.944438934326172  3225186736      -2.944438934326172  3225186736   YES
        0.95         [9.5E-1]    0      0     0.0      2.9444382190704346  1077703085      2.9444382190704346  1077703085   YES
    0.987654     [9.87654E-1]    0      0     0.0       4.381994247436523  1082931532       4.381994247436523  1082931532   YES
        0.01           [1E-2]    0      0     0.0      -4.595119953155518  3230862137      -4.595119953155518  3230862137   YES
        0.99           [9.9E-1]    0      0     0.0         4.5951247215271  1083378499         4.5951247215271  1083378499   YES
       0.001           [1E-3]    0      0     0.0      -6.906754493713379  3235709986      -6.906754493713379  3235709986   YES
   0.3333333    [3.333333E-1]    0      0     0.0     -0.6931473016738892  3207688730     -0.6931473016738892  3207688730   YES
   0.6666667    [6.666667E-1]    0      0     0.0      0.6931471824645996  1060205080      0.6931471824645996  1060205080   YES

bit-exact on all values: True    (12 fits in 0.03s = 2.1 ms/fit)
distinct boost_from_average values seen across those fits: ['0']
```

`recipe` here is the D015 / `probes/base_score.md` §9 form
`-log(f32(f32(1/f32(p)) - 1))`, **not** `log(p/(1-p))`. It is bit-exact on all 12
mid-range values, which is the baseline that lets a later mismatch be attributed to
clamping rather than to the transform.

**`boost_from_average` confirmation.** Every measuring function in this probe asserts
`bfa == "0"` on the artifact it just read, and every fit is `num_boost_round=0` with
`base_score` explicit. The assertion is `assert r["bfa"] == "0"` in `lib.fit0`,
`p02_bisect.measure_bits`, and in the loops of §3, §4, §6, §8. It never fired. The three
tables that print `bfa` per row (§1, §4 excerpt, §7) show `0` on every accepted row. The
one artifact used for the hand-edit cross-check in §5 was verified before use:

```
$ uv run python p07_handedit.py
template artifact learner_model_param: {"base_score": "[5E-1]", "boost_from_average": "0", "num_class": "0", "num_feature": "2", "num_target": "1"}
```

Every fit also asserted that the stored `base_score` string round-trips back to the exact
float32 bit pattern that was passed
(`assert lib.bits(lib.parse_stored(r["stored"])) == b`), so the value under test is known,
not assumed.

**Total measurement volume:** roughly 11,500 zero-round fits across §2–§8. Per-section
counts are printed in each pasted block.

---

## 2. The lower bound, pinned

The intercept saturates as `p` decreases. Saturated value first, from five values spread
over 23 orders of magnitude:

```
$ uv run python p02_bisect.py
====================================================================================================
A. deep-clamp saturated values (far outside any plausible bound)
  p=1e-30    stored=[1E-30]      bfa=0 measured=-13.815509796142578    mbits=3244100692  unclamped_recipe=-69.07755279541016
  p=1e-20    stored=[1E-20]      bfa=0 measured=-13.815509796142578    mbits=3244100692  unclamped_recipe=-46.051700592041016
  p=1e-12    stored=[1E-12]      bfa=0 measured=-13.815509796142578    mbits=3244100692  unclamped_recipe=-27.63102149963379
  p=1e-08    stored=[1E-8]       bfa=0 measured=-13.815509796142578    mbits=3244100692  unclamped_recipe=-18.42068099975586
  p=1e-07    stored=[1E-7]       bfa=0 measured=-13.815509796142578    mbits=3244100692  unclamped_recipe=-16.11809539794922
  SATURATED LOW value S_lo = -13.815509796142578  bits=3244100692  hex=0xC15D0C54
```

Then a bisection over float32 **bit patterns** (for positive floats the uint32 encoding is
monotone in the value, so bisecting the bit pattern bisects the float32 ordering):

```
====================================================================================================
B. LOWER BOUND -- bisect for the largest float32 p whose measured intercept still
   equals the saturated low value, and the smallest whose intercept is strictly greater

  bracket lo (start)     p_bits=0x3584E024  p=9.899999895424116e-07    stored=[9.9E-7]         measured=-13.815509796142578    mbits=3244100692  recipe(p)=-13.825559616088867    rbits=3244111230
  bracket hi (start)     p_bits=0x3593A3B6  p=1.0999999631167157e-06   stored=[1.1E-6]         measured=-13.720199584960938    mbits=3244000752  recipe(p)=-13.720199584960938    rbits=3244000752
  float32 values strictly between the brackets: 967569

  step  1  mid_bits=0x358C41ED p=1.0449999763295637e-06   measured=-13.771492958068848    == S_lo? False   -> window 967569
  step  2  mid_bits=0x35889108 p=1.0174999260925688e-06   measured=-13.798160552978516    == S_lo? False   -> window 483784
  step  3  mid_bits=0x3586B896 p=1.0037499578174902e-06   measured=-13.811766624450684    == S_lo? False   -> window 241891
  step  4  mid_bits=0x3585CC5D p=9.96874973679951e-07     measured=-13.815509796142578    == S_lo? True   -> window 120945
  step  5  mid_bits=0x35864279 p=1.0003124089053017e-06   measured=-13.815196990966797    == S_lo? False   -> window 60472
  step  6  mid_bits=0x3586076B p=9.985936912926263e-07    measured=-13.815509796142578    == S_lo? True   -> window 30235
  step  7  mid_bits=0x358624F2 p=9.99453050098964e-07     measured=-13.815509796142578    == S_lo? True   -> window 15117
  step  8  mid_bits=0x358633B5 p=9.99882672658714e-07     measured=-13.815509796142578    == S_lo? True   -> window 7558
  step  9  mid_bits=0x35863B17 p=1.0000975407820079e-06   measured=-13.815411567687988    == S_lo? False   -> window 3779
  step 10  mid_bits=0x35863766 p=9.99990106720361e-07     measured=-13.815509796142578    == S_lo? True   -> window 1889
  step 11  mid_bits=0x3586393E p=1.0000437669077655e-06   measured=-13.815465927124023    == S_lo? False   -> window 944
  step 12  mid_bits=0x35863852 p=1.0000169368140632e-06   measured=-13.815492630004883    == S_lo? False   -> window 471
  step 13  mid_bits=0x358637DC p=1.000003521767212e-06    measured=-13.815505981445312    == S_lo? False   -> window 235
  step 14  mid_bits=0x358637A1 p=9.999968142437865e-07    measured=-13.815509796142578    == S_lo? True   -> window 117
  step 15  mid_bits=0x358637BE p=1.0000001111620804e-06   measured=-13.815509796142578    == S_lo? True   -> window 58
  step 16  mid_bits=0x358637CD p=1.0000018164646463e-06   measured=-13.815507888793945    == S_lo? False   -> window 29
  step 17  mid_bits=0x358637C5 p=1.0000009069699445e-06   measured=-13.815508842468262    == S_lo? False   -> window 14
  step 18  mid_bits=0x358637C1 p=1.0000004522225936e-06   measured=-13.815508842468262    == S_lo? False   -> window 6
  step 19  mid_bits=0x358637BF p=1.0000002248489182e-06   measured=-13.815508842468262    == S_lo? False   -> window 2

  LOWER TRANSITION PAIR (adjacent float32):
  largest == S_lo        p_bits=0x358637BE  p=1.0000001111620804e-06   stored=[1.0000001E-6]   measured=-13.815509796142578    mbits=3244100692  recipe(p)=-13.815509796142578    rbits=3244100692
  smallest  > S_lo       p_bits=0x358637BF  p=1.0000002248489182e-06   stored=[1.0000002E-6]   measured=-13.815508842468262    mbits=3244100691  recipe(p)=-13.815508842468262    rbits=3244100691
  adjacent? hi_bits - lo_bits = 1
```

### The lower bound, in full

```
$ uv run python p13_constants.py
LOWER BOUND -- adjacent float32 pair on base_score
  last p whose intercept == saturated low      1.0000001111620804e-06   uint32=897988542    hex=0x358637BE
  first p whose intercept  > saturated low     1.0000002248489182e-06   uint32=897988543    hex=0x358637BF
  gap in bit patterns: 1  (adjacent)

  saturated LOW intercept (every base_score <= 0x358637BE):
  S_lo                                         -13.815509796142578      uint32=3244100692   hex=0xC15D0C54
  intercept at the first unclamped float32 0x358637BF:
  recipe(0x358637BF)                           -13.815508842468262      uint32=3244100691   hex=0xC15D0C53
```

| | decimal | uint32 | hex |
|---|---|---|---|
| largest `base_score` still at the saturated value | `1.0000001111620804e-06` | `897988542` | `0x358637BE` |
| smallest `base_score` strictly above it | `1.0000002248489182e-06` | `897988543` | `0x358637BF` |
| **saturated low intercept `S_lo`** | **`-13.815509796142578`** | `3244100692` | `0xC15D0C54` |
| intercept at the first unclamped value | `-13.815508842468262` | `3244100691` | `0xC15D0C53` |

---

## 3. The upper bound, pinned

Same method, approaching 1 from below. `1 - 1e-9` is included because it stores as `[1E0]`
and is the value at which the *unclamped* recipe raises a domain error.

```
$ uv run python p02_bisect.py
  p=0.999999999            stored=[1E0]            bfa=0 measured=13.745160102844238     mbits=1096543277  unclamped_recipe=None
  p=0.99999995             stored=[9.9999994E-1]   bfa=0 measured=13.745160102844238     mbits=1096543277  unclamped_recipe=15.942384719848633
  p=0.9999999              stored=[9.999999E-1]    bfa=0 measured=13.745160102844238     mbits=1096543277  unclamped_recipe=15.942384719848633
  p=0.9999995              stored=[9.999995E-1]    bfa=0 measured=13.745160102844238     mbits=1096543277  unclamped_recipe=14.556090354919434
  SATURATED HIGH value S_hi = 13.745160102844238  bits=1096543277  hex=0x415BEC2D

====================================================================================================
C. UPPER BOUND -- bisect for the smallest float32 p whose measured intercept has
   reached the saturated high value, and the largest whose intercept is strictly less

  bracket lo (start)     p_bits=0x3F7FFF58  p=0.9999899864196777       stored=[9.9999E-1]      measured=11.511568069458008     mbits=1094201186  recipe(p)=11.511568069458008     rbits=1094201186
  bracket hi (start)     p_bits=0x3F7FFFF8  p=0.9999995231628418       stored=[9.999995E-1]    measured=13.745160102844238     mbits=1096543277  recipe(p)=14.556090354919434     rbits=1097393599
  float32 values strictly between the brackets: 159

  step  1  mid_bits=0x3F7FFFA8 p=0.9999947547912598       measured=12.158195495605469     == S_hi? False   -> window 159
  step  2  mid_bits=0x3F7FFFD0 p=0.9999971389770508       measured=12.764330863952637     == S_hi? False   -> window 79
  step  3  mid_bits=0x3F7FFFE4 p=0.9999983310699463       measured=13.303327560424805     == S_hi? False   -> window 39
  step  4  mid_bits=0x3F7FFFEE p=0.999998927116394        measured=13.745160102844238     == S_hi? True   -> window 19
  step  5  mid_bits=0x3F7FFFE9 p=0.9999986290931702       measured=13.457478523254395     == S_hi? False   -> window 9
  step  6  mid_bits=0x3F7FFFEB p=0.9999987483024597       measured=13.544489860534668     == S_hi? False   -> window 4
  step  7  mid_bits=0x3F7FFFEC p=0.9999988079071045       measured=13.639800071716309     == S_hi? False   -> window 2
  step  8  mid_bits=0x3F7FFFED p=0.9999988675117493       measured=13.639800071716309     == S_hi? False   -> window 1

  UPPER TRANSITION PAIR (adjacent float32):
  largest   < S_hi       p_bits=0x3F7FFFED  p=0.9999988675117493       stored=[9.9999887E-1]   measured=13.639800071716309     mbits=1096432799  recipe(p)=13.639800071716309     rbits=1096432799
  smallest == S_hi       p_bits=0x3F7FFFEE  p=0.999998927116394        stored=[9.999989E-1]    measured=13.745160102844238     mbits=1096543277  recipe(p)=13.745160102844238     rbits=1096543277
  adjacent? uhi_bits - ulo_bits = 1

total fits: 42
```

### The upper bound, in full

```
$ uv run python p13_constants.py
UPPER BOUND -- adjacent float32 pair on base_score
  last p whose intercept  < saturated high     0.9999988675117493       uint32=1065353197   hex=0x3F7FFFED
  first p whose intercept == saturated high    0.999998927116394        uint32=1065353198   hex=0x3F7FFFEE
  gap in bit patterns: 1  (adjacent)

  saturated HIGH intercept (every base_score >= 0x3F7FFFEE):
  S_hi                                         13.745160102844238       uint32=1096543277   hex=0x415BEC2D
  intercept at the last unclamped float32 0x3F7FFFED:
  recipe(0x3F7FFFED)                           13.639800071716309       uint32=1096432799   hex=0x415A3C9F
```

| | decimal | uint32 | hex |
|---|---|---|---|
| largest `base_score` strictly below the saturated value | `0.9999988675117493` | `1065353197` | `0x3F7FFFED` |
| smallest `base_score` at the saturated value | `0.999998927116394` | `1065353198` | `0x3F7FFFEE` |
| **saturated high intercept `S_hi`** | **`13.745160102844238`** | `1096543277` | `0x415BEC2D` |
| intercept at the last unclamped value | `13.639800071716309` | `1096432799` | `0x415A3C9F` |

---

## 4. Exhaustive scans — the bisection's monotonicity assumption, checked

A bisection is only valid if the predicate is monotone. That was assumed, so it is
verified by brute force: every consecutive float32 in a 1026-wide window around the lower
transition, and a 532-wide window around the upper transition running all the way up to
`1.0`. **Zero violations.**

```
$ uv run python p03_exhaustive.py
====================================================================================================
D. exhaustive float32 scan around the LOWER transition
  scanned 1026 consecutive float32 values, 0x358635BE..0x358639BF
  hypothesis: measured == S_lo for every p <= 0x358637BE, and
              measured == unclamped recipe(p) for every p >= 0x358637BF
  violations: 0

  the 6 float32 either side of the transition, verbatim:
    0x358637BC p=9.99999883788405e-07     stored=[9.999999E-7]    measured=-13.815509796142578    mbits=3244100692  recipe=-13.815509796142578    rbits=3244100692
    0x358637BD p=9.999999974752427e-07    stored=[1E-6]           measured=-13.815509796142578    mbits=3244100692  recipe=-13.815509796142578    rbits=3244100692
    0x358637BE p=1.0000001111620804e-06   stored=[1.0000001E-6]   measured=-13.815509796142578    mbits=3244100692  recipe=-13.815509796142578    rbits=3244100692
    0x358637BF p=1.0000002248489182e-06   stored=[1.0000002E-6]   measured=-13.815508842468262    mbits=3244100691  recipe=-13.815508842468262    rbits=3244100691
    0x358637C0 p=1.0000003385357559e-06   stored=[1.0000003E-6]   measured=-13.815508842468262    mbits=3244100691  recipe=-13.815508842468262    rbits=3244100691
    0x358637C1 p=1.0000004522225936e-06   stored=[1.0000005E-6]   measured=-13.815508842468262    mbits=3244100691  recipe=-13.815508842468262    rbits=3244100691
    0x358637C2 p=1.0000005659094313e-06   stored=[1.0000006E-6]   measured=-13.815508842468262    mbits=3244100691  recipe=-13.815508842468262    rbits=3244100691

====================================================================================================
E. exhaustive float32 scan around the UPPER transition, up to and including 1.0
  scanned 532 consecutive float32 values, 0x3F7FFDED..0x3F800000
  hypothesis: measured == S_hi for every p >= 0x3F7FFFEE, and
              measured == unclamped recipe(p) for every p <= 0x3F7FFFED
  violations: 0

  every float32 from 0x3F7FFFE8 through 0x3F800000 (=1.0), verbatim:
    0x3F7FFFE8 p=0.9999985694885254     stored=[9.9999857E-1]   measured=13.457478523254395     mbits=1096241621  recipe=13.457478523254395     rbits=1096241621
    0x3F7FFFE9 p=0.9999986290931702     stored=[9.999986E-1]    measured=13.457478523254395     mbits=1096241621  recipe=13.457478523254395     rbits=1096241621
    0x3F7FFFEA p=0.9999986886978149     stored=[9.999987E-1]    measured=13.544489860534668     mbits=1096332859  recipe=13.544489860534668     rbits=1096332859
    0x3F7FFFEB p=0.9999987483024597     stored=[9.9999875E-1]   measured=13.544489860534668     mbits=1096332859  recipe=13.544489860534668     rbits=1096332859
    0x3F7FFFEC p=0.9999988079071045     stored=[9.999988E-1]    measured=13.639800071716309     mbits=1096432799  recipe=13.639800071716309     rbits=1096432799
    0x3F7FFFED p=0.9999988675117493     stored=[9.9999887E-1]   measured=13.639800071716309     mbits=1096432799  recipe=13.639800071716309     rbits=1096432799
    0x3F7FFFEE p=0.999998927116394      stored=[9.999989E-1]    measured=13.745160102844238     mbits=1096543277  recipe=13.745160102844238     rbits=1096543277
    0x3F7FFFEF p=0.9999989867210388     stored=[9.99999E-1]     measured=13.745160102844238     mbits=1096543277  recipe=13.745160102844238     rbits=1096543277
    0x3F7FFFF0 p=0.9999990463256836     stored=[9.9999905E-1]   measured=13.745160102844238     mbits=1096543277  recipe=13.862943649291992     rbits=1096666782
    0x3F7FFFF1 p=0.9999991059303284     stored=[9.999991E-1]    measured=13.745160102844238     mbits=1096543277  recipe=13.862943649291992     rbits=1096666782
    0x3F7FFFF2 p=0.9999991655349731     stored=[9.9999917E-1]   measured=13.745160102844238     mbits=1096543277  recipe=13.996475219726562     rbits=1096806800
    0x3F7FFFF3 p=0.9999992251396179     stored=[9.999992E-1]    measured=13.745160102844238     mbits=1096543277  recipe=13.996475219726562     rbits=1096806800
    0x3F7FFFF4 p=0.9999992847442627     stored=[9.999993E-1]    measured=13.745160102844238     mbits=1096543277  recipe=14.150625228881836     rbits=1096968438
    0x3F7FFFF5 p=0.9999993443489075     stored=[9.9999934E-1]   measured=13.745160102844238     mbits=1096543277  recipe=14.150625228881836     rbits=1096968438
    0x3F7FFFF6 p=0.9999994039535522     stored=[9.999994E-1]    measured=13.745160102844238     mbits=1096543277  recipe=14.33294677734375      rbits=1097159616
    0x3F7FFFF7 p=0.999999463558197      stored=[9.9999946E-1]   measured=13.745160102844238     mbits=1096543277  recipe=14.33294677734375      rbits=1097159616
    0x3F7FFFF8 p=0.9999995231628418     stored=[9.999995E-1]    measured=13.745160102844238     mbits=1096543277  recipe=14.556090354919434     rbits=1097393599
    0x3F7FFFF9 p=0.9999995827674866     stored=[9.999996E-1]    measured=13.745160102844238     mbits=1096543277  recipe=14.556090354919434     rbits=1097393599
    0x3F7FFFFA p=0.9999996423721313     stored=[9.9999964E-1]   measured=13.745160102844238     mbits=1096543277  recipe=14.843772888183594     rbits=1097695256
    0x3F7FFFFB p=0.9999997019767761     stored=[9.999997E-1]    measured=13.745160102844238     mbits=1096543277  recipe=14.843772888183594     rbits=1097695256
    0x3F7FFFFC p=0.9999997615814209     stored=[9.9999976E-1]   measured=13.745160102844238     mbits=1096543277  recipe=15.249238014221191     rbits=1098120417
    0x3F7FFFFD p=0.9999998211860657     stored=[9.999998E-1]    measured=13.745160102844238     mbits=1096543277  recipe=15.249238014221191     rbits=1098120417
    0x3F7FFFFE p=0.9999998807907104     stored=[9.999999E-1]    measured=13.745160102844238     mbits=1096543277  recipe=15.942384719848633     rbits=1098847234
    0x3F7FFFFF p=0.9999999403953552     stored=[9.9999994E-1]   measured=13.745160102844238     mbits=1096543277  recipe=15.942384719848633     rbits=1098847234
    0x3F800000 p=1.0                    stored=[1E0]            measured=13.745160102844238     mbits=1096543277  recipe=None(domain)           rbits=-
```

Note `0x3F800000` — `base_score = 1.0` is **accepted** and yields `S_hi`, while the
unclamped recipe has a domain error there (`f32(1/1.0) − 1 = 0.0`, `−log(0)`).

### What the transition pins, and what it does not

The transition pair pins the *boundary of observable behaviour* exactly. It does **not**
by itself pin the clamp constant to one float32, because the transform has a plateau
around each bound: several adjacent `base_score` values map to the same float32 intercept.
Any clamp constant inside that plateau produces bit-identical intercepts on **every**
float32 input, so those constants are indistinguishable by any measurement of the
intercept. The plateaus were computed exhaustively:

```
====================================================================================================
F. the observational equivalence class of each clamp constant
   c_lo must satisfy recipe(q) == S_lo for every float32 q in [c_lo, 0x358637BE].
   -> c_lo run: 0x358637B7 .. 0x358637BE   (8 float32 values)
      0x358637B6 p=9.999992016673787e-07    recipe(p)=-13.815510749816895    rbits=3244100693  == S_lo? False
      0x358637B7 p=9.999993153542164e-07    recipe(p)=-13.815509796142578    rbits=3244100692  == S_lo? True
      0x358637B8 p=9.99999429041054e-07     recipe(p)=-13.815509796142578    rbits=3244100692  == S_lo? True
      0x358637B9 p=9.999995427278918e-07    recipe(p)=-13.815509796142578    rbits=3244100692  == S_lo? True
      0x358637BA p=9.999996564147295e-07    recipe(p)=-13.815509796142578    rbits=3244100692  == S_lo? True
      0x358637BB p=9.999997701015673e-07    recipe(p)=-13.815509796142578    rbits=3244100692  == S_lo? True
      0x358637BC p=9.99999883788405e-07     recipe(p)=-13.815509796142578    rbits=3244100692  == S_lo? True
      0x358637BD p=9.999999974752427e-07    recipe(p)=-13.815509796142578    rbits=3244100692  == S_lo? True
      0x358637BE p=1.0000001111620804e-06   recipe(p)=-13.815509796142578    rbits=3244100692  == S_lo? True
      0x358637BF p=1.0000002248489182e-06   recipe(p)=-13.815508842468262    rbits=3244100691  == S_lo? False

   c_hi must satisfy recipe(q) == S_hi for every float32 q in [0x3F7FFFEE, c_hi].
   -> c_hi run: 0x3F7FFFEE .. 0x3F7FFFEF   (2 float32 values)
      0x3F7FFFED p=0.9999988675117493     recipe(p)=13.639800071716309     rbits=1096432799  == S_hi? False
      0x3F7FFFEE p=0.999998927116394      recipe(p)=13.745160102844238     rbits=1096543277  == S_hi? True
      0x3F7FFFEF p=0.9999989867210388     recipe(p)=13.745160102844238     rbits=1096543277  == S_hi? True
      0x3F7FFFF0 p=0.9999990463256836     recipe(p)=13.862943649291992     rbits=1096666782  == S_hi? False

G. where the candidate constants from the prior probe / D035 land
   f32(1e-6)                        = 9.999999974752427e-07    bits=0x358637BD  in c_lo run: True   in c_hi run: False
   double 1e-6 -> f32               = 9.999999974752427e-07    bits=0x358637BD  in c_lo run: True   in c_hi run: False
   f32(1)-f32(1e-6) in f32 arith    = 0.9999989867210388       bits=0x3F7FFFEF  in c_lo run: False   in c_hi run: True
   f32(1 - 1e-6) from double        = 0.9999989867210388       bits=0x3F7FFFEF  in c_lo run: False   in c_hi run: True
   f32(0.999999)                    = 0.9999989867210388       bits=0x3F7FFFEF  in c_lo run: False   in c_hi run: True
```

**The D035 constants are inside both equivalence classes.** `f32(1e-6) = 0x358637BD` sits
in the 8-value `c_lo` run; `f32(1) − f32(1e-6) = 0x3F7FFFEF` sits in the 2-value `c_hi`
run. So the approximation D035 records is **exact in effect**: it produces bit-identical
intercepts to the true constant on every float32 `base_score`, whatever the true constant
is within its run.

Read the equivalence classes correctly. They are **not** a failure to pin the bound. The
*bound* — the float32 at which observable behaviour changes — is pinned to an adjacent
pair, exactly. What is not pinned is which member of an 8-value (resp. 2-value) run
XGBoost's source literal is, and that distinction has **no observable consequence**, by
exhaustive construction over the plateau.

---

## 5. The clamp, stated implementably

```
p32 = f32(base_score_from_artifact)          # narrow on parse, per the float32 invariant
if p32 < f32(1e-6):        p32 = f32(1e-6)
elif p32 > f32(1-1e-6):    p32 = f32(1-1e-6)
intercept = -log( f32( f32(1/p32) - 1 ) )
```

Verified `226/226` in §6. Equivalent, and also verified `226/226`: substituting
`0x358637BE` = `1.0000001111620804e-06` for the lower constant and `0x3F7FFFEE` =
`0.999998927116394` for the upper.

**The result depends only on the float32 snap of the stored decimal.** Measured by
hand-editing the stored `base_score` string on a zero-tree, `bfa == "0"` artifact with
decimals that straddle the lower transition pair — including two decimals `1e-16` apart
that fall either side of the float32 rounding midpoint:

```
$ uv run python p07_handedit.py
============================================================================================================
K. does the float64 spelling of the stored decimal matter beyond its float32 snap?
   Decimals chosen to straddle the LOWER transition pair 0x358637BE / 0x358637BF.
   0x358637BE = 1.0000001111620804e-06
   0x358637BF = 1.0000002248489182e-06
   midpoint   = 1.0000001680054993e-06

    hand-edited base_score  f32 snap bits         restored              intercept        bits
     1.0000001111620804E-6 0x358637BE       [1.0000001E-6]    -13.815509796142578  3244100692
             1.00000015E-6 0x358637BE       [1.0000001E-6]    -13.815509796142578  3244100692
        1.0000001680055E-6 0x358637BF       [1.0000002E-6]    -13.815508842468262  3244100691
        1.0000001680056E-6 0x358637BF       [1.0000002E-6]    -13.815508842468262  3244100691
             1.00000019E-6 0x358637BF       [1.0000002E-6]    -13.815508842468262  3244100691
     1.0000002248489182E-6 0x358637BF       [1.0000002E-6]    -13.815508842468262  3244100691
                      1E-6 0x358637BD               [1E-6]    -13.815509796142578  3244100692
                  0.000001 0x358637BD               [1E-6]    -13.815509796142578  3244100692
      9.999999974752427E-7 0x358637BD               [1E-6]    -13.815509796142578  3244100692
```

`1.00000015E-6` is a float64 **above** `0x358637BE`, yet behaves exactly as `0x358637BE`.
`1.0000001680055E-6` and `1.0000001680056E-6` differ in the last decimal digit and land on
opposite sides of the float32 rounding midpoint, and the intercept follows the *snap*, not
the decimal. The value is narrowed to float32 before anything else happens. The XGBoost
stack trace at §7 names the parameter type directly — `xgboost::common::ParamArray<float>`
— which is independent corroboration that the parameter is a **float32** array.

The hand-edit instrument was cross-checked against the fit instrument first:

```
============================================================================================================
J. cross-check: hand-edited artifact vs the fit instrument, on values both can express
     stored string          fit intercept    fit bits     artifact intercept    art bits  agree
            [5E-1]                   -0.0  2147483648                   -0.0  2147483648   True
          [2.5E-1]    -1.0986123085021973  3213664084    -1.0986123085021973  3213664084   True
      [9.87654E-1]      4.381994247436523  1082931532      4.381994247436523  1082931532   True
            [1E-8]    -13.815509796142578  3244100692    -13.815509796142578  3244100692   True
     [9.999999E-1]     13.745160102844238  1096543277     13.745160102844238  1096543277   True
             [1E0]     13.745160102844238  1096543277     13.745160102844238  1096543277   True
             [0E0]    -13.815509796142578  3244100692    -13.815509796142578  3244100692   True
```

### Observationally identical formulations — all of them, named, none chosen

Four formulations were tested. **Three are indistinguishable on every input XGBoost
accepts. The fourth is indistinguishable on every input except one, and that one is
ambiguous for a separate reason.** This probe does not choose between them.

```
$ uv run python p14_formulations.py
====================================================================================================
V. candidate clamp formulations, evaluated at the one place they could differ

F1  narrow to float32, clamp to the float32 constant f32(1e-6), then transform
F2  narrow to float32, clamp to the float32 constant 0x358637BE, then transform
F3  clamp against the float64 constant 1e-6, transform starting from that float64
F4  no input clamp; clamp the DERIVED INTERCEPT to [S_lo, S_hi]

F3 at its most discriminating point -- base_score = f32(1e-6) = 9.999999974752427e-07,
which is strictly BELOW the float64 constant 1e-6 and so would be clamped by F3:
  float64: 1/1e-6 - 1        = 999999.0
           -log(that)        = -13.815509557963773
           narrowed to f32   = -13.815509796142578  bits=3244100692
  float32: f32(1/p) - 1      = 999999.0
           -log(that)        = -13.815509796142578  bits=3244100692
  XGBoost S_lo               = -13.815509796142578  bits=3244100692
  F3 == XGBoost? True    F1 == XGBoost? True
  -> F1, F2 and F3 all reproduce S_lo. NOT distinguishable by this measurement.

F4 differs from F1/F2/F3 only where the UNCLAMPED transform is undefined or
out of range. Those inputs are base_score = 1.0 (t == 0 exactly) and base_score
outside [0,1] (t < 0).  Measured behaviour at those inputs:

  base_score = 1.0   stored=[1E0]  bfa=0  XGBoost intercept=13.745160102844238 bits=1096543277
    F1/F2/F3 (clamp input to c_hi) -> 13.745160102844238 bits=1096543277   match=True
    F4: unclamped t = 0.0; -log(0.0) is a domain error in Python (math.log raises ValueError)
        under IEEE semantics -log(0.0) = inf, which clamped to S_hi gives 13.745160102844238 -> match=True
    -> F4 matches at base_score = 1.0 ONLY if log(0) yields +inf rather than raising.
       With Python's math.log it raises. Both readings are recorded; neither is chosen.

  base_score outside [0,1] would separate F4 from F1/F2/F3 decisively (F1/F2/F3
  give S_hi; F4 gives NaN). XGBoost refuses those inputs, at fit time AND at
  load_model time, so the separating experiment cannot be run. Measured refusals:
    base_score=1.0000001192092896: [13:20:10] .../regression_obj.cu:116: Check failed
    base_score=1.5: [13:20:10] .../regression_obj.cu:116: Check failed
    base_score=-1e-06: [13:20:10] .../regression_obj.cu:116: Check failed
```

**Correction, recorded rather than hidden.** An earlier draft of that script asserted that
F3 — a float64 clamp constant — was **excluded** by the measurements. That assertion was
hand-computed and **wrong**: I computed `−log(999999.0)` as rounding to
`-13.815510749816895` and it in fact rounds to `-13.815509796142578`, which is `S_lo`. The
script was rewritten to measure instead of assert, and the measurement above is what
stands. **F3 is not excluded.** No formulation is excluded.

The practical consequence, and it is the reason this matters less than it might: **F1 is
the safe choice regardless.** F1 and F2 never evaluate `log` outside its domain, because
the clamp on the input guarantees `t > 0`. F4 requires the implementation to produce and
then clamp a `+inf`, which in Python raises instead.

---

## 6. The sweep — three match counts

226 distinct float32 `base_score` values: 200 log-spaced over `1e-38` … `1.0`, plus
`1 - 1e-9`, `1 - 1e-8`, `1 - 1e-7`, `1 - 1e-6`, `0.999999`, `0.5`, `0.25`, `0.75`, `0.48`,
`0.987654`, plus **both bounds and every float32 within 6 ULP of each**, de-duplicated on
the float32 bit pattern.

```
$ uv run python p04_sweep.py
clamp constants under test
  pinned lo = 1.0000001111620804e-06  bits=0x358637BE
  pinned hi = 0.999998927116394  bits=0x3F7FFFEE
  approx lo = 9.999999974752427e-07  bits=0x358637BD   (= f32(1e-6))
  approx hi = 0.9999989867210388  bits=0x3F7FFFEF   (= f32(1)-f32(1e-6))

sweep size after de-duplicating on float32 bits: 226 values
  range of float32(base_score): [9.999999350456404e-39, 1.0]

====================================================================================================
BIT-EXACT MATCH COUNTS against XGBoost's measured intercept, n = 226
  (a) pinned clamp  [0x358637BE, 0x3F7FFFEE] then transform : 226/226
  (b) approx clamp  [f32(1e-6), f32(1)-f32(1e-6)]      then transform : 226/226
  (c) NO clamp at all                                                 : 52/226
  (d) no input clamp; clamp the derived INTERCEPT to [S_lo, S_hi]     : 225/226

  (a) vs (b) identical on every value: True
  first failure for c_noclamp: base_score=np.float64(1e-38) stored=[1E-38] xgb_bits=3244100692 cand_bits=3266248472
  first failure for d_clamp_output: base_score=np.float64(1.0) stored=[1E0] xgb_bits=1096543277 cand_bits=inf(domain)
```

| Hypothesis | Bit-exact |
|---|---|
| **(a) pinned clamp `[0x358637BE, 0x3F7FFFEE]`, then transform** | **226/226** |
| **(b) D035 approximate clamp `[f32(1e-6), f32(1)−f32(1e-6)]`, then transform** | **226/226** |
| **(c) no clamp at all** | **52/226** |
| (d) clamp the derived intercept instead of the input | 225/226 — see below |

**(a) and (b) are identical, value for value, on all 226.** The prior probe's
approximation was already exact. **D035 needs no numeric change.**

(d)'s single failure is at `base_score = 1.0`, where the unclamped recipe's `log` argument
is exactly `0.0`. It is a failure of `math.log` raising rather than of the clamp
formulation: under IEEE `log(0) = -inf` semantics (d) also reaches `226/226` (§5). Both
readings are stated; neither is chosen.

A 30-row excerpt of the raw table, showing the deep-clamp end, the transition, and the
upper end:

```
a 24-row excerpt of the raw table (arg, stored, measured, and each candidate's bits)
                     arg           stored               measured      m.bits         (a)         (b)           (c)         (d)
       np.float64(1e-38)          [1E-38]    -13.815509796142578  3244100692  3244100692  3244100692    3266248472  3244100692
np.float64(3.3700643292719385e-37)  [3.3700643E-37]    -13.815509796142578  3244100692  3244100692  3244100692    3265787424  3244100692
np.float64(1.1357333583431121e-35)  [1.1357334E-35]    -13.815509796142578  3244100692  3244100692  3244100692    3265326377  3244100692
                0.987654     [9.87654E-1]      4.381994247436523  1082931532  1082931532  1082931532    1082931532  1082931532
    9.99999087980541e-07    [9.999991E-7]    -13.815509796142578  3244100692  3244100692  3244100692    3244100693  3244100692
   9.999992016673787e-07    [9.999992E-7]    -13.815509796142578  3244100692  3244100692  3244100692    3244100693  3244100692
   9.999993153542164e-07    [9.999993E-7]    -13.815509796142578  3244100692  3244100692  3244100692    3244100692  3244100692
    9.99999883788405e-07    [9.999999E-7]    -13.815509796142578  3244100692  3244100692  3244100692    3244100692  3244100692
   9.999999974752427e-07           [1E-6]    -13.815509796142578  3244100692  3244100692  3244100692    3244100692  3244100692
  1.0000001111620804e-06   [1.0000001E-6]    -13.815509796142578  3244100692  3244100692  3244100692    3244100692  3244100692
  1.0000002248489182e-06   [1.0000002E-6]    -13.815508842468262  3244100691  3244100691  3244100691    3244100691  3244100691
      0.9999988079071045    [9.999988E-1]     13.639800071716309  1096432799  1096432799  1096432799    1096432799  1096432799
      0.9999988675117493   [9.9999887E-1]     13.639800071716309  1096432799  1096432799  1096432799    1096432799  1096432799
       0.999998927116394    [9.999989E-1]     13.745160102844238  1096543277  1096543277  1096543277    1096543277  1096543277
      0.9999990463256836   [9.9999905E-1]     13.745160102844238  1096543277  1096543277  1096543277    1096666782  1096543277
      0.9999991059303284    [9.999991E-1]     13.745160102844238  1096543277  1096543277  1096543277    1096666782  1096543277
```

Note the three rows `9.99999087980541e-07`, `9.999992016673787e-07` — below the `c_lo`
equivalence run — where the *unclamped* candidate (c) differs from XGBoost by 1 ULP
(`3244100693` vs `3244100692`). The clamp is doing real work already at 1 part in `10^7`
below the bound, not only at `1e-38`.

---

## 7. Behaviour at and beyond the extremes

**An instrument defect first, because it affects how to read this section.**
`lib.fit0` asserted `len(set(margin)) == 1` to prove the zero-tree margin is constant.
That assertion **misfires on a NaN margin**, because `NaN != NaN` makes the set have one
entry per row. In the first extremes run this surfaced as an `AssertionError` that I
initially mislabelled as "XGBoost RAISES" for `survival:cox` at negative `base_score`.
**XGBoost does not raise there — it accepts and returns NaN.** Every reading below is from
an assertion-free instrument (`lib2.fit0_raw`) that reports the number of distinct bit
patterns instead of asserting. The misfire and the corrected reading:

```
$ uv run python p10_extremes_fixed.py
======================================================================================================================
Q. the values where lib.fit0's assertion misfired -- what the margin actually is
  survival:cox base_score=-1e-06
    lib.fit0        -> AssertionError ((8,), array([nan, nan, nan, nan, nan, nan, nan, nan], dtype
    lib2.fit0_raw   -> ACCEPTED  stored=[-1E-6]    bfa=0 margin[0]=nan bits=2143289344 n_distinct_bit_patterns=1
    is nan: True
  survival:cox base_score=-0.5
    lib.fit0        -> AssertionError ((8,), array([nan, nan, nan, nan, nan, nan, nan, nan], dtype
    lib2.fit0_raw   -> ACCEPTED  stored=[-5E-1]    bfa=0 margin[0]=nan bits=2143289344 n_distinct_bit_patterns=1
    is nan: True
  survival:cox base_score=-1.0
    lib.fit0        -> AssertionError ((8,), array([nan, nan, nan, nan, nan, nan, nan, nan], dtype
    lib2.fit0_raw   -> ACCEPTED  stored=[-1E0]     bfa=0 margin[0]=nan bits=2143289344 n_distinct_bit_patterns=1
    is nan: True
  survival:cox base_score=-1e+38
    lib.fit0        -> AssertionError ((8,), array([nan, nan, nan, nan, nan, nan, nan, nan], dtype
    lib2.fit0_raw   -> ACCEPTED  stored=[-1E38]    bfa=0 margin[0]=nan bits=2143289344 n_distinct_bit_patterns=1
    is nan: True
  survival:cox base_score=-1.401298464324817e-45
    lib.fit0        -> AssertionError ((8,), array([nan, nan, nan, nan, nan, nan, nan, nan], dtype
    lib2.fit0_raw   -> ACCEPTED  stored=[-1E-45]   bfa=0 margin[0]=nan bits=2143289344 n_distinct_bit_patterns=1
    is nan: True
```

**Which readings to distrust:** none of the numbers in §2–§6 or §8 are affected. The
misfire can only occur when the margin is NaN, and the only NaN margins found anywhere in
this probe are `survival:cox` with `base_score < 0`. Every other section fits `base_score`
values for which the assertion passed, which is itself the proof that the margin was
constant there.

### `binary:logistic`

```
--- binary:logistic
                      case                   passed      fit       stored  bfa         measured intercept        bits #distinct
                       0.0                      0.0  accepts        [0E0]    0        -13.815509796142578  3244100692         1
                      -0.0                     -0.0  accepts       [-0E0]    0        -13.815509796142578  3244100692         1
                       1.0                      1.0  accepts        [1E0]    0         13.745160102844238  1096543277         1
  smallest f32 subnormal +    1.401298464324817e-45  accepts      [1E-45]    0        -13.815509796142578  3244100692         1
  smallest f32 subnormal -   -1.401298464324817e-45   RAISES   Check failed: is_valid: base_score must be in (0,1) for the logistic loss.
                    -1e-06                   -1e-06   RAISES   Check failed: is_valid: base_score must be in (0,1) for the logistic loss.
                      -0.5                     -0.5   RAISES   Check failed: is_valid: base_score must be in (0,1) for the logistic loss.
                      -1.0                     -1.0   RAISES   Check failed: is_valid: base_score must be in (0,1) for the logistic loss.
                    -1e+38                   -1e+38   RAISES   Check failed: is_valid: base_score must be in (0,1) for the logistic loss.
        nextafter(1.0) f32       1.0000001192092896   RAISES   Check failed: is_valid: base_score must be in (0,1) for the logistic loss.
                       1.5                      1.5   RAISES   Check failed: is_valid: base_score must be in (0,1) for the logistic loss.
                       2.0                      2.0   RAISES   Check failed: is_valid: base_score must be in (0,1) for the logistic loss.
                     1e+06                1000000.0   RAISES   Check failed: is_valid: base_score must be in (0,1) for the logistic loss.
                   3.4e+38                  3.4e+38   RAISES   Check failed: is_valid: base_score must be in (0,1) for the logistic loss.
                       nan                      nan   RAISES   Expecting null value "null", around character position: 3
                       inf                      inf   RAISES   Unknown construct, around character position: 0
                      -inf                     -inf  accepts        [0E0]    0        -13.815509796142578  3244100692         1
```

| `base_score` | fit | stored | intercept | bits |
|---|---|---|---|---|
| `0.0` | **accepts** | `[0E0]` | `-13.815509796142578` (`S_lo`) | `3244100692` |
| `-0.0` | **accepts** | `[-0E0]` | `-13.815509796142578` (`S_lo`) | `3244100692` |
| `1.401298464324817e-45` (min subnormal) | **accepts** | `[1E-45]` | `-13.815509796142578` (`S_lo`) | `3244100692` |
| `1.0` | **accepts** | `[1E0]` | `13.745160102844238` (`S_hi`) | `1096543277` |
| `-1.401298464324817e-45` | **raises** | — | — | — |
| `-1e-06`, `-0.5`, `-1.0`, `-1e+38` | **raises** | — | — | — |
| `1.0000001192092896` (nextafter 1.0) | **raises** | — | — | — |
| `1.5`, `2.0`, `1e+06`, `3.4e+38` | **raises** | — | — | — |
| `nan` | **raises**, in the JSON parser | — | — | — |
| `inf` | **raises**, in the JSON parser | — | — | — |
| `-inf` | **accepts**, silently stored as `0` | `[0E0]` | `-13.815509796142578` (`S_lo`) | `3244100692` |

The validity check and its full location:

```
$ uv run python p06_errors.py
base_score = -1e-06   (str() as xgboost passes it: '-1e-06')
  xgboost._c_api.XGBoostError:
  [13:06:43] /Users/runner/work/xgboost/xgboost/src/objective/regression_obj.cu:116: Check failed: is_valid: base_score must be in (0,1) for the logistic loss.
  Stack trace:
    [bt] (0) 1   libxgboost.dylib  ... dmlc::LogMessageFatal::~LogMessageFatal() + 124
    [bt] (1) 2   libxgboost.dylib  ... xgboost::obj::RegLossObj<xgboost::obj::LogisticClassification>::ProbToMargin(xgboost::linalg::Tensor<float, 1>*) const + 920
    [bt] (2) 3   libxgboost.dylib  ... xgboost::(anonymous namespace)::Intercept::InitModelParam(xgboost::LearnerTrainParam const&, bool) + 224
    [bt] (3) 4   libxgboost.dylib  ... xgboost::LearnerConfiguration::Configure() + 1840
    [bt] (4) 5   libxgboost.dylib  ... xgboost::LearnerImpl::Reset() + 36
    [bt] (5) 6   libxgboost.dylib  ... XGBoosterReset + 96
```

Two things that trace establishes, and one it does not.

- **Establishes:** the validity check and the prob→margin conversion are the same
  function, `RegLossObj<LogisticClassification>::ProbToMargin`, reached from
  `Intercept::InitModelParam` during `Configure()`. So the clamp lives at intercept
  derivation, which is consistent with `probes/output_transform.md` §10's finding that the
  *stored* value is unclamped.
- **Establishes:** the parameter is typed `xgboost::common::ParamArray<float>` (visible in
  the `nan` trace at §9.2) — a **float32** array. The stored value is float32 before the
  clamp sees it.
- **Does not establish:** the source-level clamp expression. That would need a source read,
  which this probe did not do. **INFERRED** only that the clamp sits inside
  `ProbToMargin`, from the fact that the check that guards it is there.

**The check admits a closed interval while its message says an open one.** `0.0` and `1.0`
are both accepted; the message reads `must be in (0,1)`:

```
  is 0.0 really accepted while -1e-45 is rejected?  boundary of the validity check:
    0.0                    0.0                      ACCEPTED  stored=[0E0]      measured=-13.815509796142578    bits=3244100692
    -0.0                   -0.0                     ACCEPTED  stored=[-0E0]     measured=-13.815509796142578    bits=3244100692
    smallest subnormal +   1.401298464324817e-45    ACCEPTED  stored=[1E-45]    measured=-13.815509796142578    bits=3244100692
    smallest subnormal -   -1.401298464324817e-45   RAISES    [13:06:54] .../regression_obj.cu:116: Check failed: is_valid: base_score must be in (0,1) for the logistic loss.
    1.0                    1.0                      ACCEPTED  stored=[1E0]      measured=13.745160102844238     bits=1096543277
    nextafter(1.0) in f32  1.0000001192092896       RAISES    [13:06:54] .../regression_obj.cu:116: Check failed: is_valid: base_score must be in (0,1) for the logistic loss.
```

The accepted set is exactly `[0.0, 1.0]` in float32, `-0.0` included. Adjacent float32 on
either side raise.

**A `base_score` outside `[0,1]` cannot reach `predict` at all**, so the exporter will
never read one from a legitimately-produced artifact — and cannot read one from a
hand-edited artifact either, because `load_model` runs the same check:

```
$ uv run python p07_handedit.py
============================================================================================================
L. can a base_score outside [0,1] reach predict via a hand-edited artifact?
    hand-edited base_score         restored                                                       result
                     -5E-1                - RAISES .../regression_obj.cu:116: Check failed: is_valid: base_score mu
                     -1E-6                - RAISES .../regression_obj.cu:116: Check failed: is_valid: base_score mu
                       2E0                - RAISES .../regression_obj.cu:116: Check failed: is_valid: base_score mu
                     1.5E0                - RAISES .../regression_obj.cu:116: Check failed: is_valid: base_score mu
                       1E6                - RAISES .../regression_obj.cu:116: Check failed: is_valid: base_score mu
               1.0000001E0                - RAISES .../regression_obj.cu:116: Check failed: is_valid: base_score mu
                       0E0            [0E0]                intercept=-13.815509796142578 bits=3244100692
                      -0E0           [-0E0]                intercept=-13.815509796142578 bits=3244100692
                      1E45                - RAISES .../regression_obj.cu:116: Check failed: is_valid: base_score mu
                     -1E38                - RAISES .../regression_obj.cu:116: Check failed: is_valid: base_score mu
```

### `survival:cox` and `reg:squarederror` at the same extremes

```
--- survival:cox
                      case                   passed      fit       stored  bfa         measured intercept        bits #distinct
                       0.0                      0.0  accepts        [0E0]    0                       -inf  4286578688         1
                      -0.0                     -0.0  accepts       [-0E0]    0                       -inf  4286578688         1
                       1.0                      1.0  accepts        [1E0]    0                        0.0           0         1
  smallest f32 subnormal +    1.401298464324817e-45  accepts      [1E-45]    0         -103.2789306640625  3268316880         1
  smallest f32 subnormal -   -1.401298464324817e-45  accepts     [-1E-45]    0                        nan  2143289344         1
                    -1e-06                   -1e-06  accepts      [-1E-6]    0                        nan  2143289344         1
                      -0.5                     -0.5  accepts      [-5E-1]    0                        nan  2143289344         1
                      -1.0                     -1.0  accepts       [-1E0]    0                        nan  2143289344         1
                    -1e+38                   -1e+38  accepts      [-1E38]    0                        nan  2143289344         1
        nextafter(1.0) f32       1.0000001192092896  accepts [1.0000001E0]    0     1.1920928244535389e-07   872415231         1
                       1.5                      1.5  accepts      [1.5E0]    0        0.40546509623527527  1053792543         1
                       2.0                      2.0  accepts        [2E0]    0         0.6931471824645996  1060205080         1
                     1e+06                1000000.0  accepts        [1E6]    0         13.815510749816895  1096617045         1
                   3.4e+38                  3.4e+38  accepts     [3.4E38]    0          88.72200775146484  1118925227         1
                       nan                      nan   RAISES   Expecting null value "null", around character position: 3
                       inf                      inf   RAISES   Unknown construct, around character position: 0
                      -inf                     -inf  accepts        [0E0]    0                       -inf  4286578688         1

--- reg:squarederror
                      case                   passed      fit       stored  bfa         measured intercept        bits #distinct
                       0.0                      0.0  accepts        [0E0]    0                        0.0           0         1
                      -0.0                     -0.0  accepts       [-0E0]    0                       -0.0  2147483648         1
                       1.0                      1.0  accepts        [1E0]    0                        1.0  1065353216         1
  smallest f32 subnormal +    1.401298464324817e-45  accepts      [1E-45]    0      1.401298464324817e-45           1         1
  smallest f32 subnormal -   -1.401298464324817e-45  accepts     [-1E-45]    0     -1.401298464324817e-45  2147483649         1
                    -1e-06                   -1e-06  accepts      [-1E-6]    0     -9.999999974752427e-07  3045472189         1
                      -0.5                     -0.5  accepts      [-5E-1]    0                       -0.5  3204448256         1
                      -1.0                     -1.0  accepts       [-1E0]    0                       -1.0  3212836864         1
                    -1e+38                   -1e+38  accepts      [-1E38]    0     -9.999999680285692e+37  4271273625         1
        nextafter(1.0) f32       1.0000001192092896  accepts [1.0000001E0]    0         1.0000001192092896  1065353217         1
                       1.5                      1.5  accepts      [1.5E0]    0                        1.5  1069547520         1
                       2.0                      2.0  accepts        [2E0]    0                        2.0  1073741824         1
                     1e+06                1000000.0  accepts        [1E6]    0                  1000000.0  1232348160         1
                   3.4e+38                  3.4e+38  accepts     [3.4E38]    0     3.3999999521443642e+38  2139081118         1
                       nan                      nan   RAISES   Expecting null value "null", around character position: 3
                       inf                      inf   RAISES   Unknown construct, around character position: 0
                      -inf                     -inf  accepts        [0E0]    0                        0.0           0         1
```

Two non-finite intercepts are reachable and neither raises, both **new** relative to
`probes/base_score.md` and `probes/output_transform.md`:

- **`survival:cox` with `base_score = 0.0` or `-0.0` gives intercept `-inf`**, bits
  `4286578688` = `0xFF800000`. No error, no warning.
- **`survival:cox` with any negative `base_score` gives intercept `NaN`**, bits
  `2143289344` = `0x7FC00000`. No error, no warning. `reg:squarederror` accepts the same
  negatives and returns them verbatim, which is correct for an identity link.

`FORMAT.md` §6 requires `intercept` to be "exactly representable as float32", and §9.3
covers non-finite values; whether a `-inf` or `NaN` intercept is exportable or must be
refused is a format question this probe does not decide.

---

## 8. Cox and regression are unclamped — extended well past `1e-38` / `1e38`

The prior probe checked `1e-38` and `1e38`. This extends to the **smallest positive
float32 subnormal** and the **largest finite float32**, and adds the values that would be
diagnostic if the logistic bounds were being reused (`1e-7`, `9.9e-7`, `0.999999`,
`0.9999999`, `1.0`).

```
$ uv run python p08_cox_reg.py
================================================================================================================
M. survival:cox -- is base_score clamped?   (34 values)
                       arg             stored  bfa                  measured      m.bits UNCLAMPED recipe CLAMPED recipe            verdict
     1.401298464324817e-45            [1E-45]    0        -103.2789306640625  3268316880  3268316880     3244100693          UNCLAMPED
                     1e-44            [1E-44]    0       -101.33302307128906  3268061826  3268061826     3244100693          UNCLAMPED
                     1e-42            [1E-42]    0        -96.70804595947266  3267455621  3267455621     3244100693          UNCLAMPED
                     1e-40            [1E-40]    0        -92.10340881347656  3266852082  3266852082     3244100693          UNCLAMPED
                     1e-39            [1E-39]    0        -89.80081939697266  3266550277  3266550277     3244100693          UNCLAMPED
                     1e-38            [1E-38]    0        -87.49822998046875  3266248472  3266248472     3244100693          UNCLAMPED
                     1e-30            [1E-30]    0        -69.07755279541016  3263834037  3263834037     3244100693          UNCLAMPED
                     1e-20            [1E-20]    0       -46.051700592041016  3258463473  3258463473     3244100693          UNCLAMPED
                     1e-12            [1E-12]    0        -27.63102149963379  3252489301  3252489301     3244100693          UNCLAMPED
                     1e-08             [1E-8]    0        -18.42068099975586  3247660430  3247660430     3244100693          UNCLAMPED
                     1e-07             [1E-7]    0        -16.11809539794922  3246453212  3246453212     3244100693          UNCLAMPED
                   9.9e-07           [9.9E-7]    0       -13.825560569763184  3244111231  3244111231     3244100693          UNCLAMPED
                     1e-06             [1E-6]    0       -13.815510749816895  3244100693  3244100693     3244100693        both (same)
    1.0000002248489182e-06     [1.0000002E-6]    0       -13.815510749816895  3244100693  3244100693     3244100693        both (same)
                   1.1e-06           [1.1E-6]    0       -13.720200538635254  3244000753  3244000753     3244000753        both (same)
                     1e-05             [1E-5]    0       -11.512925148010254  3241686257  3241686257     3241686257        both (same)
                     0.001             [1E-3]    0        -6.907755374908447  3235712085  3235712085     3235712085        both (same)
                      0.25           [2.5E-1]    0       -1.3862943649291992  3216077336  3216077336     3216077336        both (same)
                       0.5             [5E-1]    0       -0.6931471824645996  3207688728  3207688728     3207688728        both (same)
                  0.999999       [9.99999E-1]    0   -1.0132795296158292e-06  3045588997  3045588997     3046113285          UNCLAMPED
                 0.9999995      [9.999995E-1]    0    -4.768372718899627e-07  3036676098  3036676098     3046113285          UNCLAMPED
                 0.9999999      [9.999999E-1]    0   -1.1920928955078125e-07  3019898880  3019898881     3046113285            NEITHER
                       1.0              [1E0]    0                       0.0           0           0     3046113285          UNCLAMPED
        1.0000001192092896      [1.0000001E0]    0    1.1920928244535389e-07   872415231   872415231     3046113285          UNCLAMPED
                       1.5            [1.5E0]    0       0.40546509623527527  1053792543  1053792543     3046113285          UNCLAMPED
                       2.0              [2E0]    0        0.6931471824645996  1060205080  1060205080     3046113285          UNCLAMPED
                       7.5            [7.5E0]    0        2.0149030685424805  1073804332  1073804332     3046113285          UNCLAMPED
                    1000.0              [1E3]    0         6.907755374908447  1088228437  1088228437     3046113285          UNCLAMPED
                 1000000.0              [1E6]    0        13.815510749816895  1096617045  1096617045     3046113285          UNCLAMPED
           1000000000000.0             [1E12]    0         27.63102149963379  1105005653  1105005653     3046113285          UNCLAMPED
                     1e+20             [1E20]    0        46.051700592041016  1110979825  1110979825     3046113285          UNCLAMPED
                     1e+30             [1E30]    0         69.07755279541016  1116350389  1116350389     3046113285          UNCLAMPED
                     1e+38             [1E38]    0         87.49822998046875  1118764824  1118764824     3046113285          UNCLAMPED
    3.4028234663852886e+38     [3.4028235E38]    0         88.72283935546875  1118925336  1118925336     3046113285          UNCLAMPED

  bit-exact match counts over the 34 accepted values:
    unclamped recipe : 33/34
    logistic-bounds clamp then recipe : 7/34
    values that raised at fit time: 0
```

```
================================================================================================================
M. reg:squarederror -- is base_score clamped?   (25 values)
                       arg             stored  bfa                  measured      m.bits UNCLAMPED recipe CLAMPED recipe            verdict
   -3.4028234663852886e+38    [-3.4028235E38]    0   -3.4028234663852886e+38  4286578687  4286578687      897988542          UNCLAMPED
                    -1e+38            [-1E38]    0    -9.999999680285692e+37  4271273625  4271273625      897988542          UNCLAMPED
                    -1e+30            [-1E30]    0   -1.0000000150474662e+30  4048155338  4048155338      897988542          UNCLAMPED
                -1000000.0             [-1E6]    0                -1000000.0  3379831808  3379831808      897988542          UNCLAMPED
                      -1.0             [-1E0]    0                      -1.0  3212836864  3212836864      897988542          UNCLAMPED
                    -1e-06            [-1E-6]    0    -9.999999974752427e-07  3045472189  3045472189      897988542          UNCLAMPED
                    -1e-38           [-1E-38]    0    -9.999999350456404e-39  2154619886  2154619886      897988542          UNCLAMPED
    -1.401298464324817e-45           [-1E-45]    0    -1.401298464324817e-45  2147483649  2147483649      897988542          UNCLAMPED
                      -0.0             [-0E0]    0                      -0.0  2147483648  2147483648      897988542          UNCLAMPED
                       0.0              [0E0]    0                       0.0           0           0      897988542          UNCLAMPED
     1.401298464324817e-45            [1E-45]    0     1.401298464324817e-45           1           1      897988542          UNCLAMPED
                     1e-38            [1E-38]    0     9.999999350456404e-39     7136238     7136238      897988542          UNCLAMPED
                     1e-07             [1E-7]    0    1.0000000116860974e-07   869711765   869711765      897988542          UNCLAMPED
                     1e-06             [1E-6]    0     9.999999974752427e-07   897988541   897988541      897988542          UNCLAMPED
    1.0000002248489182e-06     [1.0000002E-6]    0    1.0000002248489182e-06   897988543   897988543      897988543        both (same)
                      0.25           [2.5E-1]    0                      0.25  1048576000  1048576000     1048576000        both (same)
                       0.5             [5E-1]    0                       0.5  1056964608  1056964608     1056964608        both (same)
                  0.999999       [9.99999E-1]    0        0.9999989867210388  1065353199  1065353199     1065353198          UNCLAMPED
                       1.0              [1E0]    0                       1.0  1065353216  1065353216     1065353198          UNCLAMPED
        1.0000001192092896      [1.0000001E0]    0        1.0000001192092896  1065353217  1065353217     1065353198          UNCLAMPED
                       2.0              [2E0]    0                       2.0  1073741824  1073741824     1065353198          UNCLAMPED
                 1000000.0              [1E6]    0                 1000000.0  1232348160  1232348160     1065353198          UNCLAMPED
                     1e+30             [1E30]    0    1.0000000150474662e+30  1900671690  1900671690     1065353198          UNCLAMPED
                     1e+38             [1E38]    0     9.999999680285692e+37  2123789977  2123789977     1065353198          UNCLAMPED
    3.4028234663852886e+38     [3.4028235E38]    0    3.4028234663852886e+38  2139095039  2139095039     1065353198          UNCLAMPED

  bit-exact match counts over the 25 accepted values:
    unclamped recipe : 25/25
    logistic-bounds clamp then recipe : 3/25
    values that raised at fit time: 0
```

| Objective | Values tried | Unclamped recipe | Logistic-bounds clamp |
|---|---|---|---|
| `survival:cox` | 34 values, `1.401298464324817e-45` … `3.4028234663852886e+38` | **33/34** | 7/34 |
| `reg:squarederror` | 25 values, `-3.4028234663852886e+38` … `3.4028234663852886e+38` | **25/25** | 3/25 |

**Verdict: neither objective is clamped.** For Cox the logistic-bounds hypothesis is wrong
on 27 of 34 values including every value below `1e-6` and every value above `1.1e-6`; for
regression on 22 of 25. The "both (same)" rows are values that happen to lie inside the
logistic clamp window, where the two hypotheses cannot differ by construction.

The one Cox `NEITHER` row is **not** a clamp. It is a 1-ULP log-rounding disagreement,
and it opens the next section.

### The Cox `ln` is a float32 log, not a float64 log narrowed once — reported loudly

`FORMAT.md` §6.1 and D015 specify `log(f32(base_score))` for `survival:cox` without
saying in which precision the `log` runs. `probes/base_score.md` §3 measured
`ln (f32 in, f64 log) 17/17` and concluded from it. **That conclusion holds on the 17
values it tested and fails outside them.** Broadened to 1547 values:

```
$ uv run python p11_log_formulation.py
================================================================================================================
R. survival:cox  --  intercept = log(f32(base_score)).  Which log?
  1547 float32 base_score values, all with boost_from_average == '0'
    f32(math.log(f32(p)))   float64 log, narrowed once (FORMAT.md 6.1 / D015) : 1538/1547
    np.log(f32(p))          float32 log                                       : 1547/1547
    f32(mpmath.log(p))      correctly rounded, one rounding                   : 1538/1547

  values where the float64-log recipe is WRONG: 9
           f32(base_score)           stored          XGBoost intercept         xgb   f64narrow      f32log      mpmath
        0.9975585341453552   [9.9755853E-1]     -0.0024444512091577053  3139449622  3139449621  3139449622  3139449621
        0.9981271028518677    [9.981271E-1]     -0.0018746532732620835  3136665325  3136665324  3136665325  3136665324
        0.9999998807907104    [9.999999E-1]    -1.1920928955078125e-07  3019898880  3019898881  3019898880  3019898881
        1.0000007152557373    [1.0000007E0]      7.152554530875932e-07   893386747   893386748   893386747   893386748
          1.00210702419281     [1.002107E0]        0.00210480741225183   990507215   990507216   990507215   990507216
        1.0022073984146118    [1.0022074E0]       0.002204965567216277   990937391   990937392   990937391   990937392
        1.0038461685180664    [1.0038462E0]      0.0038387910462915897   997954618   997954617   997954618   997954617
        1.0066554546356201    [1.0066555E0]       0.006633404642343521  1004100872  1004100873  1004100872  1004100873
         1.009063482284546    [1.0090635E0]       0.009022654965519905  1007932354  1007932355  1007932354  1007932355
```

Then narrowed to the region and quantified, 5400 more values:

```
$ uv run python p12_cox_region.py
================================================================================================================
T. survival:cox -- extent and magnitude of the float64-log 1-ULP defect
          window      n   f64log ok   f32log ok          worst abs err  worst rel err
      [0.5, 0.9]    600         599         600 2.9802322387695312e-08      8.000e-08
    [0.9, 0.999]    600         596         600  4.656612873077393e-10      1.001e-07
    [0.999, 1.0]    600         590         600  5.820766091346741e-11      1.014e-07
    [1.0, 1.001]    600         591         600  5.820766091346741e-11      1.125e-07
    [1.001, 1.1]    600         598         600  1.862645149230957e-09      1.006e-07
      [1.1, 2.0]    600         600         600                    0.0      0.000e+00
      [2.0, 1e6]    600         600         600                    0.0      0.000e+00
     [1e-6, 0.5]    600         600         600                    0.0      0.000e+00
   [1e-40, 1e-6]    600         600         600                    0.0      0.000e+00
           TOTAL   5400        5374        5400
```

Totals across both runs, `survival:cox`, all `bfa == "0"`:

| Formulation of the Cox intercept | Bit-exact |
|---|---|
| **`np.log(f32(p))` — float32 log** | **6947/6947** |
| `f32(math.log(f32(p)))` — float64 log, narrowed once | 6912/6947 (35 wrong) |
| `f32(mpmath.log(p))` — correctly rounded, one rounding | 6912/6947 in the 1547 subset |

Three things this pins down and one it does not.

- The failures are confined to `base_score` in roughly `[0.5, 1.1]`, i.e. where
  `|log(p)|` is small. Below `0.5` and above `1.1` the float64 form is `2400/2400`.
- The error is **1 ULP of the intercept**: worst absolute `2.98e-08`, worst relative
  `1.125e-07`. It does **not** approach the `1e-6` margin accuracy gate. What it breaks is
  **bit-equality against XGBoost**, which `FORMAT.md` §6.2 *requires* of the export-time
  intercept check. The failure direction is therefore a **spurious hard raise at export**
  for a Cox model with `base_score` near 1 — loud, not silent, which is the acceptable
  direction, but still wrong.
- `mpmath` at 50 digits scores the same `1538/1547` as the float64 form, so **XGBoost's
  Cox log is not correctly rounded.** It agrees with numpy's float32 `log` instead. That is
  the same conclusion `probes/output_transform.md` §2 reached for the output transform, by
  the same kind of evidence.
- **INFERRED, not measured:** that XGBoost calls the platform `logf` and that numpy's
  float32 `log` is bit-identical to it. What is *measured* is only that the two agree on
  6947/6947 sampled points including the ones where the float64 form disagrees. Confirming
  the call would need a symbol trace or a source read.

**This contradicts nothing written in `probes/base_score.md` §3 — its 17 values were all
outside the failing region — but the conclusion as phrased there reads more general than
its evidence supports.** `FORMAT.md` §6.1's `log(f32(base_score))` is under-specified in
the same way. Both are stated here so the contradiction is corrected rather than carried.
Deciding what to do about it is out of this probe's scope.

For `binary:logistic` the same question is **not** decidable, and does not matter:

```
================================================================================================================
S. binary:logistic  --  intercept = -log(f32(f32(1/p) - 1)).  Which log?
  1432 float32 base_score values inside the clamp window, all bfa == '0'
    -f32(math.log(f64(t)))  float64 log, narrowed once (FORMAT.md 6.1 / D015) : 1432/1432
    -np.log(t) in float32   float32 log                                       : 1432/1432
    -f32(mpmath.log(t))     correctly rounded, one rounding                   : 1432/1432

  values where the float64-log recipe is WRONG: 0
```

All three formulations tie at `1432/1432`. Inside the clamp window the logistic
intercept's magnitude is never small enough for the three to diverge, so `FORMAT.md`
§6.1's logistic recipe stands exactly as written.

---

## 9. Out of scope, and it looked wrong

### 9.1 The validity check's message describes an interval it does not enforce

`base_score must be in (0,1) for the logistic loss` — but `0.0`, `-0.0` and `1.0` are all
accepted (§7), and each saturates to a clamp bound. A user reading the message would
reasonably conclude `0.0` is rejected. Reported, not resolved.

### 9.2 `base_score = -inf` is silently stored as `0`

Across **all three** objectives, `float("-inf")` is accepted and stored as `[0E0]`:

```
$ uv run python p06_errors.py
the -inf anomaly, in detail
  passed -inf, str() = '-inf'
    ACCEPTED  stored='[0E0]'  bfa=0  measured=-13.815509796142578  bits=3244100692
  passed inf, str() = 'inf'
    RAISES XGBoostError: [13:06:54] /Users/runner/work/xgboost/xgboost/src/common/json.cc:376: Unknown construct, around character position: 0
```

`+inf` raises in the JSON parser; `-inf` does not, and lands on `0`. The asymmetry is in
the parameter parser, not in any objective: the `nan` trace shows the parse path is
`xgboost::common::operator>>(std::istream&, xgboost::common::ParamArray<float>&)` →
`xgboost::Json::Load`, and `nan` fails as `Expecting null value "null", around character
position: 3` — the leading `n` is being read as the start of `null`.

Consequence, and it is the project's exact failure signature: a caller who computes
`base_score` and gets `-inf` from an upstream division gets a **silently substituted `0`**,
which for `binary:logistic` then clamps to `S_lo` and for `survival:cox` becomes an `-inf`
intercept. No exception, no warning. `+inf` and `nan` crash, which is the right behaviour;
`-inf` does not.

**Not measured:** whether this reaches through the sklearn wrapper or `set_param`
differently. `scikit-learn` is not installed in this workspace and this probe installed
nothing.

### 9.3 My own instrument was wrong once, and how

Recorded because the brief asks for it and because it changed a reported answer.
`lib.fit0`'s constancy assertion `len(set(margin)) == 1` misfires on NaN and made me write
"XGBoost RAISES" for `survival:cox` at negative `base_score` when in fact **XGBoost
accepts and returns NaN**. Corrected in §7 with both readings shown side by side. Separately,
a hand-computed claim in an early draft of `p13_constants.py` asserted that a float64 clamp
constant was *excluded*; measuring it showed it is not, and the script was rewritten to
measure rather than assert (§5). Both are reasons the standing rule in D034 exists: a check
whose oracle is my own arithmetic is not a check.

---

## 10. Ambiguity register

| # | Ambiguity | Readings | What would settle it |
|---|---|---|---|
| B1 | Which float32 is the source-level lower clamp constant | Any of the 8 values `0x358637B7` … `0x358637BE`. `f32(1e-6)` = `0x358637BD` is one of them | Source read. **All 8 give bit-identical intercepts on every float32 `base_score`**, verified exhaustively over the plateau, so the choice has no observable consequence |
| B2 | Which float32 is the source-level upper clamp constant | Either `0x3F7FFFEE` or `0x3F7FFFEF`. `f32(1)−f32(1e-6)` = `0x3F7FFFEF` is one of them | Source read. Both give bit-identical intercepts on every float32 `base_score` |
| B3 | Clamp on the input `p`, or on the derived intercept | (a) clamp `p` then transform; (b) transform then clamp the intercept to `[S_lo, S_hi]` | The separating inputs are `base_score` outside `[0,1]`, where (a) gives `S_hi`/`S_lo` and (b) gives `NaN`. **XGBoost refuses those inputs at both fit time and `load_model` time**, so the experiment cannot be run. At `base_score = 1.0` the two agree iff `log(0)` yields `+inf` rather than raising |
| B4 | Clamp constant in float32 or float64 | F1/F2 (float32 constant) and F3 (float64 constant `1e-6`) all reproduce `S_lo`. Not distinguishable | Source read. Measured: the *result* depends only on the float32 snap of the stored decimal (§5) |
| B5 | Whether XGBoost's Cox `log` is the platform `logf` | (a) it is; (b) it is something bit-identical to numpy's float32 `log` on all 6947 sampled points | Symbol trace or source read. Not attempted |

**None of B1–B4 affects an implementation.** Any member of the equivalence classes, in
either precision, with the clamp on the input, reproduces XGBoost bit-for-bit on every
`base_score` XGBoost will accept. B3's clamp-the-output reading is the one variant that
needs care, because it requires `log(0) → +inf` rather than a raise.

---

## Bottom line for D035

D035 currently reads: *"clamp `p` to `[f32(1e-6), f32(1 - 1e-6)]` before applying
`-log(f32(f32(1/p) - 1))`."*

**Both constants are inside the pinned equivalence classes and the rule is bit-exact
`226/226` on the sweep, identical value-for-value to the clamp built from the measured
transition points. D035 requires no numeric change.** What could usefully be added is that
the bounds are now *pinned*, not approximate, with the transition pairs recorded here — and
that the clamp must be applied to the **float32** snap of the stored value, which §5
measures directly.

---

## Scripts

All lived outside the repository, in the session scratchpad.

| Script | What it produced |
|---|---|
| `lib.py` | the fit instrument; asserts `bfa == "0"` and the stored-value round trip |
| `lib2.py` | assertion-free instrument, for NaN margins (§7) |
| `p01_env_instrument.py` | §1 environment and instrument baseline, 12 fits |
| `p02_bisect.py` | §2, §3 bisection to both transition pairs, 42 fits |
| `p03_exhaustive.py` | §4 1026 + 532 consecutive float32, 0 violations; the plateau runs |
| `p04_sweep.py` | §6 the 226-value sweep, four hypotheses |
| `p05_extremes.py`, `p06_errors.py` | §7, §9 extremes and verbatim error text |
| `p07_handedit.py` | §5 hand-edit cross-check, float32-snap dependence, out-of-range refusal |
| `p08_cox_reg.py` | §8 Cox 34 values, regression 25 values |
| `p09_cox_ulp.py`, `p11_log_formulation.py`, `p12_cox_region.py` | §8 the Cox float32-log finding, 6947 values |
| `p10_extremes_fixed.py` | §7 corrected extremes, all three objectives |
| `p13_constants.py`, `p14_formulations.py` | §2, §3, §5 constants table and formulation discrimination |
