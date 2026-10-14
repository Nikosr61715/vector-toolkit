# vector-toolkit

A succinct bit vector supporting `rank` and `select` queries in compact space.

```python
from vector_toolkit import SuccinctBitVector

bv = SuccinctBitVector([1, 0, 1, 1, 0, 0, 1, 0, 1, 1])

# rank(value, i) — count of `value` in bits[0:i]
bv.rank(1, 5)   # 3
bv.rank(0, 5)   # 2

# select(value, k) — index of the k-th occurrence (1-indexed)
bv.select(1, 1) # 0
bv.select(0, 3) # 5

len(bv)         # 10
bv[3]           # 1
```

## Why this exists

Rank and select are the two primitives underneath suffix arrays, FM-indexes,
and wavelet trees. A plain Python list of ints answers both in O(n) by scanning,
which is fine for tiny inputs but dominates the cost of every query once the
vector grows. This library packs bits into 64-bit words and keeps a two-level
block/sample table so that `rank` is O(1) and `select` is bounded by the
sample spacing (512 occurrences). The auxiliary structures use o(n) bits on
top of the raw data.

The trade-off: construction is O(n) and builds the tables eagerly, so this is
not the right choice if you only need a single rank query on a throwaway array.
It pays off when the same vector is queried many times.

## Edge cases

`rank(value, i)` operates on the half-open prefix `bits[0:i]`, so
`rank(value, 0)` is always 0. `i` larger than the vector length is clamped to
the length. `select(value, k)` is 1-indexed and returns `-1` when the k-th
occurrence does not exist. An empty vector is legal: all queries return 0 or
-1 as appropriate.
