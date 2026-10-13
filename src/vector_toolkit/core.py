"""Succinct bit vector with rank and select.

Layout
------
Bits are packed into Python ints, 64 bits per word.  A two-level
block/sample table sits alongside the raw words:

  * ``_rank_blocks``  – cumulative popcount at every 512-bit block
    boundary (one entry per 8 words).
  * ``_select0_samples`` / ``_select1_samples`` – positions of the
    0th, 512th, 1024 … occurrence of each value, used to bound the
    linear scan inside select.

The structure is *succinct*, not *fully compressed*: it uses
o(n) bits of auxiliary space on top of the n bits of data, so
that rank runs in O(1) and select in O(n / sample_rate).
"""

from __future__ import annotations

from typing import List

_WORD_BITS = 64
_BLOCK_WORDS = 8                       # 512-bit blocks
_BLOCK_BITS = _WORD_BITS * _BLOCK_WORDS
_SELECT_SAMPLE = 512                   # sample every 512 occurrences


def _popcount64(x: int) -> int:
    """Population count of a 64-bit value.

    Python ints are arbitrary precision, so we mask to 64 bits first
    and use the classic SWAR popcount.  ``int.bit_count()`` exists in
    3.10+, but masking keeps the invariant explicit and avoids relying
    on a version-specific builtin.
    """
    x &= 0xFFFFFFFFFFFFFFFF
    x -= (x >> 1) & 0x5555555555555555
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    return ((x * 0x0101010101010101) & 0xFFFFFFFFFFFFFFFF) >> 56


class SuccinctBitVector:
    """A bit vector supporting ``rank`` and ``select`` in compact space.

    Parameters
    ----------
    bits:
        Iterable of 0/1 ints (or bools) giving the bit sequence in
        left-to-right order.  Index 0 is the first element yielded.

    Notes
    -----
    * ``rank(value, i)`` returns the number of occurrences of ``value``
      in ``bits[0:i]`` (i.e. the half-open prefix of length ``i``).
      ``rank(value, 0)`` is always 0.
    * ``select(value, k)`` returns the index of the k-th occurrence
      (1-indexed) of ``value``.  ``select(value, 1)`` is the first
      occurrence.  Returns -1 when there is no k-th occurrence.
    """

    __slots__ = (
        "_n",
        "_words",
        "_num_words",
        "_rank_blocks",
        "_total_ones",
        "_select0_samples",
        "_select1_samples",
    )

    def __init__(self, bits) -> None:
        bit_list: List[int] = [1 if b else 0 for b in bits]
        self._n = len(bit_list)

        # Pack into 64-bit words, least-significant bit = lowest index.
        num_words = (self._n + _WORD_BITS - 1) // _WORD_BITS
        words: List[int] = [0] * num_words
        for i, b in enumerate(bit_list):
            if b:
                words[i >> 6] |= 1 << (i & 63)
        self._words = words
        self._num_words = num_words

        # Block-level rank table: cumulative ones at each block start.
        num_blocks = (num_words + _BLOCK_WORDS - 1) // _BLOCK_WORDS
        rank_blocks: List[int] = [0] * num_blocks
        running = 0
        for blk in range(num_blocks):
            rank_blocks[blk] = running
            start = blk * _BLOCK_WORDS
            end = min(start + _BLOCK_WORDS, num_words)
            for w in range(start, end):
                running += _popcount64(words[w])
        self._rank_blocks = rank_blocks
        self._total_ones = running

        # Select samples: record the bit index where the j-th occurrence
        # (for j = 0, sample, 2*sample, …) appears, so the linear scan
        # inside select only walks at most ``sample`` bits.
        self._select1_samples = self._build_select_samples(1)
        self._select0_samples = self._build_select_samples(0)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _build_select_samples(self, value: int) -> List[int]:
        samples: List[int] = []
        count = 0
        next_target = 0
        for i in range(self._n):
            if self._bit_at(i) == value:
                if count == next_target:
                    samples.append(i)
                    next_target += _SELECT_SAMPLE
                count += 1
        return samples

    def _bit_at(self, i: int) -> int:
        """Return the bit at index ``i`` (0/1).  No bounds check."""
        return (self._words[i >> 6] >> (i & 63)) & 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int) -> int:
        if i < 0:
            i += self._n
        if i < 0 or i >= self._n:
            raise IndexError(i)
        return self._bit_at(i)

    def rank(self, value: int, i: int) -> int:
        """Count of ``value`` in ``bits[0:i]`` (half-open).

        ``value`` must be 0 or 1.  ``i`` is clamped to ``[0, len]``.
        """
        if value not in (0, 1):
            raise ValueError("value must be 0 or 1")
        if i < 0:
            raise ValueError("i must be non-negative")
        if i > self._n:
            i = self._n
        if i == 0:
            return 0
        ones = self._rank_ones(i)
        return ones if value == 1 else i - ones

    def _rank_ones(self, i: int) -> int:
        """Number of 1-bits in ``bits[0:i]``."""
        # Block contribution.
        blk = i // _BLOCK_BITS
        if blk >= len(self._rank_blocks):
            blk = len(self._rank_blocks) - 1
        count = self._rank_blocks[blk]
        pos = blk * _BLOCK_BITS
        # Word-level scan inside the block.
        while pos + _WORD_BITS <= i:
            count += _popcount64(self._words[pos >> 6])
            pos += _WORD_BITS
        # Remaining sub-word bits.
        if pos < i:
            rem = i - pos
            word = self._words[pos >> 6] & ((1 << rem) - 1)
            count += _popcount64(word)
        return count

    def select(self, value: int, k: int) -> int:
        """Index of the k-th (1-indexed) occurrence of ``value``.

        Returns -1 if there is no k-th occurrence.
        """
        if value not in (0, 1):
            raise ValueError("value must be 0 or 1")
        if k < 1:
            raise ValueError("k must be >= 1")
        samples = self._select1_samples if value == 1 else self._select0_samples
        total = self._total_ones if value == 1 else self._n - self._total_ones
        if k > total:
            return -1
        # Use the sample just before the k-th occurrence to bound the scan.
        sample_idx = (k - 1) // _SELECT_SAMPLE
        start = samples[sample_idx] if sample_idx < len(samples) else 0
        # ``start`` is the index of the (sample_idx*SELECT_SAMPLE)-th
        # occurrence (0-indexed count), so we have already seen
        # ``sample_idx * SELECT_SAMPLE`` occurrences before ``start``.
        remaining = k - sample_idx * _SELECT_SAMPLE
        pos = start
        while pos < self._n:
            if self._bit_at(pos) == value:
                remaining -= 1
                if remaining == 0:
                    return pos
            pos += 1
        return -1

    def count(self, value: int) -> int:
        """Total occurrences of ``value`` in the vector."""
        if value not in (0, 1):
            raise ValueError("value must be 0 or 1")
        return self._total_ones if value == 1 else self._n - self._total_ones
