import unittest

from vector_toolkit import SuccinctBitVector


class TestConstruction(unittest.TestCase):
    def test_empty_vector(self):
        bv = SuccinctBitVector([])
        self.assertEqual(len(bv), 0)
        self.assertEqual(bv.rank(1, 0), 0)
        self.assertEqual(bv.rank(0, 0), 0)
        self.assertEqual(bv.select(1, 1), -1)
        self.assertEqual(bv.select(0, 1), -1)
        self.assertEqual(bv.count(1), 0)
        self.assertEqual(bv.count(0), 0)

    def test_all_zeros(self):
        bv = SuccinctBitVector([0] * 100)
        self.assertEqual(len(bv), 100)
        self.assertEqual(bv.rank(1, 100), 0)
        self.assertEqual(bv.rank(0, 100), 100)
        self.assertEqual(bv.select(1, 1), -1)
        self.assertEqual(bv.select(0, 50), 49)
        self.assertEqual(bv.count(1), 0)
        self.assertEqual(bv.count(0), 100)

    def test_all_ones(self):
        bv = SuccinctBitVector([1] * 100)
        self.assertEqual(len(bv), 100)
        self.assertEqual(bv.rank(1, 100), 100)
        self.assertEqual(bv.rank(0, 100), 0)
        self.assertEqual(bv.select(0, 1), -1)
        self.assertEqual(bv.select(1, 50), 49)

    def test_accepts_bools(self):
        bv = SuccinctBitVector([True, False, True])
        self.assertEqual(bv.count(1), 2)
        self.assertEqual(bv.select(1, 1), 0)
        self.assertEqual(bv.select(1, 2), 2)


class TestRank(unittest.TestCase):
    def test_rank_basic(self):
        bv = SuccinctBitVector([1, 0, 1, 1, 0, 0, 1, 0, 1, 1])
        self.assertEqual(bv.rank(1, 0), 0)
        self.assertEqual(bv.rank(1, 1), 1)
        self.assertEqual(bv.rank(1, 2), 1)
        self.assertEqual(bv.rank(1, 4), 3)
        self.assertEqual(bv.rank(1, 10), 6)
        self.assertEqual(bv.rank(0, 10), 4)

    def test_rank_at_word_boundaries(self):
        # 64 bits: alternating, then a block boundary at 64.
        bits = [i % 2 for i in range(64)]
        bv = SuccinctBitVector(bits)
        self.assertEqual(bv.rank(1, 64), 32)
        self.assertEqual(bv.rank(1, 32), 16)
        self.assertEqual(bv.rank(0, 64), 32)

    def test_rank_at_block_boundary(self):
        # 512-bit block boundary plus a bit more.
        bits = [1] * 512 + [0] * 10
        bv = SuccinctBitVector(bits)
        self.assertEqual(bv.rank(1, 512), 512)
        self.assertEqual(bv.rank(1, 522), 512)
        self.assertEqual(bv.rank(0, 522), 10)

    def test_rank_clamps_past_end(self):
        bv = SuccinctBitVector([1, 0, 1])
        self.assertEqual(bv.rank(1, 999), 2)

    def test_rank_rejects_bad_value(self):
        bv = SuccinctBitVector([1, 0])
        with self.assertRaises(ValueError):
            bv.rank(2, 1)

    def test_rank_rejects_negative(self):
        bv = SuccinctBitVector([1, 0])
        with self.assertRaises(ValueError):
            bv.rank(1, -1)


class TestSelect(unittest.TestCase):
    def test_select_basic(self):
        bv = SuccinctBitVector([0, 1, 0, 1, 1, 0, 1, 0, 0, 1])
        self.assertEqual(bv.select(1, 1), 1)
        self.assertEqual(bv.select(1, 2), 3)
        self.assertEqual(bv.select(1, 3), 4)
        self.assertEqual(bv.select(1, 5), 9)
        self.assertEqual(bv.select(0, 1), 0)
        self.assertEqual(bv.select(0, 2), 2)
        self.assertEqual(bv.select(0, 5), 8)

    def test_select_out_of_range(self):
        bv = SuccinctBitVector([1, 0, 1])
        self.assertEqual(bv.select(1, 3), -1)
        self.assertEqual(bv.select(0, 2), -1)
        self.assertEqual(bv.select(1, 100), -1)

    def test_select_rejects_bad_k(self):
        bv = SuccinctBitVector([1, 0])
        with self.assertRaises(ValueError):
            bv.select(1, 0)
        with self.assertRaises(ValueError):
            bv.select(1, -1)

    def test_select_rejects_bad_value(self):
        bv = SuccinctBitVector([1, 0])
        with self.assertRaises(ValueError):
            bv.select(2, 1)

    def test_select_across_sample_boundary(self):
        # 600 ones then 600 zeros — exercises the 512-occurrence sample.
        bits = [1] * 600 + [0] * 600
        bv = SuccinctBitVector(bits)
        self.assertEqual(bv.select(1, 1), 0)
        self.assertEqual(bv.select(1, 512), 511)
        self.assertEqual(bv.select(1, 600), 599)
        self.assertEqual(bv.select(0, 1), 600)
        self.assertEqual(bv.select(0, 512), 1111)
        self.assertEqual(bv.select(0, 600), 1199)


class TestIndexing(unittest.TestCase):
    def test_getitem(self):
        bv = SuccinctBitVector([1, 0, 1, 0, 1])
        self.assertEqual(bv[0], 1)
        self.assertEqual(bv[1], 0)
        self.assertEqual(bv[4], 1)
        self.assertEqual(bv[-1], 1)
        self.assertEqual(bv[-2], 0)

    def test_getitem_out_of_range(self):
        bv = SuccinctBitVector([1, 0])
        with self.assertRaises(IndexError):
            _ = bv[2]
        with self.assertRaises(IndexError):
            _ = bv[-3]


class TestRoundTrip(unittest.TestCase):
    def test_rank_select_inverse(self):
        import random
        rng = random.Random(20240101)
        bits = [rng.randint(0, 1) for _ in range(300)]
        bv = SuccinctBitVector(bits)
        # For every position, rank then select should round-trip.
        for i in range(len(bits)):
            if bits[i] == 1:
                k = bv.rank(1, i) + 1
                self.assertEqual(bv.select(1, k), i)
            else:
                k = bv.rank(0, i) + 1
                self.assertEqual(bv.select(0, k), i)

    def test_large_vector(self):
        import random
        rng = random.Random(20240202)
        bits = [rng.randint(0, 1) for _ in range(2000)]
        bv = SuccinctBitVector(bits)
        # Spot-check rank at several points against a naive count.
        for i in [0, 1, 63, 64, 65, 511, 512, 513, 1000, 1999, 2000]:
            expected_ones = sum(bits[:i])
            self.assertEqual(bv.rank(1, i), expected_ones)
            self.assertEqual(bv.rank(0, i), i - expected_ones)
        # Spot-check select for a few k values.
        one_positions = [i for i, b in enumerate(bits) if b == 1]
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for k in [1, 2, 100, 513, len(one_positions)]:
            if k <= len(one_positions):
                self.assertEqual(bv.select(1, k), one_positions[k - 1])
        for k in [1, 2, 100, 513, len(zero_positions)]:
            if k <= len(zero_positions):
                self.assertEqual(bv.select(0, k), zero_positions[k - 1])


if __name__ == "__main__":
    unittest.main()
