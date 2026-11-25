# coding=utf-8
"""Unit tests for kivi_sink.sink_score (M1)."""

import math
import unittest

import torch

from kivi_sink.sink_score import (
    aggregate_layer_statistics,
    bootstrap_mean_ci,
    layer_sink_scores,
    mean_pairwise_spearman,
    normalize_token_scores,
    spearman_correlation,
    top_m_indices,
    top_m_share,
)


class TopMTest(unittest.TestCase):
    def test_descending_and_stable_ties(self):
        scores = torch.tensor([0.1, 0.5, 0.2, 0.2])
        self.assertEqual(top_m_indices(scores, 2).tolist(), [1, 2])
        self.assertEqual(top_m_indices(scores, 4).tolist(), [1, 2, 3, 0])

    def test_share_normalised(self):
        scores = torch.tensor([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(top_m_share(scores, 2, normalize=True), 0.7)
        self.assertAlmostEqual(top_m_share(scores, 2, normalize=False), 7.0)

    def test_padding_is_never_selected(self):
        scores = torch.tensor([[1.0, 2.0, 99.0, 99.0], [10.0, 10.0, 1.0, 1.0]])
        # Second row uses only the first two tokens; padding holds the largest
        # raw values on purpose.
        share = top_m_share(scores, 2, normalize=True, lengths=[4, 2])
        self.assertAlmostEqual(float(share[0]), (99.0 + 99.0) / 201.0, places=6)
        self.assertAlmostEqual(float(share[1]), 1.0, places=6)

    def test_batch_shapes_and_aggregation(self):
        scores = torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 3.0]])
        indices = top_m_indices(scores, 2)
        self.assertEqual(tuple(indices.shape), (2, 2))
        shares = top_m_share(scores, 2)
        self.assertEqual(tuple(shares.shape), (2,))
        self.assertAlmostEqual(float(shares[0]), 0.5)
        self.assertAlmostEqual(float(shares[1]), 1.0)

    def test_invalid_m(self):
        with self.assertRaises(ValueError):
            top_m_indices(torch.ones(4), 0)
        with self.assertRaises(ValueError):
            top_m_indices(torch.ones(4), 5)
        with self.assertRaises(TypeError):
            top_m_share(torch.ones(4), 1.5)

    def test_invalid_scores(self):
        with self.assertRaises(ValueError):
            top_m_share(torch.tensor([1.0, -1.0]), 1)
        with self.assertRaises(ValueError):
            top_m_share(torch.tensor([1.0, float("nan")]), 1)
        with self.assertRaises(ValueError):
            top_m_share(torch.zeros(4), 1, normalize=True)

    def test_does_not_mutate(self):
        scores = torch.tensor([3.0, 1.0, 2.0])
        before = scores.clone()
        top_m_share(scores, 2)
        top_m_indices(scores, 2)
        self.assertTrue(torch.equal(scores, before))


class NormalizationTest(unittest.TestCase):
    def test_sums_to_one_with_lengths(self):
        scores = torch.tensor([[1.0, 1.0, 2.0], [3.0, 1.0, 1.0]])
        normalised = normalize_token_scores(scores, lengths=[3, 2])
        self.assertAlmostEqual(float(normalised[0].sum()), 1.0)
        self.assertAlmostEqual(float(normalised[1].sum()), 1.0)
        self.assertEqual(float(normalised[1, 2]), 0.0)

    def test_1d_roundtrip(self):
        scores = torch.tensor([1.0, 3.0])
        normalised = normalize_token_scores(scores)
        self.assertTrue(torch.allclose(normalised, torch.tensor([0.25, 0.75])))


class LayerScoreTest(unittest.TestCase):
    def test_mapping_list_and_tensor_inputs_agree(self):
        layer0 = torch.tensor([0.1, 0.2, 0.3, 0.4])
        layer1 = torch.tensor([0.4, 0.3, 0.2, 0.1])
        from_mapping = layer_sink_scores({0: layer0, 1: layer1}, 2)
        from_list = layer_sink_scores([layer0, layer1], 2)
        from_tensor = layer_sink_scores(torch.stack([layer0, layer1]), 2)
        self.assertEqual(from_mapping.tolist(), from_list.tolist())
        self.assertEqual(from_mapping.tolist(), from_tensor.tolist())
        # layer0: top-2 = 0.4 + 0.3; layer1: top-2 = 0.4 + 0.3
        self.assertTrue(torch.allclose(from_mapping, torch.tensor([0.7, 0.7])))

    def test_mapping_is_sorted_by_layer(self):
        scores = {1: torch.tensor([0.0, 1.0]), 0: torch.tensor([1.0, 0.0])}
        result = layer_sink_scores(scores, 1)
        self.assertTrue(torch.allclose(result, torch.tensor([1.0, 1.0])))

    def test_empty_input(self):
        with self.assertRaises(ValueError):
            layer_sink_scores({}, 1)


class AggregateStatisticsTest(unittest.TestCase):
    def test_mean_median_std_and_ci(self):
        matrix = torch.tensor([[0.1, 0.5], [0.3, 0.7], [0.5, 0.9]])
        stats = aggregate_layer_statistics(matrix, n_boot=200, ci=0.9, seed=0)
        self.assertEqual(stats["n"], [3, 3])
        self.assertTrue(
            all(abs(a - b) < 1e-6 for a, b in zip(stats["mean"], [0.3, 0.7]))
        )
        self.assertTrue(
            all(abs(a - b) < 1e-6 for a, b in zip(stats["median"], [0.3, 0.7]))
        )
        for index in range(2):
            self.assertLessEqual(stats["ci_low"][index], stats["mean"][index])
            self.assertGreaterEqual(stats["ci_high"][index], stats["mean"][index])
            self.assertAlmostEqual(stats["std"][index], 0.2, places=6)

    def test_single_sample(self):
        stats = aggregate_layer_statistics([[0.2, 0.4]], n_boot=10)
        for actual, expected in zip(stats["mean"], [0.2, 0.4]):
            self.assertAlmostEqual(actual, expected, places=6)
        self.assertEqual(stats["std"], [0.0, 0.0])
        for actual, expected in zip(stats["ci_low"], [0.2, 0.4]):
            self.assertAlmostEqual(actual, expected, places=6)
        for actual, expected in zip(stats["ci_high"], [0.2, 0.4]):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_bootstrap_is_deterministic_per_seed(self):
        values = torch.arange(50, dtype=torch.float32)
        first = bootstrap_mean_ci(values, n_boot=100, seed=7)
        second = bootstrap_mean_ci(values, n_boot=100, seed=7)
        self.assertEqual(first, second)

    def test_bootstrap_does_not_touch_global_rng(self):
        torch.manual_seed(123)
        state = torch.get_rng_state().clone()
        bootstrap_mean_ci(torch.arange(20, dtype=torch.float32), n_boot=50, seed=3)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_invalid_bootstrap_arguments(self):
        with self.assertRaises(ValueError):
            bootstrap_mean_ci(torch.ones(4), n_boot=0)
        with self.assertRaises(ValueError):
            bootstrap_mean_ci(torch.ones(4), ci=1.0)
        with self.assertRaises(ValueError):
            bootstrap_mean_ci(torch.tensor([]))

    def test_inconsistent_layer_counts(self):
        with self.assertRaises(ValueError):
            aggregate_layer_statistics([[0.1, 0.2], [0.3]])


class SpearmanTest(unittest.TestCase):
    def test_perfect_monotone(self):
        self.assertAlmostEqual(
            spearman_correlation([1, 2, 3, 4], [10, 20, 30, 40]), 1.0
        )

    def test_reverse(self):
        self.assertAlmostEqual(
            spearman_correlation([1, 2, 3, 4], [4, 3, 2, 1]), -1.0
        )

    def test_known_value_with_a_swap(self):
        self.assertAlmostEqual(
            spearman_correlation([1, 2, 3, 4], [2, 1, 4, 3]), 0.6, places=6
        )

    def test_ties_use_average_ranks(self):
        # x ranks: [1.5, 1.5, 3, 4], y ranks: [1, 2, 3, 5] -> check against torch.
        value = spearman_correlation([1, 1, 2, 3], [1, 2, 3, 4])
        self.assertFalse(math.isnan(value))
        self.assertGreater(value, 0.9)

    def test_constant_is_nan(self):
        self.assertTrue(math.isnan(spearman_correlation([1, 1, 1], [1, 2, 3])))
        self.assertTrue(math.isnan(spearman_correlation([1, 2], [5, 5])))

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            spearman_correlation([1, 2, 3], [1, 2])

    def test_mean_pairwise(self):
        matrix = torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [0.2, 0.4, 0.6],
                [0.3, 0.6, 0.9],
            ]
        )
        self.assertAlmostEqual(mean_pairwise_spearman(matrix), 1.0, places=6)
        self.assertTrue(math.isnan(mean_pairwise_spearman(matrix[:1])))
        reversed_rows = torch.tensor([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]])
        self.assertAlmostEqual(mean_pairwise_spearman(reversed_rows), -1.0, places=6)


if __name__ == "__main__":
    unittest.main()
