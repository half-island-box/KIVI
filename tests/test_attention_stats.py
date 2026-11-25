# coding=utf-8
"""Unit tests for kivi_sink.attention_stats (M1)."""

import math
import unittest

import torch

from kivi_sink.attention_stats import (
    attention_mass_from_weights,
    causal_attention_mass,
    causal_attention_mass_streaming,
    causal_mask,
    key_norm_scores,
    key_norm_scores_from_projection,
    repeat_kv,
    select_query_count,
)


def _manual_attention_mass(query, key, query_fraction=1.0, query_offset=None):
    """Straightforward reference implementation (no chunking, no helper reuse)."""
    num_rep = query.shape[1] // key.shape[1]
    if num_rep > 1:
        key = key.repeat_interleave(num_rep, dim=1)
    head_dim = query.shape[-1]
    logits = torch.matmul(query.float(), key.float().transpose(-1, -2)) / math.sqrt(
        head_dim
    )
    query_len, key_len = query.shape[-2], key.shape[-2]
    offset = key_len - query_len if query_offset is None else query_offset
    allowed = torch.arange(key_len).unsqueeze(0) <= (
        torch.arange(offset, offset + query_len).unsqueeze(1)
    )
    logits = logits.masked_fill(~allowed, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    count = select_query_count(query_len, query_fraction)
    return probs[:, :, query_len - count :, :].mean(dim=(1, 2))


class CausalMaskTest(unittest.TestCase):
    def test_mask_matches_manual(self):
        mask = causal_mask(3, 5)
        expected = torch.tensor(
            [
                [True, True, True, False, False],
                [True, True, True, True, False],
                [True, True, True, True, True],
            ]
        )
        self.assertTrue(torch.equal(mask, expected))

    def test_explicit_offset(self):
        mask = causal_mask(2, 6, query_offset=1)
        expected = torch.tensor(
            [
                [True, True, False, False, False, False],
                [True, True, True, False, False, False],
            ]
        )
        self.assertTrue(torch.equal(mask, expected))

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            causal_mask(0, 4)
        with self.assertRaises(ValueError):
            causal_mask(3, 4, query_offset=2)  # 2 + 3 > 4
        with self.assertRaises(ValueError):
            causal_mask(3, 4, query_offset=-1)


class SelectQueryCountTest(unittest.TestCase):
    def test_fraction(self):
        self.assertEqual(select_query_count(8, 1.0), 8)
        self.assertEqual(select_query_count(8, 0.25), 2)
        self.assertEqual(select_query_count(5, 0.25), 2)  # ceil(1.25)
        self.assertEqual(select_query_count(100, 0.01), 1)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            select_query_count(0)
        with self.assertRaises(ValueError):
            select_query_count(4, 0.0)
        with self.assertRaises(ValueError):
            select_query_count(4, 1.5)


class CausalAttentionMassTest(unittest.TestCase):
    def test_scores_are_a_probability_distribution(self):
        torch.manual_seed(0)
        query = torch.randn(2, 4, 16, 8)
        key = torch.randn(2, 2, 16, 8)
        mass = causal_attention_mass(query, key)
        self.assertEqual(tuple(mass.shape), (2, 16))
        self.assertTrue(torch.allclose(mass.sum(dim=-1), torch.ones(2), atol=1e-6))
        self.assertTrue((mass >= 0).all())

    def test_future_keys_receive_no_mass(self):
        torch.manual_seed(1)
        query = torch.randn(1, 2, 1, 8)
        key = torch.randn(1, 2, 6, 8)
        mass = causal_attention_mass(query, key, query_offset=0)
        # A single query at position 0 can only see key 0.
        self.assertAlmostEqual(float(mass[0, 0]), 1.0, places=6)
        self.assertTrue(torch.allclose(mass[0, 1:], torch.zeros(5), atol=1e-7))

    def test_query_fraction_uses_last_queries(self):
        torch.manual_seed(2)
        query = torch.randn(1, 3, 8, 4)
        key = torch.randn(1, 1, 8, 4)
        full = causal_attention_mass(query, key, query_fraction=0.25)
        last_two = causal_attention_mass(
            query[:, :, -2:, :], key, query_fraction=1.0, query_offset=6
        )
        self.assertTrue(torch.allclose(full, last_two, atol=1e-6))

    def test_matches_manual_reference_with_gqa(self):
        torch.manual_seed(3)
        query = torch.randn(2, 6, 9, 4)
        key = torch.randn(2, 2, 9, 4)
        expected = _manual_attention_mass(query, key, query_fraction=1.0)
        actual = causal_attention_mass(query, key, query_fraction=1.0)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_matches_manual_reference_with_cache_offset(self):
        torch.manual_seed(4)
        query = torch.randn(1, 2, 3, 4)
        key = torch.randn(1, 2, 10, 4)
        expected = _manual_attention_mass(
            query, key, query_fraction=1.0, query_offset=5
        )
        actual = causal_attention_mass(
            query, key, query_fraction=1.0, query_offset=5
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_streaming_matches_dense(self):
        torch.manual_seed(5)
        query = torch.randn(2, 4, 37, 8)
        key = torch.randn(2, 2, 37, 8)
        dense = causal_attention_mass(query, key, query_fraction=0.6)
        for chunk in (1, 3, 8, 64):
            streamed = causal_attention_mass_streaming(
                query, key, query_fraction=0.6, query_chunk=chunk
            )
            self.assertTrue(
                torch.allclose(dense, streamed, atol=1e-5),
                msg=f"chunk={chunk} max diff={(dense - streamed).abs().max():.2e}",
            )

    def test_does_not_mutate_inputs(self):
        torch.manual_seed(6)
        query = torch.randn(1, 2, 5, 4)
        key = torch.randn(1, 1, 5, 4)
        query_before = query.clone()
        key_before = key.clone()
        causal_attention_mass(query, key)
        causal_attention_mass_streaming(query, key, query_chunk=2)
        self.assertTrue(torch.equal(query, query_before))
        self.assertTrue(torch.equal(key, key_before))

    def test_invalid_shapes(self):
        with self.assertRaises(ValueError):
            causal_attention_mass(torch.randn(2, 5), torch.randn(2, 5))
        with self.assertRaises(ValueError):
            causal_attention_mass(torch.randn(1, 3, 4, 8), torch.randn(2, 3, 4, 8))
        with self.assertRaises(ValueError):
            causal_attention_mass(torch.randn(1, 3, 4, 8), torch.randn(1, 3, 4, 16))
        with self.assertRaises(ValueError):
            causal_attention_mass(torch.randn(1, 3, 4, 8), torch.randn(1, 2, 4, 8))


class AttentionMassFromWeightsTest(unittest.TestCase):
    def test_matches_manual_average(self):
        torch.manual_seed(7)
        logits = torch.randn(2, 3, 6, 6)
        allowed = causal_mask(6, 6)
        logits = logits.masked_fill(~allowed, float("-inf"))
        weights = torch.softmax(logits, dim=-1)

        mass = attention_mass_from_weights(weights, query_fraction=0.5)
        expected = weights[:, :, -3:, :].mean(dim=(1, 2))
        self.assertTrue(torch.allclose(mass, expected, atol=1e-6))
        self.assertTrue(torch.allclose(mass.sum(-1), torch.ones(2), atol=1e-6))

    def test_invalid_rank(self):
        with self.assertRaises(ValueError):
            attention_mass_from_weights(torch.randn(2, 3))


class KeyNormScoresTest(unittest.TestCase):
    def test_matches_manual_norm(self):
        torch.manual_seed(8)
        key = torch.randn(2, 3, 5, 4)
        scores = key_norm_scores(key)
        expected = key.float().pow(2).sum(-1).sqrt().mean(dim=1)
        self.assertTrue(torch.allclose(scores, expected, atol=1e-6))
        self.assertTrue((scores >= 0).all())

    def test_orthogonal_transform_preserves_norm(self):
        # RoPE is an orthogonal rotation, so the key-norm proxy is RoPE-invariant.
        torch.manual_seed(9)
        key = torch.randn(1, 2, 5, 4)
        matrix, _ = torch.linalg.qr(torch.randn(4, 4))
        rotated = torch.matmul(key, matrix)
        self.assertTrue(
            torch.allclose(key_norm_scores(key), key_norm_scores(rotated), atol=1e-5)
        )

    def test_from_projection(self):
        torch.manual_seed(10)
        projection = torch.randn(2, 5, 6 * 4)
        scores = key_norm_scores_from_projection(projection, 6, 4)
        key = projection.reshape(2, 5, 6, 4).transpose(1, 2)
        self.assertTrue(torch.allclose(scores, key_norm_scores(key), atol=1e-6))

    def test_from_projection_invalid_size(self):
        with self.assertRaises(ValueError):
            key_norm_scores_from_projection(torch.randn(1, 5, 10), 3, 4)

    def test_repeat_kv_matches_repeat_interleave(self):
        torch.manual_seed(11)
        key = torch.randn(2, 3, 4, 5)
        repeated = repeat_kv(key, 4)
        self.assertTrue(torch.equal(repeated, key.repeat_interleave(4, dim=1)))
        self.assertIs(repeat_kv(key, 1), key)


if __name__ == "__main__":
    unittest.main()
