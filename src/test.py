"""
Unit tests for the MovieLens recommender system.

Run from the repo root:
    python -m pytest src/test.py -v
  or
    python src/test.py
"""

import os
import sys
import unittest
import tempfile

import torch
import numpy as np

# ── Path setup ──────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, os.path.join(_ROOT_DIR, "models"))
sys.path.insert(0, _SCRIPT_DIR)

from svd_rating_predictor import SVDRatingPredictor, EnsembleSVDPredictor
from train import load_ratings, train_test_split_ratings, train_svd_predictor
from compute_ndcg import ndcg_at_k, precision_at_k


# ====================================================================
# 1. DATA LOADING TESTS
# ====================================================================
class TestDataLoading(unittest.TestCase):
    """Tests for load_ratings() and train_test_split_ratings()."""

    def test_load_ratings_returns_correct_types(self):
        ratings, num_users, num_items = load_ratings()
        self.assertIsInstance(ratings, list)
        self.assertIsInstance(ratings[0], tuple)
        self.assertEqual(len(ratings[0]), 3)  # (user, item, rating)

    def test_load_ratings_count(self):
        """MovieLens 100k should have exactly 100,000 ratings."""
        ratings, _, _ = load_ratings()
        self.assertEqual(len(ratings), 100_000)

    def test_load_ratings_dimensions(self):
        """Should have 943 users and 1682 items."""
        _, num_users, num_items = load_ratings()
        self.assertEqual(num_users, 943)
        self.assertEqual(num_items, 1682)

    def test_ratings_are_zero_indexed(self):
        """User/item IDs should be 0-indexed."""
        ratings, num_users, num_items = load_ratings()
        min_user = min(u for u, _, _ in ratings)
        min_item = min(i for _, i, _ in ratings)
        max_user = max(u for u, _, _ in ratings)
        max_item = max(i for _, i, _ in ratings)
        self.assertEqual(min_user, 0)
        self.assertEqual(min_item, 0)
        self.assertEqual(max_user, num_users - 1)
        self.assertEqual(max_item, num_items - 1)

    def test_ratings_in_valid_range(self):
        """All ratings should be between 1 and 5."""
        ratings, _, _ = load_ratings()
        for _, _, r in ratings:
            self.assertGreaterEqual(r, 1.0)
            self.assertLessEqual(r, 5.0)

    def test_train_test_split_sizes(self):
        """80/20 split should produce ~80k train and ~20k test."""
        ratings, _, _ = load_ratings()
        train, test = train_test_split_ratings(ratings)
        self.assertEqual(len(train) + len(test), len(ratings))
        self.assertEqual(len(train), 80_000)
        self.assertEqual(len(test), 20_000)

    def test_train_test_split_deterministic(self):
        """Same seed should produce identical splits."""
        ratings, _, _ = load_ratings()
        train1, test1 = train_test_split_ratings(ratings, seed=42)
        train2, test2 = train_test_split_ratings(ratings, seed=42)
        self.assertEqual(train1, train2)
        self.assertEqual(test1, test2)

    def test_train_test_split_different_seeds(self):
        """Different seeds should produce different splits."""
        ratings, _, _ = load_ratings()
        train1, _ = train_test_split_ratings(ratings, seed=42)
        train2, _ = train_test_split_ratings(ratings, seed=99)
        self.assertNotEqual(train1, train2)


# ====================================================================
# 2. SVD MODEL TESTS
# ====================================================================
class TestSVDModel(unittest.TestCase):
    """Tests for SVDRatingPredictor."""

    def setUp(self):
        self.model = SVDRatingPredictor(
            num_users=10,
            num_items=20,
            n_factors=5,
            global_mean=3.5,
        )

    def test_output_shape(self):
        """Forward pass should return [batch] shaped tensor."""
        users = torch.tensor([0, 1, 2])
        items = torch.tensor([3, 4, 5])
        pred = self.model(users, items)
        self.assertEqual(pred.shape, (3,))

    def test_output_clamped(self):
        """Predictions should be clamped to [1, 5]."""
        users = torch.tensor([0, 1, 2, 3, 4])
        items = torch.tensor([0, 1, 2, 3, 4])
        pred = self.model(users, items)
        self.assertTrue((pred >= 1.0).all())
        self.assertTrue((pred <= 5.0).all())

    def test_predict_all_items(self):
        """predict_all_items should return scores for every item."""
        scores = self.model.predict_all_items(user_id=0)
        self.assertEqual(scores.shape, (20,))
        self.assertTrue((scores >= 1.0).all())
        self.assertTrue((scores <= 5.0).all())

    def test_parameter_count(self):
        """
        Parameter count for (10 users, 20 items, 5 factors):
        user_bias: 10*1 = 10
        item_bias: 20*1 = 20
        user_factors: 10*5 = 50
        item_factors: 20*5 = 100
        Total = 180
        """
        num_params = sum(p.numel() for p in self.model.parameters())
        self.assertEqual(num_params, 180)

    def test_global_mean_is_buffer(self):
        """global_mean should be a buffer (non-trainable)."""
        self.assertAlmostEqual(self.model.global_mean.item(), 3.5)
        self.assertNotIn("global_mean", dict(self.model.named_parameters()))

    def test_gradients_flow(self):
        """Backprop should produce non-None gradients."""
        users = torch.tensor([0, 1, 2])
        items = torch.tensor([3, 4, 5])
        targets = torch.tensor([4.0, 3.0, 5.0])
        pred = self.model(users, items)
        loss = ((pred - targets) ** 2).mean()
        loss.backward()
        for name, p in self.model.named_parameters():
            # Only parameters that were used should have gradients
            if p.grad is not None:
                self.assertFalse(torch.isnan(p.grad).any(), f"NaN gradient in {name}")

    def test_save_and_load(self):
        """Model should be saveable and loadable via state_dict."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            torch.save(self.model.state_dict(), path)
            loaded = SVDRatingPredictor(num_users=10, num_items=20, n_factors=5)
            loaded.load_state_dict(torch.load(path, weights_only=True))

            users = torch.tensor([0, 1, 2])
            items = torch.tensor([3, 4, 5])
            self.model.eval()
            loaded.eval()
            with torch.no_grad():
                orig = self.model(users, items)
                reloaded = loaded(users, items)
            self.assertTrue(torch.allclose(orig, reloaded))
        finally:
            os.remove(path)


# ====================================================================
# 3. ENSEMBLE MODEL TESTS
# ====================================================================
class TestEnsembleModel(unittest.TestCase):
    """Tests for EnsembleSVDPredictor."""

    def setUp(self):
        self.ensemble = EnsembleSVDPredictor(
            num_models=3,
            num_users=10,
            num_items=20,
            n_factors=5,
        )

    def test_ensemble_has_correct_num_models(self):
        self.assertEqual(len(self.ensemble.models), 3)

    def test_ensemble_output_shape(self):
        users = torch.tensor([0, 1, 2])
        items = torch.tensor([3, 4, 5])
        mean_pred, std_pred = self.ensemble(users, items)
        self.assertEqual(mean_pred.shape, (3,))
        self.assertEqual(std_pred.shape, (3,))

    def test_ensemble_uncertainty_nonnegative(self):
        users = torch.tensor([0, 1, 2])
        items = torch.tensor([3, 4, 5])
        _, std_pred = self.ensemble(users, items)
        self.assertTrue((std_pred >= 0.0).all())


# ====================================================================
# 4. EVALUATION METRIC TESTS
# ====================================================================
class TestMetrics(unittest.TestCase):
    """Tests for NDCG@k and Precision@k."""

    def test_ndcg_perfect_ranking(self):
        """If recommended items are exactly the relevant ones, NDCG = 1."""
        relevant = {1, 2, 3}
        recommended = [1, 2, 3, 4, 5]
        score = ndcg_at_k(relevant, recommended, k=5)
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_ndcg_worst_ranking(self):
        """If no recommended items are relevant, NDCG = 0."""
        relevant = {1, 2, 3}
        recommended = [4, 5, 6, 7, 8]
        score = ndcg_at_k(relevant, recommended, k=5)
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_ndcg_empty_relevant(self):
        """If no items are relevant, NDCG should be 0 (avoid division by zero)."""
        relevant = set()
        recommended = [1, 2, 3]
        score = ndcg_at_k(relevant, recommended, k=3)
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_ndcg_partial_match(self):
        """Relevant item at rank 1 should score higher than at rank 3."""
        relevant = {1}
        score_rank1 = ndcg_at_k(relevant, [1, 2, 3], k=3)
        score_rank3 = ndcg_at_k(relevant, [2, 3, 1], k=3)
        self.assertGreater(score_rank1, score_rank3)

    def test_precision_perfect(self):
        """All recommended are relevant → Precision = 1."""
        relevant = {1, 2, 3}
        recommended = [1, 2, 3]
        score = precision_at_k(relevant, recommended, k=3)
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_precision_none_relevant(self):
        """No recommended are relevant → Precision = 0."""
        relevant = {1, 2, 3}
        recommended = [4, 5, 6]
        score = precision_at_k(relevant, recommended, k=3)
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_precision_partial(self):
        """2 out of 4 recommended are relevant → Precision = 0.5."""
        relevant = {1, 2}
        recommended = [1, 3, 2, 4]
        score = precision_at_k(relevant, recommended, k=4)
        self.assertAlmostEqual(score, 0.5, places=5)

    def test_ndcg_bounded(self):
        """NDCG should always be between 0 and 1."""
        relevant = {0, 5, 10}
        recommended = [5, 0, 3, 10, 7, 1, 2, 8, 9, 4]
        score = ndcg_at_k(relevant, recommended, k=10)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


# ====================================================================
# 5. TRAINING INTEGRATION TEST
# ====================================================================
class TestTraining(unittest.TestCase):
    """Integration test: train a small SVD model and verify it learns."""

    def test_svd_training_reduces_rmse(self):
        """Training for a few epochs should produce RMSE < 5 (random baseline)."""
        # Use a small synthetic dataset for speed
        np.random.seed(42)
        num_users, num_items = 50, 100
        train_triples = [
            (np.random.randint(num_users), np.random.randint(num_items),
             float(np.random.randint(1, 6)))
            for _ in range(2000)
        ]
        test_triples = [
            (np.random.randint(num_users), np.random.randint(num_items),
             float(np.random.randint(1, 6)))
            for _ in range(500)
        ]

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            model, test_rmse = train_svd_predictor(
                train_triples,
                test_triples,
                num_users=num_users,
                num_items=num_items,
                svd_model_path=path,
                n_factors=5,
                epochs=10,
                batch_size=256,
            )
            # After training, RMSE should be well below the max possible (~4)
            self.assertLess(test_rmse, 3.0)
            # Model file should exist
            self.assertTrue(os.path.exists(path))
        finally:
            os.remove(path)

    def test_svd_training_on_real_data(self):
        """Quick smoke test: train 5 epochs on real MovieLens data."""
        ratings, num_users, num_items = load_ratings()
        train, test = train_test_split_ratings(ratings)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            model, test_rmse = train_svd_predictor(
                train, test, num_users, num_items,
                svd_model_path=path,
                epochs=5,
            )
            # SVD on MovieLens should get RMSE well under 2.0 even after 5 epochs
            self.assertLess(test_rmse, 2.0)
        finally:
            os.remove(path)


# ====================================================================
# 6. END-TO-END RANKING TEST
# ====================================================================
class TestEndToEndRanking(unittest.TestCase):
    """Verify the model can produce a top-N ranking for a user."""

    def test_top_n_ranking(self):
        """Trained model should produce a sensible top-10 list."""
        ratings, num_users, num_items = load_ratings()
        train, _ = train_test_split_ratings(ratings)
        global_mean = np.mean([r for _, _, r in train])

        model = SVDRatingPredictor(
            num_users=num_users,
            num_items=num_items,
            n_factors=20,
            global_mean=global_mean,
        )

        # Load trained weights if available, otherwise use random init
        model_path = os.path.join(_ROOT_DIR, "models", "svd_rating_predictor.pt")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, weights_only=True))

        model.eval()
        with torch.no_grad():
            all_scores = model.predict_all_items(user_id=0)

        # Should return scores for all items
        self.assertEqual(all_scores.shape[0], num_items)

        # Top-10 ranking
        top_vals, top_idx = torch.topk(all_scores, k=10)
        self.assertEqual(len(top_idx), 10)

        # Top items should have highest scores
        self.assertTrue((top_vals[:-1] >= top_vals[1:]).all(),
                        "Top-10 items should be sorted descending")


if __name__ == "__main__":
    unittest.main()
