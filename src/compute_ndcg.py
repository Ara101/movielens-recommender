"""
NDCG & Ranking Metric Computation for MovieLens Recommender
============================================================
Computes NDCG@k, Precision@k, Recall@k, and HitRate@k using the
trained PyTorch SVD model on MovieLens 100k.

Can be used standalone or imported by other scripts.

Run:
    python src/compute_ndcg.py
"""

import os
import sys
import numpy as np
import torch
from collections import defaultdict

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, os.path.join(_ROOT_DIR, "models"))

from svd_rating_predictor import SVDRatingPredictor
from train import load_ratings, train_test_split_ratings


# ------------------------------------------------------------------ #
# Core metric functions
# ------------------------------------------------------------------ #

def dcg_at_k(relevances, k):
    """
    Discounted Cumulative Gain @ k.

    DCG@k = sum_{i=1}^{k} rel_i / log2(i + 1)

    Args:
        relevances: list/array of relevance scores in ranked order
        k: cutoff position
    """
    relevances = np.asarray(relevances, dtype=float)[:k]
    if len(relevances) == 0:
        return 0.0
    positions = np.arange(1, len(relevances) + 1)
    return float(np.sum(relevances / np.log2(positions + 1)))


def ndcg_at_k(relevant_items, recommended_items, k):
    """
    Normalized Discounted Cumulative Gain @ k (binary relevance).

    Args:
        relevant_items: set of relevant (ground-truth) item IDs
        recommended_items: list of recommended item IDs **in ranked order**
        k: cutoff position

    Returns:
        NDCG@k in [0, 1]
    """
    # Binary relevance vector for the recommended list
    rel = [1.0 if item in relevant_items else 0.0
           for item in recommended_items[:k]]

    dcg = dcg_at_k(rel, k)

    # Ideal: all relevant items at the top
    n_relevant = min(len(relevant_items), k)
    ideal_rel = [1.0] * n_relevant + [0.0] * (k - n_relevant)
    idcg = dcg_at_k(ideal_rel, k)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def precision_at_k(relevant_items, recommended_items, k):
    """Fraction of top-k recommendations that are relevant."""
    hits = sum(1 for item in recommended_items[:k] if item in relevant_items)
    return hits / k


def recall_at_k(relevant_items, recommended_items, k):
    """Fraction of relevant items that appear in the top-k."""
    if len(relevant_items) == 0:
        return 0.0
    hits = sum(1 for item in recommended_items[:k] if item in relevant_items)
    return hits / len(relevant_items)


def hit_rate_at_k(relevant_items, recommended_items, k):
    """1 if at least one relevant item is in the top-k, else 0."""
    return 1.0 if any(item in relevant_items for item in recommended_items[:k]) else 0.0


# ------------------------------------------------------------------ #
# Per-user ranking helpers
# ------------------------------------------------------------------ #

def get_user_recommendations(model, user_id, rated_items, num_items, k):
    """
    Generate top-k recommendations for a user using the PyTorch SVD model.

    Args:
        model: trained SVDRatingPredictor
        user_id: 0-indexed user ID
        rated_items: set of item IDs already rated in training
        num_items: total number of items
        k: how many to recommend

    Returns:
        recommended: list of k item IDs ordered by predicted rating (desc)
        scores: corresponding predicted ratings
    """
    unrated = [i for i in range(num_items) if i not in rated_items]
    u_tensor = torch.tensor([user_id] * len(unrated), dtype=torch.long)
    i_tensor = torch.tensor(unrated, dtype=torch.long)

    with torch.no_grad():
        pred = model(u_tensor, i_tensor).numpy()

    top_idx = np.argsort(-pred)[:k]
    recommended = [unrated[j] for j in top_idx]
    scores = pred[top_idx]
    return recommended, scores


def evaluate_all_users(model, train_triples, test_triples, num_items,
                       k=10, relevance_threshold=4.0, max_users=None):
    """
    Compute ranking metrics across all users in the test set.

    Args:
        model: trained SVDRatingPredictor
        train_triples: list of (user, item, rating) from training
        test_triples: list of (user, item, rating) from test
        num_items: total item count
        k: cutoff for @k metrics
        relevance_threshold: ratings >= this are considered relevant
        max_users: cap for speed (None = all users)

    Returns:
        dict with mean NDCG@k, Precision@k, Recall@k, HitRate@k,
        num_users_evaluated, and per-user detail lists.
    """
    # Build lookup tables
    user_rated = defaultdict(set)
    for u, i, _ in train_triples:
        user_rated[u].add(i)

    user_relevant = defaultdict(set)
    for u, i, r in test_triples:
        if r >= relevance_threshold:
            user_relevant[u].add(i)

    eval_users = [u for u in user_relevant if len(user_relevant[u]) > 0]
    if max_users and len(eval_users) > max_users:
        rng = np.random.RandomState(42)
        eval_users = rng.choice(eval_users, max_users, replace=False).tolist()

    ndcg_list, prec_list, recall_list, hr_list = [], [], [], []

    model.eval()
    for u in eval_users:
        recommended, _ = get_user_recommendations(
            model, u, user_rated[u], num_items, k
        )
        rel = user_relevant[u]
        ndcg_list.append(ndcg_at_k(rel, recommended, k))
        prec_list.append(precision_at_k(rel, recommended, k))
        recall_list.append(recall_at_k(rel, recommended, k))
        hr_list.append(hit_rate_at_k(rel, recommended, k))

    return {
        "ndcg@k": float(np.mean(ndcg_list)),
        "precision@k": float(np.mean(prec_list)),
        "recall@k": float(np.mean(recall_list)),
        "hit_rate@k": float(np.mean(hr_list)),
        "num_users_evaluated": len(eval_users),
        "_ndcg_per_user": ndcg_list,
        "_prec_per_user": prec_list,
    }


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def main():
    print("=" * 60)
    print("NDCG & RANKING METRIC COMPUTATION")
    print("=" * 60)

    ratings, num_users, num_items = load_ratings()
    train, test = train_test_split_ratings(ratings)
    global_mean = np.mean([r for _, _, r in train])

    model = SVDRatingPredictor(num_users=num_users, num_items=num_items,
                               n_factors=20, global_mean=global_mean)
    model_path = os.path.join(_ROOT_DIR, "models", "svd_rating_predictor.pt")
    if not os.path.exists(model_path):
        print(f"ERROR: No trained model at {model_path}")
        print("       Run: python src/train.py")
        return
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    print(f"Loaded model from {model_path}")

    print(f"\nDataset: {len(ratings):,} ratings, {num_users} users, {num_items} items")
    print(f"Train: {len(train):,} | Test: {len(test):,}")
    print(f"Relevance threshold: rating >= 4.0\n")

    print(f"{'k':<6}{'NDCG@k':<12}{'Prec@k':<12}{'Recall@k':<12}{'HitRate@k':<12}{'Users'}")
    print("-" * 64)

    for k in [1, 3, 5, 10, 20, 50]:
        results = evaluate_all_users(model, train, test, num_items, k=k)
        print(f"{k:<6}{results['ndcg@k']:<12.4f}{results['precision@k']:<12.4f}"
              f"{results['recall@k']:<12.4f}{results['hit_rate@k']:<12.4f}"
              f"{results['num_users_evaluated']}")

    results_path = os.path.join(_ROOT_DIR, "results", "metrics.txt")
    r10 = evaluate_all_users(model, train, test, num_items, k=10)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        f.write("Ranking Metrics — SVD on MovieLens 100k\n")
        f.write("=" * 45 + "\n")
        f.write(f"NDCG@10:      {r10['ndcg@k']:.4f}\n")
        f.write(f"Precision@10: {r10['precision@k']:.4f}\n")
        f.write(f"Recall@10:    {r10['recall@k']:.4f}\n")
        f.write(f"HitRate@10:   {r10['hit_rate@k']:.4f}\n")
        f.write(f"Users eval'd: {r10['num_users_evaluated']}\n")
    print(f"\nMetrics saved to {results_path}")


if __name__ == "__main__":
    main()
