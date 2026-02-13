"""
Evaluation module for the MovieLens SVD recommender.

Computes ranking (NDCG@k, Precision@k) and rating-prediction (RMSE, MAE)
metrics using the from-scratch PyTorch SVD model.

Run:
    python src/evaluate.py
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
from compute_ndcg import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    get_user_recommendations,
    evaluate_all_users,
)


# ------------------------------------------------------------------ #
# Rating-prediction metrics
# ------------------------------------------------------------------ #

def compute_rmse(model, test_triples):
    """Root Mean Squared Error on test (user, item, rating) triples."""
    u = torch.tensor([x[0] for x in test_triples], dtype=torch.long)
    i = torch.tensor([x[1] for x in test_triples], dtype=torch.long)
    r = torch.tensor([x[2] for x in test_triples], dtype=torch.float)
    with torch.no_grad():
        pred = model(u, i)
    return float(((pred - r) ** 2).mean().sqrt())


def compute_mae(model, test_triples):
    """Mean Absolute Error on test (user, item, rating) triples."""
    u = torch.tensor([x[0] for x in test_triples], dtype=torch.long)
    i = torch.tensor([x[1] for x in test_triples], dtype=torch.long)
    r = torch.tensor([x[2] for x in test_triples], dtype=torch.float)
    with torch.no_grad():
        pred = model(u, i)
    return float((pred - r).abs().mean())


# ------------------------------------------------------------------ #
# Save / print helpers
# ------------------------------------------------------------------ #

def save_metrics(results, filepath=None, k=10):
    """Save evaluation metrics to a text file."""
    if filepath is None:
        filepath = os.path.join(_ROOT_DIR, "results", "metrics.txt")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(f"Evaluation Metrics (k={k})\n")
        f.write("=" * 45 + "\n")
        for key, val in results.items():
            if not key.startswith("_"):
                if isinstance(val, float):
                    f.write(f"{key:<22} {val:.4f}\n")
                else:
                    f.write(f"{key:<22} {val}\n")
    print(f"Metrics saved to {filepath}")


def print_results(results, k=10):
    """Pretty-print evaluation results to the console."""
    print(f"\n{'Metric':<22}{'Value'}")
    print("-" * 35)
    for key, val in results.items():
        if not key.startswith("_"):
            if isinstance(val, float):
                print(f"{key:<22}{val:.4f}")
            else:
                print(f"{key:<22}{val}")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def _load_model():
    """Load the trained SVD model + data splits."""
    ratings, num_users, num_items = load_ratings()
    train, test = train_test_split_ratings(ratings)
    global_mean = np.mean([r for _, _, r in train])

    model = SVDRatingPredictor(
        num_users=num_users, num_items=num_items,
        n_factors=20, global_mean=global_mean,
    )
    model_path = os.path.join(_ROOT_DIR, "models", "svd_rating_predictor.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run: python src/train.py"
        )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model, train, test, num_users, num_items


if __name__ == "__main__":
    print("=" * 60)
    print("MODEL EVALUATION — SVD on MovieLens 100k")
    print("=" * 60)

    model, train, test, num_users, num_items = _load_model()

    k = 10
    # Rating-prediction metrics
    rmse = compute_rmse(model, test)
    mae = compute_mae(model, test)

    # Ranking metrics
    ranking = evaluate_all_users(model, train, test, num_items, k=k)

    results = {
        "RMSE": rmse,
        "MAE": mae,
        f"NDCG@{k}": ranking["ndcg@k"],
        f"Precision@{k}": ranking["precision@k"],
        f"Recall@{k}": ranking["recall@k"],
        f"HitRate@{k}": ranking["hit_rate@k"],
        "users_evaluated": ranking["num_users_evaluated"],
    }

    print_results(results, k=k)
    save_metrics(results, k=k)