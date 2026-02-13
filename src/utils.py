"""Utility functions for the MovieLens recommender system."""

import os
import sys
import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, os.path.join(_ROOT_DIR, "models"))

from svd_rating_predictor import SVDRatingPredictor
from train import load_ratings, train_test_split_ratings

DATA_DIR = os.path.join(_ROOT_DIR, "data", "ml-100k", "ml-100k")


def ensure_data_dir(path=None):
    """Ensure data directory exists."""
    if path is None:
        path = os.path.join(_ROOT_DIR, "data")
    os.makedirs(path, exist_ok=True)


def load_movielens_data():
    """Load MovieLens 100k ratings directly from u.data."""
    return load_ratings()


def load_saved_model(model_path=None):
    """
    Load a saved PyTorch SVD model from disk.

    Returns:
        model: SVDRatingPredictor (eval mode)
    """
    if model_path is None:
        model_path = os.path.join(_ROOT_DIR, "models", "svd_rating_predictor.pt")

    ratings, num_users, num_items = load_ratings()
    train, _ = train_test_split_ratings(ratings)
    global_mean = np.mean([r for _, _, r in train])

    model = SVDRatingPredictor(
        num_users=num_users,
        num_items=num_items,
        n_factors=20,
        global_mean=global_mean,
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


def get_user_rating_history(user_id, ratings=None):
    """
    Get the rating history for a user.

    Args:
        user_id: 0-indexed user ID
        ratings: list of (user, item, rating) tuples (loaded if None)

    Returns:
        List of (item_id, rating) tuples
    """
    if ratings is None:
        ratings, _, _ = load_ratings()
    return [(i, r) for u, i, r in ratings if u == user_id]


def get_model_info(model):
    """Get basic info about the trained SVD model."""
    return {
        "num_users": model.num_users,
        "num_items": model.num_items,
        "n_factors": model.n_factors,
        "global_mean": float(model.global_mean),
        "total_params": sum(p.numel() for p in model.parameters()),
    }
