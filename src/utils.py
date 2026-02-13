"""Utility functions for the MovieLens recommender system."""

import os
import pickle
from surprise import Dataset


def ensure_data_dir(path="../data"):
    """Ensure data directory exists."""
    os.makedirs(path, exist_ok=True)


def load_movielens_data():
    """Load MovieLens 100k dataset."""
    data = Dataset.load_built('ml-100k')
    return data


def load_saved_model(model_path="../models/svd_model.pkl"):
    """Load a saved model from disk."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def load_saved_trainset(trainset_path="../models/trainset.pkl"):
    """Load a saved trainset from disk."""
    with open(trainset_path, 'rb') as f:
        return pickle.load(f)


def get_user_rating_history(trainset, user_id):
    """
    Get the rating history for a user from trainset.
    
    Args:
        trainset: Surprise trainset object
        user_id: User ID
    
    Returns:
        List of (item_id, rating) tuples
    """
    return trainset.ur[user_id]


def get_model_info(model):
    """Get basic info about the trained model."""
    info = {
        "n_factors": model.n_factors,
        "n_epochs": model.n_epochs,
        "random_state": model.random_state,
    }
    return info
