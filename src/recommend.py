"""
Recommendation engine for the MovieLens SVD recommender.

Loads the trained PyTorch SVD model and generates top-N movie
recommendations for any user, with movie titles and metadata.

Run:
    python src/recommend.py              # default: user 1
    python src/recommend.py --user 42    # specify a user
"""

import os
import sys
import argparse
import numpy as np
import torch
from collections import defaultdict

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, os.path.join(_ROOT_DIR, "models"))

from svd_rating_predictor import SVDRatingPredictor
from train import load_ratings, train_test_split_ratings

DATA_DIR = os.path.join(_ROOT_DIR, "data", "ml-100k", "ml-100k")


# ------------------------------------------------------------------ #
# Data helpers
# ------------------------------------------------------------------ #

def load_item_metadata():
    """
    Load movie titles and genres from u.item / u.genre.

    Returns:
        titles: dict {0-indexed item_id: title_string}
        genres: dict {0-indexed item_id: [genre_name, ...]}
        genre_names: ordered list of all genre names
    """
    # Genre list
    genre_names = []
    with open(os.path.join(DATA_DIR, "u.genre"), "r") as f:
        for line in f:
            line = line.strip()
            if "|" in line:
                genre_names.append(line.split("|")[0])

    # Item metadata
    titles = {}
    genres = {}
    with open(os.path.join(DATA_DIR, "u.item"), "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("|")
            iid = int(parts[0]) - 1  # 0-indexed
            titles[iid] = parts[1]
            genre_flags = [int(x) for x in parts[5:]]
            genres[iid] = [g for g, flag in zip(genre_names, genre_flags) if flag]

    return titles, genres, genre_names


def load_user_info():
    """
    Load user demographics from u.user.

    Returns:
        dict {0-indexed user_id: {"age": int, "gender": str,
                                   "occupation": str, "zipcode": str}}
    """
    users = {}
    with open(os.path.join(DATA_DIR, "u.user"), "r") as f:
        for line in f:
            parts = line.strip().split("|")
            uid = int(parts[0]) - 1
            users[uid] = {
                "age": int(parts[1]),
                "gender": parts[2],
                "occupation": parts[3],
                "zipcode": parts[4],
            }
    return users


# ------------------------------------------------------------------ #
# Recommendation functions
# ------------------------------------------------------------------ #

def load_model():
    """Load trained SVD model and return (model, num_users, num_items, global_mean)."""
    ratings, num_users, num_items = load_ratings()
    train, _ = train_test_split_ratings(ratings)
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
    return model, ratings, train, num_users, num_items


def get_top_n_recommendations(model, user_id, rated_items, num_items, n=10):
    """
    Get top-n unrated item recommendations for a user.

    Args:
        model: trained SVDRatingPredictor
        user_id: 0-indexed user ID
        rated_items: set of item IDs the user already rated (in training)
        num_items: total number of items
        n: number of recommendations

    Returns:
        list of (item_id, predicted_rating) tuples, sorted descending
    """
    unrated = [i for i in range(num_items) if i not in rated_items]
    u_tensor = torch.tensor([user_id] * len(unrated), dtype=torch.long)
    i_tensor = torch.tensor(unrated, dtype=torch.long)

    with torch.no_grad():
        preds = model(u_tensor, i_tensor).numpy()

    top_idx = np.argsort(-preds)[:n]
    return [(unrated[j], float(preds[j])) for j in top_idx]


def get_user_history(user_id, ratings):
    """Return list of (item_id, rating) for items the user rated."""
    return [(i, r) for u, i, r in ratings if u == user_id]


# ------------------------------------------------------------------ #
# Pretty-print
# ------------------------------------------------------------------ #

def print_user_profile(user_id, ratings, titles, users_info):
    """Print a user's profile: demographics + top-rated movies."""
    info = users_info.get(user_id, {})
    history = get_user_history(user_id, ratings)
    history.sort(key=lambda x: x[1], reverse=True)

    display_id = user_id + 1  # 1-indexed for display
    print(f"\n  User {display_id}  |  Age: {info.get('age','?')}  |  "
          f"Gender: {info.get('gender','?')}  |  "
          f"Occupation: {info.get('occupation','?')}")
    print(f"  Rated {len(history)} movies  |  "
          f"Avg rating: {np.mean([r for _, r in history]):.2f}")

    print(f"\n  Top-5 rated movies:")
    for rank, (iid, r) in enumerate(history[:5], 1):
        title = titles.get(iid, f"Item {iid}")
        print(f"    {rank}. [{r:.0f}/5]  {title}")


def print_recommendations(recs, titles, genres_map):
    """Print a recommendation list with titles and genres."""
    print(f"\n  {'Rank':<6}{'Pred':<7}{'Title':<42}{'Genres'}")
    print("  " + "-" * 80)
    for rank, (iid, score) in enumerate(recs, 1):
        title = titles.get(iid, f"Item {iid}")[:40]
        genre_str = ", ".join(genres_map.get(iid, []))[:30]
        print(f"  {rank:<6}{score:<7.2f}{title:<42}{genre_str}")


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MovieLens SVD Recommender")
    parser.add_argument("--user", type=int, default=1,
                        help="User ID (1-indexed, default: 1)")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of recommendations (default: 10)")
    args = parser.parse_args()

    user_id = args.user - 1  # convert to 0-indexed

    model, ratings, train, num_users, num_items = load_model()
    titles, genres_map, _ = load_item_metadata()
    users_info = load_user_info()

    # Rated items in training
    rated = {i for u, i, _ in train if u == user_id}

    print("=" * 60)
    print(f"  RECOMMENDATIONS FOR USER {args.user}")
    print("=" * 60)

    print_user_profile(user_id, ratings, titles, users_info)

    recs = get_top_n_recommendations(model, user_id, rated, num_items, n=args.n)

    print(f"\n  Top-{args.n} Recommendations:")
    print_recommendations(recs, titles, genres_map)
    print()