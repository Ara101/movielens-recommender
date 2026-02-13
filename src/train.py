import pickle
import os
import sys

import torch
import torch.nn as nn
import numpy as np

# Allow imports from the models/ directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.join(_SCRIPT_DIR, '..')
sys.path.insert(0, os.path.join(_ROOT_DIR, 'models'))
from svd_rating_predictor import SVDRatingPredictor

# Optional: scikit-surprise (needs C++ build tools on Windows)
try:
    from surprise import Dataset, SVD
    from surprise.model_selection import train_test_split
    from surprise import accuracy
    HAS_SURPRISE = True
except ImportError:
    HAS_SURPRISE = False


# ------------------------------------------------------------------ #
# Data loading — works with or without scikit-surprise
# ------------------------------------------------------------------ #

DATA_PATH = os.path.join(_ROOT_DIR, 'data', 'ml-100k', 'ml-100k', 'u.data')


def load_ratings(path=DATA_PATH):
    """
    Load MovieLens 100k ratings from u.data (tab-separated).

    Returns:
        ratings: list of (user_id, item_id, rating) tuples, 0-indexed
        num_users: total number of unique users
        num_items: total number of unique items
    """
    ratings = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            uid = int(parts[0]) - 1   # 1-indexed → 0-indexed
            iid = int(parts[1]) - 1
            r = float(parts[2])
            ratings.append((uid, iid, r))

    num_users = max(u for u, _, _ in ratings) + 1
    num_items = max(i for _, i, _ in ratings) + 1
    return ratings, num_users, num_items


def train_test_split_ratings(ratings, test_ratio=0.2, seed=42):
    """Split ratings into train/test sets."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(ratings))
    split = int(len(ratings) * (1 - test_ratio))
    train = [ratings[i] for i in indices[:split]]
    test = [ratings[i] for i in indices[split:]]
    return train, test


def train_and_save(model_path=None, trainset_path=None):
    """Train scikit-surprise SVD model on MovieLens 100k and save.
    Requires scikit-surprise to be installed (optional dependency)."""
    if not HAS_SURPRISE:
        print("[SKIP] scikit-surprise not installed — skipping library SVD baseline.")
        print("       Install via: conda install -c conda-forge scikit-surprise")
        return None, None, None

    if model_path is None:
        model_path = os.path.join(_ROOT_DIR, 'models', 'svd_model.pkl')
    if trainset_path is None:
        trainset_path = os.path.join(_ROOT_DIR, 'models', 'trainset.pkl')

    data = Dataset.load_built('ml-100k')
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    model = SVD(n_factors=20, random_state=42)
    model.fit(trainset)
    
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(trainset_path, 'wb') as f:
        pickle.dump(trainset, f)
    
    print(f"RMSE: {rmse:.4f}")
    print(f"Model saved to {model_path}")
    print(f"Trainset saved to {trainset_path}")
    
    return model, trainset, testset


# ------------------------------------------------------------------ #
# Helpers: convert data for PyTorch models
# ------------------------------------------------------------------ #

def _compute_global_mean(train_triples):
    """Compute the global mean rating from training triples."""
    return np.mean([r for _, _, r in train_triples])


# ------------------------------------------------------------------ #
# SVD Rating Predictor Training (from-scratch PyTorch)
# ------------------------------------------------------------------ #

def train_svd_predictor(
    train_triples,
    test_triples,
    num_users,
    num_items,
    svd_model_path=None,
    n_factors=20,
    lr=0.005,
    reg=0.02,
    epochs=50,
    batch_size=1024,
):
    """
    Train a from-scratch Funk SVD rating predictor and save the model.

    Args:
        train_triples: list of (user_id, item_id, rating) tuples (0-indexed)
        test_triples: list of (user_id, item_id, rating) tuples (0-indexed)
        num_users, num_items: matrix dimensions
        svd_model_path: where to save the trained SVD state dict
        n_factors: number of latent factors (matches surprise SVD)
        lr: learning rate
        reg: L2 regularisation weight
        epochs: training epochs
        batch_size: mini-batch size

    Returns:
        model: trained SVDRatingPredictor
        test_rmse: RMSE on the test set
    """
    if svd_model_path is None:
        svd_model_path = os.path.join(_ROOT_DIR, 'models', 'svd_rating_predictor.pt')

    print("\n" + "=" * 70)
    print("TRAINING SVD RATING PREDICTOR (from scratch)")
    print("=" * 70)

    # ---- Prepare data ------------------------------------------------
    global_mean = _compute_global_mean(train_triples)

    train_users = torch.tensor([u for u, _, _ in train_triples], dtype=torch.long)
    train_items = torch.tensor([i for _, i, _ in train_triples], dtype=torch.long)
    train_ratings = torch.tensor([r for _, _, r in train_triples], dtype=torch.float)

    test_users = torch.tensor([u for u, _, _ in test_triples], dtype=torch.long)
    test_items = torch.tensor([i for _, i, _ in test_triples], dtype=torch.long)
    test_ratings = torch.tensor([r for _, _, r in test_triples], dtype=torch.float)

    # ---- Initialise model --------------------------------------------
    model = SVDRatingPredictor(
        num_users=num_users,
        num_items=num_items,
        n_factors=n_factors,
        global_mean=global_mean,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
    criterion = nn.MSELoss()

    num_train = len(train_triples)
    print(f"Global mean: {global_mean:.4f}")
    print(f"Train ratings: {num_train:,}  |  Test ratings: {len(test_triples):,}")
    print(f"Model params:  {sum(p.numel() for p in model.parameters()):,}")
    print(f"Factors: {n_factors}  |  Epochs: {epochs}  |  LR: {lr}  |  Reg: {reg}")
    print("-" * 70)

    # ---- Training loop -----------------------------------------------
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(num_train)
        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, num_train, batch_size):
            idx = perm[start : start + batch_size]
            batch_u = train_users[idx]
            batch_i = train_items[idx]
            batch_r = train_ratings[idx]

            optimizer.zero_grad()
            pred = model(batch_u, batch_i)
            loss = criterion(pred, batch_r)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        # Evaluate every 10 epochs (and final epoch)
        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                test_pred = model(test_users, test_items)
                test_rmse = criterion(test_pred, test_ratings).item() ** 0.5
            print(f"Epoch {epoch:>3}/{epochs}  |  Train Loss: {avg_loss:.4f}  |  Test RMSE: {test_rmse:.4f}")

    # ---- Final evaluation --------------------------------------------
    model.eval()
    with torch.no_grad():
        test_pred = model(test_users, test_items)
        test_rmse = criterion(test_pred, test_ratings).item() ** 0.5

    print("-" * 70)
    print(f"Final Test RMSE: {test_rmse:.4f}")

    # ---- Save model --------------------------------------------------
    os.makedirs(os.path.dirname(svd_model_path), exist_ok=True)
    torch.save(model.state_dict(), svd_model_path)
    print(f"SVD model saved to {svd_model_path}")

    return model, test_rmse


if __name__ == "__main__":
    # Load data directly from files (no surprise dependency needed)
    ratings, num_users, num_items = load_ratings()
    train_triples, test_triples = train_test_split_ratings(ratings)
    print(f"MovieLens 100k: {num_users} users, {num_items} items, {len(ratings):,} ratings")
    print(f"Train: {len(train_triples):,}  |  Test: {len(test_triples):,}")

    # 1. (Optional) Train scikit-surprise SVD baseline
    train_and_save()

    # 2. Train from-scratch SVD rating predictor (PyTorch)
    svd_predictor, svd_rmse = train_svd_predictor(
        train_triples, test_triples, num_users, num_items
    )

    # ---- Summary -----------------------------------------------------
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"  PyTorch SVD   →  RMSE {svd_rmse:.4f}  →  saved to models/svd_rating_predictor.pt")