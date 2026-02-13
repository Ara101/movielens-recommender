"""
Performance visualisations for the MovieLens SVD recommender.

Generates plots and saves them to results/.

Run from the repo root:
    python src/plot.py
"""

import os
import sys

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — works without a display
import matplotlib.pyplot as plt

# ── Path setup ──────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, os.path.join(_ROOT_DIR, "models"))

from svd_rating_predictor import SVDRatingPredictor
from train import load_ratings, train_test_split_ratings, train_svd_predictor
from evaluate import ndcg_at_k, precision_at_k

RESULTS_DIR = os.path.join(_ROOT_DIR, "results")
DATA_DIR = os.path.join(_ROOT_DIR, "data", "ml-100k", "ml-100k")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})
COLORS = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974", "#64b5cd"]


# ====================================================================
# Helpers
# ====================================================================

def _load_model_and_data():
    """Load trained SVD model, ratings, and train/test split."""
    ratings, num_users, num_items = load_ratings()
    train, test = train_test_split_ratings(ratings)
    global_mean = np.mean([r for _, _, r in train])

    model = SVDRatingPredictor(
        num_users=num_users,
        num_items=num_items,
        n_factors=20,
        global_mean=global_mean,
    )
    model_path = os.path.join(_ROOT_DIR, "models", "svd_rating_predictor.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"Loaded trained model from {model_path}")
    else:
        print("No trained model found — training now …")
        model, _ = train_svd_predictor(train, test, num_users, num_items)

    model.eval()
    return model, ratings, train, test, num_users, num_items


def _load_item_names():
    """Load item ID → movie title mapping from u.item."""
    path = os.path.join(DATA_DIR, "u.item")
    names = {}
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("|")
            iid = int(parts[0]) - 1  # 0-indexed
            names[iid] = parts[1]
    return names


def _load_genres():
    """Load genre list and per-item genre vectors from u.genre / u.item."""
    genre_path = os.path.join(DATA_DIR, "u.genre")
    genres = []
    with open(genre_path, "r") as f:
        for line in f:
            line = line.strip()
            if "|" in line:
                genres.append(line.split("|")[0])

    item_path = os.path.join(DATA_DIR, "u.item")
    item_genres = {}  # iid → list of genre names
    with open(item_path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("|")
            iid = int(parts[0]) - 1
            genre_flags = [int(x) for x in parts[5:]]
            item_genres[iid] = [g for g, flag in zip(genres, genre_flags) if flag]
    return genres, item_genres


def _save(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ====================================================================
# Plot 1: Training curve (retrain for a few epochs to capture loss)
# ====================================================================

def plot_training_curve(train_triples, test_triples, num_users, num_items):
    """Train the model while recording per-epoch RMSE, then plot."""
    print("\n[1/6] Training curve …")

    global_mean = np.mean([r for _, _, r in train_triples])
    model = SVDRatingPredictor(num_users=num_users, num_items=num_items,
                               n_factors=20, global_mean=global_mean)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.02)
    criterion = torch.nn.MSELoss()

    train_u = torch.tensor([u for u, _, _ in train_triples], dtype=torch.long)
    train_i = torch.tensor([i for _, i, _ in train_triples], dtype=torch.long)
    train_r = torch.tensor([r for _, _, r in train_triples], dtype=torch.float)
    test_u = torch.tensor([u for u, _, _ in test_triples], dtype=torch.long)
    test_i = torch.tensor([i for _, i, _ in test_triples], dtype=torch.long)
    test_r = torch.tensor([r for _, _, r in test_triples], dtype=torch.float)

    epochs = 50
    train_rmses, test_rmses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(train_triples))
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(train_triples), 1024):
            idx = perm[start:start + 1024]
            optimizer.zero_grad()
            pred = model(train_u[idx], train_i[idx])
            loss = criterion(pred, train_r[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            train_rmses.append((epoch_loss / n_batches) ** 0.5)
            test_pred = model(test_u, test_i)
            test_rmses.append(criterion(test_pred, test_r).item() ** 0.5)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(1, epochs + 1), train_rmses, label="Train RMSE", color=COLORS[0], lw=2)
    ax.plot(range(1, epochs + 1), test_rmses, label="Test RMSE", color=COLORS[2], lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.set_title("SVD Training Curve — MovieLens 100k")
    ax.legend()
    _save(fig, "training_curve.png")

    return test_rmses[-1]


# ====================================================================
# Plot 2: Predicted vs Actual ratings (scatter)
# ====================================================================

def plot_predicted_vs_actual(model, test_triples):
    """Scatter plot of predicted vs actual ratings on the test set."""
    print("[2/6] Predicted vs Actual …")

    test_u = torch.tensor([u for u, _, _ in test_triples], dtype=torch.long)
    test_i = torch.tensor([i for _, i, _ in test_triples], dtype=torch.long)
    actual = np.array([r for _, _, r in test_triples])

    with torch.no_grad():
        predicted = model(test_u, test_i).numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual, predicted, alpha=0.05, s=8, color=COLORS[0], rasterized=True)
    ax.plot([1, 5], [1, 5], "--", color=COLORS[2], lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Rating")
    ax.set_ylabel("Predicted Rating")
    ax.set_title("Predicted vs Actual Ratings")
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)
    ax.set_aspect("equal")
    ax.legend()
    _save(fig, "predicted_vs_actual.png")


# ====================================================================
# Plot 3: Rating distribution (actual vs predicted)
# ====================================================================

def plot_rating_distributions(model, test_triples):
    """Side-by-side histograms of actual and predicted rating distributions."""
    print("[3/6] Rating distributions …")

    test_u = torch.tensor([u for u, _, _ in test_triples], dtype=torch.long)
    test_i = torch.tensor([i for _, i, _ in test_triples], dtype=torch.long)
    actual = [r for _, _, r in test_triples]

    with torch.no_grad():
        predicted = model(test_u, test_i).numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    bins = np.arange(0.75, 5.5, 0.5)
    ax1.hist(actual, bins=bins, color=COLORS[0], edgecolor="white", alpha=0.85)
    ax1.set_title("Actual Ratings")
    ax1.set_xlabel("Rating")
    ax1.set_ylabel("Count")

    ax2.hist(predicted, bins=50, color=COLORS[1], edgecolor="white", alpha=0.85)
    ax2.set_title("Predicted Ratings")
    ax2.set_xlabel("Rating")

    fig.suptitle("Rating Distributions — Test Set", fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, "rating_distributions.png")


# ====================================================================
# Plot 4: NDCG@k and Precision@k for varying k
# ====================================================================

def plot_metrics_at_k(model, train_triples, test_triples, num_items):
    """Evaluate NDCG@k and Precision@k for k = 1, 3, 5, 10, 20, 50."""
    print("[4/6] NDCG@k & Precision@k …")

    # Build rated-items and relevant-items maps
    from collections import defaultdict
    user_rated = defaultdict(set)
    for u, i, _ in train_triples:
        user_rated[u].add(i)

    user_relevant = defaultdict(set)
    user_test_items = defaultdict(list)
    for u, i, r in test_triples:
        user_test_items[u].append(i)
        if r >= 4.0:
            user_relevant[u].add(i)

    # Only evaluate users who have relevant test items
    eval_users = [u for u in user_relevant if len(user_relevant[u]) > 0]
    # Sample up to 200 users for speed
    rng = np.random.RandomState(42)
    if len(eval_users) > 200:
        eval_users = rng.choice(eval_users, 200, replace=False).tolist()

    ks = [1, 3, 5, 10, 20, 50]
    ndcg_means = []
    prec_means = []

    for k in ks:
        ndcg_scores = []
        prec_scores = []
        for u in eval_users:
            # Predict ratings for all items not rated in train
            unrated = [i for i in range(num_items) if i not in user_rated[u]]
            u_tensor = torch.tensor([u] * len(unrated), dtype=torch.long)
            i_tensor = torch.tensor(unrated, dtype=torch.long)
            with torch.no_grad():
                scores = model(u_tensor, i_tensor).numpy()
            # Top-k
            top_idx = np.argsort(-scores)[:k]
            recommended = [unrated[j] for j in top_idx]

            ndcg_scores.append(ndcg_at_k(user_relevant[u], recommended, k))
            prec_scores.append(precision_at_k(user_relevant[u], recommended, k))

        ndcg_means.append(np.mean(ndcg_scores))
        prec_means.append(np.mean(prec_scores))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    ax1.plot(ks, ndcg_means, "o-", color=COLORS[0], lw=2, markersize=7)
    ax1.set_xlabel("k")
    ax1.set_ylabel("NDCG@k")
    ax1.set_title("NDCG@k")
    ax1.set_xticks(ks)

    ax2.plot(ks, prec_means, "s-", color=COLORS[1], lw=2, markersize=7)
    ax2.set_xlabel("k")
    ax2.set_ylabel("Precision@k")
    ax2.set_title("Precision@k")
    ax2.set_xticks(ks)

    fig.suptitle("Ranking Metrics at Varying k", fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, "metrics_at_k.png")

    # Save numeric results
    with open(os.path.join(RESULTS_DIR, "metrics.txt"), "w") as f:
        f.write("Ranking Metrics — SVD on MovieLens 100k\n")
        f.write("=" * 45 + "\n")
        for i, k in enumerate(ks):
            f.write(f"k={k:<3}  NDCG@k={ndcg_means[i]:.4f}  Precision@k={prec_means[i]:.4f}\n")
    print(f"  Metrics written → {os.path.join(RESULTS_DIR, 'metrics.txt')}")

    return dict(zip(ks, ndcg_means)), dict(zip(ks, prec_means))


# ====================================================================
# Plot 5: Per-user error distribution
# ====================================================================

def plot_per_user_error(model, test_triples):
    """Histogram of per-user mean absolute error."""
    print("[5/6] Per-user error distribution …")

    from collections import defaultdict
    user_errors = defaultdict(list)

    test_u = torch.tensor([u for u, _, _ in test_triples], dtype=torch.long)
    test_i = torch.tensor([i for _, i, _ in test_triples], dtype=torch.long)

    with torch.no_grad():
        preds = model(test_u, test_i).numpy()

    for (u, _, r), p in zip(test_triples, preds):
        user_errors[u].append(abs(r - p))

    mae_per_user = [np.mean(errs) for errs in user_errors.values()]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(mae_per_user, bins=40, color=COLORS[3], edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(mae_per_user), color=COLORS[2], ls="--", lw=1.5,
               label=f"Mean MAE = {np.mean(mae_per_user):.3f}")
    ax.set_xlabel("Mean Absolute Error per User")
    ax.set_ylabel("Number of Users")
    ax.set_title("Per-User Prediction Error Distribution")
    ax.legend()
    _save(fig, "per_user_error.png")


# ====================================================================
# Plot 6: Top-10 recommendation example for a sample user
# ====================================================================

def plot_top10_example(model, train_triples, num_items):
    """Bar chart showing top-10 predicted ratings for a sample user."""
    print("[6/6] Top-10 recommendation example …")

    item_names = _load_item_names()

    # Pick user 0 (has many ratings)
    user_id = 0
    rated_by_user = {i for u, i, _ in train_triples if u == user_id}

    with torch.no_grad():
        all_scores = model.predict_all_items(user_id).numpy()

    # Mask out already rated items
    for i in rated_by_user:
        all_scores[i] = -1.0

    top10_idx = np.argsort(-all_scores)[:10]
    top10_scores = all_scores[top10_idx]
    top10_names = [item_names.get(i, f"Item {i}")[:35] for i in top10_idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(range(9, -1, -1), top10_scores, color=COLORS[0], edgecolor="white")
    ax.set_yticks(range(9, -1, -1))
    ax.set_yticklabels(top10_names, fontsize=9)
    ax.set_xlabel("Predicted Rating")
    ax.set_title(f"Top-10 Recommendations for User {user_id + 1}")
    ax.set_xlim(0, 5.3)

    # Annotate bars with scores
    for bar, score in zip(bars, top10_scores):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{score:.2f}", va="center", fontsize=9)

    fig.tight_layout()
    _save(fig, "top10_recommendations.png")


# ====================================================================
# Main
# ====================================================================

def main():
    print("=" * 60)
    print("GENERATING PERFORMANCE PLOTS")
    print("=" * 60)

    model, ratings, train, test, num_users, num_items = _load_model_and_data()

    # 1. Training curve (retrains to capture per-epoch metrics)
    final_rmse = plot_training_curve(train, test, num_users, num_items)

    # 2. Predicted vs Actual scatter
    plot_predicted_vs_actual(model, test)

    # 3. Rating distributions
    plot_rating_distributions(model, test)

    # 4. NDCG@k and Precision@k at varying k
    ndcg_dict, prec_dict = plot_metrics_at_k(model, train, test, num_items)

    # 5. Per-user error distribution
    plot_per_user_error(model, test)

    # 6. Top-10 example
    plot_top10_example(model, train, num_items)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Final Test RMSE:   {final_rmse:.4f}")
    print(f"  NDCG@10:           {ndcg_dict.get(10, 'N/A'):.4f}")
    print(f"  Precision@10:      {prec_dict.get(10, 'N/A'):.4f}")
    print(f"\n  All plots saved to: {os.path.abspath(RESULTS_DIR)}")
    print(f"  Files:")
    for f in sorted(os.listdir(RESULTS_DIR)):
        print(f"    - {f}")


if __name__ == "__main__":
    main()
