"""
showcase.py — End-to-end demo of the MovieLens SVD Recommender
================================================================
Ties together train, evaluate, recommend, and compute_ndcg to produce
a polished report with terminal output + plots saved to results/.

Usage:
    python src/showcase.py                 # full demo (user 1)
    python src/showcase.py --user 42       # specify user
    python src/showcase.py --user 42 --n 5 # top-5 for user 42

What it produces:
    - Console: formatted report with user profile, recommendations,
               rating predictions, and ranking metrics
    - results/showcase_report.png  — multi-panel summary figure
    - results/metrics.txt          — numeric metrics
"""

import os
import sys
import argparse
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, os.path.join(_ROOT_DIR, "models"))
sys.path.insert(0, _SCRIPT_DIR)

from svd_rating_predictor import SVDRatingPredictor
from train import load_ratings, train_test_split_ratings
from compute_ndcg import (
    ndcg_at_k, precision_at_k, recall_at_k, hit_rate_at_k,
    get_user_recommendations, evaluate_all_users,
)
from evaluate import compute_rmse, compute_mae
from recommend import (
    load_item_metadata, load_user_info,
    get_top_n_recommendations, get_user_history,
)

RESULTS_DIR = os.path.join(_ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})
C = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974", "#64b5cd"]

LINE = "=" * 65
THIN = "-" * 65


# ====================================================================
# 1. Load everything
# ====================================================================

def load_everything(model_path=None):
    ratings, num_users, num_items = load_ratings()
    train, test = train_test_split_ratings(ratings)
    global_mean = np.mean([r for _, _, r in train])

    model = SVDRatingPredictor(
        num_users=num_users, num_items=num_items,
        n_factors=20, global_mean=global_mean,
    )
    if model_path is None:
        model_path = os.path.join(_ROOT_DIR, "models", "svd_rating_predictor.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run: python src/train.py"
        )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    titles, genres_map, genre_names = load_item_metadata()
    users_info = load_user_info()

    return (model, ratings, train, test,
            num_users, num_items,
            titles, genres_map, genre_names, users_info)


# ====================================================================
# 2. Console report
# ====================================================================

def print_header():
    print()
    print(LINE)
    print("   MOVIELENS 100K — SVD RECOMMENDER SHOWCASE")
    print(LINE)


def print_model_summary(model, train, test, num_users, num_items):
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model:          Funk SVD (PyTorch, from scratch)")
    print(f"  Latent factors: {model.n_factors}")
    print(f"  Parameters:     {n_params:,}")
    print(f"  Global mean:    {model.global_mean.item():.4f}")
    print(f"  Dataset:        {num_users} users x {num_items} items")
    print(f"  Train/Test:     {len(train):,} / {len(test):,} ratings")


def print_rating_metrics(model, test):
    rmse = compute_rmse(model, test)
    mae = compute_mae(model, test)
    print(f"\n  {THIN}")
    print(f"  RATING PREDICTION (test set)")
    print(f"  {THIN}")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    MAE:  {mae:.4f}")
    return rmse, mae


def print_ranking_metrics(model, train, test, num_items):
    print(f"\n  {THIN}")
    print(f"  RANKING METRICS (relevance = rating >= 4)")
    print(f"  {THIN}")
    print(f"    {'k':<6}{'NDCG@k':<11}{'Prec@k':<11}{'Recall@k':<11}{'HitRate@k'}")
    print(f"    {'-'*50}")

    all_results = {}
    for k in [1, 5, 10, 20]:
        r = evaluate_all_users(model, train, test, num_items, k=k, max_users=300)
        all_results[k] = r
        print(f"    {k:<6}{r['ndcg@k']:<11.4f}{r['precision@k']:<11.4f}"
              f"{r['recall@k']:<11.4f}{r['hit_rate@k']:.4f}")
    return all_results


def print_user_showcase(user_id, n, model, ratings, train,
                        num_items, titles, genres_map, users_info):
    display_id = user_id + 1
    info = users_info.get(user_id, {})
    history = get_user_history(user_id, ratings)
    history.sort(key=lambda x: x[1], reverse=True)

    rated_set = {i for u, i, _ in train if u == user_id}

    print(f"\n  {THIN}")
    print(f"  USER PROFILE — User {display_id}")
    print(f"  {THIN}")
    print(f"    Age: {info.get('age','?')}  |  Gender: {info.get('gender','?')}  |  "
          f"Occupation: {info.get('occupation','?')}")
    print(f"    Total ratings: {len(history)}  |  "
          f"Avg rating: {np.mean([r for _, r in history]):.2f}")

    print(f"\n    Favourite movies (top 5 rated):")
    for rank, (iid, r) in enumerate(history[:5], 1):
        title = titles.get(iid, f"Item {iid}")[:45]
        g = ", ".join(genres_map.get(iid, []))[:25]
        print(f"      {rank}. [{r:.0f}/5]  {title:<46} {g}")

    recs = get_top_n_recommendations(model, user_id, rated_set, num_items, n=n)

    print(f"\n    Top-{n} Recommendations:")
    print(f"    {'Rank':<6}{'Pred':<7}{'Title':<43}{'Genres'}")
    print(f"    {'-'*75}")
    for rank, (iid, score) in enumerate(recs, 1):
        title = titles.get(iid, f"Item {iid}")[:42]
        g = ", ".join(genres_map.get(iid, []))[:28]
        print(f"    {rank:<6}{score:<7.2f}{title:<43}{g}")

    return recs


# ====================================================================
# 3. Showcase figure (4-panel)
# ====================================================================

def generate_showcase_figure(model, train, test, num_items, user_id,
                             recs, titles, genres_map, rmse, ranking_results):
    """Generate a polished 4-panel figure and save to results/."""
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # ------------------------------------------------------------------
    # Panel A: Predicted vs Actual scatter
    # ------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    test_u = torch.tensor([u for u, _, _ in test], dtype=torch.long)
    test_i = torch.tensor([i for _, i, _ in test], dtype=torch.long)
    actual = np.array([r for _, _, r in test])
    with torch.no_grad():
        predicted = model(test_u, test_i).numpy()

    ax_a.scatter(actual, predicted, alpha=0.04, s=6, color=C[0], rasterized=True)
    ax_a.plot([1, 5], [1, 5], "--", color=C[2], lw=1.5, label="Perfect")
    ax_a.set_xlabel("Actual Rating")
    ax_a.set_ylabel("Predicted Rating")
    ax_a.set_title(f"A. Predicted vs Actual  (RMSE={rmse:.3f})")
    ax_a.set_xlim(0.5, 5.5)
    ax_a.set_ylim(0.5, 5.5)
    ax_a.set_aspect("equal")
    ax_a.legend(fontsize=9)

    # ------------------------------------------------------------------
    # Panel B: NDCG@k / Precision@k across k
    # ------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    ks = sorted(ranking_results.keys())
    ndcg_vals = [ranking_results[k]["ndcg@k"] for k in ks]
    prec_vals = [ranking_results[k]["precision@k"] for k in ks]

    ax_b.plot(ks, ndcg_vals, "o-", color=C[0], lw=2, markersize=7, label="NDCG@k")
    ax_b.plot(ks, prec_vals, "s-", color=C[1], lw=2, markersize=7, label="Precision@k")
    ax_b.set_xlabel("k")
    ax_b.set_ylabel("Score")
    ax_b.set_title("B. Ranking Metrics at Varying k")
    ax_b.set_xticks(ks)
    ax_b.legend(fontsize=9)
    ax_b.set_ylim(0, max(max(ndcg_vals), max(prec_vals)) * 1.3)

    # ------------------------------------------------------------------
    # Panel C: Top-N recommendations bar chart
    # ------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    rec_names = [titles.get(iid, f"Item {iid}")[:30] for iid, _ in recs]
    rec_scores = [s for _, s in recs]
    n_recs = len(recs)

    bars = ax_c.barh(range(n_recs - 1, -1, -1), rec_scores,
                     color=C[0], edgecolor="white", height=0.7)
    ax_c.set_yticks(range(n_recs - 1, -1, -1))
    ax_c.set_yticklabels(rec_names, fontsize=8)
    ax_c.set_xlabel("Predicted Rating")
    ax_c.set_title(f"C. Top-{n_recs} Recs for User {user_id + 1}")
    ax_c.set_xlim(0, 5.3)
    for bar, score in zip(bars, rec_scores):
        ax_c.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                  f"{score:.2f}", va="center", fontsize=8)

    # ------------------------------------------------------------------
    # Panel D: Rating distribution (actual vs predicted)
    # ------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    bins_actual = np.arange(0.75, 5.6, 0.5)
    ax_d.hist(actual, bins=bins_actual, color=C[0], alpha=0.6,
              edgecolor="white", label="Actual", density=True)
    ax_d.hist(predicted, bins=50, color=C[1], alpha=0.6,
              edgecolor="white", label="Predicted", density=True)
    ax_d.set_xlabel("Rating")
    ax_d.set_ylabel("Density")
    ax_d.set_title("D. Rating Distributions (Test Set)")
    ax_d.legend(fontsize=9)

    fig.suptitle("MovieLens 100k — SVD Recommender Performance Report",
                 fontsize=14, fontweight="bold", y=0.98)

    path = os.path.join(RESULTS_DIR, "showcase_report.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ====================================================================
# 4. Save metrics
# ====================================================================

def save_report(rmse, mae, ranking_results):
    path = os.path.join(RESULTS_DIR, "metrics.txt")
    with open(path, "w") as f:
        f.write("MovieLens 100k — SVD Recommender Metrics\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"RMSE:            {rmse:.4f}\n")
        f.write(f"MAE:             {mae:.4f}\n\n")
        f.write(f"{'k':<6}{'NDCG@k':<12}{'Prec@k':<12}{'Recall@k':<12}{'HitRate@k'}\n")
        f.write("-" * 50 + "\n")
        for k in sorted(ranking_results):
            r = ranking_results[k]
            f.write(f"{k:<6}{r['ndcg@k']:<12.4f}{r['precision@k']:<12.4f}"
                    f"{r['recall@k']:<12.4f}{r['hit_rate@k']:.4f}\n")
    return path


# ====================================================================
# Main
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Showcase the MovieLens SVD Recommender"
    )
    parser.add_argument("--user", type=int, default=1,
                        help="User ID to showcase (1-indexed, default: 1)")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of recommendations (default: 10)")
    args = parser.parse_args()
    user_id = args.user - 1  # 0-indexed

    # Load
    (model, ratings, train, test,
     num_users, num_items,
     titles, genres_map, genre_names, users_info) = load_everything()

    # ── Console report ────────────────────────────────────────────────
    print_header()
    print_model_summary(model, train, test, num_users, num_items)
    rmse, mae = print_rating_metrics(model, test)
    ranking_results = print_ranking_metrics(model, train, test, num_items)
    recs = print_user_showcase(
        user_id, args.n, model, ratings, train,
        num_items, titles, genres_map, users_info,
    )

    # ── Save files ────────────────────────────────────────────────────
    metrics_path = save_report(rmse, mae, ranking_results)
    fig_path = generate_showcase_figure(
        model, train, test, num_items, user_id,
        recs, titles, genres_map, rmse, ranking_results,
    )

    print(f"\n  {THIN}")
    print(f"  OUTPUT FILES")
    print(f"  {THIN}")
    print(f"    Metrics:  {metrics_path}")
    print(f"    Figure:   {fig_path}")
    print(f"\n{LINE}\n")


if __name__ == "__main__":
    main()
