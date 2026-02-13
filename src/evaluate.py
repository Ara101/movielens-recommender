import numpy as np
import pickle
import os
from collections import defaultdict


def ndcg_at_k(relevant_items, recommended_items, k):
    """
    Compute Normalized Discounted Cumulative Gain at k.
    
    Args:
        relevant_items: Set or list of relevant (highly-rated) items for user
        recommended_items: List of recommended item IDs (in order)
        k: Cutoff for NDCG@k
    
    Returns:
        NDCG@k score (0 to 1)
    """
    # DCG: sum of relevance / log2(rank + 1)
    dcg = 0.0
    for rank, item in enumerate(recommended_items[:k], 1):
        if item in relevant_items:
            dcg += 1.0 / np.log2(rank + 1)
    
    # IDCG: ideal DCG (all relevant items ranked first)
    idcg = 0.0
    for rank in range(1, min(len(relevant_items), k) + 1):
        idcg += 1.0 / np.log2(rank + 1)
    
    # Avoid division by zero
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def precision_at_k(relevant_items, recommended_items, k):
    """
    Compute Precision at k.
    
    Args:
        relevant_items: Set or list of relevant (highly-rated) items for user
        recommended_items: List of recommended item IDs (in order)
        k: Cutoff for Precision@k
    
    Returns:
        Precision@k score (0 to 1)
    """
    hits = sum(1 for item in recommended_items[:k] if item in relevant_items)
    return hits / k


def evaluate_ranking_metrics(model, trainset, testset, k=10):
    """
    Evaluate ranking metrics (NDCG@k, Precision@k) on test set.
    
    Args:
        model: Trained SVD model
        trainset: Training set (to exclude rated items)
        testset: Test set with ground truth ratings
        k: Cutoff for metrics
    
    Returns:
        Dictionary with average metrics
    """
    # Build user-to-items mapping for trainset (rated items)
    user_rated_items = defaultdict(set)
    for uid, iid, rating in trainset.all_ratings():
        user_rated_items[uid].add(iid)
    
    # Build user-to-relevant-items mapping for testset (high ratings)
    user_relevant_items = defaultdict(set)
    for uid, iid, rating in testset:
        if rating >= 4.0:  # Consider rating >= 4 as relevant
            user_relevant_items[uid].add(iid)
    
    ndcg_scores = []
    precision_scores = []
    
    # For each user in test set, compute metrics
    all_users = set(uid for uid, _, _ in testset)
    for uid in all_users:
        if uid not in user_relevant_items:
            continue  # Skip users with no relevant items
        
        # Get all items not rated in training
        all_items = set(trainset.all_items())
        unrated = all_items - user_rated_items[uid]
        
        # Get predictions for unrated items
        preds = [(iid, model.predict(uid, iid).est) for iid in unrated]
        preds.sort(key=lambda x: x[1], reverse=True)
        recommended = [iid for iid, _ in preds[:k]]
        
        # Compute metrics
        relevant = user_relevant_items[uid]
        ndcg = ndcg_at_k(relevant, recommended, k)
        prec = precision_at_k(relevant, recommended, k)
        
        ndcg_scores.append(ndcg)
        precision_scores.append(prec)
    
    results = {
        "ndcg@k": np.mean(ndcg_scores) if ndcg_scores else 0.0,
        "precision@k": np.mean(precision_scores) if precision_scores else 0.0,
        "num_users_evaluated": len(ndcg_scores),
    }
    
    return results


def save_metrics(results, filepath="../results/metrics.txt", k=10):
    """Save evaluation metrics to file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(f"Evaluation Metrics (k={k})\n")
        f.write("=" * 40 + "\n")
        f.write(f"NDCG@{k}: {results['ndcg@k']:.4f}\n")
        f.write(f"Precision@{k}: {results['precision@k']:.4f}\n")
        f.write(f"Users evaluated: {results['num_users_evaluated']}\n")
    print(f"Metrics saved to {filepath}")


if __name__ == "__main__":
    # Load model and data
    with open("../models/svd_model.pkl", 'rb') as f:
        model = pickle.load(f)
    with open("../models/trainset.pkl", 'rb') as f:
        trainset = pickle.load(f)
    
    # Load testset (or recompute)
    from train import train_and_save
    _, _, testset = train_and_save()
    
    # Evaluate
    k = 10
    results = evaluate_ranking_metrics(model, trainset, testset, k=k)
    
    print(f"\nEvaluation Results (k={k})")
    print(f"NDCG@{k}: {results['ndcg@k']:.4f}")
    print(f"Precision@{k}: {results['precision@k']:.4f}")
    print(f"Users evaluated: {results['num_users_evaluated']}")
    
    save_metrics(results, k=k)