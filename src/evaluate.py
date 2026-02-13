"""Compute NDCG@10 and Precision@10 for the recommender system."""
import pickle
import os
from surprise import Dataset
from surprise.model_selection import train_test_split
from collections import defaultdict
from src.utils import ndcg_at_k
import numpy as np


def load_model(filepath='data/svd_model.pkl'):
    """Load trained SVD model from disk."""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model


def get_user_top_n_predictions(model, trainset, testset, n=10, threshold=3.5):
    """
    Generate top-N predictions for all users in the test set.
    
    Args:
        model: Trained SVD model
        trainset: Training dataset
        testset: Test dataset
        n: Number of top predictions to generate
        threshold: Rating threshold to consider an item as relevant
    
    Returns:
        Dictionary mapping user_id to list of (movie_id, predicted_rating) tuples
    """
    # Get all items
    all_items = trainset.all_items()
    
    # Get predictions for each user
    user_predictions = defaultdict(list)
    
    # Get all users from test set
    test_users = set([uid for (uid, _, _) in testset])
    
    for user_id in test_users:
        try:
            user_inner_id = trainset.to_inner_uid(user_id)
            # Get items already rated by user
            user_ratings = trainset.ur[user_inner_id]
            rated_items = {item_id for (item_id, _) in user_ratings}
        except:
            rated_items = set()
        
        # Predict ratings for unrated items
        predictions = []
        for item_inner_id in all_items:
            if item_inner_id not in rated_items:
                item_id = trainset.to_raw_iid(item_inner_id)
                pred = model.predict(user_id, item_id)
                predictions.append((item_id, pred.est))
        
        # Sort by predicted rating and get top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        user_predictions[user_id] = predictions[:n]
    
    return user_predictions


def evaluate_ndcg(model, trainset, testset, k=10, rating_threshold=4.0):
    """
    Calculate NDCG@K for the model.
    
    Args:
        model: Trained SVD model
        trainset: Training dataset
        testset: Test dataset
        k: Number of top items to consider
        rating_threshold: Threshold to consider an item relevant
    
    Returns:
        Average NDCG@K score
    """
    # Get ground truth: actual ratings from test set
    ground_truth = defaultdict(dict)
    for uid, iid, rating in testset:
        ground_truth[uid][iid] = rating
    
    # Get top-N predictions for each user
    user_predictions = get_user_top_n_predictions(model, trainset, testset, n=k)
    
    # Calculate NDCG for each user
    ndcg_scores = []
    for user_id, predictions in user_predictions.items():
        if user_id not in ground_truth:
            continue
        
        # Get relevance scores for predicted items
        relevance_scores = []
        for item_id, _ in predictions:
            if item_id in ground_truth[user_id]:
                # Item is relevant if actual rating >= threshold
                actual_rating = ground_truth[user_id][item_id]
                relevance_scores.append(1 if actual_rating >= rating_threshold else 0)
            else:
                # Item not in test set, assume not relevant
                relevance_scores.append(0)
        
        if len(relevance_scores) > 0:
            ndcg = ndcg_at_k(relevance_scores, k=k)
            ndcg_scores.append(ndcg)
    
    # Return average NDCG
    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def evaluate_precision(model, trainset, testset, k=10, rating_threshold=4.0):
    """
    Calculate Precision@K for the model.
    
    Args:
        model: Trained SVD model
        trainset: Training dataset
        testset: Test dataset
        k: Number of top items to consider
        rating_threshold: Threshold to consider an item relevant
    
    Returns:
        Average Precision@K score
    """
    # Get ground truth: actual ratings from test set
    ground_truth = defaultdict(dict)
    for uid, iid, rating in testset:
        ground_truth[uid][iid] = rating
    
    # Get top-N predictions for each user
    user_predictions = get_user_top_n_predictions(model, trainset, testset, n=k)
    
    # Calculate Precision for each user
    precision_scores = []
    for user_id, predictions in user_predictions.items():
        if user_id not in ground_truth:
            continue
        
        # Count relevant items in top-K predictions
        relevant_count = 0
        for item_id, _ in predictions:
            if item_id in ground_truth[user_id]:
                actual_rating = ground_truth[user_id][item_id]
                if actual_rating >= rating_threshold:
                    relevant_count += 1
        
        precision = relevant_count / k if k > 0 else 0.0
        precision_scores.append(precision)
    
    # Return average Precision
    return np.mean(precision_scores) if precision_scores else 0.0


if __name__ == '__main__':
    print("Loading model and data...")
    model_path = 'data/svd_model.pkl'
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run train.py first to train the model.")
        exit(1)
    
    model = load_model(model_path)
    
    # Load data and create train/test split (same as training)
    data = Dataset.load_builtin('ml-100k')
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    print("\nEvaluating NDCG@10...")
    ndcg_10 = evaluate_ndcg(model, trainset, testset, k=10)
    print(f"NDCG@10: {ndcg_10:.4f}")
    
    print("\nEvaluating Precision@10...")
    precision_10 = evaluate_precision(model, trainset, testset, k=10)
    print(f"Precision@10: {precision_10:.4f}")
    
    # Append metrics to results file
    print("\nSaving metrics to results/metrics.txt...")
    os.makedirs('results', exist_ok=True)
    with open('results/metrics.txt', 'a') as f:
        f.write(f"NDCG@10: {ndcg_10:.4f}\n")
        f.write(f"Precision@10: {precision_10:.4f}\n")
    
    print("\nEvaluation complete!")
