"""Utility functions for evaluation metrics."""
import numpy as np


def ndcg_at_k(relevance_scores, k=10):
    """
    Calculate Normalized Discounted Cumulative Gain at K.
    
    Args:
        relevance_scores: List of relevance scores (1 for relevant, 0 for not relevant)
                         ordered by predicted ranking
        k: Number of top items to consider (default: 10)
    
    Returns:
        NDCG@K score (float between 0 and 1)
    """
    relevance_scores = np.array(relevance_scores)[:k]
    
    if len(relevance_scores) == 0:
        return 0.0
    
    # Calculate DCG (Discounted Cumulative Gain)
    dcg = relevance_scores[0]
    for i in range(1, len(relevance_scores)):
        dcg += relevance_scores[i] / np.log2(i + 1)
    
    # Calculate IDCG (Ideal DCG) - sort relevance scores in descending order
    ideal_relevance = np.sort(relevance_scores)[::-1]
    idcg = ideal_relevance[0]
    for i in range(1, len(ideal_relevance)):
        idcg += ideal_relevance[i] / np.log2(i + 1)
    
    # Avoid division by zero
    if idcg == 0:
        return 0.0
    
    return dcg / idcg
