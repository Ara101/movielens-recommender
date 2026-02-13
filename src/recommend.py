import pickle
import os


def load_model_and_trainset(model_path="../models/svd_model.pkl", trainset_path="../models/trainset.pkl"):
    """Load the trained SVD model and trainset from disk."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(trainset_path, 'rb') as f:
        trainset = pickle.load(f)
    return model, trainset


def get_top_n_recommendations(model, trainset, user_id, n=10):
    """
    Get top-n unrated item recommendations for a user.
    
    Args:
        model: Trained SVD model
        trainset: Trainset with user rating history
        user_id: User ID (from trainset)
        n: Number of recommendations
    
    Returns:
        List of (item_id, predicted_rating) tuples, sorted by rating descending
    """
    all_items = set(trainset.all_items())
    # Get items already rated by user
    rated_items = {j for (j, _) in trainset.ur[user_id]}
    # Get unrated items
    unrated = all_items - rated_items
    
    # Predict rating for each unrated item
    preds = [(item_id, model.predict(user_id, item_id).est) for item_id in unrated]
    # Sort by predicted rating, descending
    preds.sort(key=lambda x: x[1], reverse=True)
    return preds[:n]


if __name__ == "__main__":
    # Example usage
    model, trainset = load_model_and_trainset()
    
    # Get recommendations for user 0 (in trainset)
    user_id = 0
    recommendations = get_top_n_recommendations(model, trainset, user_id, n=10)
    
    print(f"\nTop 10 recommendations for user {user_id}:")
    for rank, (item_id, pred_rating) in enumerate(recommendations, 1):
        print(f"{rank}. Item {item_id}: predicted rating {pred_rating:.2f}")