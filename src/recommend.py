"""Generate top-N recommendations for users."""
import pickle
from surprise import Dataset
from collections import defaultdict
import os


def load_model(filepath='data/svd_model.pkl'):
    """Load trained SVD model from disk."""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model


def get_top_n_recommendations(model, user_id, n=10, threshold=3.5):
    """
    Generate top-N movie recommendations for a user.
    
    Args:
        model: Trained SVD model
        user_id: User ID to generate recommendations for
        n: Number of recommendations to generate
        threshold: Minimum rating threshold to consider (default: 3.5)
    
    Returns:
        List of (movie_id, predicted_rating) tuples
    """
    # Load the dataset to get all movie IDs
    data = Dataset.load_builtin('ml-100k')
    trainset = data.build_full_trainset()
    
    # Get all movie IDs
    all_movie_ids = trainset.all_items()
    
    # Get movies already rated by the user
    try:
        user_inner_id = trainset.to_inner_uid(user_id)
        user_ratings = trainset.ur[user_inner_id]
        rated_movies = {item_id for (item_id, _) in user_ratings}
    except:
        rated_movies = set()
    
    # Predict ratings for all unrated movies
    predictions = []
    for movie_inner_id in all_movie_ids:
        if movie_inner_id not in rated_movies:
            movie_id = trainset.to_raw_iid(movie_inner_id)
            pred = model.predict(user_id, movie_id)
            if pred.est >= threshold:
                predictions.append((movie_id, pred.est))
    
    # Sort by predicted rating and return top N
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:n]


def recommend_for_user(user_id, model_path='data/svd_model.pkl', n=10):
    """
    Generate and display top-N recommendations for a specific user.
    
    Args:
        user_id: User ID to generate recommendations for
        model_path: Path to saved model
        n: Number of recommendations to generate
    """
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run train.py first to train the model.")
        return None
    
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    print(f"\nGenerating top-{n} recommendations for user {user_id}...")
    recommendations = get_top_n_recommendations(model, user_id, n=n)
    
    print(f"\nTop {n} movie recommendations for user {user_id}:")
    print("-" * 50)
    for i, (movie_id, rating) in enumerate(recommendations, 1):
        print(f"{i}. Movie ID: {movie_id} | Predicted Rating: {rating:.2f}")
    
    return recommendations


if __name__ == '__main__':
    # Example: Generate recommendations for user 196
    user_id = '196'
    recommendations = recommend_for_user(user_id, n=10)
