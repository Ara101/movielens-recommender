import pickle
import os
from surprise import Dataset, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy


def train_and_save(model_path="../models/svd_model.pkl", trainset_path="../models/trainset.pkl"):
    """Train SVD model on MovieLens 100k and save both model and trainset."""
    # Load built-in MovieLens 100k
    data = Dataset.load_built('ml-100k')
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    # Train SVD with 20 latent factors
    model = SVD(n_factors=20, random_state=42)
    model.fit(trainset)
    
    # Compute RMSE on test set
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)
    
    # Save model and trainset for later use
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(trainset_path, 'wb') as f:
        pickle.dump(trainset, f)
    
    print(f"RMSE: {rmse:.4f}")
    print(f"Model saved to {model_path}")
    print(f"Trainset saved to {trainset_path}")
    
    return model, trainset, testset


if __name__ == "__main__":
    model, trainset, testset = train_and_save()