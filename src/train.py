"""Train SVD model on MovieLens data and evaluate RMSE."""
import os
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
import pickle


def load_movielens_data():
    """Load MovieLens 100K dataset."""
    # Load the movielens-100k dataset (download if needed)
    data = Dataset.load_builtin('ml-100k')
    return data


def train_svd_model(data, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02):
    """
    Train SVD model on the dataset.
    
    Args:
        data: Surprise dataset object
        n_factors: Number of latent factors
        n_epochs: Number of training epochs
        lr_all: Learning rate
        reg_all: Regularization term
    
    Returns:
        Trained SVD model and test set
    """
    # Split data into train and test sets
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    # Initialize and train SVD model
    model = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all, random_state=42)
    model.fit(trainset)
    
    return model, trainset, testset


def evaluate_rmse(model, testset):
    """
    Evaluate model using RMSE.
    
    Args:
        model: Trained SVD model
        testset: Test dataset
    
    Returns:
        RMSE score
    """
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=True)
    return rmse


def save_model(model, filepath='data/svd_model.pkl'):
    """Save trained model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")


def load_model(filepath='data/svd_model.pkl'):
    """Load trained model from disk."""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model


if __name__ == '__main__':
    print("Loading MovieLens 100K dataset...")
    data = load_movielens_data()
    
    print("\nTraining SVD model...")
    model, trainset, testset = train_svd_model(data)
    
    print("\nEvaluating RMSE...")
    rmse = evaluate_rmse(model, testset)
    
    print("\nSaving model...")
    save_model(model)
    
    # Save RMSE to results
    os.makedirs('results', exist_ok=True)
    with open('results/metrics.txt', 'w') as f:
        f.write(f"RMSE: {rmse:.4f}\n")
    
    print(f"\nTraining complete! RMSE: {rmse:.4f}")
