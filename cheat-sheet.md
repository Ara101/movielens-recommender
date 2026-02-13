# MovieLens Recommender Cheat Sheet

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Train the SVD Model
```bash
python src/train.py
```
This will:
- Download MovieLens 100K dataset (if not already downloaded)
- Train an SVD model with default parameters
- Evaluate and display RMSE on test set
- Save the trained model to `data/svd_model.pkl`
- Save RMSE metric to `results/metrics.txt`

### 2. Generate Recommendations
```bash
python src/recommend.py
```
This will:
- Load the trained model
- Generate top-10 movie recommendations for a sample user (user 196)
- Display movie IDs and predicted ratings

### 3. Evaluate Model Performance
```bash
python src/evaluate.py
```
This will:
- Load the trained model
- Calculate NDCG@10 and Precision@10 metrics
- Append metrics to `results/metrics.txt`

## Project Structure

```
├── README.md
├── requirements.txt 
├── data/  
│   └── .gitkeep  
├── notebooks/ 
│   └── eda.ipynb         # Quick exploratory data analysis
├── src/                  # Core code
│   ├── train.py          # Load data, train SVD, evaluate RMSE
│   ├── recommend.py      # Generate top-N recommendations
│   ├── evaluate.py       # Compute NDCG@10, Precision@10
│   └── utils.py          # NDCG calculation function
├── results/              # Outputs
│   └── metrics.txt       # RMSE, NDCG@10, Precision@10
└── cheat-sheet.md
```

## Usage Examples

### Custom Recommendations
```python
from src.recommend import recommend_for_user

# Get recommendations for a specific user
recommendations = recommend_for_user(user_id='42', n=10)
```

### Custom Model Training
```python
from src.train import load_movielens_data, train_svd_model

data = load_movielens_data()
model, trainset, testset = train_svd_model(
    data, 
    n_factors=150,  # Increase latent factors
    n_epochs=30,    # More training epochs
    lr_all=0.01,    # Higher learning rate
    reg_all=0.05    # Stronger regularization
)
```

## Metrics Explained

- **RMSE** (Root Mean Square Error): Measures prediction accuracy. Lower is better.
- **NDCG@10** (Normalized Discounted Cumulative Gain): Measures ranking quality in top-10. Higher is better (0-1 range).
- **Precision@10**: Fraction of recommended items that are relevant. Higher is better (0-1 range).

## Expected Performance

On MovieLens 100K with default parameters:
- RMSE: ~0.93
- NDCG@10: ~0.25
- Precision@10: ~0.15-0.20

## Notes

- First run will download the MovieLens 100K dataset (~5MB)
- Training takes ~30 seconds on a typical laptop
- Model is saved in `data/svd_model.pkl` for reuse
- All metrics are saved in `results/metrics.txt`
