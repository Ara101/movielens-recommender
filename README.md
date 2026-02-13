# MovieLens Recommender

A production-ready, SVD-based collaborative filtering recommender system on MovieLens 100k with end-to-end training, inference, and ranking evaluation.

## Performance

- **NDCG@10**: 0.25 (Normalized Discounted Cumulative Gain)
- **Precision@10**: 0.30 (Top-10 recommendation precision)
- **RMSE**: ~0.93 (rating prediction accuracy)

These metrics align with published baselines for SVD on MovieLens 100k.

---

## Getting Started

### Prerequisites

- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- Git (to clone the repo)

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/movielens-recommender.git
cd movielens-recommender
```

### 2. Create the Conda Environment

```bash
# Create a new environment with Python 3.10
conda create -n movielens_env python=3.10 -y

# Activate it
conda activate movielens_env
```

### 3. Install Dependencies

**Option A** — Install from `requirements.txt` (recommended):

```bash
pip install -r requirements.txt
```

**Option B** — Install manually:

```bash
pip install scikit-surprise==1.1.4 pandas==2.3.3 numpy==1.26.4 scipy==1.15.2 jupyter
```

> **Windows note:** If `scikit-surprise` fails to build from pip (requires C++ build tools), install it via conda-forge instead:
> ```bash
> conda install -c conda-forge scikit-surprise pandas numpy scipy jupyter -y
> ```

### 4. Verify the Installation

```bash
python -c "from surprise import SVD, Dataset; print('OK')"
```

You should see `OK` printed — you're all set.

---

## Usage

All scripts live in the `src/` directory. Run them from there so relative paths resolve correctly.

```bash
cd src
```

### Step 1: Train the Model

```bash
python train.py
```

This will:
- Download MovieLens 100k automatically (first run only)
- Train an SVD model with 20 latent factors (80/20 train-test split)
- Print RMSE on the test set
- Save the model and trainset to `models/`

### Step 2: Generate Recommendations

```bash
python recommend.py
```

This will:
- Load the saved model and trainset
- Generate top-10 recommendations for a sample user
- Print item IDs with predicted ratings

To recommend for a different user, edit `user_id` in `recommend.py` or import the function:

```python
from recommend import load_model_and_trainset, get_top_n_recommendations

model, trainset = load_model_and_trainset()
recs = get_top_n_recommendations(model, trainset, user_id=5, n=10)
for rank, (item, score) in enumerate(recs, 1):
    print(f"{rank}. Item {item}: {score:.2f}")
```

### Step 3: Evaluate Ranking Metrics

```bash
python evaluate.py
```

This will:
- Compute NDCG@10 and Precision@10 across all test users
- Save results to `results/metrics.txt`

### Explore the Data (Notebook)

```bash
cd ../notebooks
jupyter notebook eda.ipynb
```

---

## Reproducing the Environment (Optional)

Export the exact conda environment for full reproducibility:

```bash
conda env export > environment.yml
```

Anyone can then recreate it with:

```bash
conda env create -f environment.yml
conda activate movielens_env
```

## Project Structure

```
movielens-recommender/
├── src/
│   ├── train.py           # Train SVD model and save to disk
│   ├── recommend.py       # Load model and generate top-N recommendations
│   ├── evaluate.py        # Evaluate NDCG and Precision metrics
│   └── utils.py          # Shared utility functions
├── models/               # Saved trained model and trainset
├── results/              # Evaluation metrics
├── data/                 # MovieLens data (auto-downloaded)
├── notebooks/            # EDA and analysis
├── requirements.txt      # Python dependencies
├── cheat-sheet.md       # Quick reference guide
└── README.md            # This file
```

## How It Works

### 1. Training (`train.py`)
- Loads MovieLens 100k (943 users, 1,682 movies)
- Trains an SVD model with 20 latent factors
- 80/20 train-test split, random_state=42 for reproducibility
- Saves model and trainset for inference

### 2. Recommendation (`recommend.py`)
- Loads the trained model and trainset
- For a given user, predicts ratings for all unrated items
- Returns top-N recommendations sorted by predicted rating
- Filters out items the user has already rated

### 3. Evaluation (`evaluate.py`)
- Computes **NDCG@k**: Ranking quality with position-weighted relevance
- Computes **Precision@k**: Fraction of top-k recommendations that are relevant
- Relevant items = test set ratings ≥ 4.0
- Reports average metrics across all test users

## Key Design Decisions

| Aspect | Choice | Why |
|--------|--------|-----|
| Algorithm | SVD (scikit-surprise) | Scalable, interpretable, strong baseline |
| Factors | 20 | Classic choice; trade-off between expressiveness and overfitting |
| Relevance threshold | Rating ≥ 4.0 | Industry standard for implicit positive feedback |
| Evaluation | NDCG + Precision | Position-aware metrics for ranking quality |
| Train/test split | 80/20 | Standard for evaluation |

## Files Reference

### `src/evaluate.py`
- `ndcg_at_k(relevant_items, recommended_items, k)`: Computes NDCG@k
- `precision_at_k(relevant_items, recommended_items, k)`: Computes Precision@k
- `evaluate_ranking_metrics(model, trainset, testset, k=10)`: Full evaluation pipeline
- `save_metrics(results, filepath, k=10)`: Persist metrics to disk

### `src/recommend.py`
- `load_model_and_trainset()`: Load saved model and trainset
- `get_top_n_recommendations(model, trainset, user_id, n=10)`: Generate recommendations

### `src/train.py`
- `train_and_save()`: End-to-end training and model persistence

### `src/utils.py`
- Helper functions for data loading and model management

## Reusable Components

The NDCG and Precision functions are reusable across different recommendation systems:
- Different algorithms (MF, neural CF, etc.)
- Different datasets (Books, Music, etc.)
- Different evaluation scenarios

This demonstrates a modular approach to building ranking systems.

## Future Improvements

- Hyperparameter tuning (n_factors, learning rate)
- Cross-validation for robust evaluation
- Implicit feedback variant
- Content-based features
- Ensemble methods

## References

- MovieLens Dataset: https://grouplens.org/datasets/movielens/100k/
- Scikit-Surprise: https://surpriselib.readthedocs.io/
- Funk SVD: "Netflix Prize Postmortem" by Simon Funk

---

**Interview Talking Point**: "I built an SVD recommender on MovieLens 100k as a standing, explicit ranking project. It achieves NDCG@10 of 0.25 and Precision@10 of 0.30, consistent with published baselines. The key wasn't state-of-the-art—it was demonstrating I can implement, evaluate, and document a ranking system end-to-end. I reused my NDCG function from the OXTR project, which shows I think in terms of reusable components."
