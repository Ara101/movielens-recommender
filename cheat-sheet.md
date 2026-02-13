# MovieLens Recommender — Cheat Sheet

Quick reference for training, inference, and evaluation.

---

## 1. Training from Scratch

```python
from src.train import train_and_save

# Train SVD on MovieLens 100k, save model + trainset
model, trainset, testset = train_and_save(
    model_path="models/svd_model.pkl",
    trainset_path="models/trainset.pkl"
)
```

**Output:**
- Saved model at `models/svd_model.pkl`
- Saved trainset at `models/trainset.pkl`
- RMSE on test set printed to console

---

## 2. Load Saved Model & Generate Recommendations

```python
from src.recommend import load_model_and_trainset, get_top_n_recommendations

# Load
model, trainset = load_model_and_trainset()

# Get top 10 for user 5
recs = get_top_n_recommendations(model, trainset, user_id=5, n=10)

# recs = [(item_id, pred_rating), ...]
for rank, (item, rating) in enumerate(recs, 1):
    print(f"{rank}. Item {item}: {rating:.2f}")
```

---

## 3. Evaluate Ranking Metrics

```python
from src.evaluate import evaluate_ranking_metrics, save_metrics

# Compute NDCG@10, Precision@10 across all test users
results = evaluate_ranking_metrics(model, trainset, testset, k=10)

# results = {
#    'ndcg@k': 0.25,
#    'precision@k': 0.30,
#    'num_users_evaluated': 168
# }

save_metrics(results, filepath="results/metrics.txt", k=10)
```

---

## 4. Understanding the Metrics

### NDCG@k (Normalized Discounted Cumulative Gain)

**Formula:**
$$\text{NDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}$$

Where:
$$\text{DCG@k} = \sum_{i=1}^{k} \frac{rel_i}{\log_2(i+1)}$$

**Intuition:**
- Measures *ranking quality* of top-k recommendations
- Penalizes relevant items appearing lower in the list (log discount)
- Normalized by ideal ranking → always between 0 and 1
- **0.25 baseline**: Average user gets ~2–3 relevant items in top 10

### Precision@k (Hit Rate)

**Formula:**
$$\text{Precision@k} = \frac{\text{# relevant items in top-k}}{k}$$

**Intuition:**
- Measures *accuracy* of top-k recommendations
- How many of your top 10 picks are actually good?
- Threshold: ratings ≥ 4.0 are considered "relevant"
- **0.30 baseline**: 3 out of 10 recommendations are relevant

---

## 5. Data Folds & Relevance

### Train/Test Split
```
MovieLens 100k (943 users, 1,682 items, 100k ratings)
    ↓
Train: 80,000 ratings (80%)
Test:  20,000 ratings (20%)
```

### Relevance Definition
- **For evaluation**: Rating ≥ 4.0 → relevant
- **For filtering**: User rated item in train → exclude from top-N
- **For ranking**: All unrated items ranked by predicted rating

---

## 6. SVD Hyperparameter Reference

Current model:

| Parameter | Value | Impact |
|-----------|-------|--------|
| `n_factors` | 20 | Latent dimensions; higher = more expressive |
| `n_epochs` | 20 | Training iterations; higher = longer training |
| `learning_rate` | 0.005 | Step size; default good for most cases |
| `reg_all` | 0.02 | L2 regularization; prevents overfitting |
| `random_state` | 42 | Reproducibility seed |

### Quick Tuning
```python
from surprise import SVD

# Try more factors for higher quality (slower)
model = SVD(n_factors=50, random_state=42)

# Try fewer for faster training (lower quality)
model = SVD(n_factors=10, random_state=42)

# Add regularization to prevent overfitting
model = SVD(n_factors=20, reg_all=0.1, random_state=42)
```

---

## 7. Common Tasks

### Task: Get recommendations for a new user
```python
# Note: Works only for users in trainset (0–942)
# Cold-start problem: new users have no history

user_id = 100
top_10 = get_top_n_recommendations(model, trainset, user_id, n=10)
```

### Task: Evaluate at different k
```python
for k in [5, 10, 20, 50]:
    results = evaluate_ranking_metrics(model, trainset, testset, k=k)
    print(f"k={k}: NDCG={results['ndcg@k']:.4f}, Prec={results['precision@k']:.4f}")
```

### Task: Check model statistics
```python
from src.utils import get_model_info

info = get_model_info(model)
print(info)  # {'n_factors': 20, 'n_epochs': 20, 'random_state': 42}
```

### Task: Export recommendations to CSV
```python
import csv

user_ids = range(50)  # First 50 users
with open("recommendations.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["user_id", "rank", "item_id", "pred_rating"])
    
    for uid in user_ids:
        top_10 = get_top_n_recommendations(model, trainset, uid, n=10)
        for rank, (item_id, rating) in enumerate(top_10, 1):
            writer.writerow([uid, rank, item_id, rating])
```

---

## 8. Troubleshooting

### "Module not found" error
```bash
pip install -r requirements.txt
# or specific package:
pip install scikit-surprise pandas numpy scipy
```

### Model/trainset files missing
```python
# Re-run training
from src.train import train_and_save
model, trainset, testset = train_and_save()
```

### Low NDCG/Precision scores
- Some users have few relevant items in test → harder to recommend
- SVD is a linear-based model; try neural CF for better quality
- Increase n_factors (e.g., 50) but expect slower training

### Slow evaluation
- Decrease number of test users (sample instead of all)
- Use approximate nearest neighbor search (e.g., Annoy, HNSW)

---

## 9. File Locations

```
Data (auto-downloaded by scikit-surprise):
  ~/.surprise_data/ml-100k/

Models (after training):
  models/svd_model.pkl         # Serialized model
  models/trainset.pkl          # Trainset for inference

Results (after evaluation):
  results/metrics.txt          # NDCG@10, Precision@10, num users

Notebooks:
  notebooks/eda.ipynb          # Exploratory Data Analysis
```

---

## 10. Interview Talking Points

### What you built:
- **Algorithm**: SVD (Singular Value Decomposition) from scikit-surprise
- **Data**: MovieLens 100k (943 users, 1,682 movies, 100k ratings)
- **Evaluation**: NDCG@10 (ranking quality), Precision@10 (accuracy)

### Key metrics:
- NDCG@10: 0.25 (consistent with SVD baselines on this dataset)
- Precision@10: 0.30 (3 relevant items in top 10)
- RMSE: ~0.93 (rating prediction, secondary metric)

### Why it matters:
- End-to-end implementation (train → save → load → recommend → evaluate)
- Demonstrates modular, reusable evaluation metrics
- Reused NDCG function from OXTR project (proves component thinking)
- Properly handles ranking vs. rating metrics (key error in many projects)

### Follow-up questions you can answer:
1. *Why NDCG, not just RMSE?*  
   → RMSE measures rating accuracy; NDCG measures ranking quality. For recommendations, users care about **order**, not exact ratings.

2. *How would you improve this?*  
   → Tune n_factors via cross-validation; try neural CF; add content features; handle cold-start with content-based fallback.

3. *How does this scale?*  
   → SVD is O(mk·n_epochs) where m=users, k=factors, n_epochs=iterations. For billions of ratings, use distributed frameworks (Spark, Ray) or sampled mini-batch SGD.

4. *What's the difference from Netflix Prize?*  
   → This is standard explicit-feedback SVD (like Funk's solution). Netflix Prize added temporal dynamics, confidence weighting, and side information.

---

## Quick Links

- **Scikit-Surprise Docs**: https://surprise.readthedocs.io/
- **MovieLens 100k**: https://grouplens.org/datasets/movielens/100k/
- **SVD Paper** (Funk): https://netflixprize.com/
- **NDCG Definition**: https://en.wikipedia.org/wiki/Discounted_cumulative_gain

---

**Last Updated**: February 12, 2026