"""
SVD (Funk SVD) for MovieLens Rating Prediction & Ranking
From-scratch PyTorch implementation of matrix factorisation, the same
algorithm behind scikit-surprise's SVD and the Netflix Prize winner.

Explores
- Matrix factorisation fundamentals (user/item latent factors)
- SGD-based optimisation for collaborative filtering
- Bias terms (user bias, item bias, global mean)
- How SVD produces a ranking via predicted ratings
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class SVDRatingPredictor(nn.Module):
    """
    Funk SVD (Matrix Factorisation) for rating prediction.

    Prediction formula:
        r̂(u, i) = μ + b_u + b_i + p_u · q_i

    Where:
    - μ   = global mean rating
    - b_u = user bias   (does this user rate higher/lower than average?)
    - b_i = item bias   (is this movie rated higher/lower than average?)
    - p_u = user latent factor vector  [n_factors]
    - q_i = item latent factor vector  [n_factors]

    Designed for MovieLens 100k (943 users, 1,682 items, 100k ratings).
    """

    def __init__(
        self,
        num_users: int = 943,
        num_items: int = 1682,
        n_factors: int = 20,         # Latent factor dimension (matches surprise default)
        global_mean: float = 3.53,   # MovieLens 100k global mean ≈ 3.53
    ):
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.n_factors = n_factors

        # Global mean (fixed, set from training data)
        self.register_buffer("global_mean", torch.tensor(global_mean))

        # Bias terms
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        # Latent factor embeddings
        self.user_factors = nn.Embedding(num_users, n_factors)
        self.item_factors = nn.Embedding(num_items, n_factors)

        # Initialise with small random values (like surprise default)
        nn.init.normal_(self.user_bias.weight, std=0.01)
        nn.init.normal_(self.item_bias.weight, std=0.01)
        nn.init.normal_(self.user_factors.weight, std=0.05)
        nn.init.normal_(self.item_factors.weight, std=0.05)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict ratings for (user, item) pairs.

        Args:
            user_ids: [batch] user indices (0-based)
            item_ids: [batch] item indices (0-based)

        Returns:
            ratings: [batch] predicted ratings clamped to [1, 5]
        """
        # Bias terms
        bu = self.user_bias(user_ids).squeeze(-1)     # [batch]
        bi = self.item_bias(item_ids).squeeze(-1)      # [batch]

        # Latent factor dot product
        pu = self.user_factors(user_ids)               # [batch, n_factors]
        qi = self.item_factors(item_ids)               # [batch, n_factors]
        dot = (pu * qi).sum(dim=-1)                    # [batch]

        # r̂ = μ + b_u + b_i + p_u · q_i
        rating = self.global_mean + bu + bi + dot

        return rating.clamp(1.0, 5.0)

    def predict_all_items(self, user_id: int) -> torch.Tensor:
        """
        Score every item for a single user (for top-N ranking).

        Args:
            user_id: scalar user index

        Returns:
            ratings: [num_items] predicted ratings for all items
        """
        user_ids = torch.full((self.num_items,), user_id, dtype=torch.long,
                              device=self.user_factors.weight.device)
        item_ids = torch.arange(self.num_items, dtype=torch.long,
                                device=self.item_factors.weight.device)
        return self.forward(user_ids, item_ids)


class EnsembleSVDPredictor(nn.Module):
    """
    Ensemble of multiple SVD models for improved ranking and uncertainty.

    - Average predictions reduce variance → more stable top-N lists
    - Std across models estimates prediction uncertainty
    - Works well when individual models are trained with different seeds
    """

    def __init__(self, num_models: int = 5, **svd_kwargs):
        super().__init__()

        self.models = nn.ModuleList([
            SVDRatingPredictor(**svd_kwargs) for _ in range(num_models)
        ])

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mean_pred: [batch] ensemble-averaged predicted ratings
            std_pred:  [batch] prediction uncertainty (std across models)
        """
        preds = torch.stack(
            [m(user_ids, item_ids) for m in self.models],
            dim=1,
        )  # [batch, num_models]

        return preds.mean(dim=1), preds.std(dim=1)


def main():
    """
    Demo: Initialise models, show architecture, and run a forward pass.
    """
    print("=" * 70)
    print("SVD RATING PREDICTOR FOR MOVIELENS RANKING")
    print("=" * 70)

    num_users = 943
    num_items = 1682

    # ------------------------------------------------------------------ #
    # 1. Single SVD model
    # ------------------------------------------------------------------ #
    print("\n[1] SINGLE SVD RATING PREDICTOR")
    print("-" * 70)
    svd = SVDRatingPredictor(
        num_users=num_users,
        num_items=num_items,
        n_factors=20,
    )
    print(svd)

    num_params = sum(p.numel() for p in svd.parameters())
    print(f"\nTotal parameters: {num_params:,}")
    print(f"Trainable: {sum(p.numel() for p in svd.parameters() if p.requires_grad):,}")

    # ------------------------------------------------------------------ #
    # 2. Ensemble model
    # ------------------------------------------------------------------ #
    print("\n\n[2] ENSEMBLE SVD (5 models)")
    print("-" * 70)
    ensemble = EnsembleSVDPredictor(
        num_models=5,
        num_users=num_users,
        num_items=num_items,
        n_factors=20,
    )
    print(f"Ensemble with {len(ensemble.models)} models")
    print(f"Total params: {sum(p.numel() for p in ensemble.parameters()):,}")

    # ------------------------------------------------------------------ #
    # 3. Demo forward pass
    # ------------------------------------------------------------------ #
    print("\n\n[3] DEMO FORWARD PASS")
    print("-" * 70)

    batch_users = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    batch_items = torch.tensor([10, 20, 30, 40], dtype=torch.long)

    print(f"Batch: {len(batch_users)} (user, item) pairs")

    svd.eval()
    with torch.no_grad():
        pred_ratings = svd(batch_users, batch_items)

    print(f"Predicted ratings: {pred_ratings.numpy()}")
    print(f"(Clamped to [1, 5] — MovieLens rating scale)")

    # ------------------------------------------------------------------ #
    # 4. Ensemble forward pass
    # ------------------------------------------------------------------ #
    print("\n\n[4] ENSEMBLE FORWARD PASS")
    print("-" * 70)

    ensemble.eval()
    with torch.no_grad():
        mean_pred, std_pred = ensemble(batch_users, batch_items)

    print(f"Mean ratings:   {mean_pred.numpy()}")
    print(f"Uncertainties:  {std_pred.numpy()}")

    # ------------------------------------------------------------------ #
    # 5. Top-10 ranking demo
    # ------------------------------------------------------------------ #
    print("\n\n[5] TOP-10 RANKING DEMO (user 0)")
    print("-" * 70)

    svd.eval()
    with torch.no_grad():
        all_ratings = svd.predict_all_items(user_id=0)

    top10_vals, top10_idx = torch.topk(all_ratings, k=10)
    print(f"{'Rank':<6}{'Item ID':<10}{'Pred Rating':<12}")
    for rank, (idx, val) in enumerate(zip(top10_idx, top10_vals), 1):
        print(f"{rank:<6}{idx.item():<10}{val.item():.2f}")

    # ------------------------------------------------------------------ #
    # 6. SVD key concepts
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("INTERVIEW TALKING POINTS — SVD")
    print("=" * 70)

    print("""
1. SVD PREDICTION FORMULA
   r̂(u, i) = μ + b_u + b_i + p_u · q_i
   - μ     = global mean rating (3.53 for MovieLens 100k)
   - b_u   = user bias (does this user rate high/low?)
   - b_i   = item bias (is this movie generally liked?)
   - p_u·q_i = dot product of latent factors (taste match)

2. WHY SVD IS THE RIGHT BASELINE
   - MovieLens 100k is small (100k ratings) → SVD trains in seconds
   - RMSE ~0.93 matches published baselines
   - 20 latent factors = only ~52k parameters
   - scikit-surprise's SVD uses the same algorithm under the hood

3. HOW SVD RANKS ITEMS
   - For user u, compute r̂(u, i) for all unrated items i
   - Sort by predicted rating descending → top-N recommendation list
   - Evaluate with NDCG@10 + Precision@10

4. SVD LIMITATIONS
   - Linear model: can't capture non-linear taste patterns
   - No side information: ignores user age, item genre, timestamps
   - No higher-order signals: only direct user-item interactions

5. PARAMETER COUNT (MovieLens 100k)
   - SVD (20 factors): ~52k params, trains in seconds on CPU
    """)


if __name__ == "__main__":
    main()
