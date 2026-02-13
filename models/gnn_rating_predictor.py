"""
Graph Neural Network (GNN) for MovieLens Rating Prediction & Ranking
Production-ready architecture for collaborative filtering on a user-item
bipartite graph, predicting ratings and ranking items for recommendation.

For interview: demonstrates understanding of:
- Graph neural networks for collaborative filtering
- Bipartite user-item graph construction
- Rating prediction for top-N ranking
- Ensemble methods for improved recommendations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.data import Data
import numpy as np
from typing import Tuple


class RatingPredictorGNN(nn.Module):
    """
    GCN-based collaborative filtering on a user-item bipartite graph.

    Architecture:
    - Input: Bipartite graph (users + items as nodes, ratings as edges)
    - Learnable user & item embeddings (initial node features)
    - 3 GCN layers propagate information across the graph
    - Rating decoder: dot-product + MLP maps (user_emb, item_emb) -> rating
    - Output: Predicted rating in [1, 5]

    Designed for MovieLens 100k (943 users, 1,682 items, 100k ratings).
    """

    def __init__(
        self,
        num_users: int = 943,
        num_items: int = 1682,
        embed_dim: int = 64,        # Latent embedding dimension
        hidden_dim: int = 64,       # GCN hidden dimension
        num_layers: int = 3,
        dropout: float = 0.3,
        mlp_hidden: int = 32,
    ):
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Learnable embeddings for users and items
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)

        # GCN layers: message passing on the bipartite graph
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(SAGEConv(embed_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.conv_layers.append(SAGEConv(hidden_dim, hidden_dim))

        # Rating prediction head: concatenated user+item embeddings -> rating
        self.rating_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def encode(self, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Run GCN message passing on the full bipartite graph to produce
        refined node embeddings for all users and items.

        Args:
            edge_index: [2, num_edges] edge indices of the user-item graph.
                        User node IDs are 0..num_users-1, item node IDs are
                        num_users..num_users+num_items-1.

        Returns:
            x: [num_users + num_items, hidden_dim] refined node embeddings
        """
        # Concatenate user and item embeddings as initial node features
        x = torch.cat([self.user_embedding.weight,
                        self.item_embedding.weight], dim=0)

        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def decode(
        self,
        z: torch.Tensor,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict ratings for given (user, item) pairs using refined embeddings.

        Args:
            z: [num_nodes, hidden_dim] node embeddings from encode()
            user_ids: [batch] user indices (0-based)
            item_ids: [batch] item indices (0-based, offset by num_users
                      in the graph but passed here as raw item IDs)

        Returns:
            ratings: [batch] predicted ratings clamped to [1, 5]
        """
        user_emb = z[user_ids]
        item_emb = z[item_ids + self.num_users]  # offset into item nodes

        # Concatenate user and item embeddings, then predict rating
        pair_emb = torch.cat([user_emb, item_emb], dim=-1)
        rating = self.rating_head(pair_emb).squeeze(-1)

        # Clamp to valid MovieLens rating range
        return rating.clamp(1.0, 5.0)

    def forward(
        self,
        edge_index: torch.Tensor,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        End-to-end forward: encode graph, then decode specific (user, item) pairs.

        Args:
            edge_index: [2, num_edges] bipartite graph edges
            user_ids: [batch] user indices to predict for
            item_ids: [batch] item indices to predict for

        Returns:
            ratings: [batch] predicted ratings in [1, 5]
        """
        z = self.encode(edge_index)
        return self.decode(z, user_ids, item_ids)


class EnsembleRatingPredictor(nn.Module):
    """
    Ensemble of multiple GNN rating predictors for improved ranking.

    Why ensemble for recommendations?
    - Reduces variance of predicted ratings → more stable rankings
    - Uncertainty estimation: high std → model is unsure about a recommendation
    - Use case: Rank top-100 items, surface top-10 with lowest uncertainty

    Matches the SVD baseline approach but with graph-based learning.
    """

    def __init__(self, num_models: int = 5, **gnn_kwargs):
        super().__init__()

        self.models = nn.ModuleList([
            RatingPredictorGNN(**gnn_kwargs) for _ in range(num_models)
        ])

    def forward(
        self,
        edge_index: torch.Tensor,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mean_pred: [batch] ensemble-averaged predicted ratings
            std_pred:  [batch] prediction uncertainty (std across models)
        """
        preds = torch.stack(
            [m(edge_index, user_ids, item_ids) for m in self.models],
            dim=1,
        )  # [batch, num_models]

        return preds.mean(dim=1), preds.std(dim=1)


class MultiTaskRatingHead(nn.Module):
    """
    Multi-task prediction head for recommendation.

    Tasks:
    1. Rating prediction (primary): What rating will the user give? (1-5)
    2. Engagement prediction (auxiliary): Will the user click/watch? (0-1)

    Why multi-task?
    - Sparse explicit ratings benefit from auxiliary implicit signals
    - Shared representations improve generalisation on MovieLens 100k
    - Matches industry practice (Netflix, YouTube combine rating + engagement)
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 32):
        super().__init__()

        # Rating head: predicts explicit rating [1, 5]
        self.rating_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

        # Engagement head: predicts implicit click/watch probability [0, 1]
        self.engagement_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, pair_embedding: torch.Tensor):
        """
        Args:
            pair_embedding: [batch, input_dim] concatenated user+item embeddings

        Returns:
            rating: predicted rating (clamped to [1, 5])
            engagement: predicted engagement probability [0, 1]
        """
        rating = self.rating_head(pair_embedding).squeeze(-1).clamp(1.0, 5.0)
        engagement = self.engagement_head(pair_embedding).squeeze(-1)
        return rating, engagement


def build_movielens_graph(
    ratings: list[tuple[int, int, float]],
    num_users: int = 943,
    num_items: int = 1682,
) -> Data:
    """
    Build a PyTorch Geometric bipartite graph from MovieLens ratings.

    Nodes:
    - 0 .. num_users-1            → user nodes
    - num_users .. num_users+num_items-1 → item nodes

    Edges:
    - Each (user, item, rating) becomes two directed edges
      (user→item and item→user) for undirected message passing.

    Args:
        ratings: list of (user_id, item_id, rating) tuples (0-indexed)
        num_users: total number of users
        num_items: total number of items

    Returns:
        PyG Data object with edge_index, edge_attr (ratings), and metadata.
    """
    src, dst, vals = [], [], []
    for uid, iid, r in ratings:
        u_node = uid
        i_node = iid + num_users
        # Undirected: add both directions
        src += [u_node, i_node]
        dst += [i_node, u_node]
        vals += [r, r]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(vals, dtype=torch.float)

    return Data(
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_users + num_items,
    )


def main():
    """
    Demo: Initialise models and show architecture for MovieLens rating prediction.
    """
    print("=" * 70)
    print("GNN RATING PREDICTOR FOR MOVIELENS RANKING")
    print("=" * 70)

    num_users = 943
    num_items = 1682

    # ------------------------------------------------------------------ #
    # 1. Single GNN model
    # ------------------------------------------------------------------ #
    print("\n[1] SINGLE GNN RATING PREDICTOR")
    print("-" * 70)
    gnn = RatingPredictorGNN(
        num_users=num_users,
        num_items=num_items,
        embed_dim=64,
        hidden_dim=64,
        num_layers=3,
        dropout=0.3,
    )
    print(gnn)

    num_params = sum(p.numel() for p in gnn.parameters())
    print(f"\nTotal parameters: {num_params:,}")
    print(f"Trainable: {sum(p.numel() for p in gnn.parameters() if p.requires_grad):,}")

    # ------------------------------------------------------------------ #
    # 2. Ensemble model
    # ------------------------------------------------------------------ #
    print("\n\n[2] ENSEMBLE GNN (5 models)")
    print("-" * 70)
    ensemble = EnsembleRatingPredictor(
        num_models=5,
        num_users=num_users,
        num_items=num_items,
        embed_dim=64,
        hidden_dim=64,
        num_layers=3,
        dropout=0.3,
    )
    print(f"Ensemble with {len(ensemble.models)} models")
    print(f"Total params: {sum(p.numel() for p in ensemble.parameters()):,}")

    # ------------------------------------------------------------------ #
    # 3. Multi-task head
    # ------------------------------------------------------------------ #
    print("\n\n[3] MULTI-TASK RATING + ENGAGEMENT HEAD")
    print("-" * 70)
    mtl_head = MultiTaskRatingHead(input_dim=128, hidden_dim=32)
    print(mtl_head)

    # ------------------------------------------------------------------ #
    # 4. Demo forward pass with synthetic MovieLens-like graph
    # ------------------------------------------------------------------ #
    print("\n\n[4] DEMO FORWARD PASS")
    print("-" * 70)

    # Synthetic ratings: (user, item, rating)
    np.random.seed(42)
    num_ratings = 500
    fake_ratings = [
        (np.random.randint(0, num_users),
         np.random.randint(0, num_items),
         float(np.random.randint(1, 6)))
        for _ in range(num_ratings)
    ]
    graph = build_movielens_graph(fake_ratings, num_users, num_items)

    # Predict ratings for a batch of (user, item) pairs
    batch_users = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    batch_items = torch.tensor([10, 20, 30, 40], dtype=torch.long)

    print(f"Graph: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges")
    print(f"Batch: {len(batch_users)} (user, item) pairs")

    gnn.eval()
    with torch.no_grad():
        pred_ratings = gnn(graph.edge_index, batch_users, batch_items)

    print(f"\nPredicted ratings: {pred_ratings.numpy()}")
    print(f"(Clamped to [1, 5] — movieLens rating scale)")

    # ------------------------------------------------------------------ #
    # 5. Ensemble forward pass
    # ------------------------------------------------------------------ #
    print("\n\n[5] ENSEMBLE FORWARD PASS")
    print("-" * 70)

    ensemble.eval()
    with torch.no_grad():
        mean_pred, std_pred = ensemble(graph.edge_index, batch_users, batch_items)

    print(f"Mean ratings:   {mean_pred.numpy()}")
    print(f"Uncertainties:  {std_pred.numpy()}")

    # ------------------------------------------------------------------ #
    # 6. Ranking demo: top-10 for a user
    # ------------------------------------------------------------------ #
    print("\n\n[6] TOP-10 RANKING DEMO (user 0)")
    print("-" * 70)

    user_id = 0
    all_items = torch.arange(num_items, dtype=torch.long)
    user_ids = torch.full_like(all_items, user_id)

    gnn.eval()
    with torch.no_grad():
        all_ratings = gnn(graph.edge_index, user_ids, all_items)

    top10_vals, top10_idx = torch.topk(all_ratings, k=10)
    print(f"{'Rank':<6}{'Item ID':<10}{'Pred Rating':<12}")
    for rank, (idx, val) in enumerate(zip(top10_idx, top10_vals), 1):
        print(f"{rank:<6}{idx.item():<10}{val.item():.2f}")

    # ------------------------------------------------------------------ #
    # Interview talking points
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("INTERVIEW TALKING POINTS")
    print("=" * 70)

    print("""
1. MODEL ARCHITECTURE
   - Bipartite graph: users + items as nodes, ratings as edges
   - SAGEConv layers propagate collaborative signals across the graph
   - Rating decoder: concat(user_emb, item_emb) -> MLP -> rating [1, 5]
   - Same embedding dim (64) as SVD n_factors for fair comparison

2. WHY GNN OVER SVD?
   - SVD is a strong linear baseline (RMSE ~0.93 on MovieLens 100k)
   - GNN captures higher-order interactions (friend-of-friend patterns)
   - GNN can incorporate side info (user age, item genre) as node features
   - Trade-off: GNN is more powerful but harder to train & tune

3. RANKING PIPELINE (mirrors SVD pipeline in src/)
   - Train: Learn user & item embeddings via GCN message passing
   - Predict: Score all unrated items for a user
   - Rank: Sort by predicted rating, return top-N
   - Evaluate: NDCG@10 + Precision@10 (same metrics as SVD baseline)

4. ENSEMBLE STRATEGY
   - 5 GNN models with different random seeds
   - Average predictions for stable ranking
   - Uncertainty (std): flag low-confidence recommendations
   - Reduces variance compared to a single model

5. METRICS (consistent with project README)
   - NDCG@10: Ranking quality (position-aware)
   - Precision@10: Fraction of top-10 that are relevant (rating >= 4)
   - RMSE: Rating prediction accuracy (secondary)
   - Relevance threshold: rating >= 4.0 (industry standard)
    """)


if __name__ == "__main__":
    main()
