"""
NDCG Computation for Ranking Evaluation
Demonstrates how to evaluate molecule ranking quality using NDCG@k metric
(Used for interview: shows ranking metrics understanding)
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
import os


def compute_qed_score(smiles):
    """
    Compute QED (Quantitative Estimate of Drug-likeness) score.
    Higher = more drug-like. Used as dummy relevance label.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        
        # Simplified QED components (Lipinski-based)
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        
        # Penalty scoring (simplified)
        qed = 1.0
        
        # Molecular weight: optimal ~300-400
        if mw < 200 or mw > 600:
            qed *= 0.5
        
        # LogP: optimal -1 to 5
        if logp < -1 or logp > 5:
            qed *= 0.7
        
        # H-bond donors/acceptors: optimal < 5
        if hbd > 5 or hba > 10:
            qed *= 0.8
        
        return min(qed, 1.0)
    except:
        return 0.0


def dcg_at_k(relevances, k):
    """
    Discounted Cumulative Gain @ k
    DCG = sum(relevance[i] / log2(i+1)) for i in 0..k-1
    """
    relevances = np.array(relevances)[:k]
    positions = np.arange(1, len(relevances) + 1)
    return np.sum(relevances / np.log2(positions + 1))


def ndcg_at_k(scores, labels, k=10):
    """
    Normalized Discounted Cumulative Gain @ k
    
    Args:
        scores: array of predicted scores (to rank items)
        labels: array of true relevance labels
        k: cutoff (evaluate top-k)
    
    Returns:
        NDCG@k score (0-1)
    """
    # Sort by scores (descending)
    sorted_indices = np.argsort(-scores)
    sorted_labels = labels[sorted_indices]
    
    # Compute DCG@k
    dcg = dcg_at_k(sorted_labels, k)
    
    # Compute ideal DCG@k (best possible ranking)
    ideal_sorted_labels = np.sort(-labels)[:k]
    idcg = dcg_at_k(-ideal_sorted_labels, k)
    
    # Avoid division by zero
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def precision_at_k(scores, labels, k=10, threshold=0.5):
    """
    Precision @ k: fraction of top-k items that are relevant.
    Items with label >= threshold are considered relevant.
    """
    sorted_indices = np.argsort(-scores)[:k]
    sorted_labels = labels[sorted_indices]
    
    relevant_count = np.sum(sorted_labels >= threshold)
    return relevant_count / k


def main():
    # Load generated molecules
    csv_path = "results/oxtr_generation/generated_ligands.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        print("   Run: python scripts/simple_generate.py")
        return
    
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} generated molecules")
    
    # Compute QED scores (dummy relevance labels)
    print("\n📊 Computing QED (drug-likeness) scores...")
    qed_scores = df['SMILES'].apply(compute_qed_score).values
    
    df['QED_Score'] = qed_scores
    
    # Display some examples
    print("\n   Sample molecules & their QED scores:")
    print(df[['SMILES', 'QED_Score']].head(10).to_string(index=False))
    print(f"\n   Mean QED: {qed_scores.mean():.3f} ± {qed_scores.std():.3f}")
    
    # Simulate predicted scores (normally would come from affinity predictor)
    # For demo: add some noise to QED scores to create a "predictions" column
    np.random.seed(42)
    noise = np.random.normal(0, 0.1, len(qed_scores))
    predicted_scores = np.clip(qed_scores + noise, 0, 1)
    
    df['Predicted_Score'] = predicted_scores
    
    # Compute ranking metrics
    print("\n" + "="*60)
    print("🎯 RANKING METRICS")
    print("="*60)
    
    for k in [5, 10, 20]:
        ndcg = ndcg_at_k(predicted_scores, qed_scores, k=k)
        prec = precision_at_k(predicted_scores, qed_scores, k=k, threshold=0.6)
        
        print(f"\n@k={k}:")
        print(f"  NDCG@{k:2d}: {ndcg:.3f}")
        print(f"  Precision@{k:2d} (rel>=0.6): {prec:.3f}")
    
    # Show top-10 ranked molecules
    print("\n" + "="*60)
    print("🏆 TOP-10 MOLECULES (by predicted ranking)")
    print("="*60)
    
    sorted_df = df.sort_values('Predicted_Score', ascending=False).head(10)
    print("\nRank | SMILES (truncated) | Predicted | True QED")
    print("-" * 70)
    for idx, (i, row) in enumerate(sorted_df.iterrows(), 1):
        smiles_short = row['SMILES'][:40] + "..."
        print(f"{idx:4d} | {smiles_short:43s} | {row['Predicted_Score']:9.3f} | {row['QED_Score']:.3f}")
    
    # Save results
    output_csv = "results/oxtr_generation/ndcg_evaluation.csv"
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Results saved to: {output_csv}")
    
    print("\n" + "="*60)
    print("💡 INTERVIEW TAKEAWAY")
    print("="*60)
    print("""
When asked about NDCG :

"NDCG (Normalized Discounted Cumulative Gain) measures ranking quality.
It rewards correct ranking of relevant items AND assigns higher value to
ranking relevant items higher (discounting by log position).

At , NDCG@k evaluates how well we rank ads by predicted CTR:
- Top-1 ad: weight = 1.0
- Top-5 ad: weight = 1/log2(6) ≈ 0.43
- Top-10 ad: weight = 1/log2(11) ≈ 0.30

In this project, NDCG@{k} measures how well our ranking (predicted scores)
matches true relevance (QED drug-likeness). Higher = better ranking quality.

Precision@k is simpler: fraction of top-k that are relevant. But NDCG
accounts for ranking position, making it better for relevance problems."
    """)


if __name__ == "__main__":
    main()
