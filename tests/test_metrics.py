# tests/test_metrics.py
import os
import sys
import torch
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.metrics import compute_distance_matrix, evaluate_reid

def test_metrics():
    # Simulate embeddings for 3 Query images and 5 Gallery images (Embedding Dim = 128)
    query_features = torch.randn(3, 128)
    gallery_features = torch.randn(5, 128)
    
    # Assign dummy Person IDs mimicking Wi-PER81 formatting
    query_pids = np.array(['01', '02', '03'])
    gallery_pids = np.array(['03', '01', '04', '02', '01'])
    
    print("1. Testing Distance Matrix Calculation...")
    dist_mat = compute_distance_matrix(query_features, gallery_features, metric='euclidean')
    print(f"   Distance Matrix Shape: {dist_mat.shape}")  # Target: (3, 5)
    assert dist_mat.shape == (3, 5)
    
    print("\n2. Testing mAP and Rank-K CMC Evaluation...")
    cmc, mAP = evaluate_reid(dist_mat, query_pids, gallery_pids, max_rank=5)
    
    print(f"   Rank-1 Accuracy: {cmc[0]:.4f}")
    print(f"   Rank-5 Accuracy: {cmc[4]:.4f}")
    print(f"   mAP Score: {mAP:.4f}")
    
    print("\n--- Metrics Validation Passed! ---")

if __name__ == "__main__":
    test_metrics()