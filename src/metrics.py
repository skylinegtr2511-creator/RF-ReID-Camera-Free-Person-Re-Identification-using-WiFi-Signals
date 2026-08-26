import torch
import numpy as np

def compute_distance_matrix(qf, gf, metric='euclidean'):
    """
    Computes the distance matrix between Query (Probe) and Gallery feature embeddings.
    
    Args:
        qf (torch.Tensor): Query embeddings, shape [num_query, feature_dim]
        gf (torch.Tensor): Gallery embeddings, shape [num_gallery, feature_dim]
        metric (str): 'euclidean' or 'cosine'
        
    Returns:
        dist_mat (torch.Tensor): Distance matrix of shape [num_query, num_gallery]
    """
    if metric == 'euclidean':
        # Built-in PyTorch function for pairwise Euclidean distances
        dist_mat = torch.cdist(qf, gf, p=2)
    elif metric == 'cosine':
        # Ensure L2 normalization before calculating cosine distance
        qf_norm = torch.nn.functional.normalize(qf, p=2, dim=1)
        gf_norm = torch.nn.functional.normalize(gf, p=2, dim=1)
        # Cosine distance = 1 - Cosine Similarity
        dist_mat = 1 - torch.mm(qf_norm, gf_norm.t())
    else:
        raise ValueError(f"Unknown metric: {metric}")
        
    return dist_mat

def evaluate_reid(dist_mat, q_pids, g_pids, max_rank=10):
    """
    Evaluates the Rank-K (CMC) and mean Average Precision (mAP) metrics.
    
    Args:
        dist_mat (numpy.ndarray or torch.Tensor): Distance matrix [num_query, num_gallery]
        q_pids (numpy.ndarray): Array of Query Person IDs
        g_pids (numpy.ndarray): Array of Gallery Person IDs
        max_rank (int): Max rank to compute for CMC curve (default: 10)
        
    Returns:
        cmc (numpy.ndarray): CMC curve up to max_rank (Rank-1 is cmc[0], Rank-5 is cmc[4])
        mAP (float): mean Average Precision score
    """
    # Convert to NumPy for efficient index slicing if tensor is passed
    if isinstance(dist_mat, torch.Tensor):
        dist_mat = dist_mat.cpu().numpy()
        
    num_query, num_gallery = dist_mat.shape
    
    # Sort the distances for each query to rank the gallery images (lowest distance first)
    indices = np.argsort(dist_mat, axis=1)
    
    # Create a boolean matrix where True (1) means it's a correct identity match
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
    all_cmc = []
    all_AP = []
    
    for q_idx in range(num_query):
        # Extract the binary match array for the current query
        q_matches = matches[q_idx]
        
        num_true_matches = np.sum(q_matches)
        # If there are no ground-truth matches for this query in the gallery, skip it
        if num_true_matches == 0:
            continue
            
        # --- 1. Compute Average Precision (AP) ---
        # Find the rank positions (indices) where the true matches occurred
        match_ranks = np.where(q_matches == 1)[0]
        
        ap = 0.0
        for i, rank in enumerate(match_ranks):
            # Precision at k = (number of correct matches up to rank k) / rank
            ap += (i + 1) / (rank + 1)
        ap /= num_true_matches
        all_AP.append(ap)
        
        # --- 2. Compute Cumulative Matching Characteristics (CMC) ---
        # A query is considered successful at Rank-K if the FIRST true match is at rank <= K
        first_match_rank = match_ranks[0]
        cmc = np.zeros(num_gallery)
        cmc[first_match_rank:] = 1
        all_cmc.append(cmc)
        
    # Average across all queries to get mAP
    mAP = np.mean(all_AP) if all_AP else 0.0
    
    # Average across all queries to get mean CMC curve
    if all_cmc:
        all_cmc = np.asarray(all_cmc).astype(np.float32)
        mean_cmc = np.mean(all_cmc, axis=0)
    else:
        mean_cmc = np.zeros(max_rank)
        
    return mean_cmc[:max_rank], mAP