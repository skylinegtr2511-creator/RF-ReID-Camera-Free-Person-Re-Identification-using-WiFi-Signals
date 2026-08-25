import torch
import numpy as np
from tqdm import tqdm
from src.metrics import compute_distance_matrix, evaluate_reid

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Executes one epoch of training using Triplet Loss.
    """
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training Epoch", leave=False)
    
    for batch in pbar:
        anchor = batch['anchor'].to(device)
        positive = batch['positive'].to(device)
        negative = batch['negative'].to(device)
        
        optimizer.zero_grad()
        
        anchor_embed = model(anchor)
        positive_embed = model(positive)
        negative_embed = model(negative)
        
        loss = criterion(anchor_embed, positive_embed, negative_embed)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss

@torch.no_grad()
def evaluate_loss(model, dataloader, criterion, device):
    """
    Computes the Triplet Loss on a validation/test dataset to check for overfitting.
    No gradients are calculated.
    """
    model.eval()
    running_loss = 0.0
    
    for batch in dataloader:
        anchor = batch['anchor'].to(device)
        positive = batch['positive'].to(device)
        negative = batch['negative'].to(device)
        
        anchor_embed = model(anchor)
        positive_embed = model(positive)
        negative_embed = model(negative)
        
        loss = criterion(anchor_embed, positive_embed, negative_embed)
        running_loss += loss.item()
        
    val_loss = running_loss / len(dataloader)
    return val_loss

@torch.no_grad()
def extract_features(model, dataloader, device):
    """
    Extracts embeddings for an entire dataset (Gallery or Probe).
    """
    model.eval()
    all_features = []
    all_pids = []
    
    for imgs, pids, _ in tqdm(dataloader, desc="Extracting Features", leave=False):
        imgs = imgs.to(device)
        features = model(imgs)
        all_features.append(features.cpu().numpy())
        all_pids.extend(pids)
        
    all_features = np.vstack(all_features)
    all_pids = np.array(all_pids)
    
    return torch.tensor(all_features), all_pids

def evaluate_model(model, gallery_loader, probe_loader, device):
    """
    Extracts features for Gallery and Probe sets, computes the distance matrix, 
    and returns mAP and CMC scores.
    """
    print("Extracting Features for Validation...")
    gf, g_pids = extract_features(model, gallery_loader, device)
    qf, q_pids = extract_features(model, probe_loader, device)
    
    print("Computing Distance Matrix...")
    dist_mat = compute_distance_matrix(qf, gf, metric='euclidean')
    
    print("Calculating mAP and Rank-K...")
    cmc, mAP = evaluate_reid(dist_mat, q_pids, g_pids, max_rank=10)
    
    return cmc, mAP