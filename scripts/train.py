import os
import sys
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import joblib

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import WiPER81TripletDataset, WiPER81EvalDataset
from src.models import TriNetModel
from src.losses import TripletLoss
from src.train_utils import train_one_epoch, evaluate_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train Re-ID Siamese Network on Wi-PER81")
    parser.add_argument('--backbone', type=str, default='densenet121', 
                        choices=['densenet121', 'mobilenet_v3', 'efficientnet_b0'],
                        help="CNN backbone to use inside the TriNet model")
    parser.add_argument('--batch_size', type=int, default=16, help="Training batch size")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--margin', type=float, default=0.3, help="Triplet loss margin")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Starting Training on {device.type.upper()} ---")
    print(f"Model Backbone: {args.backbone}")
    
    # Setup DataLoaders
    train_dir = os.path.join("data", "Wi-PER81Dataset", "ImageData", "D1")
    train_dataset = WiPER81TripletDataset(data_dir=train_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    val_gallery_dir = os.path.join("data", "Wi-PER81Dataset", "ImageData", "D2v", "Gallery")
    val_probe_dir = os.path.join("data", "Wi-PER81Dataset", "ImageData", "D2v", "Probe")
    val_gallery_dataset = WiPER81EvalDataset(folder_path=val_gallery_dir)
    val_probe_dataset = WiPER81EvalDataset(folder_path=val_probe_dir)
    val_gallery_loader = DataLoader(val_gallery_dataset, batch_size=32, shuffle=False)
    val_probe_loader = DataLoader(val_probe_dataset, batch_size=32, shuffle=False)
    
    # Setup Model, Loss, Optimizer
    model = TriNetModel(backbone_name=args.backbone, embedding_dim=128).to(device)
    criterion = TripletLoss(margin=args.margin)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Ensure directories exist
    os.makedirs(os.path.join("outputs", "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join("outputs", "logs"), exist_ok=True)
    
    # Initialize history trackers
    history = {
        'train_loss': [],
        'val_mAP': [],
        'val_rank1': [],
        'val_rank5': []
    }
    best_mAP = 0.0
    
    # Main Training Loop
    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch}/{args.epochs}]")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Training Triplet Loss: {train_loss:.4f}")
        
        # Evaluate
        cmc, mAP = evaluate_model(model, val_gallery_loader, val_probe_loader, device)
        print(f"Validation mAP: {mAP:.4f} | Rank-1: {cmc[0]:.4f} | Rank-5: {cmc[4]:.4f}")
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_mAP'].append(mAP)
        history['val_rank1'].append(cmc[0])
        history['val_rank5'].append(cmc[4])
        
        # Save Best Model
        if mAP > best_mAP:
            best_mAP = mAP
            save_path = os.path.join("outputs", "checkpoints", f"best_model_{args.backbone}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"--> New best model saved to {save_path}!")

    # Save History using joblib
    log_path = os.path.join("outputs", "logs", f"history_{args.backbone}.joblib")
    joblib.dump(history, log_path)
    print(f"\nTraining metrics saved to {log_path}")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()