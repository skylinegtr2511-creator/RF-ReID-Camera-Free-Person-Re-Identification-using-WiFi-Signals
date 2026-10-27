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
from src.models import TriNetModel, DuATMModel
from src.losses import TripletLoss
from src.train_utils import train_one_epoch, evaluate_loss, evaluate_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train Re-ID Siamese Network on Wi-PER81")
    parser.add_argument('--model_type', type=str, default='duatm', choices=['trinet', 'duatm'],
                        help="Architecture type: trinet or duatm")
    parser.add_argument('--backbone', type=str, default='resnet50', 
                        choices=['densenet121', 'mobilenet_v3', 'efficientnet_b0', 
                                 'resnet50', 'efficientnet_v2_s', 'convnext_tiny'],
                        help="CNN backbone to use inside the model")
    parser.add_argument('--batch_size', type=int, default=32, help="Training batch size")
    parser.add_argument('--epochs', type=int, default=50, help="Max training epochs")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--margin', type=float, default=0.3, help="Triplet loss margin")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="L2 Regularization penalty")
    parser.add_argument('--patience', type=int, default=7, help="Epochs to wait before early stopping")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Starting Training on {device.type.upper()} ---")
    print(f"Model Architecture: {args.model_type.upper()} | Backbone: {args.backbone}")
    
    # DataLoaders
    train_dir = os.path.join("data", "Wi-PER81Dataset", "ImageData", "D1")
    train_dataset = WiPER81TripletDataset(data_dir=train_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    val_triplet_dir = os.path.join("data", "Wi-PER81Dataset", "ImageData", "D2v", "Gallery")
    val_triplet_dataset = WiPER81TripletDataset(data_dir=val_triplet_dir)
    val_triplet_loader = DataLoader(val_triplet_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    val_gallery_dir = os.path.join("data", "Wi-PER81Dataset", "ImageData", "D2v", "Gallery")
    val_probe_dir = os.path.join("data", "Wi-PER81Dataset", "ImageData", "D2v", "Probe")
    val_gallery_dataset = WiPER81EvalDataset(folder_path=val_gallery_dir)
    val_probe_dataset = WiPER81EvalDataset(folder_path=val_probe_dir)
    val_gallery_loader = DataLoader(val_gallery_dataset, batch_size=32, shuffle=False)
    val_probe_loader = DataLoader(val_probe_dataset, batch_size=32, shuffle=False)
    
    # Initialize Architecture Selection
    if args.model_type == 'duatm':
        model = DuATMModel(backbone_name=args.backbone, embedding_dim=128).to(device)
        model_tag = f"duatm_{args.backbone}"
    else:
        model = TriNetModel(backbone_name=args.backbone, embedding_dim=128).to(device)
        model_tag = f"trinet_{args.backbone}"
    
    criterion = TripletLoss(margin=args.margin)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    os.makedirs(os.path.join("outputs", "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join("outputs", "logs"), exist_ok=True)
    
    history = {'train_loss': [], 'val_loss': [], 'val_mAP': [], 'val_rank1': [], 'val_rank5': []}
    best_mAP = 0.0
    epochs_no_improve = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch}/{args.epochs}]")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_loss(model, val_triplet_loader, criterion, device)
        cmc, mAP = evaluate_model(model, val_gallery_loader, val_probe_loader, device)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val mAP: {mAP:.4f} | Rank-1: {cmc[0]:.4f} | Rank-5: {cmc[4]:.4f}")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mAP'].append(mAP)
        history['val_rank1'].append(cmc[0])
        history['val_rank5'].append(cmc[4])
        
        if mAP > best_mAP:
            best_mAP = mAP
            epochs_no_improve = 0
            save_path = os.path.join("outputs", "checkpoints", f"best_model_{model_tag}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"--> New best model saved to {save_path}! (mAP: {best_mAP:.4f})")
        else:
            epochs_no_improve += 1
            print(f"--> No improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= args.patience:
                print(f"\n[!] Early stopping triggered after {epoch} epochs.")
                break

    log_path = os.path.join("outputs", "logs", f"history_{model_tag}.joblib")
    joblib.dump(history, log_path)
    print(f"\nAll metrics saved to {log_path}")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()