import os
import sys
import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import WiPER81EvalDataset
from src.models import TriNetModel, DuATMModel
from src.train_utils import evaluate_model

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Re-ID Models on Test Set")
    parser.add_argument('--model_type', type=str, default='trinet', choices=['trinet', 'duatm'],
                        help="Which architecture family to evaluate")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Starting Final Test Set Evaluation for {args.model_type.upper()} ---")

    # 1. Load the Unseen D2t Test Data
    test_gallery_dir = os.path.join("data", "Wi-PER81Dataset", "ImageData", "D2t", "Gallery")
    test_probe_dir = os.path.join("data", "Wi-PER81Dataset", "ImageData", "D2t", "Probe")

    test_gallery_dataset = WiPER81EvalDataset(folder_path=test_gallery_dir)
    test_probe_dataset = WiPER81EvalDataset(folder_path=test_probe_dir)

    test_gallery_loader = DataLoader(test_gallery_dataset, batch_size=32, shuffle=False)
    test_probe_loader = DataLoader(test_probe_dataset, batch_size=32, shuffle=False)

    # 2. Setup Backbones Based on Architecture
    if args.model_type == 'duatm':
        backbones = ['resnet50', 'efficientnet_v2_s', 'convnext_tiny']
        prefix = 'duatm_'
    else:
        backbones = ['densenet121', 'efficientnet_b0', 'mobilenet_v3']
        prefix = '' # Original TriNet models were saved without a prefix

    results = []

    # 3. Evaluate Each Model
    for backbone in backbones:
        print(f"\n=======================================================")
        print(f"Evaluating Backbone: {backbone.upper()}")
        print(f"=======================================================")
        
        checkpoint_name = f"best_model_{prefix}{backbone}.pth"
        
        # Fallback check in case you retrained TriNet using the new script logic
        if args.model_type == 'trinet' and not os.path.exists(os.path.join("outputs", "checkpoints", checkpoint_name)):
            checkpoint_name = f"best_model_trinet_{backbone}.pth"
            
        checkpoint_path = os.path.join("outputs", "checkpoints", checkpoint_name)

        if not os.path.exists(checkpoint_path):
            print(f"[!] Checkpoint not found: {checkpoint_path}. Skipping.")
            continue

        # Initialize correct model architecture
        if args.model_type == 'duatm':
            model = DuATMModel(backbone_name=backbone, embedding_dim=128).to(device)
        else:
            model = TriNetModel(backbone_name=backbone, embedding_dim=128).to(device)
        
        # Load the best weights
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        # Run evaluation
        cmc, mAP = evaluate_model(model, test_gallery_loader, test_probe_loader, device)

        # Store metrics
        results.append({
            "Architecture": args.model_type.upper(),
            "Backbone": backbone.upper(),
            "mAP": f"{mAP:.4f}",
            "Accuracy (Rank-1)": f"{cmc[0]:.4f}",
            "CMC (R1 / R5 / R10)": f"{cmc[0]:.3f} / {cmc[4]:.3f} / {cmc[9]:.3f}"
        })

    # 4. Generate and Save Table
    if results:
        df = pd.DataFrame(results)
        
        print("\n\n#######################################################")
        print(f"        FINAL D2T TEST SET PERFORMANCE ({args.model_type.upper()})")
        print("#######################################################")
        print(df.to_string(index=False))
        print("#######################################################\n")
        
        # Save to CSV based on model type
        os.makedirs(os.path.join("outputs", "results"), exist_ok=True)
        csv_path = os.path.join("outputs", "results", f"final_test_metrics_{args.model_type}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Metrics table saved to {csv_path}")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()