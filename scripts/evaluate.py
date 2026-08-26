import os
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import WiPER81EvalDataset
from src.models import TriNetModel
from src.train_utils import evaluate_model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Starting Final Test Set Evaluation on {device.type.upper()} ---")

    # 1. Load the Unseen D2t Test Data
    test_gallery_dir = os.path.join("data", "Wi-PER81Dataset", "ImageData", "D2t", "Gallery")
    test_probe_dir = os.path.join("data", "Wi-PER81Dataset", "ImageData", "D2t", "Probe")

    test_gallery_dataset = WiPER81EvalDataset(folder_path=test_gallery_dir)
    test_probe_dataset = WiPER81EvalDataset(folder_path=test_probe_dir)

    test_gallery_loader = DataLoader(test_gallery_dataset, batch_size=32, shuffle=False)
    test_probe_loader = DataLoader(test_probe_dataset, batch_size=32, shuffle=False)

    backbones = ['densenet121', 'efficientnet_b0', 'mobilenet_v3']
    results = []

    # 2. Evaluate Each Model
    for backbone in backbones:
        print(f"\n=======================================================")
        print(f"Evaluating Backbone: {backbone.upper()}")
        print(f"=======================================================")
        
        checkpoint_path = os.path.join("outputs", "checkpoints", f"best_model_{backbone}.pth")

        if not os.path.exists(checkpoint_path):
            print(f"[!] Checkpoint not found: {checkpoint_path}. Skipping.")
            continue

        # Initialize Model architecture (must match what was saved)
        model = TriNetModel(backbone_name=backbone, embedding_dim=128).to(device)
        
        # Load the best weights
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        # Run the evaluation loop on the test set
        cmc, mAP = evaluate_model(model, test_gallery_loader, test_probe_loader, device)

        # Store the metrics
        results.append({
            "Model Backbone": backbone.upper(),
            "Test mAP": round(mAP, 4),
            "Test Rank-1": round(cmc[0], 4),
            "Test Rank-5": round(cmc[4], 4),
            "Test Rank-10": round(cmc[9], 4)
        })

    # 3. Generate and Save the Final Performance Table
    if results:
        df = pd.DataFrame(results)
        
        print("\n\n#######################################################")
        print("             FINAL D2T TEST SET PERFORMANCE            ")
        print("#######################################################")
        print(df.to_string(index=False))
        print("#######################################################\n")
        
        # Save table to a CSV file for your assignment report
        os.makedirs(os.path.join("outputs", "results"), exist_ok=True)
        csv_path = os.path.join("outputs", "results", "final_test_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"Metrics table saved to {csv_path}")

if __name__ == "__main__":
    # Workaround for Windows multiprocessing
    import multiprocessing
    multiprocessing.freeze_support()
    main()