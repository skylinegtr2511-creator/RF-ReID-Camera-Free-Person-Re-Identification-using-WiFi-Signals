import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import WiPER81TripletDataset
from torch.utils.data import DataLoader


def main():
    print("Loading dataset...")
    # Initialize training dataset
    dataset = WiPER81TripletDataset(data_dir="data/Wi-PER81Dataset/ImageData/D1")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Fetch a single batch
    batch = next(iter(dataloader))
    print("\n--- Success! Data loaded correctly ---")
    print("Anchor batch shape:", batch['anchor'].shape)      # Target: [4, 3, 224, 224]
    print("Positive batch shape:", batch['positive'].shape)  # Target: [4, 3, 224, 224]
    print("Negative batch shape:", batch['negative'].shape)  # Target: [4, 3, 224, 224]
    print("Anchor PIDs in batch:", batch['anchor_pid'])

if __name__ == "__main__":
    main()