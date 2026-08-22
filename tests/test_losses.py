import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.losses import TripletLoss, PCBLoss

def test_triplet_loss():
    print("Testing Triplet Loss...")
    loss_fn = TripletLoss(margin=0.3)
    
    # Simulate batch size of 4, embedding dimension of 128
    anchor = torch.randn(4, 128)
    positive = torch.randn(4, 128)
    negative = torch.randn(4, 128)
    
    loss = loss_fn(anchor, positive, negative)
    print(f"--- Success! Triplet Loss computed: {loss.item():.4f} ---\n")

def test_pcb_loss():
    print("Testing PCB Loss...")
    loss_fn = PCBLoss()
    
    # Simulate batch size of 4, 81 classes (Wi-PER81), and 6 spatial parts
    batch_size = 4
    num_classes = 81
    num_parts = 6
    
    # Simulate a list of 6 tensors, each representing logits for a spatial part
    part_logits_list = [torch.randn(batch_size, num_classes) for _ in range(num_parts)]
    
    # Simulate ground truth labels
    targets = torch.randint(0, num_classes, (batch_size,))
    
    loss = loss_fn(part_logits_list, targets)
    print(f"--- Success! PCB Loss computed: {loss.item():.4f} ---")

if __name__ == "__main__":
    test_triplet_loss()
    test_pcb_loss()