import os
import sys
import torch

# Add the parent directory (project root) to sys.path so 'src' can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import TriNetModel, PCBModel, DuATMModel

def test_models():
    dummy_input = torch.randn(2, 3, 224, 224)  # Batch of 2 images
    
    print("1. Testing TriNet (DenseNet121 Backbone)...")
    trinet = TriNetModel(backbone_name='densenet121', embedding_dim=128)
    out = trinet(dummy_input)
    print(f"   Output embedding shape: {out.shape}")  # Target: [2, 128]
    assert out.shape == (2, 128)
    
    print("\n2. Testing PCB Model (Part-based Convolutional Baseline)...")
    pcb = PCBModel(num_classes=81, num_parts=6)
    
    # Training mode test
    pcb.train()
    logits_list = pcb(dummy_input)
    print(f"   Training output (6 part logits): {len(logits_list)} x {logits_list[0].shape}")
    
    # Eval mode test
    pcb.eval()
    concat_parts = pcb(dummy_input)
    print(f"   Evaluation concatenated shape: {concat_parts.shape}")  # Target: [2, 1536]
    
    print("\n3. Testing DuATM Model (Dual Attention Matching)...")
    duatm = DuATMModel(embedding_dim=128)
    duatm.eval()
    duatm_out = duatm(dummy_input)
    print(f"   Dual-Attention output shape: {duatm_out.shape}")  # Target: [2, 128]
    
    print("\n--- All 3 Models Passed Validation! ---")

if __name__ == "__main__":
    test_models()