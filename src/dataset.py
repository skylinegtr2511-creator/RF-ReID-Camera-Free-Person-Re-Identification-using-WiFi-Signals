import os
import glob
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# =====================================================================
# 1. Data Augmentations Pipeline
# =====================================================================

def get_train_transforms(image_size=(224, 224)):
    """
    Data Augmentation for CSI Heatmap Training.
    Applies Random Cropping (temporal/spatial shift) and 
    Random Rectangular Erasing (simulates dropped packets / interference).
    """
    return T.Compose([
        T.Resize(image_size),
        # Random cropping with edge padding to preserve edge subcarrier boundaries
        T.RandomCrop(image_size, padding=4, padding_mode='edge'),
        T.ToTensor(),
        # Random Erasing to simulate packet loss / RF multipath fading
        T.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_eval_transforms(image_size=(224, 224)):
    """
    Clean transform pipeline for Validation (D2v) and Testing (D2t).
    No augmentations are applied during evaluation.
    """
    return T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# =====================================================================
# 2. Triplet Dataset Class (For Training: TriNet, PCB, DuATM)
# =====================================================================

class WiPER81TripletDataset(Dataset):
    """
    PyTorch Dataset that generates (Anchor, Positive, Negative) triplets 
    from the D1 training set for Metric Learning.
    """
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Path to D1 folder (e.g., 'data/Wi-PER81Dataset/ImageData/D1')
            transform (callable, optional): PyTorch torchvision transforms
        """
        self.data_dir = data_dir
        self.transform = transform if transform else get_train_transforms()
        
        # Dictionary mapping person_id -> list of image file paths
        self.identity_to_images = {}
        self._build_dataset()
        
        self.identities = list(self.identity_to_images.keys())
        self.all_samples = []
        for pid, img_list in self.identity_to_images.items():
            for img_path in img_list:
                self.all_samples.append((img_path, pid))

    def _extract_pid(self, filepath):
        """
        Extracts identity class ID from folder structure or filename.
        Works if subfolders exist or extracts from filename patterns.
        """
        parent_dir = os.path.basename(os.path.dirname(filepath))
        # If parent directory is a identity class folder (e.g., '01', 'p1', etc.)
        if parent_dir not in ['Gallery', 'Probe']:
            return parent_dir
        
        # Fallback: extract PID from filename pattern
        filename = os.path.basename(filepath)
        # Assuming filename formatting or metadata index mapping
        pid = filename.split('_')[0] if '_' in filename else filename.split('.')[0]
        return pid

    def _build_dataset(self):
        # Gather all PNG images from both Gallery and Probe subdirectories in D1
        search_pattern = os.path.join(self.data_dir, '**', '*.png')
        image_paths = glob.glob(search_pattern, recursive=True)
        
        if len(image_paths) == 0:
            raise FileNotFoundError(f"No .png heatmaps found in directory: {self.data_dir}")

        for img_path in image_paths:
            pid = self._extract_pid(img_path)
            if pid not in self.identity_to_images:
                self.identity_to_images[pid] = []
            self.identity_to_images[pid].append(img_path)

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, index):
        # 1. Select Anchor
        anchor_path, anchor_pid = self.all_samples[index]

        # 2. Select Positive (Same person, different heatmap)
        pos_candidates = self.identity_to_images[anchor_pid]
        if len(pos_candidates) > 1:
            pos_path = random.choice([p for p in pos_candidates if p != anchor_path])
        else:
            pos_path = anchor_path  # Fallback if only 1 heatmap exists for identity

        # 3. Select Negative (Different person)
        neg_pid = random.choice([pid for pid in self.identities if pid != anchor_pid])
        neg_path = random.choice(self.identity_to_images[neg_pid])

        # Load PIL Images
        anchor_img = Image.open(anchor_path).convert('RGB')
        pos_img = Image.open(pos_path).convert('RGB')
        neg_img = Image.open(neg_path).convert('RGB')

        # Apply Transforms
        if self.transform:
            anchor_img = self.transform(anchor_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)

        return {
            'anchor': anchor_img,
            'positive': pos_img,
            'negative': neg_img,
            'anchor_pid': anchor_pid,
            'neg_pid': neg_pid
        }


# =====================================================================
# 3. Evaluation Dataset Class (For Probe vs Gallery Testing)
# =====================================================================

class WiPER81EvalDataset(Dataset):
    """
    PyTorch Dataset for evaluation on D2v (Validation) or D2t (Test).
    Loads individual images alongside person IDs for Rank-K / mAP metrics.
    """
    def __init__(self, folder_path, transform=None):
        """
        Args:
            folder_path (str): Path to Gallery or Probe folder
            transform (callable, optional): Evaluation transforms
        """
        self.folder_path = folder_path
        self.transform = transform if transform else get_eval_transforms()
        
        search_pattern = os.path.join(self.folder_path, '**', '*.png')
        self.image_paths = glob.glob(search_pattern, recursive=True)

    def _extract_pid(self, filepath):
        parent_dir = os.path.basename(os.path.dirname(filepath))
        if parent_dir not in ['Gallery', 'Probe']:
            return parent_dir
        filename = os.path.basename(filepath)
        return filename.split('_')[0] if '_' in filename else filename.split('.')[0]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        pid = self._extract_pid(img_path)
        
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, pid, img_path