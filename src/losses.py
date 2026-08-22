import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """
    Computes the Triplet Margin Loss for Metric Learning.
    Ensures the distance between the Anchor and Positive is smaller 
    than the distance between the Anchor and Negative by a defined margin.
    """
    def __init__(self, margin=0.3, p=2):
        """
        Args:
            margin (float): The margin of separation.
            p (int): The norm degree for pairwise distance (2 for Euclidean).
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.p = p
        # We use MarginRankingLoss internally to handle the max(0, dist_ap - dist_an + margin)
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor (Tensor): Feature embeddings of the anchor images.
            positive (Tensor): Feature embeddings of the positive images.
            negative (Tensor): Feature embeddings of the negative images.
        Returns:
            Tensor: The calculated loss scalar.
        """
        # Calculate Euclidean distances
        dist_ap = F.pairwise_distance(anchor, positive, p=self.p)
        dist_an = F.pairwise_distance(anchor, negative, p=self.p)
        
        # y is the target tensor. We set it to 1 because we want the first input (dist_an) 
        # to be strictly larger than the second input (dist_ap) by the margin.
        y = torch.ones_like(dist_an)
        
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


class PCBLoss(nn.Module):
    """
    Computes the loss for the Part-based Convolutional Baseline (PCB).
    Calculates the Cross Entropy Loss for each spatial part independently 
    and sums them together.
    """
    def __init__(self):
        super(PCBLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, part_logits_list, targets):
        """
        Args:
            part_logits_list (list of Tensors): A list where each element is the 
                                                logits tensor for a specific spatial part.
            targets (Tensor): The ground truth identity labels.
        Returns:
            Tensor: The accumulated loss scalar.
        """
        loss = 0.0
        # Accumulate loss over all spatial parts (e.g., 6 horizontal stripes)
        for part_logits in part_logits_list:
            loss += self.criterion(part_logits, targets)
            
        return loss