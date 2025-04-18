import torch
import torch.nn as nn
import torch.nn.functional as functional

class ContrastiveLoss(nn.Module):
    # Trains the Siamese netowkr to make genuine-par embeddings close together (d -> 0) \
    # and forged pair embeddings at least margin apart (d >= margin), without over \
    # penalizing negatives that are already far enough

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embedding_a, embedding_b, labels):
        # Computes the Euclidean distance between corresponding rows \
        # yielding a 1D tensor length batch_size
        distances = functional.pairwise_distance(
            embedding_a, embedding_b)
        positive_loss = labels * distances.pow(2)
        negative_loss = (1 - labels) * functional.relu(self.margin - distances).pow(2)
        # return L = y * d^2 + (1-y) * max(0, m -d )^2
        return (positive_loss + negative_loss).mean()