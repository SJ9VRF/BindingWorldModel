# Contrastive loss for training
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for Multimodal Embeddings
    Implements SOTA techniques like NT-Xent Loss with temperature scaling.
    """

    def __init__(self, temperature=0.07, use_hard_negatives=False):
        """
        Initialize the Contrastive Loss.
        
        Args:
            temperature (float): Scaling factor for logits in the softmax.
            use_hard_negatives (bool): Whether to use hard-negative mining.
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.use_hard_negatives = use_hard_negatives

    def forward(self, embeddings_a, embeddings_b, negatives=None):
        """
        Forward pass for the contrastive loss.
        
        Args:
            embeddings_a (Tensor): Embeddings from modality A, shape (batch_size, embedding_dim).
            embeddings_b (Tensor): Embeddings from modality B, shape (batch_size, embedding_dim).
            negatives (Tensor, optional): Negative samples, shape (num_negatives, embedding_dim).
        
        Returns:
            Tensor: Computed contrastive loss.
        """
        batch_size = embeddings_a.size(0)

        # Normalize embeddings to unit length
        embeddings_a = F.normalize(embeddings_a, p=2, dim=1)
        embeddings_b = F.normalize(embeddings_b, p=2, dim=1)

        # Compute similarity matrix
        positive_sim = torch.sum(embeddings_a * embeddings_b, dim=1)  # Shape: (batch_size,)
        positive_sim = positive_sim / self.temperature

        # Compute logits for cross-entropy
        logits = torch.mm(embeddings_a, embeddings_b.T) / self.temperature  # Shape: (batch_size, batch_size)

        # Add hard negatives if provided
        if self.use_hard_negatives and negatives is not None:
            negatives = F.normalize(negatives, p=2, dim=1)
            negative_logits = torch.mm(embeddings_a, negatives.T) / self.temperature  # Shape: (batch_size, num_negatives)
            logits = torch.cat([logits, negative_logits], dim=1)  # Shape: (batch_size, batch_size + num_negatives)

        # Create labels (diagonal is positive pair)
        labels = torch.arange(batch_size).to(embeddings_a.device)  # Shape: (batch_size,)

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss

    def pairwise_loss(self, embeddings_a, embeddings_b):
        """
        Pairwise contrastive loss for all positive pairs.
        
        Args:
            embeddings_a (Tensor): Embeddings from modality A, shape (batch_size, embedding_dim).
            embeddings_b (Tensor): Embeddings from modality B, shape (batch_size, embedding_dim).
        
        Returns:
            Tensor: Pairwise contrastive loss.
        """
        # Normalize embeddings to unit length
        embeddings_a = F.normalize(embeddings_a, p=2, dim=1)
        embeddings_b = F.normalize(embeddings_b, p=2, dim=1)

        # Compute pairwise similarity
        similarity_matrix = torch.mm(embeddings_a, embeddings_b.T)  # Shape: (batch_size, batch_size)

        # Positive pairs (diagonal elements)
        positive_similarity = torch.diagonal(similarity_matrix)  # Shape: (batch_size,)

        # Mask to ignore diagonal elements
        mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device).bool()
        negative_similarity = similarity_matrix[~mask].view(similarity_matrix.size(0), -1)  # Non-diagonal elements

        # NT-Xent loss
        positive_logits = positive_similarity / self.temperature
        negative_logits = negative_similarity / self.temperature
        logits = torch.cat([positive_logits.unsqueeze(1), negative_logits], dim=1)  # Shape: (batch_size, 1 + negatives)

        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)

        return loss


# Example Usage
if __name__ == "__main__":
    # Example embeddings
    batch_size = 8
    embedding_dim = 512

    embeddings_a = torch.rand(batch_size, embedding_dim)  # Modality A embeddings
    embeddings_b = torch.rand(batch_size, embedding_dim)  # Modality B embeddings
    negatives = torch.rand(16, embedding_dim)  # Negative samples (optional)

    # Initialize the contrastive loss
    contrastive_loss = ContrastiveLoss(temperature=0.07, use_hard_negatives=True)

    # Compute loss with hard negatives
    loss = contrastive_loss(embeddings_a, embeddings_b, negatives=negatives)

    print(f"Contrastive Loss: {loss.item():.4f}")
