# Fusion logic for multimodal embeddings

import torch
import torch.nn as nn


class FusionLayer(nn.Module):
    """
    Fusion Layer for Multimodal Embeddings
    Combines embeddings from multiple modalities using SOTA techniques.
    """

    def __init__(self, modality_dims, fusion_method='transformer', embedding_dim=512, attention_heads=4, num_layers=2):
        """
        Initialize the Fusion Layer.
        
        Args:
            modality_dims (list of int): List of embedding dimensions for each modality.
            fusion_method (str): Fusion method ('concat', 'weighted', 'transformer').
            embedding_dim (int): Output embedding dimension for the fused representation.
            attention_heads (int): Number of attention heads for transformer-based fusion.
            num_layers (int): Number of layers for transformer-based fusion.
        """
        super(FusionLayer, self).__init__()
        self.fusion_method = fusion_method
        self.modality_dims = modality_dims
        self.embedding_dim = embedding_dim

        if fusion_method == 'concat':
            # Concatenation-based fusion
            self.fusion_layer = nn.Linear(sum(modality_dims), embedding_dim)
        elif fusion_method == 'weighted':
            # Weighted sum-based fusion with learnable weights
            self.weights = nn.Parameter(torch.ones(len(modality_dims), dtype=torch.float32))
            self.projection_layer = nn.Linear(modality_dims[0], embedding_dim)
        elif fusion_method == 'transformer':
            # Transformer-based attention fusion
            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=max(modality_dims),
                nhead=attention_heads,
                dim_feedforward=embedding_dim * 2,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
            self.projection_layer = nn.Linear(max(modality_dims), embedding_dim)
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")

    def forward(self, modality_embeddings):
        """
        Forward pass for the Fusion Layer.
        
        Args:
            modality_embeddings (list of Tensor): List of tensors from each modality, 
                                                  each of shape (batch_size, embedding_dim).
        
        Returns:
            Tensor: Fused embedding of shape (batch_size, embedding_dim).
        """
        if self.fusion_method == 'concat':
            # Concatenate embeddings and project to shared space
            fused_embedding = torch.cat(modality_embeddings, dim=1)  # Shape: (batch_size, sum(modality_dims))
            fused_embedding = self.fusion_layer(fused_embedding)
        elif self.fusion_method == 'weighted':
            # Weighted sum of embeddings
            weighted_embeddings = [
                w * e for w, e in zip(torch.softmax(self.weights, dim=0), modality_embeddings)
            ]
            fused_embedding = torch.sum(torch.stack(weighted_embeddings, dim=0), dim=0)  # Shape: (batch_size, embedding_dim)
            fused_embedding = self.projection_layer(fused_embedding)
        elif self.fusion_method == 'transformer':
            # Transformer-based attention fusion
            modality_stack = torch.stack(modality_embeddings, dim=1)  # Shape: (batch_size, num_modalities, embedding_dim)
            fused_embedding = self.transformer(modality_stack)  # Shape: (batch_size, num_modalities, embedding_dim)
            fused_embedding = fused_embedding.mean(dim=1)  # Aggregate across modalities
            fused_embedding = self.projection_layer(fused_embedding)
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")

        # Normalize the fused embedding
        fused_embedding = torch.nn.functional.normalize(fused_embedding, p=2, dim=1)
        return fused_embedding

    def summary(self):
        """
        Prints a summary of the Fusion Layer's structure and parameters.
        """
        print(f"Fusion Layer")
        print(f"Fusion Method: {self.fusion_method}")
        print(f"Modality Dimensions: {self.modality_dims}")
        print(f"Output Embedding Dimension: {self.embedding_dim}")
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable Parameters: {total_params}")


# Example Usage
if __name__ == "__main__":
    # Example embeddings from three modalities
    batch_size = 4
    modality1_dim = 512
    modality2_dim = 512
    modality3_dim = 512

    modality1_embedding = torch.rand(batch_size, modality1_dim)
    modality2_embedding = torch.rand(batch_size, modality2_dim)
    modality3_embedding = torch.rand(batch_size, modality3_dim)

    # Initialize the fusion layer with transformer-based fusion
    fusion_layer = FusionLayer(
        modality_dims=[modality1_dim, modality2_dim, modality3_dim],
        fusion_method='transformer',
        embedding_dim=512,
        attention_heads=4,
        num_layers=2
    )

    # Forward pass and summary
    fused_embedding = fusion_layer([modality1_embedding, modality2_embedding, modality3_embedding])
    fusion_layer.summary()
    print("Fused Embedding Shape:", fused_embedding.shape)
