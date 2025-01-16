# Base class for modality

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseModality(ABC, nn.Module):
    """
    Abstract Base Class for all modalities.
    Provides a common structure and utilities for modality-specific encoders.
    """

    def __init__(self, input_dim=None, embedding_dim=512, device='cuda'):
        """
        Initializes the base modality.
        
        Args:
            input_dim (int): Input dimension of raw modality data (if applicable).
            embedding_dim (int): Dimension to project features into the shared embedding space.
            device (str): Device to run the model ('cuda' or 'cpu').
        """
        super(BaseModality, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Projection layer to map features into the shared embedding space
        if input_dim:
            self.projection_layer = nn.Linear(input_dim, embedding_dim)
        else:
            self.projection_layer = None

        # Move the module to the specified device
        self.to(self.device)

    def preprocess(self, data):
        """
        Preprocess raw modality data.
        
        Args:
            data: Raw data specific to the modality (e.g., image tensor, text string, etc.).
        
        Returns:
            Preprocessed data ready for feature extraction.
        """
        if isinstance(data, torch.Tensor):
            # Example: Normalize tensor input if it's raw data
            data = data / torch.norm(data, p=2, dim=1, keepdim=True)
        else:
            raise ValueError("Unsupported data type for preprocessing. Must be a PyTorch Tensor.")
        return data

    def extract_features(self, preprocessed_data):
        """
        Extract features from preprocessed data.
        
        Args:
            preprocessed_data: Preprocessed modality-specific data.
        
        Returns:
            Raw feature vector before projection.
        """
        if self.input_dim:
            # Example: Apply a simple feed-forward network for feature extraction
            feature_extractor = nn.Sequential(
                nn.Linear(self.input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU()
            ).to(self.device)
            return feature_extractor(preprocessed_data)
        else:
            raise ValueError("Input dimension is not defined for feature extraction.")

    def forward(self, data):
        """
        Defines the forward pass for the modality encoder.
        
        Args:
            data: Raw input data for the modality.
        
        Returns:
            Tensor: Features projected into the shared embedding space.
        """
        preprocessed_data = self.preprocess(data)
        raw_features = self.extract_features(preprocessed_data)
        if self.projection_layer:
            projected_features = self.projection_layer(raw_features)
        else:
            projected_features = raw_features
        return torch.nn.functional.normalize(projected_features, p=2, dim=1)

    def initialize_parameters(self):
        """
        Initializes parameters of the modality encoder using Xavier initialization.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def summary(self):
        """
        Prints a summary of the modality encoder's structure and parameters.
        """
        print(f"Modality: {self.__class__.__name__}")
        print(f"Input Dimension: {self.input_dim}")
        print(f"Embedding Dimension: {self.embedding_dim}")
        print(f"Device: {self.device}")
        print(f"Model Structure: {self}")
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total Parameters: {total_params}")


# Example Implementation for Derived Classes
class ExampleModality(BaseModality):
    """
    Example implementation of a modality extending BaseModality.
    """

    def __init__(self, input_dim=1024, embedding_dim=512, device='cuda'):
        super(ExampleModality, self).__init__(input_dim, embedding_dim, device)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.projection_layer = nn.Linear(128, embedding_dim)  # Projection to shared embedding space

    def preprocess(self, data):
        """
        Normalize data (example preprocessing).
        """
        # Assuming data is a tensor with shape (batch_size, input_dim)
        return data / torch.norm(data, p=2, dim=1, keepdim=True)

    def extract_features(self, preprocessed_data):
        """
        Extract features using the feature extractor.
        """
        return self.feature_extractor(preprocessed_data)


# Test ExampleModality
if __name__ == "__main__":
    # Example data
    example_data = torch.rand(10, 1024)  # Batch of 10 samples with 1024 features

    # Initialize the modality
    modality = ExampleModality(input_dim=1024, embedding_dim=512, device='cpu')
    modality.initialize_parameters()

    # Run the model
    output = modality(example_data)

    # Print summary and output
    modality.summary()
    print("Output Shape:", output.shape)
