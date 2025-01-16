# Touch interaction processing class
import torch
import torch.nn as nn
import torch.nn.functional as F


class TouchModality(nn.Module):
    """
    Touch Modality Encoder
    Encodes tactile sensory data into a shared embedding space.
    """

    def __init__(self, input_dim=6, embedding_dim=512, feature_dim=256, device='cuda'):
        """
        Initialize the Touch Modality Encoder.
        
        Args:
            input_dim (int): Number of input features (e.g., pressure, temperature, shear).
            embedding_dim (int): Output embedding dimension.
            feature_dim (int): Dimensionality of features extracted by the encoder.
            device (str): Device to run the model ('cuda' or 'cpu').
        """
        super(TouchModality, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Feature Extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
            nn.ReLU()
        )

        # Projection Layer to map features into shared embedding space
        self.projection_layer = nn.Linear(feature_dim, embedding_dim)

        # Move model to device
        self.to(self.device)

    def preprocess(self, touch_data):
        """
        Preprocess raw touch sensory data.
        
        Args:
            touch_data (Tensor): Raw touch data of shape (batch_size, num_sensors, input_dim).
        
        Returns:
            Tensor: Normalized touch data.
        """
        if not isinstance(touch_data, torch.Tensor):
            raise ValueError("Touch data must be a PyTorch Tensor.")

        # Normalize touch data to zero mean and unit variance
        mean = torch.mean(touch_data, dim=1, keepdim=True)  # Shape: (batch_size, 1, input_dim)
        std = torch.std(touch_data, dim=1, keepdim=True) + 1e-7  # Avoid division by zero
        normalized_data = (touch_data - mean) / std
        return normalized_data.to(self.device)

    def extract_features(self, preprocessed_data):
        """
        Extract features from preprocessed touch sensory data.
        
        Args:
            preprocessed_data (Tensor): Normalized touch data of shape (batch_size, num_sensors, input_dim).
        
        Returns:
            Tensor: Feature vectors of shape (batch_size, feature_dim).
        """
        # Flatten across the sensor axis and extract features
        batch_size, num_sensors, input_dim = preprocessed_data.shape
        flattened_data = preprocessed_data.view(batch_size * num_sensors, input_dim)  # Flatten
        features = self.feature_extractor(flattened_data)
        features = features.view(batch_size, num_sensors, -1)  # Reshape back
        aggregated_features = torch.max(features, dim=1)[0]  # Max pooling across sensors
        return aggregated_features

    def forward(self, touch_data):
        """
        Forward pass for the touch modality encoder.
        
        Args:
            touch_data (Tensor): Raw touch data of shape (batch_size, num_sensors, input_dim).
        
        Returns:
            Tensor: Projected features in the shared embedding space.
        """
        preprocessed_data = self.preprocess(touch_data)
        raw_features = self.extract_features(preprocessed_data)
        projected_features = self.projection_layer(raw_features)
        return torch.nn.functional.normalize(projected_features, p=2, dim=1)

    def initialize_parameters(self):
        """
        Initializes parameters of the model using Xavier initialization.
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
        print(f"Touch Modality Encoder")
        print(f"Input Dimension: {self.input_dim}")
        print(f"Feature Dimension: {self.feature_dim}")
        print(f"Embedding Dimension: {self.embedding_dim}")
        print(f"Device: {self.device}")
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable Parameters: {total_params}")


# Example Usage
if __name__ == "__main__":
    # Example touch data (batch of 4 samples, 10 sensors, 6 features per sensor)
    batch_size = 4
    num_sensors = 10
    input_dim = 6
    example_touch_data = torch.rand(batch_size, num_sensors, input_dim)

    # Initialize the touch modality
    touch_modality = TouchModality(input_dim=input_dim, embedding_dim=512, feature_dim=256, device='cuda')

    # Preprocess, encode, and summarize
    touch_modality.initialize_parameters()
    embeddings = touch_modality(example_touch_data)

    # Print summary and embeddings
    touch_modality.summary()
    print("Output Embedding Shape:", embeddings.shape)
