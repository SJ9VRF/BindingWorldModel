# Point cloud processing class

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
    """
    PointNet Encoder: Encodes point cloud data into high-dimensional feature vectors.
    """

    def __init__(self, feature_dim=1024):
        super(PointNetEncoder, self).__init__()
        self.feature_dim = feature_dim

        # MLP Layers for Point Feature Extraction
        self.conv1 = nn.Conv1d(3, 64, 1)  # Input: (x, y, z)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, feature_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(feature_dim)

    def forward(self, x):
        """
        Forward pass for PointNetEncoder.
        
        Args:
            x (Tensor): Input point cloud of shape (batch_size, num_points, 3).
        
        Returns:
            Tensor: Encoded feature vector of shape (batch_size, feature_dim).
        """
        x = x.transpose(1, 2)  # Transpose to (batch_size, 3, num_points)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))  # Shape: (batch_size, feature_dim, num_points)
        x = torch.max(x, dim=2)[0]  # Max pooling across points, Shape: (batch_size, feature_dim)
        return x


class PointCloudModality(nn.Module):
    """
    Point Cloud Modality Encoder
    Encodes point cloud data into a shared embedding space using PointNet.
    """

    def __init__(self, embedding_dim=512, feature_dim=1024, device='cuda'):
        """
        Initialize the Point Cloud Modality Encoder.
        
        Args:
            embedding_dim (int): Output embedding dimension.
            feature_dim (int): Dimensionality of features extracted by the encoder.
            device (str): Device to run the model ('cuda' or 'cpu').
        """
        super(PointCloudModality, self).__init__()
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # PointNet Encoder
        self.encoder = PointNetEncoder(feature_dim=feature_dim)

        # Projection Layer to map features into shared embedding space
        self.projection_layer = nn.Linear(feature_dim, embedding_dim)

        # Move model to device
        self.to(self.device)

    def preprocess(self, point_cloud):
        """
        Preprocess raw point cloud data.
        
        Args:
            point_cloud (Tensor): Raw point cloud data of shape (batch_size, num_points, 3).
        
        Returns:
            Tensor: Preprocessed point cloud data.
        """
        if not isinstance(point_cloud, torch.Tensor):
            raise ValueError("Point cloud data must be a PyTorch Tensor.")

        # Normalize point cloud to zero mean and unit variance
        centroid = torch.mean(point_cloud, dim=1, keepdim=True)  # Shape: (batch_size, 1, 3)
        point_cloud = point_cloud - centroid
        scale = torch.max(torch.norm(point_cloud, p=2, dim=2, keepdim=True), dim=1, keepdim=True)[0]
        point_cloud = point_cloud / scale
        return point_cloud.to(self.device)

    def extract_features(self, preprocessed_point_cloud):
        """
        Extract features from preprocessed point cloud data.
        
        Args:
            preprocessed_point_cloud (Tensor): Preprocessed point cloud data of shape (batch_size, num_points, 3).
        
        Returns:
            Tensor: Raw feature vectors of shape (batch_size, feature_dim).
        """
        return self.encoder(preprocessed_point_cloud)

    def forward(self, point_cloud):
        """
        Forward pass for the point cloud modality encoder.
        
        Args:
            point_cloud (Tensor): Raw point cloud data of shape (batch_size, num_points, 3).
        
        Returns:
            Tensor: Projected features in the shared embedding space.
        """
        preprocessed_point_cloud = self.preprocess(point_cloud)
        raw_features = self.extract_features(preprocessed_point_cloud)
        projected_features = self.projection_layer(raw_features)
        return torch.nn.functional.normalize(projected_features, p=2, dim=1)

    def initialize_parameters(self):
        """
        Initializes parameters of the projection layer using Xavier initialization.
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
        print(f"Point Cloud Modality Encoder")
        print(f"Encoder: PointNet")
        print(f"Feature Dimension: {self.feature_dim}")
        print(f"Embedding Dimension: {self.embedding_dim}")
        print(f"Device: {self.device}")
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable Parameters: {total_params}")


# Example Usage
if __name__ == "__main__":
    # Example point cloud data (batch of 2, 1024 points each, 3 coordinates per point)
    batch_size = 2
    num_points = 1024
    example_point_cloud = torch.rand(batch_size, num_points, 3)

    # Initialize the point cloud modality
    point_cloud_modality = PointCloudModality(embedding_dim=512, feature_dim=1024, device='cuda')

    # Preprocess, encode, and summarize
    point_cloud_modality.initialize_parameters()
    embeddings = point_cloud_modality(example_point_cloud)

    # Print summary and embeddings
    point_cloud_modality.summary()
    print("Output Embedding Shape:", embeddings.shape)
