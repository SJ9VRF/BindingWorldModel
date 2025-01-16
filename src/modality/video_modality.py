# Video processing class

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models.video import r3d_18
from timm import create_model  # For TimeSformer

class VideoModality(nn.Module):
    """
    Video Modality Encoder
    Encodes video data into a shared embedding space using SOTA video models (e.g., ResNet3D, TimeSformer).
    """

    def __init__(self, embedding_dim=512, backbone='r3d_18', frame_size=(224, 224), num_frames=16, device='cuda'):
        """
        Initialize the Video Modality Encoder.
        
        Args:
            embedding_dim (int): Output embedding dimension.
            backbone (str): Backbone model ('r3d_18' or 'timesformer').
            frame_size (tuple): Target frame size for preprocessing (height, width).
            num_frames (int): Number of frames to sample from each video.
            device (str): Device to run the model ('cuda' or 'cpu').
        """
        super(VideoModality, self).__init__()
        self.embedding_dim = embedding_dim
        self.frame_size = frame_size
        self.num_frames = num_frames
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Backbone Model for Feature Extraction
        if backbone == 'r3d_18':
            self.backbone = r3d_18(pretrained=True)
            self.feature_dim = self.backbone.fc.in_features  # Output dimension of ResNet3D
            self.backbone.fc = nn.Identity()  # Remove classification head
        elif backbone == 'timesformer':
            self.backbone = create_model('timesformer', pretrained=True)
            self.feature_dim = self.backbone.embed_dim  # Output dimension of TimeSformer
            self.backbone.head = nn.Identity()  # Remove classification head
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Projection Layer to map features into shared embedding space
        self.projection_layer = nn.Linear(self.feature_dim, embedding_dim)

        # Video Preprocessing Pipeline
        self.preprocess_pipeline = T.Compose([
            T.Resize(self.frame_size),  # Resize frames to target size
            T.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])  # Video normalization
        ])

        # Move model to device
        self.to(self.device)

    def preprocess(self, video):
        """
        Preprocess raw video data.
        
        Args:
            video (Tensor): Raw video tensor of shape (batch_size, num_frames, channels, height, width).
        
        Returns:
            Tensor: Preprocessed video tensor.
        """
        if not isinstance(video, torch.Tensor):
            raise ValueError("Video data must be a PyTorch Tensor.")
        
        # Normalize each frame using the preprocessing pipeline
        batch_size, num_frames, channels, height, width = video.shape
        video = video.view(-1, channels, height, width)  # Reshape to (batch_size * num_frames, channels, height, width)
        video = self.preprocess_pipeline(video)
        video = video.view(batch_size, num_frames, channels, self.frame_size[0], self.frame_size[1])
        return video.to(self.device)

    def extract_features(self, preprocessed_video):
        """
        Extract features from preprocessed video data using the backbone model.
        
        Args:
            preprocessed_video (Tensor): Preprocessed video tensor.
        
        Returns:
            Tensor: Raw feature vectors from the backbone model.
        """
        with torch.no_grad():
            features = self.backbone(preprocessed_video)
        return features

    def forward(self, video):
        """
        Forward pass for the video modality encoder.
        
        Args:
            video (Tensor): Raw video tensor of shape (batch_size, num_frames, channels, height, width).
        
        Returns:
            Tensor: Projected features in the shared embedding space.
        """
        preprocessed_video = self.preprocess(video)
        raw_features = self.extract_features(preprocessed_video)
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
        print(f"Video Modality Encoder")
        print(f"Backbone: {self.backbone.__class__.__name__}")
        print(f"Feature Dimension: {self.feature_dim}")
        print(f"Embedding Dimension: {self.embedding_dim}")
        print(f"Frame Size: {self.frame_size}")
        print(f"Number of Frames: {self.num_frames}")
        print(f"Device: {self.device}")
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable Parameters: {total_params}")


# Example Usage
if __name__ == "__main__":
    # Example video data (batch of 2 videos, 16 frames, 3 channels, 256x256 resolution)
    batch_size = 2
    num_frames = 16
    channels = 3
    height, width = 256, 256
    example_video = torch.rand(batch_size, num_frames, channels, height, width)

    # Initialize the video modality
    video_modality = VideoModality(embedding_dim=512, backbone='r3d_18', frame_size=(224, 224), num_frames=16, device='cuda')

    # Preprocess, encode, and summarize
    video_modality.initialize_parameters()
    embeddings = video_modality(example_video)

    # Print summary and embeddings
    video_modality.summary()
    print("Output Embedding Shape:", embeddings.shape)
