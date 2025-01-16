# Image processing class
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from timm import create_model  # PyTorch Image Models (timm)

class ImageModality(nn.Module):
    """
    Image Modality Encoder
    Encodes images into a shared embedding space using a SOTA backbone (EfficientNet or ViT).
    """

    def __init__(self, embedding_dim=512, backbone='efficientnet_b3', device='cuda'):
        """
        Initialize the Image Modality Encoder.
        
        Args:
            embedding_dim (int): Output embedding dimension.
            backbone (str): Backbone model ('efficientnet_b3', 'vit_base_patch16_224', etc.).
            device (str): Device to run the model ('cuda' or 'cpu').
        """
        super(ImageModality, self).__init__()
        self.embedding_dim = embedding_dim
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Backbone Model for Feature Extraction
        if backbone == 'efficientnet_b3':
            self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
            self.feature_dim = self.backbone.classifier[1].in_features  # EfficientNet output dim
            self.backbone.classifier = nn.Identity()  # Remove classification head
        elif backbone == 'vit_base_patch16_224':
            self.backbone = create_model('vit_base_patch16_224', pretrained=True)
            self.feature_dim = self.backbone.head.in_features  # ViT output dim
            self.backbone.head = nn.Identity()  # Remove classification head
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Projection Layer to map features into shared embedding space
        self.projection_layer = nn.Linear(self.feature_dim, embedding_dim)

        # Image Preprocessing Pipeline
        self.preprocess_pipeline = T.Compose([
            T.Resize((224, 224)),  # Resize images to match model input size
            T.ToTensor(),  # Convert PIL image or numpy array to tensor
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])

        # Move model to device
        self.to(self.device)

    def preprocess(self, images):
        """
        Preprocess raw images.
        
        Args:
            images (list of PIL Images or Torch Tensors): Raw image inputs.
        
        Returns:
            Tensor: Preprocessed image tensor ready for feature extraction.
        """
        if isinstance(images, list):  # Handle batch of images
            images = [self.preprocess_pipeline(img) for img in images]
            images = torch.stack(images)
        elif isinstance(images, torch.Tensor):  # Assume already in tensor form
            images = images / 255.0  # Normalize pixel values to [0, 1]
            images = self.preprocess_pipeline(images)
        else:
            raise ValueError("Images must be a list of PIL Images or a Torch Tensor.")
        return images.to(self.device)

    def extract_features(self, preprocessed_images):
        """
        Extract features from preprocessed images using the backbone model.
        
        Args:
            preprocessed_images (Tensor): Preprocessed image tensor.
        
        Returns:
            Tensor: Raw feature vectors from the backbone.
        """
        with torch.no_grad():
            features = self.backbone(preprocessed_images)
        return features

    def forward(self, images):
        """
        Forward pass for the image modality encoder.
        
        Args:
            images: Raw image inputs.
        
        Returns:
            Tensor: Projected features in the shared embedding space.
        """
        preprocessed_images = self.preprocess(images)
        raw_features = self.extract_features(preprocessed_images)
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
        print(f"Image Modality Encoder")
        print(f"Backbone: {self.backbone.__class__.__name__}")
        print(f"Feature Dimension: {self.feature_dim}")
        print(f"Embedding Dimension: {self.embedding_dim}")
        print(f"Device: {self.device}")
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable Parameters: {total_params}")
