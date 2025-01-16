# Audio processing class
import torch
import torch.nn as nn
import torchaudio
from torchaudio.models import wav2vec2_base
from torchaudio.transforms import Resample


class AudioModality(nn.Module):
    """
    Audio Modality Encoder
    Encodes audio data into a shared embedding space using a SOTA backbone (e.g., Wav2Vec2).
    """

    def __init__(self, embedding_dim=512, backbone='wav2vec2_base', sample_rate=16000, device='cuda'):
        """
        Initialize the Audio Modality Encoder.
        
        Args:
            embedding_dim (int): Output embedding dimension.
            backbone (str): Backbone model (e.g., 'wav2vec2_base').
            sample_rate (int): Target sample rate for audio preprocessing.
            device (str): Device to run the model ('cuda' or 'cpu').
        """
        super(AudioModality, self).__init__()
        self.embedding_dim = embedding_dim
        self.sample_rate = sample_rate
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Backbone Model for Feature Extraction
        if backbone == 'wav2vec2_base':
            self.backbone = wav2vec2_base(num_out=768)
            self.feature_dim = 768  # Wav2Vec2 output dimension
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Projection Layer to map features into shared embedding space
        self.projection_layer = nn.Linear(self.feature_dim, embedding_dim)

        # Audio Preprocessing Pipeline
        self.resample = Resample(orig_freq=None, new_freq=sample_rate)  # Resample to target sample rate

        # Move model to device
        self.to(self.device)

    def preprocess(self, waveform, orig_sample_rate):
        """
        Preprocess raw audio waveform.
        
        Args:
            waveform (Tensor): Raw audio waveform (shape: [batch_size, num_channels, num_samples]).
            orig_sample_rate (int): Original sample rate of the waveform.
        
        Returns:
            Tensor: Resampled and normalized waveform ready for feature extraction.
        """
        # Resample waveform to target sample rate if necessary
        if orig_sample_rate != self.sample_rate:
            waveform = self.resample(waveform)

        # Normalize waveform to [-1, 1]
        waveform = (waveform - waveform.mean(dim=-1, keepdim=True)) / (waveform.std(dim=-1, keepdim=True) + 1e-7)
        return waveform.to(self.device)

    def extract_features(self, preprocessed_waveform):
        """
        Extract features from preprocessed audio using the backbone model.
        
        Args:
            preprocessed_waveform (Tensor): Preprocessed audio waveform (shape: [batch_size, num_samples]).
        
        Returns:
            Tensor: Raw feature vectors from the backbone model.
        """
        with torch.no_grad():
            # Forward pass through the backbone model
            features = self.backbone(preprocessed_waveform)
        return features

    def forward(self, waveform, orig_sample_rate):
        """
        Forward pass for the audio modality encoder.
        
        Args:
            waveform (Tensor): Raw audio waveform (shape: [batch_size, num_channels, num_samples]).
            orig_sample_rate (int): Original sample rate of the waveform.
        
        Returns:
            Tensor: Projected features in the shared embedding space.
        """
        preprocessed_waveform = self.preprocess(waveform, orig_sample_rate)
        raw_features = self.extract_features(preprocessed_waveform)
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
        print(f"Audio Modality Encoder")
        print(f"Backbone: {self.backbone.__class__.__name__}")
        print(f"Feature Dimension: {self.feature_dim}")
        print(f"Embedding Dimension: {self.embedding_dim}")
        print(f"Sample Rate: {self.sample_rate}")
        print(f"Device: {self.device}")
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable Parameters: {total_params}")


# Example Usage
if __name__ == "__main__":
    # Example audio data (simulate with random waveform)
    batch_size = 2
    num_channels = 1
    num_samples = 16000 * 5  # 5 seconds of audio at 16kHz
    example_waveform = torch.randn(batch_size, num_channels, num_samples)
    example_sample_rate = 44100  # Original sample rate

    # Initialize the audio modality
    audio_modality = AudioModality(embedding_dim=512, backbone='wav2vec2_base', sample_rate=16000, device='cuda')

    # Preprocess, encode, and summarize
    audio_modality.initialize_parameters()
    embeddings = audio_modality(example_waveform, orig_sample_rate=example_sample_rate)

    # Print summary and embeddings
    audio_modality.summary()
    print("Output Embedding Shape:", embeddings.shape)
