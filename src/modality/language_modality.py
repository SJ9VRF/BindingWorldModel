# Language processing class
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class LanguageModality(nn.Module):
    """
    Language Modality Encoder
    Encodes text data into a shared embedding space using a SOTA transformer-based language model.
    """

    def __init__(self, embedding_dim=512, backbone='bert-base-uncased', device='cuda'):
        """
        Initialize the Language Modality Encoder.
        
        Args:
            embedding_dim (int): Output embedding dimension.
            backbone (str): Transformer model backbone (e.g., 'bert-base-uncased', 'roberta-base').
            device (str): Device to run the model ('cuda' or 'cpu').
        """
        super(LanguageModality, self).__init__()
        self.embedding_dim = embedding_dim
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load pre-trained transformer backbone and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.backbone = AutoModel.from_pretrained(backbone)
        self.feature_dim = self.backbone.config.hidden_size  # Extract feature dimension from model config

        # Projection layer to map transformer outputs into shared embedding space
        self.projection_layer = nn.Linear(self.feature_dim, embedding_dim)

        # Move model to the specified device
        self.to(self.device)

    def preprocess(self, texts):
        """
        Preprocess raw text inputs.
        
        Args:
            texts (list of str): List of text strings to preprocess.
        
        Returns:
            Dict: Tokenized inputs ready for the transformer model.
        """
        if not isinstance(texts, list):
            raise ValueError("Input to preprocess must be a list of strings.")
        
        # Tokenize input texts with padding and truncation
        tokenized_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        )
        
        # Move tokenized inputs to the specified device
        tokenized_inputs = {key: val.to(self.device) for key, val in tokenized_inputs.items()}
        return tokenized_inputs

    def extract_features(self, tokenized_inputs):
        """
        Extract features from preprocessed text using the transformer model.
        
        Args:
            tokenized_inputs (Dict): Tokenized inputs from the preprocess method.
        
        Returns:
            Tensor: Raw feature vectors from the transformer model.
        """
        with torch.no_grad():
            outputs = self.backbone(**tokenized_inputs)
        
        # Use the [CLS] token representation for the entire sequence
        return outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, feature_dim)

    def forward(self, texts):
        """
        Forward pass for the language modality encoder.
        
        Args:
            texts (list of str): List of text inputs.
        
        Returns:
            Tensor: Projected features in the shared embedding space.
        """
        tokenized_inputs = self.preprocess(texts)
        raw_features = self.extract_features(tokenized_inputs)
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
        print(f"Language Modality Encoder")
        print(f"Backbone: {self.backbone.config.model_type} ({self.backbone.name_or_path})")
        print(f"Feature Dimension: {self.feature_dim}")
        print(f"Embedding Dimension: {self.embedding_dim}")
        print(f"Device: {self.device}")
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable Parameters: {total_params}")


# Example Usage
if __name__ == "__main__":
    # Example text inputs
    example_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Multimodal learning integrates multiple data types.",
        "Transformers are revolutionizing NLP."
    ]

    # Initialize the language modality
    language_modality = LanguageModality(embedding_dim=512, backbone='bert-base-uncased', device='cuda')

    # Preprocess, encode, and summarize
    language_modality.initialize_parameters()
    embeddings = language_modality(example_texts)

    # Print summary and embeddings
    language_modality.summary()
    print("Output Embedding Shape:", embeddings.shape)
