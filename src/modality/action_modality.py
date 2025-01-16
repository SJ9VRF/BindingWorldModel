# Action sequence processing class
import torch
import torch.nn as nn


class ActionModality(nn.Module):
    """
    Action Modality Encoder
    Encodes sequential action data into a shared embedding space using SOTA sequence models.
    """

    def __init__(self, input_dim=10, embedding_dim=512, hidden_dim=256, num_layers=2, model_type='transformer', device='cuda'):
        """
        Initialize the Action Modality Encoder.
        
        Args:
            input_dim (int): Number of features per timestep in the action sequence.
            embedding_dim (int): Output embedding dimension.
            hidden_dim (int): Hidden dimension for the sequence model.
            num_layers (int): Number of layers in the sequence model.
            model_type (str): Sequence model type ('transformer' or 'lstm').
            device (str): Device to run the model ('cuda' or 'cpu').
        """
        super(ActionModality, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Sequence Model (Transformer or LSTM)
        if model_type == 'transformer':
            self.sequence_model = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, dim_feedforward=hidden_dim),
                num_layers=num_layers
            )
        elif model_type == 'lstm':
            self.sequence_model = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        # Projection Layer to map sequence features into the shared embedding space
        self.projection_layer = nn.Linear(hidden_dim * (2 if model_type == 'lstm' else 1), embedding_dim)

        # Move model to device
        self.to(self.device)

    def preprocess(self, action_sequence):
        """
        Preprocess raw action sequence data.
        
        Args:
            action_sequence (Tensor): Raw action sequence data of shape (batch_size, sequence_length, input_dim).
        
        Returns:
            Tensor: Normalized action sequence data.
        """
        if not isinstance(action_sequence, torch.Tensor):
            raise ValueError("Action sequence data must be a PyTorch Tensor.")

        # Normalize each feature across the sequence
        mean = torch.mean(action_sequence, dim=1, keepdim=True)  # Shape: (batch_size, 1, input_dim)
        std = torch.std(action_sequence, dim=1, keepdim=True) + 1e-7  # Avoid division by zero
        normalized_sequence = (action_sequence - mean) / std
        return normalized_sequence.to(self.device)

    def extract_features(self, preprocessed_sequence):
        """
        Extract features from preprocessed action sequence data.
        
        Args:
            preprocessed_sequence (Tensor): Normalized sequence data of shape (batch_size, sequence_length, input_dim).
        
        Returns:
            Tensor: Encoded sequence features.
        """
        if isinstance(self.sequence_model, nn.LSTM):
            # LSTM: Extract features from the last hidden state
            _, (hidden, _) = self.sequence_model(preprocessed_sequence)
            features = hidden[-2:] if self.sequence_model.bidirectional else hidden[-1:]  # Bidirectional: use both directions
            features = features.transpose(0, 1).reshape(preprocessed_sequence.size(0), -1)  # Flatten hidden states
        else:
            # Transformer: Global average pooling of all sequence outputs
            features = self.sequence_model(preprocessed_sequence.transpose(0, 1))  # Shape: (sequence_length, batch_size, hidden_dim)
            features = features.mean(dim=0)  # Average across the sequence length
        return features

    def forward(self, action_sequence):
        """
        Forward pass for the action modality encoder.
        
        Args:
            action_sequence (Tensor): Raw action sequence data of shape (batch_size, sequence_length, input_dim).
        
        Returns:
            Tensor: Projected features in the shared embedding space.
        """
        preprocessed_sequence = self.preprocess(action_sequence)
        raw_features = self.extract_features(preprocessed_sequence)
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
        print(f"Action Modality Encoder")
        print(f"Input Dimension: {self.input_dim}")
        print(f"Hidden Dimension: {self.hidden_dim}")
        print(f"Embedding Dimension: {self.embedding_dim}")
        print(f"Model Type: {'Transformer' if isinstance(self.sequence_model, nn.TransformerEncoder) else 'LSTM'}")
        print(f"Device: {self.device}")
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable Parameters: {total_params}")


# Example Usage
if __name__ == "__main__":
    # Example action data (batch of 4 samples, sequence length of 20, 10 features per timestep)
    batch_size = 4
    sequence_length = 20
    input_dim = 10
    example_action_sequence = torch.rand(batch_size, sequence_length, input_dim)

    # Initialize the action modality
    action_modality = ActionModality(
        input_dim=input_dim, embedding_dim=512, hidden_dim=256, num_layers=2, model_type='transformer', device='cuda'
    )

    # Preprocess, encode, and summarize
    action_modality.initialize_parameters()
    embeddings = action_modality(example_action_sequence)

    # Print summary and embeddings
    action_modality.summary()
    print("Output Embedding Shape:", embeddings.shape)
