# Main script to orchestrate training and evaluation

from src.modality import LanguageModality, ImageModality, VideoModality
from src.fusion.fusion_layer import FusionLayer
from src.loss.contrastive_loss import ContrastiveLoss
from src.trainer.trainer import MultimodalTrainer
from src.utils.data_loader import get_data_loader

def main():
    # Initialize modalities
    model = {
        'text': LanguageModality(),
        'image': ImageModality(),
        'video': VideoModality(),
        # Add other modalities here
    }

    # Fusion layer (optional)
    fusion_layer = FusionLayer(modalities_count=len(model))

    # Loss and optimizer
    loss_fn = ContrastiveLoss()
    optimizer = torch.optim.Adam(fusion_layer.parameters(), lr=0.001)

    # Data loaders
    data_loaders = {
        'text-image': get_data_loader('text_image_dataset'),
        'image-video': get_data_loader('image_video_dataset'),
    }

    # Define training phases
    phases = [
        ('text', 'image'),
        ('image', 'video'),
    ]

    # Trainer
    trainer = MultimodalTrainer(model, fusion_layer, loss_fn, optimizer)
    trainer.train(data_loaders, epochs=5, modality_pairs=phases)

if __name__ == "__main__":
    main()
