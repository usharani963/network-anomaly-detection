import torch

from hybrid_model import HybridAnomalyDetector


# ============================================================
# Model Configuration
# ============================================================

MODEL_PATH = "best_model_CICIDS2017.pth"

SEQUENCE_LENGTH = 10
NUM_CLASSES = 2
LATENT_DIM = 64


# ============================================================
# Load Trained Model
# ============================================================

def load_model(input_dim: int):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Create the same architecture used during training
    model = HybridAnomalyDetector(
        input_dim=input_dim,
        sequence_length=SEQUENCE_LENGTH,
        num_classes=NUM_CLASSES,
        latent_dim=LATENT_DIM
    )

    # Load your already-trained weights
    state_dict = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(state_dict)

    model.to(device)

    # Important for inference
    model.eval()

    print("Model loaded successfully!")

    return model, device