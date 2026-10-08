import torch

from preprocess import preprocess_csv
from model_loader import load_model


CSV_PATH = "CIC_test.csv"

# -----------------------------
# 1. Preprocess CSV
# -----------------------------
X_sequences = preprocess_csv(
    CSV_PATH,
    sequence_length=10
)

print("\nPreprocessing successful!")
print("Input shape:", X_sequences.shape)


# -----------------------------
# 2. Load trained model
# -----------------------------
model, device = load_model(78)

print("\nModel loaded successfully!")
print("Device:", device)


# -----------------------------
# 3. Take a small batch
# -----------------------------
batch_size = 32

X_batch = X_sequences[:batch_size]

X_tensor = torch.tensor(
    X_batch,
    dtype=torch.float32
).to(device)

print("\nModel input shape:", X_tensor.shape)


# -----------------------------
# 4. Run prediction
# -----------------------------
with torch.no_grad():

    outputs = model(X_tensor)


# -----------------------------
# 5. Read outputs
# -----------------------------
classification = outputs["classification"]

anomaly_score = outputs["anomaly_score"]

reconstruction = outputs["reconstruction"]


print("\n===== MODEL OUTPUT =====")

print(
    "Classification shape:",
    classification.shape
)

print(
    "Anomaly score shape:",
    anomaly_score.shape
)

print(
    "Reconstruction shape:",
    reconstruction.shape
)


# -----------------------------
# 6. Classification probability
# -----------------------------
probabilities = torch.softmax(
    classification,
    dim=1
)

attack_probability = probabilities[:, 1]


# -----------------------------
# 7. Predictions
# -----------------------------
predictions = (
    attack_probability >= 0.5
).long()


print("\n===== PREDICTIONS =====")

for i in range(min(10, batch_size)):

    label = (
        "ATTACK"
        if predictions[i].item() == 1
        else "NORMAL"
    )

    print(
        f"Sample {i + 1}: "
        f"{label} | "
        f"Attack Probability: "
        f"{attack_probability[i].item():.4f} | "
        f"Anomaly Score: "
        f"{anomaly_score[i].item():.4f}"
    )