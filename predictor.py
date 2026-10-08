import torch
import torch.nn.functional as F
from model_loader import load_model


model, device = load_model(78)


def predict(X_sequences, batch_size=256):

    results = []

    total_samples = len(X_sequences)

    for start in range(0, total_samples, batch_size):

        end = min(
            start + batch_size,
            total_samples
        )

        X_batch = X_sequences[start:end]

        X_tensor = torch.tensor(
            X_batch,
            dtype=torch.float32
        ).to(device)

        with torch.no_grad():

            outputs = model(X_tensor)

            # Classification probability
            cls_prob = F.softmax(
                outputs["classification"],
                dim=1
            )[:, 1]

            # Anomaly score
            anomaly_score = (
                outputs["anomaly_score"]
                .squeeze(1)
            )

            # Reconstruction error
            reconstruction_error = F.mse_loss(
                outputs["reconstruction"],
                X_tensor[:, -1, :],
                reduction="none"
            ).mean(dim=1)

            # Original project ensemble score
            ensemble_score = (
                0.4 * cls_prob
                + 0.4 * anomaly_score
                + 0.2 * reconstruction_error
            )

            # Final decision
            predictions = (
                ensemble_score > 0.5
            ).long()

        for i in range(len(predictions)):

            prediction = predictions[i].item()

            results.append({
                "prediction": (
                    "ATTACK"
                    if prediction == 1
                    else "NORMAL"
                ),

                "attack_probability": round(
                    cls_prob[i].item(),
                    4
                ),

                "anomaly_score": round(
                    anomaly_score[i].item(),
                    4
                ),

                "reconstruction_error": round(
                    reconstruction_error[i].item(),
                    4
                ),

                "ensemble_score": round(
                    ensemble_score[i].item(),
                    4
                )
            })

        print(
            f"Processed {end}/{total_samples} sequences"
        )

    return results