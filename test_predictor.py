from preprocess import preprocess_csv
from predictor import predict


CSV_PATH = "CIC_test.csv"


# -----------------------------
# Preprocessing
# -----------------------------
X_sequences = preprocess_csv(
    CSV_PATH,
    sequence_length=10
)

print("\nPreprocessing complete.")


# -----------------------------
# Test only first 32 samples
# -----------------------------
X_test = X_sequences[:32]


# -----------------------------
# Prediction
# -----------------------------
results = predict(X_test)


# -----------------------------
# Display results
# -----------------------------
print("\n===== PREDICTION RESULTS =====")

for i, result in enumerate(results[:10]):

    print(
        f"Sample {i + 1}: "
        f"{result['prediction']} | "
        f"Attack Probability: "
        f"{result['attack_probability']} | "
        f"Anomaly Score: "
        f"{result['anomaly_score']}"
    )