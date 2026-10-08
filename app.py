from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import io

from preprocess import CICIDSPreprocessor, create_sequences
from predictor import predict


app = FastAPI(
    title="CICIDS2017 Anomaly Detection API",
    description="Hybrid VAE + Transformer + Spatial CNN",
    version="1.0.0"
)


# Load preprocessing artifacts once
preprocessor = CICIDSPreprocessor()


@app.get("/")
def home():
    return {
        "message": "CICIDS2017 Anomaly Detection API is running",
        "status": "success"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
async def predict_csv(
    file: UploadFile = File(...)
):

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file."
        )

    try:

        # -----------------------------
        # Read CSV
        # -----------------------------
        contents = await file.read()

        df = pd.read_csv(
            io.BytesIO(contents)
        )

        original_rows = len(df)

        # -----------------------------
        # Preprocess
        # -----------------------------
        X_scaled = preprocessor.preprocess(df)

        # -----------------------------
        # Create sequences
        # -----------------------------
        X_sequences = create_sequences(
            X_scaled,
            sequence_length=10
        )

        # -----------------------------
        # Prediction
        # -----------------------------
        results = predict(X_sequences)

        # -----------------------------
        # Summary
        # -----------------------------
        total_sequences = len(results)

        attack_count = sum(
            result["prediction"] == "ATTACK"
            for result in results
        )

        normal_count = (
            total_sequences - attack_count
        )

        attack_percentage = (
            attack_count / total_sequences * 100
            if total_sequences > 0
            else 0
        )

        # -----------------------------
        # Return complete results
        # -----------------------------
        return {
            "filename": file.filename,
            "original_rows": original_rows,
            "processed_rows": len(X_scaled),
            "total_sequences": total_sequences,
            "normal_count": normal_count,
            "attack_count": attack_count,
            "attack_percentage": round(
                attack_percentage,
                2
            ),
            "results": results
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )