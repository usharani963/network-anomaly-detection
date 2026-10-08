import os
import joblib
import numpy as np
import pandas as pd


class CICIDSPreprocessor:

    def __init__(
        self,
        scaler_path="cicids_scaler.pkl",
        protocol_encoder_path="cicids_protocol_encoder.pkl",
        feature_names_path="cicids_feature_names.pkl"
    ):
        self.scaler_path = scaler_path
        self.protocol_encoder_path = protocol_encoder_path
        self.feature_names_path = feature_names_path

        # Load preprocessing artifacts
        self.scaler = joblib.load(self.scaler_path)
        self.protocol_encoder = joblib.load(
            self.protocol_encoder_path
        )
        self.feature_names = joblib.load(
            self.feature_names_path
        )

        print("Preprocessing artifacts loaded successfully!")
        print("Number of features:", len(self.feature_names))

    def preprocess(self, df):
        """
        Preprocess one CICIDS2017 dataframe.

        Returns:
            X_scaled: numpy array with shape (n_samples, 78)
        """

        # -----------------------------------
        # 1. Clean column names
        # -----------------------------------
        df = df.copy()
        df.columns = df.columns.str.strip().str.lower()

        # -----------------------------------
        # 2. Remove duplicates
        # -----------------------------------
        df = df.drop_duplicates()

        # -----------------------------------
        # 3. Remove infinity and NaN
        # -----------------------------------
        df = df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        df = df.dropna()

        # -----------------------------------
        # 4. Remove non-feature columns
        # -----------------------------------
        drop_cols = [
            "timestamp",
            "flow id",
            "src ip",
            "src port",
            "dst ip",
            "dst port",
            "datetime",
            "label"
        ]

        X = df.drop(
            columns=[
                col
                for col in drop_cols
                if col in df.columns
            ],
            errors="ignore"
        )

        # -----------------------------------
        # 5. Handle Protocol
        # -----------------------------------
        if "protocol" in X.columns:

            if X["protocol"].dtype == "object":

                X["protocol"] = (
                    self.protocol_encoder.transform(
                        X["protocol"]
                    )
                )

        # -----------------------------------
        # 6. Ensure exact training columns
        # -----------------------------------

        # Add missing columns
        for column in self.feature_names:

            if column not in X.columns:
                X[column] = 0

        # Keep only training columns
        X = X[self.feature_names]

        # -----------------------------------
        # 7. Convert everything to numeric
        # -----------------------------------
        X = X.apply(
            pd.to_numeric,
            errors="coerce"
        )

        X = X.replace(
            [np.inf, -np.inf],
            np.nan
        )

        X = X.fillna(0)

        # -----------------------------------
        # 8. Use the TRAINING scaler
        # -----------------------------------
        X_scaled = self.scaler.transform(X)

        # -----------------------------------
        # 9. Convert to float32
        # -----------------------------------
        X_scaled = np.asarray(
            X_scaled,
            dtype=np.float32
        )

        print("Preprocessing complete!")
        print("Input shape:", X.shape)
        print("Scaled shape:", X_scaled.shape)

        return X_scaled


def create_sequences(
    X,
    sequence_length=10
):
    """
    Convert feature rows into sequences.

    Example:

        X = (100, 78)

        sequence_length = 10

        Output = (91, 10, 78)
    """

    if len(X) < sequence_length:

        raise ValueError(
            f"Need at least {sequence_length} rows, "
            f"but received only {len(X)} rows."
        )

    sequences = []

    for i in range(
        len(X) - sequence_length + 1
    ):

        sequences.append(
            X[i:i + sequence_length]
        )

    return np.array(
        sequences,
        dtype=np.float32
    )


def preprocess_csv(
    csv_path,
    sequence_length=10
):
    """
    Complete preprocessing pipeline:

    CSV
      ↓
    Cleaning
      ↓
    78 features
      ↓
    Training scaler
      ↓
    Sequences of 10
      ↓
    Model input
    """

    # -----------------------------------
    # 1. Check CSV
    # -----------------------------------
    if not os.path.exists(csv_path):

        raise FileNotFoundError(
            f"CSV file not found: {csv_path}"
        )

    # -----------------------------------
    # 2. Load CSV
    # -----------------------------------
    df = pd.read_csv(csv_path)

    print("CSV loaded successfully!")
    print("Original shape:", df.shape)

    # -----------------------------------
    # 3. Create preprocessor
    # -----------------------------------
    preprocessor = CICIDSPreprocessor()

    # -----------------------------------
    # 4. Preprocess
    # -----------------------------------
    X_scaled = preprocessor.preprocess(df)

    # -----------------------------------
    # 5. Create sequences
    # -----------------------------------
    X_sequences = create_sequences(
        X_scaled,
        sequence_length
    )

    print(
        "Sequence shape:",
        X_sequences.shape
    )

    return X_sequences