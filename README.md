# Anomaly Detection in Network Activities Using Generative AI-Enhanced Hybrid Deep Learning

A deep learning-based network intrusion detection system that analyzes network traffic and classifies activities as **NORMAL** or **ATTACK**.

The system combines a **Variational Autoencoder (VAE)**, **Transformer**, and **Spatial CNN** to learn reconstruction, temporal, and spatial patterns from network traffic.

## Features

- Network traffic preprocessing
- CICIDS2017 dataset support
- Feature scaling using the training preprocessing artifacts
- Sequential traffic analysis
- VAE-based representation learning
- Transformer-based temporal feature extraction
- CNN-based feature extraction
- Hybrid feature fusion
- Attack probability prediction
- Anomaly score generation
- Ensemble anomaly scoring
- FastAPI prediction API
- Streamlit interactive dashboard
- CSV upload and prediction
- Prediction results visualization

## Architecture

```text
                  Network Traffic
                         │
                         ▼
                  Data Preprocessing
                         │
                         ▼
                Feature Standardization
                         │
                         ▼
                Sequence Generation
                  (10 × 78 features)
                         │
             ┌───────────┼───────────┐
             │           │           │
             ▼           ▼           ▼
            VAE     Transformer   Spatial CNN
             │           │           │
             │           │           │
             └───────────┼───────────┘
                         │
                         ▼
                   Feature Fusion
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       Classification         Anomaly Scorer
              │                     │
              └──────────┬──────────┘
                         ▼
                Ensemble Anomaly Score
                         │
                         ▼
                 NORMAL / ATTACK
```

## Model Components

### 1. Variational Autoencoder

The VAE learns a compact latent representation of the network traffic.

The model uses:

- Input dimension: 78
- Latent dimension: 64
- Hidden dimensions: 256 → 128 → 64

The latent representation is used as one of the inputs to the hybrid model.

### 2. Transformer

The Transformer captures temporal relationships between network traffic observations.

Configuration:

- Sequence length: 10
- `d_model`: 128
- Attention heads: 8
- Transformer layers: 4
- Feed-forward dimension: 512

### 3. Spatial CNN

The CNN extracts local patterns from the sequence of network features.

The convolutional layers use:

- 64 filters
- 128 filters
- 256 filters

The CNN produces classification-related features that are combined with the VAE and Transformer representations.

### 4. Feature Fusion

The VAE latent representation, Transformer features, and CNN classification output are concatenated.

These combined features are passed through fully connected layers to generate the final classification.

### 5. Anomaly Scoring

The model also produces an anomaly score from the VAE latent representation and Transformer features.

The final project evaluation uses an ensemble score:

```text
Ensemble Score =
0.4 × Classification Probability
+
0.4 × Anomaly Score
+
0.2 × Reconstruction Error
```

The final decision is based on the configured threshold.

## Dataset

The project was trained using the CICIDS2017 network intrusion dataset.

The preprocessing pipeline removes non-predictive fields such as:

- Timestamp
- Flow ID
- Source IP
- Destination IP
- Source Port
- Destination Port

The resulting model input contains **78 features**.

The traffic labels are converted into binary classes:

```text
BENIGN → NORMAL
Other attack labels → ATTACK
```

## Preprocessing

The inference pipeline uses the same preprocessing artifacts created during training:

```text
cicids_scaler.pkl
cicids_protocol_encoder.pkl
cicids_feature_names.pkl
```

This ensures that uploaded data is transformed using the same feature order and scaling used during model training.

## Backend

The backend is implemented using FastAPI.

Main endpoint:

```text
POST /predict
```

The API accepts a CSV file and performs:

```text
CSV
 ↓
Preprocessing
 ↓
Scaling
 ↓
Sequence creation
 ↓
Model inference
 ↓
Prediction
```

## Frontend

The Streamlit dashboard provides:

- CSV upload
- Dataset preview
- Detection summary
- Normal traffic count
- Attack count
- Attack percentage
- Prediction results
- Suspicious traffic records
- Prediction download

## Technologies

### Programming

- Python

### Machine Learning / Deep Learning

- PyTorch
- Variational Autoencoder
- Transformer
- Convolutional Neural Network
- Scikit-learn

### Backend

- FastAPI
- Uvicorn

### Frontend

- Streamlit

### Data Processing

- Pandas
- NumPy
- Joblib

### Development

- Git
- GitHub
- VS Code

## Running the Project

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd anomaly-detection
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start FastAPI

```bash
uvicorn app:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

### 5. Start Streamlit

Open another terminal:

```bash
streamlit run streamlit_app.py
```

## Project Workflow

```text
User uploads CICIDS2017 CSV
            ↓
       Streamlit UI
            ↓
       FastAPI API
            ↓
      Data Cleaning
            ↓
       Feature Selection
            ↓
       Feature Scaling
            ↓
     Create 10-row Sequences
            ↓
     Hybrid Deep Learning Model
            ↓
     ┌──────┼──────┐
     ▼      ▼      ▼
    VAE Transformer CNN
     │      │      │
     └──────┼──────┘
            ▼
       Feature Fusion
            ↓
      Anomaly Detection
            ↓
       NORMAL / ATTACK
            ↓
       Dashboard Result
```

## Results

The trained model achieved approximately **92.22% accuracy** during the reported evaluation.

The project also measured inference performance and reported approximately **1.42 ms per sample** under the evaluation setup.

These metrics depend on the dataset split, preprocessing, hardware, and evaluation configuration.

## Future Improvements

- Real-time network traffic monitoring
- Streaming packet analysis
- Improved anomaly-score calibration
- Support for additional intrusion datasets
- Cloud deployment
- Authentication for the API
- Batch prediction optimization
- Monitoring and logging
- Model retraining pipeline

## Author

**Meenuga Usha Rani**

B.Tech Computer Science and Engineering (Artificial Intelligence)

GitHub: https://github.com/usharani963

Portfolio: https://meenuga-portfolio.vercel.app/
