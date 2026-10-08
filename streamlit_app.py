import streamlit as st
import pandas as pd
import requests


API_URL = "http://127.0.0.1:8000/predict"


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Network Anomaly Detection",
    page_icon="🛡️",
    layout="wide"
)


# =========================================================
# HEADER
# =========================================================

st.title("🛡️ Network Anomaly Detection")

st.markdown(
    """
    **AI-powered network traffic analysis**

    Hybrid Deep Learning model using **VAE + Transformer + Spatial CNN**
    to identify normal and anomalous network activities.
    """
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ System")

    st.success("Model: Loaded")
    st.success("API: FastAPI")
    st.info("Sequence Length: 10")
    st.info("Features: 78")

    st.markdown("---")

    st.markdown(
        """
        **Model Architecture**

        • Variational Autoencoder  
        • Transformer Encoder  
        • Spatial CNN  
        • Fusion Layer  
        • Anomaly Scorer
        """
    )


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📂 Upload CICIDS2017 CSV",
    type=["csv"]
)


if uploaded_file is not None:

    # -----------------------------------------------------
    # Read CSV
    # -----------------------------------------------------

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Information")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Rows",
        f"{len(df):,}"
    )

    col2.metric(
        "Columns",
        len(df.columns)
    )

    col3.metric(
        "Expected Features",
        78
    )


    # -----------------------------------------------------
    # Dataset Preview
    # -----------------------------------------------------

    with st.expander(
        "👁️ View Dataset Preview"
    ):

        st.dataframe(
            df.head(10),
            use_container_width=True
        )


    # -----------------------------------------------------
    # Detect Button
    # -----------------------------------------------------

    if st.button(
        "🔍 Detect Network Anomalies",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Analyzing network traffic... This may take some time for large datasets."
        ):

            uploaded_file.seek(0)

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file,
                    "text/csv"
                )
            }

            try:

                response = requests.post(
                    API_URL,
                    files=files,
                    timeout=600
                )


                # =================================================
                # API ERROR
                # =================================================

                if response.status_code != 200:

                    st.error(
                        f"API Error: {response.text}"
                    )

                    st.stop()


                result = response.json()


                # =================================================
                # SUMMARY
                # =================================================

                total = result["total_sequences"]

                normal = result["normal_count"]

                attacks = result["attack_count"]


                attack_percentage = (
                    attacks / total * 100
                    if total > 0
                    else 0
                )

                normal_percentage = (
                    normal / total * 100
                    if total > 0
                    else 0
                )


                # =================================================
                # DETECTION SUMMARY
                # =================================================

                st.success(
                    "Network traffic analysis completed successfully!"
                )

                st.subheader(
                    "📈 Detection Summary"
                )


                col1, col2, col3, col4 = st.columns(4)


                col1.metric(
                    "Total Sequences",
                    f"{total:,}"
                )

                col2.metric(
                    "Normal Traffic",
                    f"{normal:,}",
                    f"{normal_percentage:.2f}%"
                )

                col3.metric(
                    "Attack Traffic",
                    f"{attacks:,}",
                    f"{attack_percentage:.2f}%"
                )

                col4.metric(
                    "Attack Rate",
                    f"{attack_percentage:.2f}%"
                )


                # =================================================
                # CLASSIFICATION CHART
                # =================================================

                st.subheader(
                    "📊 Traffic Classification"
                )


                chart_data = pd.DataFrame(
                    {
                        "Traffic Type": [
                            "Normal",
                            "Attack"
                        ],
                        "Count": [
                            normal,
                            attacks
                        ]
                    }
                )


                st.bar_chart(
                    chart_data.set_index(
                        "Traffic Type"
                    )
                )


                # =================================================
                # RESULTS DATAFRAME
                # =================================================

                results_df = pd.DataFrame(
                    result["results"]
                )


                if len(results_df) > 0:

                    results_df.insert(
                        0,
                        "Sample",
                        range(
                            1,
                            len(results_df) + 1
                        )
                    )


                    # =================================================
                    # RESULT METRICS
                    # =================================================

                    attack_results = results_df[
                        results_df["prediction"] == "ATTACK"
                    ]


                    normal_results = results_df[
                        results_df["prediction"] == "NORMAL"
                    ]


                    st.subheader(
                        "🔎 Prediction Analysis"
                    )


                    c1, c2, c3 = st.columns(3)


                    c1.metric(
                        "Displayed Results",
                        len(results_df)
                    )

                    c2.metric(
                        "Displayed Attacks",
                        len(attack_results)
                    )

                    c3.metric(
                        "Displayed Normal",
                        len(normal_results)
                    )


                    # =================================================
                    # SUSPICIOUS RECORDS
                    # =================================================

                    st.subheader(
                        "🚨 Suspicious Network Activities"
                    )


                    suspicious_df = results_df[
                        results_df["prediction"] == "ATTACK"
                    ].copy()


                    if len(suspicious_df) > 0:

                        suspicious_df = (
                            suspicious_df
                            .sort_values(
                                "attack_probability",
                                ascending=False
                            )
                        )


                        st.dataframe(
                            suspicious_df,
                            use_container_width=True
                        )

                    else:

                        st.info(
                            "No attack records were detected "
                            "in the displayed prediction results."
                        )


                    # =================================================
                    # ALL PREDICTIONS
                    # =================================================

                    st.subheader(
                        "📋 Prediction Results"
                    )


                    st.dataframe(
                        results_df,
                        use_container_width=True
                    )


                    # =================================================
                    # DOWNLOAD RESULTS
                    # =================================================

                    csv_data = results_df.to_csv(
                        index=False
                    )


                    st.download_button(
                        label="⬇️ Download Prediction Results",
                        data=csv_data,
                        file_name="network_anomaly_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )


            # =====================================================
            # CONNECTION ERROR
            # =====================================================

            except requests.exceptions.ConnectionError:

                st.error(
                    """
                    ❌ Could not connect to FastAPI.

                    Please make sure the FastAPI server is running:

                    `python -m uvicorn app:app --reload`
                    """
                )


            # =====================================================
            # TIMEOUT
            # =====================================================

            except requests.exceptions.Timeout:

                st.error(
                    "⏳ The analysis took too long. Please try again."
                )


            # =====================================================
            # OTHER ERROR
            # =====================================================

            except Exception as e:

                st.error(
                    f"❌ Unexpected error: {str(e)}"
                )