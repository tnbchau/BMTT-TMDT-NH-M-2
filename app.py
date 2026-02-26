import streamlit as st
import pandas as pd
import os

# =============================
# IMPORT PROJECT MODULES
# =============================
from src.data.preprocessing import run_preprocessing
from src.features.feature_extraction import FeatureExtractor, FeatureAdapter
from src.features.sequence_extraction import SequenceExtractor
from src.models.ml_experiment import MLExperiment
from src.models.dl_experiment import LSTMExperiment

# =============================
# CONFIG
# =============================
st.set_page_config(
    page_title="Phishing Email Detection",
    page_icon="📧",
    layout="centered"
)

MODEL_DIR = "models"
FEATURE_DIR = os.path.join(MODEL_DIR, "features")

# =============================
# LOAD MODELS
# =============================
@st.cache_resource
def load_feature_tools():
    extractor = FeatureExtractor.load(
        os.path.join(FEATURE_DIR, "feature_extractor.pkl")
    )
    seq_extractor = SequenceExtractor.load(
        os.path.join(FEATURE_DIR, "sequence_extractor.pkl")
    )
    return extractor, seq_extractor


@st.cache_resource
def load_ml_models():
    return {
        "svm": MLExperiment.load(os.path.join(MODEL_DIR, "svm_tuned.pkl")),
        "logistic": MLExperiment.load(os.path.join(MODEL_DIR, "logistic_tuned.pkl")),
        "random_forest": MLExperiment.load(os.path.join(MODEL_DIR, "random_forest_tuned.pkl")),
        "naive_bayes": MLExperiment.load(os.path.join(MODEL_DIR, "naive_bayes_tuned.pkl")),
    }


@st.cache_resource
def load_lstm():
    return LSTMExperiment.load(os.path.join(MODEL_DIR, "lstm_tuned"))


feature_extractor, seq_extractor = load_feature_tools()
ml_models = load_ml_models()
lstm_model = load_lstm()

# =============================
# SAMPLE EMAILS
# =============================
PHISHING_SAMPLE = {
    "subject": "URGENT: Verify Your Account Immediately",
    "body": """Dear customer,

Your account has been suspended.
Click here immediately to verify:
http://192.168.1.1/login

Failure to act will result in permanent closure."""
}

LEGIT_SAMPLE = {
    "subject": "Team Meeting Tomorrow",
    "body": """Hi team,

We have a scheduled meeting tomorrow at 10AM.
Please find the attached agenda.

Best regards,
HR Department"""
}

# =============================
# CALLBACK FUNCTIONS
# =============================
def load_phishing():
    st.session_state.subject = PHISHING_SAMPLE["subject"]
    st.session_state.body = PHISHING_SAMPLE["body"]

def load_legit():
    st.session_state.subject = LEGIT_SAMPLE["subject"]
    st.session_state.body = LEGIT_SAMPLE["body"]

# =============================
# INIT SESSION STATE
# =============================
if "subject" not in st.session_state:
    st.session_state.subject = ""

if "body" not in st.session_state:
    st.session_state.body = ""

# =============================
# UI
# =============================
st.title("📧 Phishing Email Detection System")
st.markdown("### Compare All Models")

st.subheader("Email Input")

col1, col2 = st.columns(2)

with col1:
    st.button("⚠ Load Phishing Sample", on_click=load_phishing)

with col2:
    st.button("✅ Load Legit Sample", on_click=load_legit)

subject = st.text_input("Email Subject", key="subject")
body = st.text_area("Email Body", height=250, key="body")

st.divider()

# =============================
# PREDICTION
# =============================
if st.button("🔍 Analyze Email (All Models)"):

    if not st.session_state.subject.strip() and not st.session_state.body.strip():
        st.warning("Please enter subject and/or body.")
        st.stop()

    combined_text = st.session_state.subject + " " + st.session_state.body

    input_df = pd.DataFrame({
        "combined_text": [combined_text],
        "label": [0]
    })

    processed_df = run_preprocessing(input_df)

    results = []

    # ML models
    features = feature_extractor.transform(processed_df)

    for model_name, model in ml_models.items():

        adapted = FeatureAdapter.adapt_for_model(features, model_name)

        if hasattr(model.model, "predict_proba"):
            prob = model.predict_proba(adapted)[0][1]
        else:
            prob = float(model.predict(adapted)[0])

        pred = int(prob >= 0.5)

        results.append({
            "Model": model_name.upper(),
            "Prediction": "PHISHING" if pred else "LEGIT",
            "Probability": round(float(prob), 4)
        })

    # LSTM
    X_seq = seq_extractor.transform(
        processed_df["cleaned_text"].tolist()
    )

    lstm_prob = float(lstm_model.predict_proba(X_seq)[0])
    lstm_pred = int(lstm_prob >= 0.5)

    results.append({
        "Model": "LSTM",
        "Prediction": "PHISHING" if lstm_pred else "LEGIT",
        "Probability": round(lstm_prob, 4)
    })

    # Display
    st.subheader("Model Comparison")
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)

    phishing_votes = sum(r["Prediction"] == "PHISHING" for r in results)

    st.divider()

    if phishing_votes >= 3:
        st.error("🚨 Majority Vote: PHISHING")
    else:
        st.success("✅ Majority Vote: LEGIT")