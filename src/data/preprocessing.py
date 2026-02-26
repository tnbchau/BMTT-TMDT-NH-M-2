import re
import pandas as pd
from typing import Tuple

import nltk
from nltk.corpus import stopwords

# Download stopwords if needed
try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

EN_STOPWORDS = set(stopwords.words("english"))

# =========================
# CONSTANTS
# =========================
TEXT_COL = "combined_text"
LABEL_COL = "label"
TEXT_LENGTH_THRESHOLD = 10000  # adjust if needed


# =========================
# TEXT CLEANING
# =========================
def clean_text(text: str) -> str:
    """
    Basic text cleaning for phishing detection.
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Replace URLs & emails with tokens
    text = re.sub(r"http[s]?://\S+", " URL_TOKEN ", text)
    text = re.sub(r"\S+@\S+", " EMAIL_TOKEN ", text)

    # Remove non-alphabetic characters (keep tokens)
    text = re.sub(r"[^a-zA-Z_ ]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =========================
# STOPWORD REMOVAL
# =========================
def remove_stopwords(text: str) -> str:
    """
    Remove common English stopwords.
    """
    tokens = text.split()
    tokens = [t for t in tokens if t not in EN_STOPWORDS]
    return " ".join(tokens)


# =========================
# URL FEATURE EXTRACTION
# =========================
def extract_url_features(text: str) -> Tuple[int, int, int]:
    """
    Extract URL-based phishing features.

    Returns:
        num_urls, has_url, num_ip_urls
    """
    if pd.isna(text):
        return 0, 0, 0

    text = str(text)

    urls = re.findall(r"http[s]?://\S+", text)
    ip_urls = re.findall(
        r"http[s]?://(?:\d{1,3}\.){3}\d{1,3}", text
    )

    num_urls = len(urls)
    has_url = int(num_urls > 0)
    num_ip_urls = len(ip_urls)

    return num_urls, has_url, num_ip_urls


# =========================
# MAIN PREPROCESSING PIPELINE
# =========================
def preprocess(
    df: pd.DataFrame,
    clean_text_flag: bool = True,
    remove_stopwords_flag: bool = True
) -> pd.DataFrame:
    """
    Full preprocessing pipeline.

    Args:
        df: Raw dataframe
        clean_text_flag: Apply text cleaning
        remove_stopwords_flag: Remove English stopwords

    Returns:
        Preprocessed dataframe for modeling
    """

    df = df.copy()

    # =========================
    # BASIC SANITY CHECK
    # =========================
    df[TEXT_COL] = df[TEXT_COL].astype(str).fillna("").str.strip()

    # =========================
    # URL-BASED FEATURES
    # =========================
    url_features = df[TEXT_COL].apply(extract_url_features)

    df["num_urls"] = url_features.apply(lambda x: x[0])
    df["has_url"] = url_features.apply(lambda x: x[1])
    df["num_ip_urls"] = url_features.apply(lambda x: x[2])
    df["has_ip_url"] = (df["num_ip_urls"] > 0).astype(int)

    # =========================
    # TEXT CLEANING
    # =========================
    if clean_text_flag:
        df["cleaned_text"] = df[TEXT_COL].apply(clean_text)
    else:
        df["cleaned_text"] = df[TEXT_COL]

    # =========================
    # STOPWORD REMOVAL
    # =========================
    if remove_stopwords_flag:
        df["cleaned_text"] = df["cleaned_text"].apply(remove_stopwords)

    # =========================
    # TEXT LENGTH & OUTLIER REMOVAL
    # =========================
    df["text_length"] = df["cleaned_text"].str.len()
    initial_len = len(df)

    df = df[df["text_length"] <= TEXT_LENGTH_THRESHOLD]

    removed = initial_len - len(df)
    print(f"[INFO] Removed {removed} samples exceeding {TEXT_LENGTH_THRESHOLD} chars")

    # =========================
    # FINAL FEATURE SET
    # =========================
    final_cols = [
        "cleaned_text",
        LABEL_COL,
        "num_urls",
        "has_url",
        "num_ip_urls",
        "has_ip_url",
    ]

    df_final = df[final_cols].reset_index(drop=True)

    return df_final

def run_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wrapper for full preprocessing.
    """
    return preprocess(
        df,
        clean_text_flag=True,
        remove_stopwords_flag=True
    )
# =========================
if __name__ == "__main__":
    print("[INFO] Running preprocessing test...")

    sample_df = pd.DataFrame({
        "combined_text": [
            "Verify your account at http://192.168.1.1/login now!",
            "Meeting scheduled tomorrow at 10am."
        ],
        "label": [1, 0]
    })

    processed_df = run_preprocessing(sample_df)

    print("\n===== PREPROCESSED SAMPLE =====")
    print(processed_df.head())