"""
SEQUENCE EXTRACTION MODULE
- For LSTM-based deep learning models
- Uses Keras Tokenizer + padding
- Shares preprocessing output with ML models
"""

import numpy as np
import pandas as pd
from typing import List
import joblib

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class SequenceExtractor:
    """
    Converts cleaned text into padded integer sequences for LSTM.
    """

    def __init__(
        self,
        max_words: int = 20000,
        max_len: int = 200,
        oov_token: str = "<OOV>"
    ):
        self.max_words = max_words
        self.max_len = max_len
        self.oov_token = oov_token

        self.tokenizer = Tokenizer(
            num_words=max_words,
            oov_token=oov_token
        )
        self.is_fitted = False

    # -------------------------
    # FIT / TRANSFORM
    # -------------------------

    def fit(self, texts: List[str]):
        self.tokenizer.fit_on_texts(texts)
        self.is_fitted = True
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("SequenceExtractor must be fitted first")

        sequences = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(
            sequences,
            maxlen=self.max_len,
            padding="post",
            truncating="post",
            value=0
        )

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)

    # -------------------------
    # UTILITIES
    # -------------------------

    def vocab_size(self) -> int:
        if not self.is_fitted:
            raise RuntimeError("Extractor not fitted")
        return min(self.max_words, len(self.tokenizer.word_index) + 1)

    # -------------------------
    # SAVE / LOAD
    # -------------------------

    def save(self, path: str):
        joblib.dump({
            "max_words": self.max_words,
            "max_len": self.max_len,
            "oov_token": self.oov_token,
            "tokenizer": self.tokenizer
        }, path)

    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        extractor = cls(
            max_words=data["max_words"],
            max_len=data["max_len"],
            oov_token=data["oov_token"]
        )
        extractor.tokenizer = data["tokenizer"]
        extractor.is_fitted = True
        return extractor


# ==================================================
# QUICK TEST
# ==================================================
if __name__ == "__main__":
    df = pd.DataFrame({
        "cleaned_text": [
            "verify account now",
            "meeting tomorrow",
            "urgent password update",
            "monthly newsletter"
        ],
        "num_urls": [1, 0, 1, 0],
        "num_ip_urls": [1, 0, 0, 0],
        "has_url": [1, 0, 1, 0],
        "has_ip_url": [1, 0, 0, 0],
        "label": [1, 0, 1, 0]
    })
    extractor = SequenceExtractor(
        max_words=1000,
        max_len=10
    )

    X = extractor.fit_transform(df["cleaned_text"])

    print("Shape:", X.shape)
    print("Sequences:\n", X)
    print("Vocab size:", extractor.vocab_size())

    extractor.save("sequence_extractor.joblib")
    loaded = SequenceExtractor.load("sequence_extractor.joblib")

    X2 = loaded.transform(df["cleaned_text"])
    print("Load check:", (X == X2).all())