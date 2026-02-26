import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict


def split_data(
    df: pd.DataFrame,
    label_col: str = "label",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Dict[str, pd.DataFrame | pd.Series]:
    """
    Stratified split into train / validation / test sets.

    Args:
        df: Preprocessed DataFrame
        label_col: Label column name
        test_size: Proportion for test set
        val_size: Proportion for validation set (of total data)
        random_state: Random seed

    Returns:
        Dictionary with X_train, y_train, X_val, y_val, X_test, y_test
    """

    # =========================
    # SANITY CHECK
    # =========================
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataframe")

    # =========================
    # SPLIT FEATURES / LABEL
    # =========================
    X = df.drop(columns=[label_col])
    y = df[label_col]

    # =========================
    # TRAIN+VAL vs TEST
    # =========================
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # =========================
    # TRAIN vs VALIDATION
    # =========================
    val_ratio = val_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio,
        stratify=y_train_val,
        random_state=random_state
    )

    # =========================
    # SUMMARY
    # =========================
    print("[INFO] Data split completed")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val  : {len(X_val)} samples")
    print(f"  Test : {len(X_test)} samples")
    print(f"  Label mean (phishing ratio):")
    print(f"    Train: {y_train.mean():.3f}")
    print(f"    Val  : {y_val.mean():.3f}")
    print(f"    Test : {y_test.mean():.3f}")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


# =========================
# QUICK TEST
# =========================
if __name__ == "__main__":
    sample_df = pd.DataFrame({
    "cleaned_text": [f"text {i}" for i in range(20)],
    "num_urls": [i % 2 for i in range(20)],
    "has_url": [i % 2 for i in range(20)],
    "label": [i % 2 for i in range(20)]
})
   
    print("Testing split_data function...")
    splits = split_data(sample_df, test_size=0.25, val_size=0.125)
    
    print("\nSample from each split:")
    for name in ["X_train", "X_val", "X_test"]:
        print(f"\n{name}:")
        print(splits[name].head())