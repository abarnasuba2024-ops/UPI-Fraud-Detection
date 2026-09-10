import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score
)
from sklearn.model_selection import train_test_split


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "upi_transactions.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "fraud_model.pkl"
)


FEATURES = [
    "amount",
    "hour",
    "transactions_last_24h",
    "device_changed",
    "location_changed",
    "account_age_days",
    "failed_attempts",
    "previous_fraud_count",
    "merchant_risk",
    "is_new_payee"
]


def generate_dataset(n=12000, seed=42):

    rng = np.random.default_rng(seed)

    # Transaction amount
    amount = np.round(
        np.clip(
            rng.lognormal(7.0, 1.0, n),
            10,
            250000
        ),
        2
    )

    # Transaction hour
    hour = rng.integers(0, 24, n)

    # Number of transactions in last 24 hours
    transactions_last_24h = rng.poisson(3, n)

    # New device
    device_changed = rng.binomial(
        1,
        0.10,
        n
    )

    # New location
    location_changed = rng.binomial(
        1,
        0.08,
        n
    )

    # Account age
    account_age_days = rng.integers(
        1,
        2500,
        n
    )

    # Failed login/payment attempts
    failed_attempts = np.clip(
        rng.poisson(0.35, n),
        0,
        8
    )

    # Previous fraud count
    previous_fraud_count = np.clip(
        rng.poisson(0.08, n),
        0,
        5
    )

    # Merchant risk score
    merchant_risk = rng.integers(
        1,
        11,
        n
    )

    # New payee
    is_new_payee = rng.binomial(
        1,
        0.20,
        n
    )

    # Night-time transactions
    night_transaction = (
        (hour <= 5) |
        (hour >= 23)
    ).astype(int)

    # Synthetic fraud score
    logit = (
        -5.0

        + 0.000018 * amount

        + 0.16 * transactions_last_24h

        + 1.65 * device_changed

        + 1.35 * location_changed

        - 0.00035 * account_age_days

        + 0.42 * failed_attempts

        + 1.15 * previous_fraud_count

        + 0.20 * merchant_risk

        + 1.10 * is_new_payee

        + 0.65 * night_transaction
    )

    probability = 1 / (
        1 + np.exp(-logit)
    )

    fraud = rng.binomial(
        1,
        probability
    )

    df = pd.DataFrame({

        "amount":
            amount,

        "hour":
            hour,

        "transactions_last_24h":
            transactions_last_24h,

        "device_changed":
            device_changed,

        "location_changed":
            location_changed,

        "account_age_days":
            account_age_days,

        "failed_attempts":
            failed_attempts,

        "previous_fraud_count":
            previous_fraud_count,

        "merchant_risk":
            merchant_risk,

        "is_new_payee":
            is_new_payee,

        "fraud":
            fraud
    })

    os.makedirs(
        os.path.dirname(DATA_PATH),
        exist_ok=True
    )

    df.to_csv(
        DATA_PATH,
        index=False
    )

    return df


def train_and_save_model():

    print("Generating dataset...")

    df = generate_dataset()

    X = df[FEATURES]

    y = df["fraud"]

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )

    print("Training Random Forest...")

    model = RandomForestClassifier(

        n_estimators=250,

        max_depth=14,

        min_samples_leaf=3,

        class_weight="balanced",

        random_state=42,

        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    print()
    print("========== MODEL RESULTS ==========")
    print(
        "Accuracy:",
        round(accuracy, 4)
    )

    print(
        "ROC-AUC:",
        round(auc, 4)
    )

    print()
    print(
        classification_report(
            y_test,
            predictions
        )
    )

    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True
    )

    joblib.dump(
        {
            "model": model,
            "features": FEATURES
        },
        MODEL_PATH
    )

    print(
        "Model saved to:",
        MODEL_PATH
    )

    return model


if __name__ == "__main__":

    train_and_save_model()