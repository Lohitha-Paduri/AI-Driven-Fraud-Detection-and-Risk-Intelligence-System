
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.data import generate_transactions


# Features used by the machine learning model
FEATURES = [
    "amount",
    "hour",
    "account_age_days",
    "txn_velocity_1h",
    "failed_attempts",
    "distance_km",
    "international",
    "merchant_risk",
    "device_trust",
    "channel_Mobile",
    "channel_POS",
    "channel_Web"
]


# Convert transaction data into model-compatible features
def encode_features(df):

    x = df[
        [
            "amount",
            "hour",
            "account_age_days",
            "txn_velocity_1h",
            "failed_attempts",
            "distance_km",
            "international",
            "merchant_risk",
            "device_trust",
            "channel"
        ]
    ].copy()

    x = pd.get_dummies(x, columns=["channel"])

    # Add missing columns if necessary
    for col in FEATURES:
        if col not in x.columns:
            x[col] = 0

    # Maintain consistent feature order
    return x[FEATURES]


# Train the ensemble machine learning model
def get_model_bundle():

    # Generate synthetic transaction data
    df = generate_transactions(7000, seed=42)

    X = encode_features(df)
    y = df["is_fraud"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # Random Forest model
    rf = RandomForestClassifier(
        n_estimators=180,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    # Gradient Boosting model
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.06,
        max_depth=2,
        random_state=42
    )

    # Combine both models using soft voting
    model = VotingClassifier(
        estimators=[
            ("random_forest", rf),
            ("gradient_boosting", gb)
        ],
        voting="soft",
        weights=[1, 1]
    )

    # Train the ensemble model
    model.fit(X_train, y_train)

    # Predict fraud probabilities
    p = model.predict_proba(X_test)[:, 1]

    # Convert probabilities into binary predictions
    pred = (p >= 0.5).astype(int)

    # Calculate model evaluation metrics
    metrics = {
        "roc_auc": float(
            roc_auc_score(y_test, p)
        ),

        "precision": float(
            precision_score(y_test, pred, zero_division=0)
        ),

        "recall": float(
            recall_score(y_test, pred, zero_division=0)
        ),

        "f1": float(
            f1_score(y_test, pred, zero_division=0)
        ),

        "confusion_matrix": confusion_matrix(
            y_test,
            pred
        ).tolist()
    }

    # Extract feature importance from Random Forest
    importances = model.named_estimators_[
        "random_forest"
    ].feature_importances_

    feature_importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": importances
    }).sort_values(
        by="importance",
        ascending=False
    )

    return {
        "model": model,
        "metrics": metrics,
        "feature_importance": feature_importance
    }


# Predict fraud risk for a new transaction
def predict_transaction(model, payload):

    row = pd.DataFrame([payload])

    X = encode_features(row)

    # Get fraud probability
    score = float(
        model.predict_proba(X)[0, 1]
    )

    # Assign risk category
    if score >= 0.70:
        band = "High"

    elif score >= 0.35:
        band = "Medium"

    else:
        band = "Low"

    # Recommendation based on risk category
    recommendation = {
        "High": "Prioritize manual review",
        "Medium": "Request additional verification",
        "Low": "Continue standard monitoring"
    }[band]

    return {
        "risk_score": score,
        "risk_band": band,
        "recommendation": recommendation
    }


# Generate a human-readable local explanation
def get_local_explanation(payload, result):

    reasons = []

    if payload["amount"] >= 25000:
        reasons.append(
            "The transaction amount is relatively large (₹25,000 or more)."
        )

    if payload["txn_velocity_1h"] >= 5:
        reasons.append(
            "Several transactions occurred within one hour, "
            "which may indicate unusual transaction velocity."
        )

    if payload["failed_attempts"] >= 2:
        reasons.append(
            "Multiple recent failed attempts were reported."
        )

    if payload["distance_km"] >= 500:
        reasons.append(
            "The transaction is far from the account's usual location."
        )

    if payload["international"]:
        reasons.append(
            "The transaction is marked as international."
        )

    if payload["merchant_risk"] == 2:
        reasons.append(
            "The merchant is marked as high risk."
        )

    if payload["device_trust"] == 0:
        reasons.append(
            "The device is marked as untrusted."
        )

    if payload["hour"] <= 4:
        reasons.append(
            "The transaction occurred during very early hours."
        )

    if payload["account_age_days"] < 30:
        reasons.append(
            "The account is relatively new (less than 30 days old)."
        )

    if not reasons:
        reasons.append(
            "No strong individual warning signal crossed the "
            "simple explanation thresholds; the model combines "
            "all features together."
        )

    reasons.append(
        f"The ensemble model estimates a "
        f"{result['risk_score']:.1%} fraud probability. "
        "This is a risk estimate, not proof of fraud."
    )

    return reasons
