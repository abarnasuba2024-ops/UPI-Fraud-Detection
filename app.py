import os
import sqlite3
import threading
import webbrowser

import joblib
import numpy as np
import pandas as pd

from flask import (
    Flask,
    request,
    redirect,
    url_for,
    render_template_string
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(BASE_DIR, "fraud_model.pkl")
DATABASE_FILE = os.path.join(BASE_DIR, "transactions.db")

HOST = "127.0.0.1"
PORT = 5000

app = Flask(__name__)


# ============================================================
# MACHINE LEARNING FEATURES
# ============================================================

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


# ============================================================
# GENERATE SYNTHETIC DATASET
# ============================================================

def generate_dataset(n=12000, seed=42):

    rng = np.random.default_rng(seed)

    amount = np.round(
        np.clip(
            rng.lognormal(7.0, 1.0, n),
            10,
            250000
        ),
        2
    )

    hour = rng.integers(0, 24, n)

    transactions_last_24h = rng.poisson(3, n)

    device_changed = rng.binomial(
        1,
        0.10,
        n
    )

    location_changed = rng.binomial(
        1,
        0.08,
        n
    )

    account_age_days = rng.integers(
        1,
        2500,
        n
    )

    failed_attempts = np.clip(
        rng.poisson(0.35, n),
        0,
        8
    )

    previous_fraud_count = np.clip(
        rng.poisson(0.08, n),
        0,
        5
    )

    merchant_risk = rng.integers(
        1,
        11,
        n
    )

    is_new_payee = rng.binomial(
        1,
        0.20,
        n
    )

    night_transaction = (
        (hour <= 5) |
        (hour >= 23)
    ).astype(int)

    # Fraud scoring logic
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

    data = pd.DataFrame({
        "amount": amount,
        "hour": hour,
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
        "fraud": fraud
    })

    return data


# ============================================================
# TRAIN MACHINE LEARNING MODEL
# ============================================================

def train_model():

    print()
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

    print("Training Random Forest model...")

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

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        "Model accuracy:",
        round(accuracy * 100, 2),
        "%"
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    print("Model saved successfully.")

    return model


# ============================================================
# LOAD MODEL
# ============================================================

def get_model():

    if not os.path.exists(MODEL_FILE):

        return train_model()

    return joblib.load(MODEL_FILE)


model = get_model()


# ============================================================
# DATABASE
# ============================================================

def init_database():

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    connection.execute("""
        CREATE TABLE IF NOT EXISTS transactions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            created_at TEXT,

            amount REAL,

            hour INTEGER,

            transactions_last_24h INTEGER,

            device_changed INTEGER,

            location_changed INTEGER,

            account_age_days INTEGER,

            failed_attempts INTEGER,

            previous_fraud_count INTEGER,

            merchant_risk INTEGER,

            is_new_payee INTEGER,

            fraud_probability REAL,

            prediction TEXT,

            risk_level TEXT

        )
    """)

    connection.commit()

    connection.close()


init_database()


# ============================================================
# COMMON HTML
# ============================================================

CSS = """

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, Helvetica, sans-serif;
    background: #f4f7fb;
    color: #222;
}

.navbar {
    background: #111827;
    color: white;
    padding: 18px 40px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.logo {
    font-size: 23px;
    font-weight: bold;
}

.nav-links a {
    color: white;
    text-decoration: none;
    margin-left: 25px;
    font-size: 15px;
}

.container {
    width: 92%;
    max-width: 1200px;
    margin: 30px auto;
}

h1 {
    margin-bottom: 5px;
}

.subtitle {
    color: #6b7280;
}

.cards {
    display: grid;
    grid-template-columns:
        repeat(4, 1fr);
    gap: 20px;
    margin-top: 30px;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow:
        0 4px 15px rgba(0,0,0,0.08);
}

.card-title {
    color: #6b7280;
    font-size: 14px;
}

.card-value {
    font-size: 32px;
    font-weight: bold;
    margin-top: 10px;
}

.blue {
    color: #2563eb;
}

.red {
    color: #dc2626;
}

.green {
    color: #16a34a;
}

.orange {
    color: #ea580c;
}

.form-card {
    background: white;
    padding: 30px;
    border-radius: 15px;
    box-shadow:
        0 4px 15px rgba(0,0,0,0.08);
    margin-top: 25px;
}

.form-grid {
    display: grid;
    grid-template-columns:
        repeat(2, 1fr);
    gap: 18px;
}

.form-group {
    display: flex;
    flex-direction: column;
}

label {
    font-weight: bold;
    margin-bottom: 7px;
}

input,
select {
    padding: 12px;
    border: 1px solid #d1d5db;
    border-radius: 8px;
    font-size: 15px;
}

button {
    margin-top: 25px;
    padding: 13px 25px;
    border: none;
    border-radius: 8px;
    background: #2563eb;
    color: white;
    font-size: 16px;
    cursor: pointer;
}

button:hover {
    background: #1d4ed8;
}

.result {
    margin-top: 25px;
    padding: 30px;
    background: white;
    border-radius: 15px;
    text-align: center;
    box-shadow:
        0 4px 15px rgba(0,0,0,0.08);
}

.fraud {
    color: #dc2626;
}

.legitimate {
    color: #16a34a;
}

.high {
    background: #dc2626;
    color: white;
}

.medium {
    background: #f59e0b;
    color: white;
}

.low {
    background: #16a34a;
    color: white;
}

.badge {
    display: inline-block;
    padding: 8px 18px;
    border-radius: 20px;
    font-weight: bold;
}

table {
    width: 100%;
    border-collapse: collapse;
    background: white;
    margin-top: 25px;
}

th,
td {
    padding: 14px;
    border-bottom: 1px solid #e5e7eb;
    text-align: left;
}

th {
    background: #f9fafb;
}

@media(max-width: 800px) {

    .cards {
        grid-template-columns:
            repeat(2, 1fr);
    }

    .form-grid {
        grid-template-columns: 1fr;
    }
}

</style>

"""


# ============================================================
# DASHBOARD HTML
# ============================================================

DASHBOARD_HTML = """

<!DOCTYPE html>

<html>

<head>

<title>UPI Fraud Detection</title>

{{ css|safe }}

</head>

<body>

<div class="navbar">

    <div class="logo">
        🛡️ UPI FraudGuard
    </div>

    <div class="nav-links">

        <a href="/">
            Dashboard
        </a>

        <a href="/predict">
            Check Transaction
        </a>

        <a href="/history">
            History
        </a>

    </div>

</div>


<div class="container">

    <h1>
        UPI Fraud Detection Dashboard
    </h1>

    <p class="subtitle">
        Machine Learning Based Transaction Risk Monitoring
    </p>


    <div class="cards">

        <div class="card">

            <div class="card-title">
                Total Transactions
            </div>

            <div class="card-value blue">
                {{ total }}
            </div>

        </div>


        <div class="card">

            <div class="card-title">
                Fraud Detected
            </div>

            <div class="card-value red">
                {{ fraud }}
            </div>

        </div>


        <div class="card">

            <div class="card-title">
                Legitimate
            </div>

            <div class="card-value green">
                {{ legitimate }}
            </div>

        </div>


        <div class="card">

            <div class="card-title">
                Fraud Rate
            </div>

            <div class="card-value orange">
                {{ "%.1f"|format(fraud_rate) }}%
            </div>

        </div>

    </div>


    <div class="form-card">

        <h2>
            Recent Transactions
        </h2>

        <table>

            <tr>

                <th>
                    Time
                </th>

                <th>
                    Amount
                </th>

                <th>
                    Probability
                </th>

                <th>
                    Prediction
                </th>

                <th>
                    Risk
                </th>

            </tr>


            {% for t in recent %}

            <tr>

                <td>
                    {{ t["created_at"] }}
                </td>

                <td>
                    ₹{{ "%.2f"|format(t["amount"]) }}
                </td>

                <td>
                    {{ "%.2f"|format(
                        t["fraud_probability"] * 100
                    ) }}%
                </td>

                <td>

                    {% if t["prediction"] == "FRAUD" %}

                    <span class="badge fraud">
                        FRAUD
                    </span>

                    {% else %}

                    <span class="badge legitimate">
                        LEGITIMATE
                    </span>

                    {% endif %}

                </td>

                <td>

                    {% if t["risk_level"] == "HIGH" %}

                    <span class="badge high">
                        HIGH
                    </span>

                    {% elif t["risk_level"] == "MEDIUM" %}

                    <span class="badge medium">
                        MEDIUM
                    </span>

                    {% else %}

                    <span class="badge low">
                        LOW
                    </span>

                    {% endif %}

                </td>

            </tr>

            {% endfor %}


            {% if not recent %}

            <tr>

                <td
                    colspan="5"
                    style="text-align:center"
                >

                    No transactions yet.

                </td>

            </tr>

            {% endif %}

        </table>

    </div>

</div>

</body>

</html>

"""


# ============================================================
# PREDICTION PAGE
# ============================================================

PREDICT_HTML = """

<!DOCTYPE html>

<html>

<head>

<title>Check Transaction</title>

{{ css|safe }}

</head>

<body>

<div class="navbar">

    <div class="logo">
        🛡️ UPI FraudGuard
    </div>

    <div class="nav-links">

        <a href="/">
            Dashboard
        </a>

        <a href="/predict">
            Check Transaction
        </a>

        <a href="/history">
            History
        </a>

    </div>

</div>


<div class="container">

    <h1>
        Check UPI Transaction
    </h1>

    <p class="subtitle">
        Enter transaction details for fraud analysis.
    </p>


    <div class="form-card">

        <form method="POST">

            <div class="form-grid">


                <div class="form-group">

                    <label>
                        Transaction Amount (₹)
                    </label>

                    <input
                        type="number"
                        name="amount"
                        value="1000"
                        min="1"
                        step="0.01"
                        required
                    >

                </div>


                <div class="form-group">

                    <label>
                        Transaction Hour (0-23)
                    </label>

                    <input
                        type="number"
                        name="hour"
                        value="14"
                        min="0"
                        max="23"
                        required
                    >

                </div>


                <div class="form-group">

                    <label>
                        Transactions Last 24 Hours
                    </label>

                    <input
                        type="number"
                        name="transactions_last_24h"
                        value="3"
                        min="0"
                        required
                    >

                </div>


                <div class="form-group">

                    <label>
                        Account Age (Days)
                    </label>

                    <input
                        type="number"
                        name="account_age_days"
                        value="500"
                        min="1"
                        required
                    >

                </div>


                <div class="form-group">

                    <label>
                        Failed Attempts
                    </label>

                    <input
                        type="number"
                        name="failed_attempts"
                        value="0"
                        min="0"
                        required
                    >

                </div>


                <div class="form-group">

                    <label>
                        Previous Fraud Count
                    </label>

                    <input
                        type="number"
                        name="previous_fraud_count"
                        value="0"
                        min="0"
                        required
                    >

                </div>


                <div class="form-group">

                    <label>
                        Merchant Risk (1-10)
                    </label>

                    <input
                        type="number"
                        name="merchant_risk"
                        value="3"
                        min="1"
                        max="10"
                        required
                    >

                </div>


                <div class="form-group">

                    <label>
                        Device Changed?
                    </label>

                    <select name="device_changed">

                        <option value="0">
                            No
                        </option>

                        <option value="1">
                            Yes
                        </option>

                    </select>

                </div>


                <div class="form-group">

                    <label>
                        Location Changed?
                    </label>

                    <select name="location_changed">

                        <option value="0">
                            No
                        </option>

                        <option value="1">
                            Yes
                        </option>

                    </select>

                </div>


                <div class="form-group">

                    <label>
                        New Payee?
                    </label>

                    <select name="is_new_payee">

                        <option value="0">
                            No
                        </option>

                        <option value="1">
                            Yes
                        </option>

                    </select>

                </div>


            </div>


            <button type="submit">

                🔍 Analyze Transaction

            </button>

        </form>

    </div>


    {% if result %}

    <div class="result">

        {% if result.prediction == "FRAUD" %}

        <h1 class="fraud">
            🚨 FRAUD DETECTED
        </h1>

        {% else %}

        <h1 class="legitimate">
            ✅ LEGITIMATE TRANSACTION
        </h1>

        {% endif %}


        <h2>
            Fraud Probability:
            {{ "%.2f"|format(
                result.probability
            ) }}%
        </h2>


        <h3>
            Risk Level
        </h3>


        {% if result.risk_level == "HIGH" %}

        <span class="badge high">
            HIGH RISK
        </span>

        {% elif result.risk_level == "MEDIUM" %}

        <span class="badge medium">
            MEDIUM RISK
        </span>

        {% else %}

        <span class="badge low">
            LOW RISK
        </span>

        {% endif %}

    </div>

    {% endif %}


</div>

</body>

</html>

"""


# ============================================================
# HISTORY PAGE
# ============================================================

HISTORY_HTML = """

<!DOCTYPE html>

<html>

<head>

<title>Transaction History</title>

{{ css|safe }}

</head>

<body>

<div class="navbar">

    <div class="logo">
        🛡️ UPI FraudGuard
    </div>

    <div class="nav-links">

        <a href="/">
            Dashboard
        </a>

        <a href="/predict">
            Check Transaction
        </a>

        <a href="/history">
            History
        </a>

    </div>

</div>


<div class="container">

    <h1>
        Transaction History
    </h1>

    <p class="subtitle">
        Previously analyzed transactions
    </p>


    <table>

        <tr>

            <th>ID</th>

            <th>Date & Time</th>

            <th>Amount</th>

            <th>Fraud Probability</th>

            <th>Prediction</th>

            <th>Risk</th>

        </tr>


        {% for t in transactions %}

        <tr>

            <td>
                {{ t["id"] }}
            </td>

            <td>
                {{ t["created_at"] }}
            </td>

            <td>
                ₹{{ "%.2f"|format(t["amount"]) }}
            </td>

            <td>
                {{ "%.2f"|format(
                    t["fraud_probability"] * 100
                ) }}%
            </td>

            <td>

                {% if t["prediction"] == "FRAUD" %}

                <span class="badge fraud">
                    FRAUD
                </span>

                {% else %}

                <span class="badge legitimate">
                    LEGITIMATE
                </span>

                {% endif %}

            </td>

            <td>

                {% if t["risk_level"] == "HIGH" %}

                <span class="badge high">
                    HIGH
                </span>

                {% elif t["risk_level"] == "MEDIUM" %}

                <span class="badge medium">
                    MEDIUM
                </span>

                {% else %}

                <span class="badge low">
                    LOW
                </span>

                {% endif %}

            </td>

        </tr>

        {% endfor %}


        {% if not transactions %}

        <tr>

            <td
                colspan="6"
                style="text-align:center"
            >

                No transactions found.

            </td>

        </tr>

        {% endif %}

    </table>

</div>

</body>

</html>

"""


# ============================================================
# DASHBOARD ROUTE
# ============================================================

@app.route("/")
def dashboard():

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    connection.row_factory = sqlite3.Row

    total = connection.execute(
        "SELECT COUNT(*) AS count FROM transactions"
    ).fetchone()["count"]

    fraud = connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM transactions
        WHERE prediction = 'FRAUD'
        """
    ).fetchone()["count"]

    legitimate = total - fraud

    if total > 0:

        fraud_rate = (
            fraud / total
        ) * 100

    else:

        fraud_rate = 0

    recent = connection.execute(
        """
        SELECT *
        FROM transactions
        ORDER BY id DESC
        LIMIT 10
        """
    ).fetchall()

    connection.close()

    return render_template_string(
        DASHBOARD_HTML,
        css=CSS,
        total=total,
        fraud=fraud,
        legitimate=legitimate,
        fraud_rate=fraud_rate,
        recent=recent
    )


# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.route(
    "/predict",
    methods=["GET", "POST"]
)
def predict():

    result = None

    if request.method == "POST":

        try:

            amount = float(
                request.form["amount"]
            )

            hour = int(
                request.form["hour"]
            )

            transactions_last_24h = int(
                request.form[
                    "transactions_last_24h"
                ]
            )

            device_changed = int(
                request.form[
                    "device_changed"
                ]
            )

            location_changed = int(
                request.form[
                    "location_changed"
                ]
            )

            account_age_days = int(
                request.form[
                    "account_age_days"
                ]
            )

            failed_attempts = int(
                request.form[
                    "failed_attempts"
                ]
            )

            previous_fraud_count = int(
                request.form[
                    "previous_fraud_count"
                ]
            )

            merchant_risk = int(
                request.form[
                    "merchant_risk"
                ]
            )

            is_new_payee = int(
                request.form[
                    "is_new_payee"
                ]
            )


            values = {

                "amount": amount,

                "hour": hour,

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
                    is_new_payee

            }


            input_data = pd.DataFrame(
                [values],
                columns=FEATURES
            )


            probability = float(
                model.predict_proba(
                    input_data
                )[0][1]
            )


            if probability >= 0.50:

                prediction = "FRAUD"

            else:

                prediction = "LEGITIMATE"


            if probability >= 0.75:

                risk_level = "HIGH"

            elif probability >= 0.40:

                risk_level = "MEDIUM"

            else:

                risk_level = "LOW"


            connection = sqlite3.connect(
                DATABASE_FILE
            )


            connection.execute(
                """
                INSERT INTO transactions (

                    created_at,
                    amount,
                    hour,
                    transactions_last_24h,
                    device_changed,
                    location_changed,
                    account_age_days,
                    failed_attempts,
                    previous_fraud_count,
                    merchant_risk,
                    is_new_payee,
                    fraud_probability,
                    prediction,
                    risk_level

                )

                VALUES (
                    datetime('now'),
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,

                (

                    amount,

                    hour,

                    transactions_last_24h,

                    device_changed,

                    location_changed,

                    account_age_days,

                    failed_attempts,

                    previous_fraud_count,

                    merchant_risk,

                    is_new_payee,

                    probability,

                    prediction,

                    risk_level

                )
            )


            connection.commit()

            connection.close()


            result = {

                "prediction":
                    prediction,

                "probability":
                    probability * 100,

                "risk_level":
                    risk_level

            }


        except Exception as error:

            print(
                "Prediction error:",
                error
            )


    return render_template_string(
        PREDICT_HTML,
        css=CSS,
        result=result
    )


# ============================================================
# HISTORY ROUTE
# ============================================================

@app.route("/history")
def history():

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    connection.row_factory = sqlite3.Row

    transactions = connection.execute(
        """
        SELECT *
        FROM transactions
        ORDER BY id DESC
        """
    ).fetchall()

    connection.close()

    return render_template_string(
        HISTORY_HTML,
        css=CSS,
        transactions=transactions
    )


# ============================================================
# AUTOMATICALLY OPEN BROWSER
# ============================================================

def open_browser():

    url = (
        f"http://{HOST}:{PORT}/"
    )

    webbrowser.open_new(
        url
    )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("       UPI FRAUD DETECTION SYSTEM")
    print("=" * 60)
    print()
    print(
        "Starting application..."
    )
    print()
    print(
        f"URL: http://{HOST}:{PORT}"
    )
    print()
    print(
        "Browser will open automatically..."
    )
    print()

    # Open browser after Flask starts
    threading.Timer(
        2,
        open_browser
    ).start()

    # IMPORTANT:
    # debug=False prevents the browser from
    # opening twice.
    app.run(
        host=HOST,
        port=PORT,
        debug=False
    )