import os
import sqlite3
import threading
import webbrowser

import joblib
import numpy as np
import pandas as pd

from flask import Flask, request, redirect, url_for, render_template_string
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# =========================================================
# APP CONFIGURATION
# =========================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(BASE_DIR, "fraud_model.pkl")
DATABASE_FILE = os.path.join(BASE_DIR, "transactions.db")

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 5000))

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


# =========================================================
# MACHINE LEARNING DATASET
# =========================================================

def generate_dataset(n=12000, seed=42):

    rng = np.random.default_rng(seed)

    amount = rng.lognormal(mean=6.0, sigma=1.0, size=n)
    amount = np.clip(amount, 10, 100000)

    hour = rng.integers(0, 24, n)

    transactions_last_24h = rng.poisson(4, n)
    transactions_last_24h = np.clip(
        transactions_last_24h, 0, 30
    )

    device_changed = rng.binomial(1, 0.12, n)

    location_changed = rng.binomial(1, 0.10, n)

    account_age_days = rng.integers(
        1, 2500, n
    )

    failed_attempts = rng.poisson(0.5, n)
    failed_attempts = np.clip(
        failed_attempts, 0, 8
    )

    previous_fraud_count = rng.poisson(0.15, n)
    previous_fraud_count = np.clip(
        previous_fraud_count, 0, 5
    )

    merchant_risk = rng.uniform(0, 1, n)

    is_new_payee = rng.binomial(1, 0.18, n)

    # Fraud scoring logic
    score = (
        -3.2
        + 0.000025 * amount
        + 0.07 * transactions_last_24h
        + 1.10 * device_changed
        + 0.90 * location_changed
        - 0.00025 * account_age_days
        + 0.40 * failed_attempts
        + 0.70 * previous_fraud_count
        + 2.00 * merchant_risk
        + 0.90 * is_new_payee
        + 0.45 * ((hour <= 5) | (hour >= 23))
    )

    probability = 1 / (1 + np.exp(-score))

    fraud = rng.binomial(1, probability)

    data = pd.DataFrame({
        "amount": amount,
        "hour": hour,
        "transactions_last_24h": transactions_last_24h,
        "device_changed": device_changed,
        "location_changed": location_changed,
        "account_age_days": account_age_days,
        "failed_attempts": failed_attempts,
        "previous_fraud_count": previous_fraud_count,
        "merchant_risk": merchant_risk,
        "is_new_payee": is_new_payee,
        "fraud": fraud
    })

    return data


# =========================================================
# TRAIN MODEL
# =========================================================

def train_model():

    print("Training fraud detection model...")

    data = generate_dataset()

    X = data[FEATURES]
    y = data["fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=14,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"Model accuracy: {accuracy * 100:.2f}%"
    )

    joblib.dump(model, MODEL_FILE)

    return model


def get_model():

    if os.path.exists(MODEL_FILE):

        try:
            return joblib.load(MODEL_FILE)

        except Exception:

            print(
                "Existing model could not be loaded. Retraining..."
            )

    return train_model()


model = get_model()


# =========================================================
# DATABASE
# =========================================================

def init_database():

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            amount REAL,
            hour INTEGER,
            transactions_last_24h INTEGER,
            device_changed INTEGER,
            location_changed INTEGER,
            account_age_days INTEGER,
            failed_attempts INTEGER,
            previous_fraud_count INTEGER,
            merchant_risk REAL,
            is_new_payee INTEGER,
            fraud_probability REAL,
            prediction TEXT,
            risk_level TEXT
        )
    """)

    connection.commit()
    connection.close()


init_database()


# =========================================================
# COMMON CSS
# =========================================================

CSS = """
<style>

* {
    box-sizing: border-box;
}

html {
    width: 100%;
    overflow-x: hidden;
}

body {
    margin: 0;
    padding: 0;
    width: 100%;
    background: #f4f7fb;
    color: #17233c;
    font-family: Georgia, "Times New Roman", serif;
    overflow-x: hidden;
}


/* ================= NAVBAR ================= */

.navbar {
    width: 100%;
    min-height: 70px;
    background: #101a2d;
    color: white;

    display: flex;
    align-items: center;
    justify-content: space-between;

    padding: 15px 5%;

    gap: 20px;
}

.logo {
    font-size: 24px;
    font-weight: bold;
    white-space: nowrap;
}

.logo span {
    margin-right: 8px;
}

.nav-links {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 25px;
    flex-wrap: wrap;
}

.nav-links a {
    color: white;
    text-decoration: none;
    font-size: 16px;
}

.nav-links a:hover {
    color: #5da9ff;
}


/* ================= MAIN CONTAINER ================= */

.container {
    width: 100%;
    max-width: 1400px;

    margin: 0 auto;

    padding: 45px 5%;
}


/* ================= HEADINGS ================= */

h1 {
    margin: 0 0 12px;

    font-size: 44px;
    line-height: 1.2;

    color: #16233d;
}

.subtitle {
    color: #6d7d96;

    font-size: 19px;

    margin-bottom: 35px;
}


/* ================= STAT CARDS ================= */

.stats {
    width: 100%;

    display: grid;

    grid-template-columns:
        repeat(4, minmax(0, 1fr));

    gap: 22px;

    margin-bottom: 35px;
}

.card {
    background: white;

    border-radius: 15px;

    padding: 30px;

    min-height: 145px;

    box-shadow:
        0 8px 25px rgba(25, 45, 80, 0.08);

    display: flex;
    flex-direction: column;
    justify-content: center;
}

.card-title {
    color: #65738a;
    font-size: 17px;
    margin-bottom: 12px;
}

.card-value {
    font-size: 42px;
    font-weight: bold;
}

.blue {
    color: #1769e0;
}

.red {
    color: #d62929;
}

.green {
    color: #169b50;
}

.orange {
    color: #df6c13;
}


/* ================= SECTION ================= */

.section {
    background: white;

    border-radius: 15px;

    padding: 35px;

    margin-top: 25px;

    box-shadow:
        0 8px 25px rgba(25, 45, 80, 0.08);

    overflow-x: auto;
}

.section-title {
    font-size: 30px;

    margin-bottom: 25px;

    color: #17233c;
}


/* ================= TABLE ================= */

.table-wrapper {
    width: 100%;
    overflow-x: auto;
}

table {
    width: 100%;

    border-collapse: collapse;

    min-width: 650px;
}

th {
    background: #f5f7fa;

    color: #34435c;

    text-align: left;

    padding: 15px;

    font-size: 15px;
}

td {
    padding: 15px;

    border-bottom: 1px solid #edf0f4;

    color: #536176;
}

.empty {
    text-align: center;

    padding: 45px;

    color: #6d7d96;
}


/* ================= FORM ================= */

.form-grid {
    display: grid;

    grid-template-columns:
        repeat(2, minmax(0, 1fr));

    gap: 20px;
}

.form-group {
    display: flex;

    flex-direction: column;
}

.form-group label {
    margin-bottom: 8px;

    color: #34435c;

    font-weight: bold;
}

.form-group input,
.form-group select {
    width: 100%;

    padding: 13px;

    border: 1px solid #d8dee8;

    border-radius: 8px;

    font-size: 15px;

    background: white;
}

.form-group input:focus,
.form-group select:focus {
    outline: none;

    border-color: #1769e0;
}


/* ================= BUTTON ================= */

.button {
    display: inline-block;

    border: none;

    border-radius: 8px;

    padding: 14px 24px;

    margin-top: 25px;

    background: #1769e0;

    color: white;

    font-size: 16px;

    cursor: pointer;

    text-decoration: none;
}

.button:hover {
    background: #0d55bd;
}


/* ================= RESULT ================= */

.result {
    background: white;

    border-radius: 15px;

    padding: 35px;

    margin-top: 30px;

    box-shadow:
        0 8px 25px rgba(25, 45, 80, 0.08);
}

.result-title {
    font-size: 30px;

    margin-bottom: 20px;
}

.result-item {
    margin: 12px 0;

    font-size: 18px;
}

.high {
    color: #d62929;
    font-weight: bold;
}

.medium {
    color: #df6c13;
    font-weight: bold;
}

.low {
    color: #169b50;
    font-weight: bold;
}


/* ================= FOOTER ================= */

.footer {
    text-align: center;

    color: #8290a6;

    padding: 35px 20px;

    margin-top: 30px;

    border-top: 1px solid #dce2eb;
}


/* =========================================================
   MOBILE RESPONSIVE
   ========================================================= */

@media (max-width: 768px) {

    .navbar {
        width: 100%;

        padding: 15px 20px;

        flex-direction: column;

        align-items: center;

        justify-content: center;

        gap: 14px;

        text-align: center;
    }

    .logo {
        font-size: 23px;
    }

    .nav-links {
        width: 100%;

        justify-content: center;

        gap: 12px 20px;
    }

    .nav-links a {
        font-size: 15px;
    }


    .container {
        width: 100%;

        max-width: 100%;

        padding: 30px 24px;
    }


    h1 {
        font-size: 34px;

        line-height: 1.2;
    }

    .subtitle {
        font-size: 17px;

        line-height: 1.5;

        margin-bottom: 25px;
    }


    /* 2 x 2 cards */

    .stats {
        grid-template-columns: 1fr 1fr;

        gap: 16px;
    }

    .card {
        width: 100%;

        min-width: 0;

        min-height: 150px;

        padding: 22px 18px;
    }

    .card-title {
        font-size: 15px;
    }

    .card-value {
        font-size: 38px;
    }


    .section {
        width: 100%;

        padding: 25px 20px;

        margin-top: 25px;

        overflow-x: auto;
    }

    .section-title {
        font-size: 27px;
    }


    .form-grid {
        grid-template-columns: 1fr;
    }


    .footer {
        font-size: 14px;
    }
}


/* ================= SMALL MOBILE ================= */

@media (max-width: 480px) {

    .container {
        padding: 25px 18px;
    }

    h1 {
        font-size: 30px;
    }

    .subtitle {
        font-size: 16px;
    }

    .navbar {
        padding: 14px 12px;
    }

    .logo {
        font-size: 21px;
    }

    .nav-links {
        gap: 10px 14px;
    }

    .nav-links a {
        font-size: 14px;
    }

    .stats {
        grid-template-columns: 1fr 1fr;

        gap: 12px;
    }

    .card {
        padding: 20px 14px;

        min-height: 135px;
    }

    .card-title {
        font-size: 14px;
    }

    .card-value {
        font-size: 34px;
    }

    .section {
        padding: 22px 16px;
    }

    .section-title {
        font-size: 24px;
    }
}

</style>
"""


# =========================================================
# BASE HTML
# =========================================================

BASE_HTML = """
<!DOCTYPE html>

<html lang="en">

<head>

    <meta charset="UTF-8">

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >

    <title>{{ title }}</title>

    {{ css|safe }}

</head>

<body>

    <div class="navbar">

        <div class="logo">
            🛡️ UPI FraudGuard
        </div>

        <div class="nav-links">

            <a href="{{ url_for('dashboard') }}">
                Dashboard
            </a>

            <a href="{{ url_for('predict') }}">
                Check Transaction
            </a>

            <a href="{{ url_for('history') }}">
                History
            </a>

        </div>

    </div>


    {{ content|safe }}


    <div class="footer">
        UPI FraudGuard &nbsp; | &nbsp;
        Secure Transactions. Safer Tomorrow.
    </div>

</body>

</html>
"""


# =========================================================
# DASHBOARD
# =========================================================

DASHBOARD_CONTENT = """

<div class="container">

    <h1>
        UPI Fraud Detection Dashboard
    </h1>

    <div class="subtitle">
        Machine Learning Based Transaction Risk Monitoring
    </div>


    <div class="stats">

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
                {{ fraud_rate }}%
            </div>

        </div>

    </div>


    <div class="section">

        <div class="section-title">
            Recent Transactions
        </div>

        <div class="table-wrapper">

            <table>

                <thead>

                    <tr>
                        <th>Time</th>
                        <th>Amount</th>
                        <th>Probability</th>
                        <th>Prediction</th>
                        <th>Risk</th>
                    </tr>

                </thead>

                <tbody>

                {% if transactions %}

                    {% for row in transactions %}

                    <tr>

                        <td>
                            {{ row[1] }}
                        </td>

                        <td>
                            ₹{{ "%.2f"|format(row[2]) }}
                        </td>

                        <td>
                            {{ "%.2f"|format(row[13] * 100) }}%
                        </td>

                        <td>
                            {{ row[14] }}
                        </td>

                        <td>
                            {{ row[15] }}
                        </td>

                    </tr>

                    {% endfor %}

                {% else %}

                    <tr>

                        <td
                            colspan="5"
                            class="empty"
                        >
                            No transactions yet.
                        </td>

                    </tr>

                {% endif %}

                </tbody>

            </table>

        </div>

    </div>

</div>

"""


# =========================================================
# PREDICTION PAGE
# =========================================================

PREDICT_CONTENT = """

<div class="container">

    <h1>
        Check UPI Transaction
    </h1>

    <div class="subtitle">
        Enter transaction details to estimate fraud risk.
    </div>


    <div class="section">

        <form method="POST">

            <div class="form-grid">


                <div class="form-group">

                    <label>
                        Transaction Amount
                    </label>

                    <input
                        type="number"
                        name="amount"
                        step="0.01"
                        min="1"
                        required
                        placeholder="Example: 5000"
                    >

                </div>


                <div class="form-group">

                    <label>
                        Transaction Hour
                    </label>

                    <input
                        type="number"
                        name="hour"
                        min="0"
                        max="23"
                        required
                        placeholder="0 - 23"
                    >

                </div>


                <div class="form-group">

                    <label>
                        Transactions Last 24 Hours
                    </label>

                    <input
                        type="number"
                        name="transactions_last_24h"
                        min="0"
                        required
                        placeholder="Example: 5"
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
                        Account Age (Days)
                    </label>

                    <input
                        type="number"
                        name="account_age_days"
                        min="1"
                        required
                        placeholder="Example: 500"
                    >

                </div>


                <div class="form-group">

                    <label>
                        Failed Attempts
                    </label>

                    <input
                        type="number"
                        name="failed_attempts"
                        min="0"
                        required
                        placeholder="Example: 1"
                    >

                </div>


                <div class="form-group">

                    <label>
                        Previous Fraud Count
                    </label>

                    <input
                        type="number"
                        name="previous_fraud_count"
                        min="0"
                        required
                        placeholder="Example: 0"
                    >

                </div>


                <div class="form-group">

                    <label>
                        Merchant Risk
                    </label>

                    <input
                        type="number"
                        name="merchant_risk"
                        min="0"
                        max="1"
                        step="0.01"
                        required
                        placeholder="0.00 - 1.00"
                    >

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


            <button
                type="submit"
                class="button"
            >
                Analyze Transaction
            </button>

        </form>

    </div>


    {% if result %}

    <div class="result">

        <div class="result-title">
            Prediction Result
        </div>


        <div class="result-item">
            Fraud Probability:
            <strong>
                {{ probability }}%
            </strong>
        </div>


        <div class="result-item">

            Prediction:

            {% if prediction == "FRAUD" %}

                <strong class="high">
                    FRAUD
                </strong>

            {% else %}

                <strong class="low">
                    LEGITIMATE
                </strong>

            {% endif %}

        </div>


        <div class="result-item">

            Risk Level:

            {% if risk == "HIGH" %}

                <strong class="high">
                    HIGH
                </strong>

            {% elif risk == "MEDIUM" %}

                <strong class="medium">
                    MEDIUM
                </strong>

            {% else %}

                <strong class="low">
                    LOW
                </strong>

            {% endif %}

        </div>

    </div>

    {% endif %}

</div>

"""


# =========================================================
# HISTORY PAGE
# =========================================================

HISTORY_CONTENT = """

<div class="container">

    <h1>
        Transaction History
    </h1>

    <div class="subtitle">
        Previously analyzed UPI transactions
    </div>


    <div class="section">

        <div class="table-wrapper">

            <table>

                <thead>

                    <tr>

                        <th>ID</th>
                        <th>Time</th>
                        <th>Amount</th>
                        <th>Probability</th>
                        <th>Prediction</th>
                        <th>Risk</th>

                    </tr>

                </thead>

                <tbody>

                {% if transactions %}

                    {% for row in transactions %}

                    <tr>

                        <td>
                            {{ row[0] }}
                        </td>

                        <td>
                            {{ row[1] }}
                        </td>

                        <td>
                            ₹{{ "%.2f"|format(row[2]) }}
                        </td>

                        <td>
                            {{ "%.2f"|format(row[13] * 100) }}%
                        </td>

                        <td>
                            {{ row[14] }}
                        </td>

                        <td>
                            {{ row[15] }}
                        </td>

                    </tr>

                    {% endfor %}

                {% else %}

                    <tr>

                        <td
                            colspan="6"
                            class="empty"
                        >
                            No transactions found.
                        </td>

                    </tr>

                {% endif %}

                </tbody>

            </table>

        </div>

    </div>

</div>

"""


# =========================================================
# DASHBOARD ROUTE
# =========================================================

@app.route("/")
def dashboard():

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    cursor = connection.cursor()

    cursor.execute(
        "SELECT COUNT(*) FROM transactions"
    )

    total = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM transactions
        WHERE prediction = 'FRAUD'
        """
    )

    fraud = cursor.fetchone()[0]

    legitimate = total - fraud

    fraud_rate = (
        round((fraud / total) * 100, 2)
        if total > 0
        else 0
    )

    cursor.execute(
        """
        SELECT *
        FROM transactions
        ORDER BY id DESC
        LIMIT 10
        """
    )

    transactions = cursor.fetchall()

    connection.close()


    content = render_template_string(
        DASHBOARD_CONTENT,

        total=total,
        fraud=fraud,
        legitimate=legitimate,
        fraud_rate=fraud_rate,
        transactions=transactions
    )


    return render_template_string(
        BASE_HTML,

        title="UPI Fraud Detection",
        css=CSS,
        content=content
    )


# =========================================================
# PREDICT ROUTE
# =========================================================

@app.route(
    "/predict",
    methods=["GET", "POST"]
)
def predict():

    result = False

    probability = 0
    prediction = ""
    risk = ""


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
                request.form["device_changed"]
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

            merchant_risk = float(
                request.form["merchant_risk"]
            )

            is_new_payee = int(
                request.form["is_new_payee"]
            )


            values = [[
                amount,
                hour,
                transactions_last_24h,
                device_changed,
                location_changed,
                account_age_days,
                failed_attempts,
                previous_fraud_count,
                merchant_risk,
                is_new_payee
            ]]


            input_data = pd.DataFrame(
                values,
                columns=FEATURES
            )


            fraud_probability = model.predict_proba(
                input_data
            )[0][1]


            probability = round(
                fraud_probability * 100,
                2
            )


            if fraud_probability >= 0.50:

                prediction = "FRAUD"

            else:

                prediction = "LEGITIMATE"


            if fraud_probability >= 0.75:

                risk = "HIGH"

            elif fraud_probability >= 0.40:

                risk = "MEDIUM"

            else:

                risk = "LOW"


            from datetime import datetime

            timestamp = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )


            connection = sqlite3.connect(
                DATABASE_FILE
            )

            cursor = connection.cursor()


            cursor.execute(
                """
                INSERT INTO transactions (
                    timestamp,
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

                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,

                (
                    timestamp,
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
                    risk
                )
            )


            connection.commit()
            connection.close()


            result = True


        except Exception as error:

            print(
                "Prediction error:",
                error
            )


    content = render_template_string(
        PREDICT_CONTENT,

        result=result,
        probability=probability,
        prediction=prediction,
        risk=risk
    )


    return render_template_string(
        BASE_HTML,

        title="Check Transaction",
        css=CSS,
        content=content
    )


# =========================================================
# HISTORY ROUTE
# =========================================================

@app.route("/history")
def history():

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT *
        FROM transactions
        ORDER BY id DESC
        """
    )

    transactions = cursor.fetchall()

    connection.close()


    content = render_template_string(
        HISTORY_CONTENT,

        transactions=transactions
    )


    return render_template_string(
        BASE_HTML,

        title="Transaction History",
        css=CSS,
        content=content
    )


# =========================================================
# LOCAL BROWSER AUTO OPEN
# =========================================================

def open_browser():

    webbrowser.open_new(
        "http://127.0.0.1:5000/"
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    print()
    print("=" * 55)
    print("UPI FraudGuard")
    print("UPI Fraud Detection System")
    print("=" * 55)
    print()
    print(
        f"Running on http://127.0.0.1:{PORT}"
    )
    print()


    # Open browser only for local development
    if os.environ.get("RENDER") is None:

        threading.Timer(
            2,
            open_browser
        ).start()


    app.run(
        host=HOST,
        port=PORT,
        debug=False
    )