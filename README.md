# UPI Fraud Detection System

A machine-learning-based UPI fraud detection web application developed using:

- Python
- Flask
- Scikit-learn
- Random Forest
- Pandas
- NumPy
- SQLite
- Bootstrap

## Modules

1. Login
2. Dashboard
3. Transaction analysis
4. Fraud prediction
5. Risk classification
6. Transaction history
7. Machine learning model
8. Synthetic dataset generation

## Machine Learning Algorithm

Random Forest Classifier is used.

## Input Features

- Transaction amount
- Transaction hour
- Transactions in last 24 hours
- Device change
- Location change
- Account age
- Failed attempts
- Previous fraud count
- Merchant risk
- New payee

## Output

The system produces:

- Fraud probability
- FRAUD / LEGITIMATE prediction
- LOW / MEDIUM / HIGH risk

## Installation

Create virtual environment:

python -m venv venv

Activate it.

Windows:

venv\Scripts\activate

Linux/macOS:

source venv/bin/activate

Install packages:

pip install -r requirements.txt

Run:

python app.py

Open:

http://127.0.0.1:5000

## Login

Username:

admin

Password:

admin123

## Important

This project uses synthetic data for educational purposes. It is not a real UPI/banking fraud-detection service.