# Phishing Website Detection

A local phishing website detection dashboard using URL, content, and domain signals with an XGBoost classifier.

## Dataset

This project uses the real **Phishing Websites** dataset from the UCI Machine Learning Repository:

- Source: https://archive.ics.uci.edu/dataset/327/phishing+websites
- Download: https://archive.ics.uci.edu/static/public/327/phishing+websites.zip
- Records: 11,055
- Features: 30 integer features plus `Result`
- Labels: `-1` phishing (4,898), `1` legitimate (6,157)
- License: CC BY 4.0

Citation:

> Mohammad, R. & McCluskey, L. (2012). Phishing Websites. UCI Machine Learning Repository. https://doi.org/10.24432/C51W2X

The downloaded archive is stored in `data/uci-phishing-websites/`.

## Setup

From the project directory, install the Python dependencies:

```powershell
& "C:/Program Files/Python313/python.exe" -m pip install -r requirements.txt
```

## Train and evaluate

```powershell
& "C:/Program Files/Python313/python.exe" src/train_model.py
& "C:/Program Files/Python313/python.exe" src/evaluate.py
```

Training creates these files in `models/`:

- `model.pkl`
- `scaler.pkl`
- `feature_names.json`
- `feature_importance.png` (created by evaluation)

## Run the application

Open two PowerShell terminals in the project directory.

Terminal 1, start the API:

```powershell
& "C:/Program Files/Python313/python.exe" -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Terminal 2, start the frontend:

```powershell
& "C:/Program Files/Python313/python.exe" -m http.server 5173 --directory frontend
```

Open the dashboard at http://127.0.0.1:5173/.

## API

Check a URL:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body '{"url":"https://www.example.com"}'
```

Get model metrics:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/metrics" -Method Get
```

`POST /predict` accepts `url` and optional `html` fields. The response includes the prediction, phishing probability, confidence, and all extracted feature values.

## Model results

The reproducible held-out test results are:

```text
Accuracy:  0.9715
Precision: 0.9741
Recall:    0.9612
F1:        0.9676
ROC-AUC:   0.9962
```
