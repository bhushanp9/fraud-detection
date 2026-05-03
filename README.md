# 🛡️ Credit Card Fraud Detection

A machine learning system that detects fraudulent credit card transactions in real time using a **Random Forest Classifier** trained on 284,807 transactions. Exposes predictions via a **REST API** built with Flask.

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 99.9% |
| Fraud Precision | 96% |
| Fraud Recall | 74% |
| ROC-AUC Score | **0.9529** |

> ⚠️ The dataset is highly imbalanced (0.17% fraud). `class_weight='balanced'` was used to handle this without oversampling.

---

## 🧠 Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.12 |
| ML Model | Scikit-learn — RandomForestClassifier |
| API | Flask + Flask-CORS |
| Serialization | Joblib |
| Dataset | [Kaggle Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |

---

## 📁 Project Structure

```
fraud-detection/
├── app.py                  # Flask REST API
├── train.py                # Model training script
├── fraud_detection_model.pkl  # Trained model
├── requirements.txt        # Python dependencies
└── .gitignore
```

---

## ⚙️ Setup & Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/bhushanp9/fraud-detection.git
cd fraud-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root.

> The CSV is excluded from this repo (150MB — exceeds GitHub's limit).

### 4. Train the model (optional — `.pkl` already included)
```bash
python train.py
```

### 5. Start the API
```bash
python app.py
```
API will be running at `http://localhost:5000`

---

## 🌐 API Endpoints

### `GET /`
Health check — returns API status and available routes.

---

### `GET /features`
Returns the list of 30 input features the model expects.

---

### `POST /predict`
Predict whether a single transaction is fraudulent.

**Request body:**
```json
{
  "Time": 0,
  "V1": -1.3598071336738,
  "V2": -0.0727811733098497,
  "V3": 2.53634673796914,
  "V4": 1.37815522427443,
  "V5": -0.338320769942518,
  "V6": 0.462387777762292,
  "V7": 0.239598554061257,
  "V8": 0.0986979012610507,
  "V9": 0.363786969611213,
  "V10": 0.0907941719789316,
  "V11": -0.551599533260813,
  "V12": -0.617800855762348,
  "V13": -0.991389847235408,
  "V14": -0.311169353699879,
  "V15": 1.46817697209427,
  "V16": -0.470400525259478,
  "V17": 0.207971241929242,
  "V18": 0.0257905801985591,
  "V19": 0.403992960255733,
  "V20": 0.251412098239705,
  "V21": -0.018306777944153,
  "V22": 0.277837575558899,
  "V23": -0.110473910188767,
  "V24": 0.0669280749146731,
  "V25": 0.128539358273528,
  "V26": -0.189114843888824,
  "V27": 0.133558376740387,
  "V28": -0.0210530534538215,
  "Amount": 149.62
}
```

**Response:**
```json
{
  "prediction": 0,
  "label": "Legitimate",
  "fraud_probability": 0.0,
  "legitimate_probability": 1.0
}
```

---

### `POST /predict/batch`
Predict fraud for up to **1000 transactions** at once.

**Request body:**
```json
{
  "transactions": [
    { "Time": 0, "V1": -1.35, "V2": -0.07, "..": "...", "Amount": 149.62 },
    { "Time": 1, "V1": 1.19, "V2": 0.26, "..": "...", "Amount": 2.69 }
  ]
}
```

**Response:**
```json
{
  "total": 2,
  "processed": 2,
  "failed": 0,
  "results": [
    { "index": 0, "prediction": 0, "label": "Legitimate", "fraud_probability": 0.0 },
    { "index": 1, "prediction": 0, "label": "Legitimate", "fraud_probability": 0.01 }
  ],
  "errors": []
}
```

---

## 🔍 How It Works

1. **Data** — 284,807 real credit card transactions (European cardholders, Sept 2013)
2. **Features** — `Time`, `Amount`, and `V1–V28` (PCA-transformed for confidentiality)
3. **Preprocessing** — `Time` and `Amount` scaled with `StandardScaler`; V1–V28 already normalized
4. **Imbalance handling** — `class_weight='balanced'` penalizes misclassifying fraud more heavily
5. **Model** — 100 decision trees voting together (Random Forest)
6. **Top predictors** — V14, V10, V12, V4, V17 had the highest feature importance

---

## 🚀 Deployment

This API can be deployed for free on:
- [Render](https://render.com) — connect your GitHub repo, set start command to `python app.py`
- [Railway](https://railway.app) — one-click deploy from GitHub

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

**Bhushan** — [github.com/bhushanp9](https://github.com/bhushanp9)
