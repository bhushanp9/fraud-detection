from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Allow frontend requests from any origin

# Load model once at startup
MODEL_PATH = os.getenv("MODEL_PATH", "fraud_detection_model.pkl")
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"❌ Failed to load model: {e}")

# Feature names expected by the model (30 features, no 'Class')
FEATURES = (
    ["Time"] +
    [f"V{i}" for i in range(1, 29)] +
    ["Amount"]
)


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "endpoints": {
            "POST /predict": "Predict fraud for a single transaction",
            "POST /predict/batch": "Predict fraud for multiple transactions",
            "GET /features": "List expected input features",
        }
    })


@app.route("/features", methods=["GET"])
def get_features():
    """Return the list of features the model expects."""
    return jsonify({"features": FEATURES, "count": len(FEATURES)})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict whether a single transaction is fraudulent.

    Expected JSON body:
    {
        "Time": 0,
        "V1": -1.35, "V2": -0.07, ..., "V28": 0.01,
        "Amount": 149.62
    }

    Returns:
    {
        "prediction": 0 or 1,
        "label": "Legitimate" or "Fraudulent",
        "fraud_probability": 0.03,
        "legitimate_probability": 0.97
    }
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    # Validate all features are present
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({
            "error": "Missing features",
            "missing": missing,
            "hint": f"Send all {len(FEATURES)} features. Check GET /features for the full list."
        }), 400

    try:
        input_array = np.array([[data[f] for f in FEATURES]])
        prediction = int(model.predict(input_array)[0])
        probabilities = model.predict_proba(input_array)[0]

        return jsonify({
            "prediction": prediction,
            "label": "Fraudulent" if prediction == 1 else "Legitimate",
            "fraud_probability": round(float(probabilities[1]), 4),
            "legitimate_probability": round(float(probabilities[0]), 4),
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Predict fraud for multiple transactions at once.

    Expected JSON body:
    {
        "transactions": [
            {"Time": 0, "V1": -1.35, ..., "Amount": 149.62},
            {"Time": 1, "V1": 1.19, ..., "Amount": 2.69}
        ]
    }
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json(silent=True)
    if not data or "transactions" not in data:
        return jsonify({"error": "Expected JSON with a 'transactions' list"}), 400

    transactions = data["transactions"]
    if not isinstance(transactions, list) or len(transactions) == 0:
        return jsonify({"error": "'transactions' must be a non-empty list"}), 400

    if len(transactions) > 1000:
        return jsonify({"error": "Batch size limit is 1000 transactions"}), 400

    results = []
    errors = []

    for i, txn in enumerate(transactions):
        missing = [f for f in FEATURES if f not in txn]
        if missing:
            errors.append({"index": i, "error": f"Missing features: {missing}"})
            continue
        try:
            row = np.array([[txn[f] for f in FEATURES]])
            pred = int(model.predict(row)[0])
            proba = model.predict_proba(row)[0]
            results.append({
                "index": i,
                "prediction": pred,
                "label": "Fraudulent" if pred == 1 else "Legitimate",
                "fraud_probability": round(float(proba[1]), 4),
            })
        except Exception as e:
            errors.append({"index": i, "error": str(e)})

    return jsonify({
        "total": len(transactions),
        "processed": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV", "development") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
