from flask import Flask, request, jsonify
import joblib
import numpy as np
from pathlib import Path

# Load trained model
model = joblib.load(Path("models/model.pkl"))

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    prediction = model.predict([np.array(data)])
    return jsonify({"prediction": int(prediction[0])})

@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    data = request.json["features"]
    prediction = model.predict([np.array(features) for features in data])
    return jsonify({"predictions": [int(p) for p in prediction]})

if __name__ == "__main__":
    print("🚀 Starting Flask API on http://0.0.0.0:8000")
    app.run(host="0.0.0.0", port=8000)