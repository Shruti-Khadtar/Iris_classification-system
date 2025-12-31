# app.py
from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import numpy as np
import os

MODEL_PATH = "model_pipeline.pkl"
META_PATH = "meta.pkl"

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "replace-with-a-random-string")

# Load model and metadata at startup
if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
    raise FileNotFoundError("Run `python train_model.py` first to generate model files.")

model = joblib.load(MODEL_PATH)
meta = joblib.load(META_PATH)
target_names = meta.get("target_names", ["setosa", "versicolor", "virginica"])

@app.route("/", methods=["GET"])
def index():
    default = {"sepal_length": "", "sepal_width": "", "petal_length": "", "petal_width": ""}
    return render_template("index.html", result=None, default=default)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        sepal_length = float(request.form.get("sepal_length", "").strip())
        sepal_width = float(request.form.get("sepal_width", "").strip())
        petal_length = float(request.form.get("petal_length", "").strip())
        petal_width = float(request.form.get("petal_width", "").strip())
    except Exception:
        flash("Please enter valid numeric values.", "error")
        return redirect(url_for("index"))

    X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    pred_idx = model.predict(X)[0]
    pred_proba = model.predict_proba(X)[0]

    predicted_name = target_names[int(pred_idx)]
    proba_dict = {target_names[i]: float(pred_proba[i]) for i in range(len(target_names))}

    result = {
        "predicted": predicted_name,
        "probabilities": proba_dict,
        "inputs": {
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width
        }
    }

    return render_template("index.html", result=result, default=result["inputs"])

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
