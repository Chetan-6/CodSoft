from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load Model

model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "Titanic Survival API Running"

@app.route("/predict", methods = ["POST"])
def predict():
    data = request.get_json()

    # Example input: [Pclass, Sex, Age, Fare]
    features = np.array(data["features"]).reshape(1, -1)

    prediction = model.predict(features)

    return jsonify({
        "prediction": int(prediction[0])
    })

if __name__ == "__main__":
    app.run(debug=True)
