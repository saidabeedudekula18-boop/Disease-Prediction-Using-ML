from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Disease Information
disease_info = {
    "Arthritis": {
        "description": "Arthritis is a condition that causes pain, swelling and stiffness in the joints.",
        "medicine": ["Ibuprofen", "Naproxen"],
        "diet": ["Milk", "Leafy Vegetables", "Vitamin D Foods"],
        "precautions": [
            "Regular Exercise",
            "Avoid Heavy Weight Lifting",
            "Take Proper Rest"
        ],
        "doctor": "Orthopedic Doctor"
    },

    "Migraine": {
        "description": "Migraine is a neurological disorder that causes severe headaches, nausea and sensitivity to light.",
        "medicine": ["Paracetamol", "Sumatriptan"],
        "diet": ["Drink Plenty of Water", "Avoid Caffeine"],
        "precautions": [
            "Sleep Well",
            "Reduce Stress",
            "Avoid Bright Light"
        ],
        "doctor": "Neurologist"
    }
}

# Load trained model
model = joblib.load("models/disease_model.pkl")

# Load dataset
df = pd.read_csv("dataset/Training.csv")
df = df.drop(columns=["Unnamed: 133"], errors="ignore")

symptoms = list(df.drop("prognosis", axis=1).columns)


@app.route("/")
def home():
    return render_template(
        "index.html",
        symptoms=symptoms,
        disease_info=disease_info
    )


@app.route("/predict", methods=["POST"])
def predict():

    selected = request.form.getlist("symptoms")

    input_data = [1 if symptom in selected else 0 for symptom in symptoms]

    print("Selected Symptoms:", selected)
    print("Input Data:", input_data)

    input_df = pd.DataFrame([input_data], columns=symptoms)

    prediction = model.predict(input_df)[0]

    print("Predicted Disease:", prediction)
    print("-" * 50)

    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)
        confidence = round(max(probabilities[0]) * 100, 2)

        prediction_note = ""

        if confidence is not None and confidence < 50:
            prediction_note = (
                "⚠️ This prediction has low confidence. "
                "Please consult a qualified medical professional for an accurate diagnosis."
            )

    info = disease_info.get(
        prediction,
        {
            "description": "No detailed description is available for this disease. Please consult a qualified doctor.",
            "medicine": ["Consult Doctor"],
            "diet": ["Healthy Balanced Diet"],
            "precautions": ["Take Proper Rest"],
            "doctor": "General Physician"
        }
    )

    return render_template(
        "index.html",
        symptoms=symptoms,
        prediction=prediction,
        selected=selected,
        info=info,
        disease_info=disease_info,
        confidence=confidence,
        prediction_note=prediction_note
    )


if __name__ == "__main__":
    app.run(debug=True)