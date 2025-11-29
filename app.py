from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# -------------------------
# Load saved model & scaler
# -------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        hours_studied = float(request.form['hours_studied'])
        hours_sleep = float(request.form['hours_sleep'])
        attendance = float(request.form['attendance'])
        previous_score = float(request.form['previous_score'])

        # Convert to array
        input_data = np.array([[hours_studied, hours_sleep, attendance, previous_score]])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]

        # Ensure prediction stays between 0 and 100
        prediction = max(0, min(100, prediction))

        return render_template("index.html",
                               prediction_text=f"Predicted Exam Score: {prediction:.2f}")

    except Exception as e:
        return render_template("index.html",
                               prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
