from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
import joblib
import numpy as np

app = Flask(__name__)
CORS(app,origins=['https://frontdeployment.netlify.app'])

# Load trained model and scaler
rf_model = joblib.load("cloudburst_rf_model.pkl")
scaler = joblib.load("scaler.pkl")

DB_CONFIG = {
    "host": "localhost",
    "dbname": "postgres",
    "user": "postgres",
    "password": "Nisha@84"
}

def connect_db():
    return psycopg2.connect(**DB_CONFIG)

@app.route("/")
def home():
    return "Cloudburst Prediction API is running!"

@app.route("/predict", methods=["GET"])
def predict():
    city = request.args.get("city")
    if not city:
        return jsonify({"error": "City parameter is required"}), 400
    
    try:
        conn = connect_db()
        cur = conn.cursor()
        cur.execute("SELECT temperature, humidity, pressure, wind_speed FROM weather WHERE city = %s", (city,))
        row = cur.fetchone()
        conn.close()
        
        if not row:
            return jsonify({"error": "City data not found"}), 404
        
        # Debugging: Print the extracted data
        print(f"Extracted Data for {city}: {row}")

        # Prepare data for prediction
        input_data = np.array([row])
        input_scaled = scaler.transform(input_data)

        # Debugging: Print scaled data
        print(f"Scaled Features: {input_scaled}")

        # Make prediction
        prediction = rf_model.predict(input_scaled)[0]
        risk_level = "High Risk" if prediction == 1 else "Low Risk"

        # Debugging: Print prediction output
        print(f"Prediction for {city}: Risk Level = {risk_level}, Probability = {prediction}")

        return jsonify({"city": city, "risk_level": risk_level, "probability": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
