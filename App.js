import React, { useState } from "react";
import axios from "axios";

export default function CloudburstApp() {
  const [city, setCity] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchPrediction = async () => {
    if (!city.trim()) {
      setError("Please enter a valid city name.");
      return;
    }

    setLoading(true);
    setError(null); // Clear previous errors

    try {
      const response = await axios.get(`https://cloudburst-q09n.onrender.com/predict?city=${city}`);
      setPrediction(response.data);
    } catch (error) {
      setError("Failed to fetch prediction. Please try again.");
      console.error("Error fetching prediction:", error);
    }

    setLoading(false);
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-6">
      <h1 className="text-3xl font-bold mb-6">Cloudburst Prediction System</h1>

      <div className="flex gap-4">
        <input
          type="text"
          placeholder="Enter city name"
          value={city}
          onChange={(e) => setCity(e.target.value)}
          className="p-2 border rounded w-64"
          disabled={loading}
        />
        <button
          onClick={fetchPrediction}
          className={`px-4 py-2 rounded text-white ${loading ? "bg-gray-400 cursor-not-allowed" : "bg-blue-500 hover:bg-blue-600"}`}
          disabled={loading}
        >
          {loading ? "Loading..." : "Get Prediction"}
        </button>
      </div>

      {error && <p className="text-red-500 mt-4">{error}</p>}

      {prediction && !error && (
        <div className="mt-6 p-4 border rounded bg-white shadow-lg">
          <h2 className="text-lg font-semibold">Prediction Result:</h2>
          <p><strong>Risk Level:</strong> {prediction.risk_level}</p>
          <p><strong>Probability:</strong> {prediction.probability}</p>
        </div>
      )}
    </div>
  );
}