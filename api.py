from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

# Initialize FastAPI app
app = FastAPI()

# Load trained model
try:
    model_path = "fraud_detection_model.pkl"
    with open(model_path, "rb") as file:
        model = pickle.load(file)

    # Ensure model is an instance of RandomForestClassifier
    if not isinstance(model, RandomForestClassifier):
        raise ValueError("Loaded object is not a valid RandomForestClassifier model.")

    print(f"Model loaded successfully from {model_path}")

except Exception as e:
    print(f"Error loading model: {e}")
    raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")

# Define request body schema
class RequestInput(BaseModel):
    features: list[list[float]]  # Expecting a list of lists

# Health check route
@app.get("/")
def home():
    return {"message": "Fraud detection API is running"}

# Prediction route
@app.post("/predict/")
def predict(data: RequestInput):
    try:
        # Convert input to NumPy array
        X = np.array(data.features)

        # Debugging: Print input shape
        print("Input Shape:", X.shape)

        # Validate input shape
        expected_features = 8  # Update based on your dataset
        if X.shape[1] != expected_features:
            raise HTTPException(status_code=400, detail=f"Expected {expected_features} features, but got {X.shape[1]}.")

        # Make prediction
        prediction = model.predict(X)

        return {"prediction": prediction.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
