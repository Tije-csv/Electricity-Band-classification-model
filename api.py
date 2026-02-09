import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import os

# Import functions from elec.py
from elec import generate_raw_data, prepare_dataset, train_and_evaluate

app = FastAPI(title="Electricity Supply Band Predictor", 
              description="Predicts electricity supply bands for Nigerian power grid feeders")

# Global variables for model and encoder
model = None
label_encoder = None
feature_names = None

class FeederInput(BaseModel):
    disco: str  # IKEDC, AEDC, EKEDC, KEDCO, IBEDC
    zone: str   # Urban, Suburban, Rural
    feeder_age: float
    transformer_issue: bool

class PredictionResponse(BaseModel):
    supply_band: str
    confidence: float

def load_or_train_model():
    """Load saved model or train a new one"""
    global model, label_encoder, feature_names
    
    model_path = "electricity_model.pkl"
    encoder_path = "label_encoder.pkl"
    features_path = "feature_names.pkl"
    
    if os.path.exists(model_path) and os.path.exists(encoder_path) and os.path.exists(features_path):
        print("Loading saved model...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        with open(features_path, 'rb') as f:
            feature_names = pickle.load(f)
    else:
        print("Training new model...")
        raw_df = generate_raw_data(2000)
        data_splits, label_encoder = prepare_dataset(raw_df)
        model = train_and_evaluate(data_splits, label_encoder)
        
        # Store feature names
        X_train, X_test, y_train, y_test = data_splits
        feature_names = X_train.columns.tolist()
        
        # Save model and encoder
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(encoder_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        with open(features_path, 'wb') as f:
            pickle.dump(feature_names, f)
        
        print("Model saved successfully!")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_or_train_model()
    print("Model loaded and API ready!")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Electricity Supply Band Predictor API",
        "docs": "/docs",
        "version": "1.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(feeder: FeederInput):
    """
    Predict the electricity supply band for a feeder.
    
    - **disco**: Distribution company (IKEDC, AEDC, EKEDC, KEDCO, IBEDC)
    - **zone**: Area type (Urban, Suburban, Rural)
    - **feeder_age**: Age of feeder in years
    - **transformer_issue**: Whether feeder has transformer issues
    """
    
    # Create input dataframe
    input_data = pd.DataFrame([{
        'disco': feeder.disco,
        'zone': feeder.zone,
        'feeder_age': feeder.feeder_age,
        'transformer_issue': feeder.transformer_issue
    }])
    
    # One-hot encode categorical features
    input_data = pd.get_dummies(input_data, columns=['disco', 'zone'], drop_first=True)
    
    # Ensure all required features are present
    for col in feature_names:
        if col not in input_data.columns:
            input_data[col] = 0
    
    # Reorder columns to match training data
    input_data = input_data[feature_names]
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    confidence = float(max(probabilities))
    
    # Convert prediction to band label
    supply_band = label_encoder.classes_[prediction]
    
    return PredictionResponse(
        supply_band=supply_band,
        confidence=round(confidence, 4)
    )

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
