from fastapi import FastAPI
import pandas as pd
import pickle
import os
import json
from tensorflow.keras.models import model_from_json
import numpy as np
from pydantic import BaseModel
from typing import List

# Initialize FastAPI app
app = FastAPI(title="ML Prediction Service")

# Global variables for model and scaler
model = None
scaler = None

# List of features used by the model
selected_features = [
    "nb_funding_rounds",
    "nb_investors", 
    "nb_offices",
    "ipo",
    "acquired",
    "milestones",
    "relationships",
    "funding_rounds"
]

# Pydantic model for input validation
class PredictionInput(BaseModel):
    nb_funding_rounds: float = 0
    nb_investors: float = 0
    nb_offices: float = 0
    ipo: float = 0
    acquired: float = 0
    milestones: float = 0
    relationships: float = 0
    funding_rounds: float = 0

# Function to load ML model and scaler
def load_ml_model():
    """Load ML model and scaler from files"""
    global model, scaler
    
    try:
        print("🔍 Searching for model files...")
        
        # List of possible file paths (tries different locations)
        possible_paths = [
            # 1. Current directory
            {
                "arch": "mlp_model_architecture.json",
                "weights": "mlp_model.weights.h5", 
                "scaler": "mlp_model_scaler.pkl"
            },
            # 2. In Projects folder next to current directory
            {
                "arch": "../Projects/mlp_model_architecture.json",
                "weights": "../Projects/mlp_model.weights.h5",
                "scaler": "../Projects/mlp_model_scaler.pkl"
            },
            # 3. In Projects folder inside container
            {
                "arch": "Projects/mlp_model_architecture.json",
                "weights": "Projects/mlp_model.weights.h5",
                "scaler": "Projects/mlp_model_scaler.pkl"
            }
        ]
        
        loaded = False
        for paths in possible_paths:
            arch_path = paths["arch"]
            weights_path = paths["weights"]
            scaler_path = paths["scaler"]
            
            # Check if all files exist
            if (os.path.exists(arch_path) and 
                os.path.exists(weights_path) and 
                os.path.exists(scaler_path)):
                
                print(f"✅ Found files at: {arch_path}")
                
                # Load model architecture
                with open(arch_path, "r") as f:
                    model_json = f.read()
                model = model_from_json(model_json)
                
                # Load model weights
                model.load_weights(weights_path)
                
                # Load scaler
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
                
                loaded = True
                print("🎯 Model loaded successfully!")
                break
        
        # If no model files found, create a dummy model for testing
        if not loaded:
            print("⚠️ Model files not found, creating dummy model...")
            create_dummy_model()
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        create_dummy_model()

def create_dummy_model():
    """Create dummy model for testing when real model is not available"""
    global model, scaler
    print("🔄 Creating dummy model for testing...")
    
    # Dummy model
    model = "dummy_model"
    
    # Dummy scaler
    import sklearn.preprocessing
    scaler = sklearn.preprocessing.StandardScaler()
    import numpy as np
    scaler.fit(np.random.rand(10, 8))
    
    print("✅ Dummy model created for testing")

# Load model when application starts
@app.on_event("startup")
def startup_event():
    print("🚀 Starting ML service...")
    load_ml_model()
    print(f"📊 Features: {selected_features}")

# Home endpoint
@app.get("/")
def home():
    return {
        "message": "🎯 ML Prediction Service",
        "status": "running",
        "model": "loaded" if model else "not_loaded",
        "features": selected_features,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict (POST)",
            "test": "/test (GET)"
        }
    }

# Health check endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "loaded" if model and model != "dummy_model" else "dummy",
        "service": "ml_prediction_api",
        "features_count": len(selected_features)
    }

# Prediction endpoint
@app.post("/predict")
def predict(input_data: List[PredictionInput]):
    """
    Make predictions for multiple inputs
    """
    try:
        # Convert input data to DataFrame
        data = [item.dict() for item in input_data]
        df = pd.DataFrame(data)
        
        # Ensure all features are present
        for feature in selected_features:
            if feature not in df.columns:
                df[feature] = 0
        
        # If using dummy model, return random predictions
        if model == "dummy_model":
            import numpy as np
            predictions = np.random.rand(len(df), 1).tolist()
            return {
                "status": "success",
                "model": "dummy",
                "predictions": predictions,
                "count": len(predictions)
            }
        
        # Real prediction with actual model
        X = df[selected_features]
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled).tolist()
        
        return {
            "status": "success",
            "predictions": predictions,
            "count": len(predictions),
            "features_used": selected_features
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "error_type": type(e).__name__
        }

# Test endpoint with sample data
@app.get("/test")
def test_endpoint():
    """
    Test the model with dummy data
    """
    test_data = [
        PredictionInput(
            nb_funding_rounds=2.0,
            nb_investors=4.0,
            nb_offices=1.0,
            ipo=0.0,
            acquired=0.0,
            milestones=8.0,
            relationships=12.0,
            funding_rounds=2.0
        ),
        PredictionInput(
            nb_funding_rounds=5.0,
            nb_investors=8.0,
            nb_offices=3.0,
            ipo=1.0,
            acquired=0.0,
            milestones=15.0,
            relationships=20.0,
            funding_rounds=4.0
        )
    ]
    
    return predict(test_data)

# Show required input schema
@app.get("/input_schema")
def input_schema():
    """Display required input structure"""
    return {
        "required_fields": selected_features,
        "example": {
            "nb_funding_rounds": 3.0,
            "nb_investors": 5.0,
            "nb_offices": 2.0,
            "ipo": 0.0,
            "acquired": 0.0,
            "milestones": 10.0,
            "relationships": 15.0,
            "funding_rounds": 3.0
        }
    }

# Main entry point for running the application
if __name__ == "__main__":
    import uvicorn
    # Get port from environment variable or use default 8080
    port = int(os.environ.get("PORT", 9000))
    print(f"🚀 Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)