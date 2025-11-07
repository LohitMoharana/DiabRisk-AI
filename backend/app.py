import os
import joblib
import pandas as pd
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any

# --- Import your custom advice module ---
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
try:
    from nlp.advice import get_lifestyle_advice
except ImportError:
    print("Error: Could not import 'get_lifestyle_advice'. Using dummy fallback.")


    def get_lifestyle_advice(prob, feats):
        return "Error", [{"topic": "Error", "message": "Advice module not loaded."}]

# --- Initialize FastAPI ---
app = FastAPI(title="DiabRisk AI API")

# --- Enable CORS for frontend ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load model artifacts ---
# Note: Adjusted path to be relative to app.py
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'src/models')
MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
IMPUTER_PATH = os.path.join(MODEL_DIR, 'imputer.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'encoder.pkl')
FEATURES_PATH = os.path.join(MODEL_DIR, 'feature_columns.json')

artifacts = {}

try:
    artifacts['model'] = joblib.load(MODEL_PATH)
    artifacts['scaler'] = joblib.load(SCALER_PATH)
    artifacts['imputer'] = joblib.load(IMPUTER_PATH)
    artifacts['encoder'] = joblib.load(ENCODER_PATH)
    with open(FEATURES_PATH, 'r') as f:
        artifacts['model_features'] = json.load(f)
    artifacts['categorical_features'] = ['AGE_BIN', 'BMI_CLASS']
    print("--- All model artifacts loaded successfully! ---")
except Exception as e:
    print(f"Error loading model artifacts: {e}")
    artifacts = None


# --- Pydantic model for request validation ---
# This defines the exact structure the API expects
class RiskInput(BaseModel):
    AGE: Optional[float] = None
    AGE_BIN: Optional[str] = None
    SEX: Optional[int] = None
    BMI: Optional[float] = None
    BMI_CLASS: Optional[str] = None
    WAIST_HT_RATIO: Optional[float] = None
    SBP_MEAN: Optional[float] = None
    DBP_MEAN: Optional[float] = None
    HYPERTENSION_FLAG: Optional[int] = None
    TCHOL: Optional[float] = None
    HDL: Optional[float] = None
    TRIG: Optional[float] = None
    CHOL_HDL_RATIO: Optional[float] = None
    TG_TO_HDL: Optional[float] = None
    ACR: Optional[float] = None
    SMOKER: Optional[int] = None
    PA_SCORE: Optional[float] = None


# --- Health Check Endpoint ---
@app.get("/health")
def health():
    if not artifacts:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")
    return {"status": "healthy", "artifacts_loaded": True}


# --- Prediction Endpoint ---
@app.post("/predict")
def predict(data: RiskInput):
    if not artifacts:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")

    try:
        # Convert Pydantic model to dictionary
        input_dict = data.dict()
        input_df = pd.DataFrame([input_dict])
        input_df_for_advice = input_df.copy()  # For the advice function

        # Re-order columns to match the training set
        try:
            input_df = input_df[artifacts['model_features']]
        except KeyError as e:
            raise HTTPException(status_code=400, detail=f"Missing feature in input: {str(e)}")

        # 3. --- Start Preprocessing Pipeline ---

        # Step 3a: Encode Categorical Features
        cat_features = artifacts['categorical_features']
        # Handle potential 'None' values before encoding
        for col in cat_features:
            if input_df[col].isnull().any():
                # Encoder might fail on None, replace with a placeholder if necessary
                # Or ensure your encoder was trained to handle NaNs/None
                # For OrdinalEncoder, 'None' might be treated as a new category
                # We'll fill with a known "missing" string if needed, but let's try
                input_df[col] = input_df[col].fillna('None')  # A safer way

        input_df[cat_features] = artifacts['encoder'].transform(input_df[cat_features])

        # Step 3b: Impute Missing Values
        input_imputed = artifacts['imputer'].transform(input_df)
        input_df_imputed = pd.DataFrame(input_imputed, columns=artifacts['model_features'])

        # Step 3c: Scale Features
        input_scaled = artifacts['scaler'].transform(input_df_imputed)

        # 4. --- Make Prediction ---
        prediction_proba = artifacts['model'].predict_proba(input_scaled)
        risk_probability = float(prediction_proba[0][1])

        # 5. --- Generate Personalized Advice ---
        risk_level, advice_list = get_lifestyle_advice(risk_probability, input_df_for_advice)

        # 6. --- Return Full Response ---
        return {
            "risk_probability": round(risk_probability, 4),
            "risk_level": risk_level,
            "advice": advice_list,
            "message": "Prediction successful"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        # Return a more specific error if possible
        if "transform" in str(e):
            return HTTPException(status_code=422, detail=f"Encoding error. Check categorical values. {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")