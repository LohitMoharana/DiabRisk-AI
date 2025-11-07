# This file contains the logic for generating personalized lifestyle advice
# based on a user's risk score and input features.

import pandas as pd


def define_risk_categories(probability):
    """Categorizes probability into a readable risk level."""
    if probability < 0.20:
        return "Low"
    elif probability < 0.50:
        return "Moderate"
    else:
        return "High"


def get_lifestyle_advice(probability, features_df):
    """
    Generates personalized advice.

    Args:
        probability (float): The predicted risk probability (e.g., 0.65).
        features_df (pd.DataFrame): The input features (single row DataFrame)
                                    *before* scaling, encoding, or imputing.

    Returns:
        tuple: (risk_level, advice_list)
    """

    risk_level = define_risk_categories(probability)
    advice = []

    # We use .iloc[0] because features_df is a 1-row DataFrame
    features = features_df.iloc[0]

    # --- 1. General Advice Based on Risk Level ---
    if risk_level == "Low":
        advice.append({
            "topic": "General",
            "message": "Your risk is currently low. Keep up your healthy lifestyle! Maintaining a balanced diet and regular physical activity is the best way to keep your risk low."
        })
    elif risk_level == "Moderate":
        advice.append({
            "topic": "General",
            "message": "Your risk is moderate. This is an important time to make positive lifestyle changes. Focus on improving your diet and increasing your physical activity."
        })
    else:  # High Risk
        advice.append({
            "topic": "Urgent",
            "message": "Your risk is high. Please schedule an appointment with your doctor or a healthcare provider to discuss these results and get professional medical advice."
        })

    # --- 2. Personalized Advice Based on Features ---
    # Try/except blocks handle potential 'None' values if imputer isn't used

    # Check BMI
    try:
        if features['BMI'] is not None and features['BMI'] > 25:
            advice.append({
                "topic": "Weight Management",
                "message": f"Your BMI is {features['BMI']:.1f}, which is in the overweight/obese range. Losing even 5-7% of your body weight can significantly reduce your diabetes risk."
            })
    except (TypeError, KeyError):
        pass

    # Check Blood Pressure
    try:
        if (features['HYPERTENSION_FLAG'] is not None and features['HYPERTENSION_FLAG'] == 1) or \
                (features['SBP_MEAN'] is not None and features['SBP_MEAN'] > 130):
            advice.append({
                "topic": "Blood Pressure",
                "message": "Your blood pressure appears elevated. Focus on reducing sodium (salt) in your diet, and discuss this with your doctor."
            })
    except (TypeError, KeyError):
        pass

    # Check Physical Activity
    try:
        # Assuming 150 MET-minutes/week is a good threshold
        if features['PA_SCORE'] is not None and features['PA_SCORE'] < 150:
            advice.append({
                "topic": "Physical Activity",
                "message": "Your physical activity level appears low. Aim for at least 150 minutes of moderate-intensity exercise, like brisk walking, per week."
            })
    except (TypeError, KeyError):
        pass

    # Check Smoking
    try:
        if features['SMOKER'] is not None and features['SMOKER'] == 1:  # Assuming 1 is 'Current Smoker'
            advice.append({
                "topic": "Smoking",
                "message": "Smoking significantly increases diabetes risk. Quitting is one of the most powerful steps you can take to improve your health. Seek resources to help you quit."
            })
    except (TypeError, KeyError):
        pass

    # Check Lipids (TG_TO_HDL)
    try:
        if features['TG_TO_HDL'] is not None and features['TG_TO_HDL'] > 3.0:  # A common clinical threshold
            advice.append({
                "topic": "Cholesterol & Fats",
                "message": "Your Triglyceride/HDL ratio is high, a strong indicator of insulin resistance. Focus on reducing refined carbohydrates (sugar, white bread) and saturated fats."
            })
    except (TypeError, KeyError):
        pass

    return risk_level, advice