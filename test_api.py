import requests
import json

# This is the sample data your frontend will send.
# We set 'TRIG' and 'TG_TO_HDL' to None (null)
# to prove that our API's imputer is working.
sample_data = {
    "AGE": 55,
    "AGE_BIN": "50-59",  # The encoder will handle this string
    "SEX": 1,
    "BMI": 29.5,
    "BMI_CLASS": "Obese",  # The encoder will handle this string
    "WAIST_HT_RATIO": 0.58,
    "SBP_MEAN": 135,
    "DBP_MEAN": 88,
    "HYPERTENSION_FLAG": 1,
    "TCHOL": 210,
    "HDL": 45,
    "TRIG": None,  # <-- Set to None (null) to test the imputer
    "CHOL_HDL_RATIO": 4.6,
    "TG_TO_HDL": None,  # <-- Set to None (null) to test the imputer
    "ACR": 15,
    "SMOKER": 1,
    "PA_SCORE": 100
}

# The URL of your local Flask API
url = "http://127.0.0.1:5000/predict"

print(f"Sending test prediction to {url}...")
try:
    response = requests.post(url, json=sample_data)

    if response.status_code == 200:
        print("\n--- ✅ Prediction Successful! ---")
        print("Received response from server:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"\n--- ❌ Error: {response.status_code} ---")
        print("Server returned an error:")
        print(response.text)

except requests.exceptions.ConnectionError:
    print("\n--- ❌ Connection Error ---")
    print("Error: Could not connect.")
    print("Is the 'app.py' server running in your 'backend/' folder?")