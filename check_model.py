import joblib
import os

path = r'C:\Users\emna1\Documents\3IA\ML_BI\static\satisfaction_model.joblib'
try:
    model = joblib.load(path)
    print("Model type:", type(model))
    print("Model loaded successfully!")
    print("Available methods/attributes:")
    print([attr for attr in dir(model) if not attr.startswith('_')][:20])
except Exception as e:
    print(f"Error: {e}")
