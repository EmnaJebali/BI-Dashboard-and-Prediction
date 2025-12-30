import joblib

path = r'C:\Users\emna1\Documents\3IA\ML_BI\static\satisfaction_model.joblib'
try:
    model = joblib.load(path)
    print("Feature names:", model.feature_names_in_)
    print("Number of features:", len(model.feature_names_in_))
except Exception as e:
    print(f"Error: {e}")
