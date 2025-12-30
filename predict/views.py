from django.shortcuts import render

import joblib
import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from django.conf import settings
import os

MODEL_PATH = os.path.join(
    settings.BASE_DIR,
    "static",
    "FlightsPrediction.pkl"
)

# Safely load the Prophet model for forecasting total number of flights
model = None
model_loaded = False
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        model_loaded = True
except Exception as e:
    print(f"Error loading model: {e}")   



class ForecastAPIView(APIView):
    def get(self, request):
        if not model_loaded:
            return Response(
                {"error": "Model not loaded"},
                status=500
            )
        
        try:
            periods = int(request.GET.get("months", 6))
            future = model.make_future_dataframe(
                periods=periods,
                freq='M'
            )
            forecast = model.predict(future)
            result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']] \
                        .tail(periods)
            return Response(
                result.to_dict(orient="records")
            )
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=500
            )


def prediction_form(request):
    prediction = None
    error = None

    if request.method == "POST" and model_loaded:
        try:
            months = int(request.POST.get("months", 6))
            future = model.make_future_dataframe(periods=months, freq="M")
            forecast = model.predict(future)
            prediction = forecast[["ds", "yhat"]].tail(months)
        except Exception as e:
            error = str(e)

    return render(
        request,
        "prediction_form.html",
        {
            "model_loaded": model_loaded,
            "prediction": prediction.to_dict(orient="records")
            if prediction is not None else None,
            "error": error
        }
    )
