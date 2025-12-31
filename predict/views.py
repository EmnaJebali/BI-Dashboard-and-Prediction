from django.shortcuts import render

import joblib
import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from django.conf import settings
import os
from .forms import SatisfactionForm


FLIGHTS_MODEL_PATH = os.path.join(
    settings.BASE_DIR,
    "ml_models",
    "FlightsPrediction.pkl"
)

SATISFACTION_MODEL_PATH = os.path.join(
    settings.BASE_DIR,
    "ml_models",
    "satisfaction_model.joblib"
)


# Safely load the Prophet model for forecasting total number of flights
model = None
model_loaded = False
try:
    if os.path.exists(FLIGHTS_MODEL_PATH):
        model = joblib.load(FLIGHTS_MODEL_PATH)
        model_loaded = True
except Exception as e:
    print(f"Error loading flights model: {e}")

# Safely load the satisfaction model
satisfaction_model = None
satisfaction_model_loaded = False
try:
    if os.path.exists(SATISFACTION_MODEL_PATH):
        satisfaction_model = joblib.load(SATISFACTION_MODEL_PATH)
        satisfaction_model_loaded = True
except Exception as e:
    print(f"Error loading satisfaction model: {e}")




def prediction_selector(request):
    """Display prediction type selection page"""
    return render(request, 'prediction_selector.html')   



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
    """Flights prediction form"""
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
            "error": error,
            "prediction_type": "Flights"
        }
    )


def flights_prediction(request):
    """Flights prediction form"""
    return prediction_form(request)


def satisfaction_prediction(request):
    """Customer satisfaction prediction form"""
    prediction = None
    error = None
    form = SatisfactionForm()

    if request.method == "POST" and satisfaction_model_loaded:
        form = SatisfactionForm(request.POST)
        if form.is_valid():
            try:
                # Extract form data
                data = form.cleaned_data
                
                # Map form fields to model feature order (24 features total)
                gender_male = 1 if data.get('gender') == 'Male' else 0
                customer_type_disloyal = 1 if data.get('customer_type') == 'disloyal Customer' else 0
                travel_personal = 1 if data.get('type_of_travel') == 'Personal Travel' else 0
                class_eco = 1 if data.get('flight_class') == 'Eco' else 0
                class_eco_plus = 1 if data.get('flight_class') == 'Eco Plus' else 0
                
                departure_delay = data.get('departure_delay', 0) or 0
                arrival_delay = data.get('arrival_delay', 0) or 0
                total_delay = departure_delay + arrival_delay
                
                features = [
                    data.get('age'),  # Age
                    data.get('flight_distance'),  # Flight Distance
                    data.get('seat_comfort'),  # Seat comfort
                    0,  # Departure/Arrival time convenient (not in form, default 0)
                    0,  # Food and drink (not in form, default 0)
                    0,  # Gate location (not in form, default 0)
                    0,  # Inflight wifi service (not in form, default 0)
                    data.get('inflight_entertainment'),  # Inflight entertainment
                    data.get('online_support'),  # Online support
                    data.get('ease_of_online_booking'),  # Ease of Online booking
                    data.get('on_board_service'),  # On-board service
                    data.get('leg_room_service'),  # Leg room service
                    data.get('baggage_handling'),  # Baggage handling
                    data.get('checkin_service'),  # Checkin service
                    0,  # Cleanliness (not in form, default 0)
                    data.get('online_boarding'),  # Online boarding
                    departure_delay,  # Departure Delay in Minutes
                    arrival_delay,  # Arrival Delay in Minutes
                    gender_male,  # Gender_Male
                    customer_type_disloyal,  # Customer Type_disloyal Customer
                    travel_personal,  # Type of Travel_Personal Travel
                    class_eco,  # Class_Eco
                    class_eco_plus,  # Class_Eco Plus
                    total_delay,  # Total Delay
                ]
                
                # Make prediction
                prediction_result = satisfaction_model.predict([features])
                prediction_proba = satisfaction_model.predict_proba([features])
                
                # Get class names/labels
                classes = satisfaction_model.classes_
                
                prediction = {
                    "predicted_class": int(prediction_result[0]),
                    "class_name": str(classes[int(prediction_result[0])]),
                    "probabilities": {
                        str(classes[i]): float(prob) for i, prob in enumerate(prediction_proba[0])
                    }
                }
            except Exception as e:
                error = f"Prediction error: {str(e)}"
        else:
            error = "Please fill in all required fields correctly"

    return render(
        request,
        "satisfaction_form.html",
        {
            "form": form,
            "model_loaded": satisfaction_model_loaded,
            "prediction": prediction,
            "error": error,
            "prediction_type": "Customer Satisfaction"
        }
    )


def revenue_prediction(request):
    """Revenue prediction form - placeholder"""
    return render(
        request,
        "prediction_form.html",
        {
            "model_loaded": False,
            "prediction": None,
            "error": "Revenue prediction model coming soon",
            "prediction_type": "Revenue"
        }
    )
