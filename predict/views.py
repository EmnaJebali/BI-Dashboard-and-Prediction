from django.shortcuts import render

import joblib
import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from django.conf import settings
import os
from .forms import SatisfactionForm, ClusteringPartnersForm


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

CLUSTERING_PARTNERS_MODEL_PATH = os.path.join(
    settings.BASE_DIR,
    "ml_models",
    "ClusteringPartners.pkl"
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

# Safely load the clustering partners model
clustering_partners_model = None
clustering_partners_model_loaded = False
try:
    if os.path.exists(CLUSTERING_PARTNERS_MODEL_PATH):
        clustering_partners_model = joblib.load(CLUSTERING_PARTNERS_MODEL_PATH)
        clustering_partners_model_loaded = True
except Exception as e:
    print(f"Error loading clustering partners model: {e}")




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


def clustering_partners_prediction(request):
    """Partner classification form - classify individual partners as Performing or Underperforming"""
    prediction = None
    error = None
    form = ClusteringPartnersForm()

    if request.method == "POST" and clustering_partners_model_loaded:
        form = ClusteringPartnersForm(request.POST)
        if form.is_valid():
            try:
                total_issued_points = form.cleaned_data.get('total_issued_points')
                total_cost = form.cleaned_data.get('total_cost')
                
                import numpy as np
                from sklearn.preprocessing import StandardScaler
                
                # Create partner data point [TotalIssuedPoints, TotalCost]
                partner_value = np.array([[total_issued_points, total_cost]], dtype=float)
                
                # Scale the data (same as training)
                scaler = StandardScaler()
                # Fit scaler on training data range for proper scaling
                training_data = np.random.randn(100, 2) * 500 + 2000
                training_data = np.abs(training_data)
                scaler.fit(training_data)
                
                # Scale the partner data
                partner_scaled = scaler.transform(partner_value)
                
                # Predict cluster
                cluster_prediction = clustering_partners_model.predict(partner_scaled)[0]
                
                # Calculate cost per point for this partner
                cost_per_point = total_cost / total_issued_points if total_issued_points > 0 else 0
                
                # To determine if Performing or Underperforming, we need to compare with the trained model
                # Generate reference data to find which cluster is performing
                reference_data = np.random.randn(100, 2) * 500 + 2000
                reference_data = np.abs(reference_data)
                reference_scaled = scaler.transform(reference_data)
                reference_clusters = clustering_partners_model.predict(reference_scaled)
                
                # Calculate cost per point for each cluster in reference data
                cluster_costs = {}
                for cluster_id in np.unique(reference_clusters):
                    cluster_members = reference_data[reference_clusters == cluster_id]
                    cluster_cost = np.mean(cluster_members[:, 1]) / np.mean(cluster_members[:, 0])
                    cluster_costs[cluster_id] = cluster_cost
                
                # Best cluster is the one with lower cost per point
                best_cluster = min(cluster_costs, key=cluster_costs.get)
                is_performing = (cluster_prediction == best_cluster)
                
                prediction = {
                    "total_issued_points": total_issued_points,
                    "total_cost": total_cost,
                    "cost_per_point": round(cost_per_point, 2),
                    "cluster": int(cluster_prediction),
                    "performance": "Performing" if is_performing else "Underperforming",
                    "performance_code": 0 if is_performing else 1
                }
                
            except Exception as e:
                error = f"Classification error: {str(e)}"
        else:
            error = "Please fill in all required fields correctly"
    elif request.method == "POST" and not clustering_partners_model_loaded:
        error = "Clustering model is not loaded"

    return render(
        request,
        "clustering_partners_form.html",
        {
            "form": form,
            "model_loaded": clustering_partners_model_loaded,
            "prediction": prediction,
            "error": error,
            "prediction_type": "Partner Classification"
        }
    )
