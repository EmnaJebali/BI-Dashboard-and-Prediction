from django.urls import path
from .views import (
    ForecastAPIView, 
    prediction_form,
    prediction_selector,
    flights_prediction,
    satisfaction_prediction,
    revenue_prediction
)

app_name = "predict"

urlpatterns = [
    path("", prediction_selector, name="prediction_selector"),
    path("flights/", flights_prediction, name="flights_prediction"),
    path("satisfaction/", satisfaction_prediction, name="satisfaction_prediction"),
    path("revenue/", revenue_prediction, name="revenue_prediction"),
    path("api/forecast/", ForecastAPIView.as_view(), name="forecast_api"),
]
