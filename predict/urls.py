from django.urls import path
from .views import ForecastAPIView, prediction_form

app_name = "predict"

urlpatterns = [
    path("api/forecast/", ForecastAPIView.as_view(),name="forecast_api"),
    path("predict/", prediction_form, name="prediction_form"),
]
