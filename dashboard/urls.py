from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    path('', views.home, name='home'),
    path('dashboard/', views.powerbi_dashboard, name='powerbi_dashboard'),
    path('predict/', views.prediction_form, name='prediction_form'),
    path('api/predict/', views.predict_api, name='predict_api'),
]

