# Airline Fidelity Program Analysis Dashboard

A Django web application for analyzing airline customer fidelity and satisfaction through machine learning and business intelligence. This application provides:

1. **Power BI Dashboard Embedding** - Visualize airline performance metrics and trends through interactive Power BI dashboards
2. **ML Prediction Models** - Make predictions using trained machine learning models for:
   - Flight Volume Forecasting
   - Customer Satisfaction Prediction
   - Revenue Forecasting

## Features

- 🎨 Modern, responsive UI with Bootstrap 5
- 📊 Embedded Power BI dashboard visualization
- 🤖 Multiple ML model prediction services
- 🎯 Sidebar navigation with organized model selection
- 📱 Mobile-responsive design
- ⚡ Fast and efficient prediction API endpoints
- 🔧 Easy configuration for Power BI URL and model paths

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Power BI Dashboard

You have two options to set your Power BI embed URL:

**Option A: Environment Variable**
```bash
# Windows PowerShell
$env:POWERBI_EMBED_URL="https://app.powerbi.com/view?r=YOUR_EMBED_URL"

# Windows CMD
set POWERBI_EMBED_URL=https://app.powerbi.com/view?r=YOUR_EMBED_URL
```

**Option B: Settings File**
Edit `airline/settings.py` and set:
```python
POWERBI_EMBED_URL = 'https://app.powerbi.com/view?r=YOUR_EMBED_URL'
```

**How to get your Power BI embed URL:**
1. Open your Power BI report in Power BI Service
2. Click "File" → "Embed report" → "Website or portal"
3. Copy the embed URL

### 3. Add Your ML Models

Place your trained model files in the `static/` directory:
- `FlightsPrediction.pkl` - Prophet model for flight volume forecasting
- `satisfaction_model.joblib` - Satisfaction prediction model
- Additional models can be added and referenced in `predict/views.py`

**Supported model formats:**
- `.pkl` (pickle)
- `.joblib` (joblib)

**Model requirements:**
- Flight model: Prophet forecasting model with `.make_future_dataframe()` and `.predict()` methods
- Satisfaction model: Scikit-learn compatible model with `.predict_proba()` method
- Custom models can be added by extending the prediction views

### 4. Customize Prediction Forms

Edit the relevant form and view files to match your model's input features:

**For Flight Predictions:**
- `predict/views.py` - `prediction_form()` function
- `templates/prediction_form.html` - Flight prediction form

**For Satisfaction Predictions:**
- `predict/forms.py` - `SatisfactionForm` class
- `predict/views.py` - `satisfaction_prediction()` function
- `templates/satisfaction_form.html` - Satisfaction form

### 5. Run Migrations

```bash
python manage.py migrate
```

### 6. Run the Development Server

```bash
python manage.py runserver
```

Visit `http://127.0.0.1:8000/` to see the application.

## Usage

### Navigation

The application features a persistent sidebar with organized navigation:
- **Airline Fidelity** - Branding and home link
- **Dashboard Section** - Power BI Dashboard link
- **Predict Section** - Dropdown list of all available prediction models

### Home Page
The landing page provides quick access to:
- Power BI Dashboard - View comprehensive business intelligence dashboards
- ML Prediction - Access machine learning prediction models

### Power BI Dashboard
View your embedded Power BI dashboard with:
- Interactive visualizations
- Real-time data updates
- Drill-down capabilities
- Export and sharing features

### ML Predictions

#### Flight Volume Forecast
1. Navigate to Predict → Models → Flights Forecast
2. Enter the number of months to forecast (1-24)
3. Click "Predict" to generate forecasts
4. View the predicted flight volumes in the results table

#### Customer Satisfaction Prediction
1. Navigate to Predict → Models → Satisfaction
2. Fill out the comprehensive form with:
   - Basic information (gender, age, customer type, travel type)
   - Flight information (class, distance)
   - Service ratings (14 different rating scales)
   - Flight delay information
3. Click "Predict Satisfaction"
4. View satisfaction prediction with confidence scores

#### Revenue Forecast
1. Navigate to Predict → Models → Revenue
2. Enter historical revenue data
3. Click "Predict" to generate revenue forecasts
4. View trends and projections

## Project Structure

```
ML_BI/
├── airline/                    # Django project settings
│   ├── settings.py             # Configuration
│   ├── urls.py                 # Main URL routing
│   └── wsgi.py
├── dashboard/                  # Dashboard app
│   ├── views.py                # Dashboard and Power BI views
│   ├── urls.py                 # Dashboard URL routing
│   └── migrations/
├── predict/                    # Prediction app
│   ├── views.py                # Prediction views and API
│   ├── forms.py                # Prediction forms
│   ├── urls.py                 # Prediction URL routing
│   └── migrations/
├── templates/                  # HTML templates
│   ├── base.html               # Base template with sidebar
│   ├── home.html               # Home page
│   ├── powerbi_dashboard.html  # Power BI embedding
│   ├── prediction_selector.html # Model selection page
│   ├── prediction_form.html    # Flight forecast form
│   └── satisfaction_form.html  # Satisfaction prediction form
├── static/                     # Static files
│   ├── FlightsPrediction.pkl   # Flight forecasting model
│   ├── satisfaction_model.joblib
│   └── train_prophet.ipynb
├── db.sqlite3                  # SQLite database
├── manage.py                   # Django management script
└── requirements.txt            # Python dependencies
```

## Customization

### Adding a New Prediction Model

1. **Create a new view in `predict/views.py`:**
```python
def new_model_prediction(request):
    """New prediction view"""
    prediction = None
    error = None
    
    if request.method == "POST":
        try:
            # Load model and make prediction
            model = joblib.load(MODEL_PATH)
            prediction = model.predict(...)
        except Exception as e:
            error = str(e)
    
    return render(request, 'new_model_form.html', {
        'prediction': prediction,
        'error': error
    })
```

2. **Add URL routing in `predict/urls.py`:**
```python
path('new-model/', views.new_model_prediction, name='new_model_prediction'),
```

3. **Create form template in `templates/new_model_form.html`**

4. **Update sidebar in `templates/base.html`** to add link to new model

### Customizing the Sidebar

Edit the sidebar section in `templates/base.html` to:
- Add new prediction models to the dropdown
- Modify dashboard links
- Change branding text

## Troubleshooting

**Power BI dashboard not showing:**
- Verify the embed URL is correct and not expired
- Check that the Power BI report is published and accessible
- Ensure you have proper authentication
- Try the "Open in New Tab" button to debug

**Models not loading:**
- Check that model files exist in `static/` directory
- Verify file paths in `predict/views.py`
- Ensure model file format (.pkl or .joblib) is correct
- Check console for error messages

**Prediction errors:**
- Verify form input values are in expected ranges
- Check that model input features match form fields
- Review error messages in the UI and console
- Ensure required Python libraries are installed

**Sidebar navigation issues:**
- Clear browser cache and refresh
- Check browser console for JavaScript errors (F12)
- Ensure Bootstrap 5 CDN is loading correctly

## License

This project is open source and available for use.

## Support

For issues or questions, please check:
- Browser console (F12) for client-side errors
- Django terminal output for server-side errors
- Power BI admin portal for dashboard access issues

