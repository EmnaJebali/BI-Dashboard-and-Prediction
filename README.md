# ML & BI Dashboard Application

A Django web application that provides:
1. **Power BI Dashboard Embedding** - Visualize your published Power BI dashboards
2. **ML Prediction Service** - Make predictions using your trained machine learning model

## Features

- 🎨 Modern, responsive UI with Bootstrap 5
- 📊 Embedded Power BI dashboard visualization
- 🤖 ML model prediction form with API endpoint
- 🔧 Easy configuration for Power BI URL and model path

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

### 3. Add Your ML Model

Place your trained model file in one of these locations:
- `model.pkl` or `model.joblib` in the project root
- `models/model.pkl` or `models/model.joblib`

Or set the path in `airline/settings.py`:
```python
MODEL_PATH = '/path/to/your/model.pkl'
```

**Supported model formats:**
- `.pkl` (pickle)
- `.joblib` (joblib)

**Model requirements:**
- The model should have a `.predict()` method
- If using scikit-learn, the model should have `feature_names_in_` attribute for automatic feature mapping
- For other frameworks, you may need to customize the `make_prediction()` function in `dashboard/views.py`

### 4. Customize Prediction Form

Edit `dashboard/forms.py` and `templates/dashboard/prediction_form.html` to match your model's input features:

1. Update the form fields in `dashboard/forms.py` to match your model's expected inputs
2. Update the HTML form in `templates/dashboard/prediction_form.html` with the same fields
3. If needed, modify the `make_prediction()` function in `dashboard/views.py` to handle your specific model format

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

### Home Page
Navigate to the home page to access:
- Power BI Dashboard
- ML Prediction Form

### Power BI Dashboard
View your embedded Power BI dashboard with interactive visualizations.

### ML Prediction
1. Fill out the prediction form with your input features
2. Click "Predict" to get predictions from your trained model
3. View the prediction result and probabilities (if available)

### API Endpoint
You can also make predictions via API:

```bash
POST /api/predict/
Content-Type: application/json

{
    "feature1": 1.5,
    "feature2": 2.3
}
```

Response:
```json
{
    "success": true,
    "prediction": {
        "value": 0.85,
        "probabilities": [0.15, 0.85]
    }
}
```

## Project Structure

```
ML_BI/
├── airline/              # Django project settings
│   ├── settings.py       # Configuration (Power BI URL, Model Path)
│   └── urls.py           # Main URL routing
├── dashboard/            # Main application
│   ├── views.py          # Views for dashboard and predictions
│   ├── forms.py          # Prediction form (customize for your model)
│   └── urls.py           # App URL routing
├── templates/            # HTML templates
│   └── dashboard/
│       ├── home.html
│       ├── powerbi_dashboard.html
│       └── prediction_form.html
├── static/               # Static files (CSS, JS, images)
├── model.pkl            # Your trained model (place here)
└── requirements.txt     # Python dependencies
```

## Customization

### Adding More Form Fields

1. Edit `dashboard/forms.py`:
```python
feature3 = forms.FloatField(
    label='Feature 3',
    required=True,
    widget=forms.NumberInput(attrs={'class': 'form-control'})
)
```

2. Edit `templates/dashboard/prediction_form.html`:
```html
<div class="mb-3">
    <label for="feature3" class="form-label">Feature 3</label>
    <input type="number" class="form-control" id="feature3" name="feature3" required>
</div>
```

### Custom Model Prediction Logic

If your model has special requirements, modify the `make_prediction()` function in `dashboard/views.py` to handle:
- Feature preprocessing
- Different model frameworks (TensorFlow, PyTorch, etc.)
- Custom prediction formats

## Troubleshooting

**Power BI dashboard not showing:**
- Verify the embed URL is correct
- Check that the Power BI report is published and accessible
- Ensure the URL doesn't contain `YOUR_EMBED_URL_HERE`

**Model not loading:**
- Check the model file path
- Verify the model file format (.pkl or .joblib)
- Ensure required libraries (joblib, pickle) are installed
- Check console for error messages

**Prediction errors:**
- Verify form fields match model's expected features
- Check feature names and types match training data
- Review the `make_prediction()` function for compatibility

## License

This project is open source and available for use.

