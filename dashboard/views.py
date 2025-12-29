from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import os
from django.conf import settings

# Import model loading utilities
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    import pickle
    HAS_PICKLE = True
except ImportError:
    HAS_PICKLE = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Global variable to store loaded model
_loaded_model = None
_model_path = None


def load_model(model_path=None):
    """Load the ML model from file"""
    global _loaded_model, _model_path
    
    if model_path is None:
        # Try to find model in common locations
        possible_paths = [
            os.path.join(settings.BASE_DIR, 'model.pkl'),
            os.path.join(settings.BASE_DIR, 'model.joblib'),
            os.path.join(settings.BASE_DIR, 'models', 'model.pkl'),
            os.path.join(settings.BASE_DIR, 'models', 'model.joblib'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
    
    if model_path is None or not os.path.exists(model_path):
        return None
    
    if _loaded_model is not None and _model_path == model_path:
        return _loaded_model
    
    try:
        if model_path.endswith('.joblib') and HAS_JOBLIB:
            _loaded_model = joblib.load(model_path)
        elif model_path.endswith('.pkl') and HAS_PICKLE:
            with open(model_path, 'rb') as f:
                _loaded_model = pickle.load(f)
        else:
            # Try both methods
            if HAS_JOBLIB:
                _loaded_model = joblib.load(model_path)
            elif HAS_PICKLE:
                with open(model_path, 'rb') as f:
                    _loaded_model = pickle.load(f)
            else:
                return None
        
        _model_path = model_path
        return _loaded_model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def home(request):
    """Home page with navigation to dashboard and prediction"""
    return render(request, 'dashboard/home.html')


def powerbi_dashboard(request):
    """View to display embedded Power BI dashboard"""
    import urllib.parse
    
    # Get Power BI embed URL from settings or environment variable
    powerbi_url = getattr(settings, 'POWERBI_EMBED_URL', 
                         os.environ.get('POWERBI_EMBED_URL', ''))
    
    # Fallback to hardcoded URL if settings are empty (for debugging)
    if not powerbi_url or powerbi_url.strip() == '':
        powerbi_url = 'https://app.powerbi.com/reportEmbed?reportId=6535d1ba-f49c-4722-9ca5-f93c23e84051&autoAuth=true&ctid=604f1a96-cbe8-43f8-abbf-f8eaf5d85730'
    
    # Parse URL to extract report ID and workspace ID
    report_id = None
    workspace_id = None
    tenant_id = None
    
    if powerbi_url:
        try:
            parsed = urllib.parse.urlparse(powerbi_url)
            params = urllib.parse.parse_qs(parsed.query)
            
            # Extract reportId
            if 'reportId' in params:
                report_id = params['reportId'][0]
            
            # Extract workspace ID (groupId) if present
            if 'groupId' in params:
                workspace_id = params['groupId'][0]
            
            # Extract tenant ID (ctid)
            if 'ctid' in params:
                tenant_id = params['ctid'][0]
        except Exception as e:
            print(f"Error parsing Power BI URL: {e}")
    
    # Debug: print URL to console
    print(f"Power BI URL from settings: {powerbi_url}")
    print(f"Power BI URL type: {type(powerbi_url)}")
    print(f"Power BI URL length: {len(powerbi_url) if powerbi_url else 0}")
    
    # Ensure powerbi_url is a string and not empty
    if not powerbi_url:
        powerbi_url = ''
    else:
        powerbi_url = str(powerbi_url).strip()
    
    context = {
        'powerbi_url': powerbi_url,
        'report_id': report_id,
        'workspace_id': workspace_id,
        'tenant_id': tenant_id,
    }
    print(f"Context powerbi_url: {context['powerbi_url']}")
    return render(request, 'dashboard/powerbi_dashboard.html', context)


def prediction_form(request):
    """View to display prediction form"""
    # Load model if available
    model = load_model()
    model_loaded = model is not None
    
    context = {
        'model_loaded': model_loaded,
    }
    
    if request.method == 'POST':
        # Handle form submission
        try:
            data = json.loads(request.body) if request.content_type == 'application/json' else request.POST
            prediction_result = make_prediction(data, model)
            
            if request.content_type == 'application/json':
                return JsonResponse({
                    'success': True,
                    'prediction': prediction_result
                })
            else:
                context['prediction'] = prediction_result
                context['form_data'] = data
        except Exception as e:
            error_msg = str(e)
            if request.content_type == 'application/json':
                return JsonResponse({
                    'success': False,
                    'error': error_msg
                }, status=400)
            else:
                context['error'] = error_msg
    
    return render(request, 'dashboard/prediction_form.html', context)


@csrf_exempt
def predict_api(request):
    """API endpoint for predictions"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    model = load_model()
    if model is None:
        return JsonResponse({'error': 'Model not loaded'}, status=500)
    
    try:
        if request.content_type == 'application/json':
            data = json.loads(request.body)
        else:
            data = request.POST.dict()
        
        prediction = make_prediction(data, model)
        
        return JsonResponse({
            'success': True,
            'prediction': prediction
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)


def make_prediction(data, model):
    """Make prediction using the loaded model"""
    if model is None:
        raise ValueError("Model not loaded")
    
    # Extract features from data
    # This is a generic implementation - you'll need to adapt it to your model
    # The data should match the features your model expects
    
    # Convert data to numpy array if needed
    # You'll need to adjust this based on your model's input requirements
    try:
        # Try to get feature values from data
        # Adjust feature names based on your model
        features = []
        
        # Example: if your model expects specific features
        # You should modify this section based on your actual model
        feature_names = getattr(model, 'feature_names_in_', None)
        
        if feature_names is not None:
            # Model has feature names (e.g., scikit-learn)
            for feature_name in feature_names:
                value = data.get(feature_name, 0)
                try:
                    features.append(float(value))
                except (ValueError, TypeError):
                    features.append(0.0)
        else:
            # Generic approach - extract numeric values
            for key, value in data.items():
                if key not in ['csrfmiddlewaretoken']:
                    try:
                        features.append(float(value))
                    except (ValueError, TypeError):
                        pass
        
        if not features:
            raise ValueError("No valid features found in input data")
        
        # Convert to numpy array if numpy is available
        if HAS_NUMPY:
            features_array = np.array(features).reshape(1, -1)
        else:
            features_array = [features]
        
        # Make prediction
        prediction = model.predict(features_array)
        
        # Handle different prediction output formats
        if HAS_NUMPY and isinstance(prediction, np.ndarray):
            prediction_value = float(prediction[0]) if len(prediction) > 0 else None
        else:
            prediction_value = float(prediction) if isinstance(prediction, (int, float)) else str(prediction)
        
        # If model has predict_proba, include probabilities
        result = {'value': prediction_value}
        
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(features_array)
                if HAS_NUMPY and isinstance(probabilities, np.ndarray):
                    result['probabilities'] = probabilities[0].tolist()
                else:
                    result['probabilities'] = probabilities
            except:
                pass
        
        return result
        
    except Exception as e:
        raise ValueError(f"Prediction error: {str(e)}")
