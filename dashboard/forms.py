from django import forms


class PredictionForm(forms.Form):
    """
    Generic prediction form. 
    You should customize this based on your model's input features.
    """
    
    # Example fields - customize these based on your model
    # Replace with your actual feature names and types
    
    # Numeric fields
    feature1 = forms.FloatField(
        label='Feature 1',
        required=True,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.01',
            'placeholder': 'Enter value'
        })
    )
    
    feature2 = forms.FloatField(
        label='Feature 2',
        required=True,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.01',
            'placeholder': 'Enter value'
        })
    )
    
    # Add more fields as needed based on your model
    # Example:
    # feature3 = forms.IntegerField(label='Feature 3', required=True)
    # feature4 = forms.ChoiceField(label='Feature 4', choices=[('A', 'Option A'), ('B', 'Option B')])
    
    def __init__(self, *args, **kwargs):
        # You can dynamically add fields based on model requirements
        super().__init__(*args, **kwargs)
        
        # Example: Add fields dynamically if needed
        # This is useful if you want to load feature names from the model

