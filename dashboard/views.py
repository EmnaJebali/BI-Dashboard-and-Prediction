from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import os
from django.conf import settings



def home(request):
    """Home page with navigation to dashboard and prediction"""
    return render(request, 'home.html')


def powerbi_dashboard(request):
    """View to redirect to Power BI dashboard in the same tab for proper authentication"""
    import urllib.parse
    
    # Get Power BI embed URL from settings or environment variable
    powerbi_url = getattr(settings, 'POWERBI_EMBED_URL', 
                         os.environ.get('POWERBI_EMBED_URL', ''))
    
    # Fallback to hardcoded URL if settings are empty (for debugging)
    if not powerbi_url or powerbi_url.strip() == '':
        powerbi_url = 'https://app.powerbi.com/reportEmbed?reportId=6535d1ba-f49c-4722-9ca5-f93c23e84051&autoAuth=true&ctid=604f1a96-cbe8-43f8-abbf-f8eaf5d85730'
    
    # Ensure powerbi_url is a string and not empty
    if not powerbi_url:
        powerbi_url = ''
    else:
        powerbi_url = str(powerbi_url).strip()
    
    # If URL is configured, redirect directly to PowerBI in the same tab
    # This allows proper authentication flow without iframe limitations
    if powerbi_url and powerbi_url.strip():
        print(f"Redirecting to Power BI URL: {powerbi_url}")
        return redirect(powerbi_url)
    
    # If no URL is configured, show configuration message
    context = {
        'powerbi_url': '',
    }
    return render(request, 'powerbi_dashboard.html', context)


