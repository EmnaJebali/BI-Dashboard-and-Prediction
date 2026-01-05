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
    """View to provide Power BI dashboard access options"""
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
    
    # Check if user wants to redirect to PowerBI (query parameter)
    redirect_mode = request.GET.get('redirect', 'false').lower() == 'true'
    
    if redirect_mode and powerbi_url and powerbi_url.strip():
        # Redirect to PowerBI in the same tab
        return redirect(powerbi_url)
    
    context = {
        'powerbi_url': powerbi_url,
    }
    return render(request, 'powerbi_dashboard.html', context)


