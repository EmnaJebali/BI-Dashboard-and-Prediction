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
    return render(request, 'powerbi_dashboard.html', context)


