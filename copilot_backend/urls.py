from django.contrib import admin
from django.http import JsonResponse
from django.urls import include, path


def health(_request):
    return JsonResponse({"status": "ok"})


urlpatterns = [
    path('', health),
    path('health/', health),
    path('admin/', admin.site.urls),
    path('auth/', include('auth_app.urls')),
    path('api/', include('api_app.urls')),
]
