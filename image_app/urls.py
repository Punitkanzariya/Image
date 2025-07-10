from django.urls import path
from .views import upscale_view

urlpatterns = [
    path('', upscale_view, name='upscale'),
]
    