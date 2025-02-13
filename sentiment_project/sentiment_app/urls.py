from django.urls import path
from . import views

urlpatterns = [
    path('analyze/', views.sentiment_analysis_view, name='sentiment_analysis'),
]
