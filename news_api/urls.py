from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict_news, name='predict_news'),
]
