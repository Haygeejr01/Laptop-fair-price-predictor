from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('predict_page/', views.predict_page, name='predict_page'),
]