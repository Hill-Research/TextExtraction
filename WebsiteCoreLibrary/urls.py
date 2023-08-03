"""
URL configuration for WebsiteCoreLibrary project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
import sys

from IndexProcesser import views_main
from NovartisProcesser import views_text
from PaddleProcesser import views_figure
from CSVProcesser import views_csv

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views_main.index),
    path('text/', views_text.PatientSelection.text),
    path('text/extraction/', views_text.PatientSelection.extraction),
    path('text/normalization/', views_text.PatientSelection.normalization),
    path('text/generation/', views_text.PatientSelection.generation),
    path('figure/', views_figure.FigureClassification.figure),
    path('figure/read/', views_figure.FigureClassification.read),
    path('csv/', views_csv.CSVHandler.csv),
    path('csv/read/', views_csv.CSVHandler.read),
    path('csv/desensitization/', views_csv.CSVHandler.desensitization),
]
