from django.urls import path
from . import views

# URLConf
urlpatterns = [
    path('make_n_model/', views.get_make_n_model)
]
