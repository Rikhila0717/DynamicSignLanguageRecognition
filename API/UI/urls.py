from django.urls import path

from . import views

urlpatterns = [
    path('',views.home_page,name='home'),
    path('asl/',views.test_asl,name='asl'),
    path('isl/',views.test_isl,name='isl'),
    path('bsl/',views.test_bsl,name='bsl'),
    path('fsl/',views.test_fsl,name='fsl'),
]