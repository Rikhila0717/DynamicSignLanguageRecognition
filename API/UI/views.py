from django.shortcuts import render
from django.http import HttpResponse
import sys

sys.path.append("../../FinalProject")

from pyscripts import testing

#views in ui
#1) select from and to languages
#(once this is done, opencv will capture real-time data)
#2) display result

# Create your views here.

def index(request):
    return HttpResponse("Home page")

def home_page(request):
    return render(request,"home.html")

def test_project(request):
    return testing.executable()
    