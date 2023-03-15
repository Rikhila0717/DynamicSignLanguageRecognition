from django.shortcuts import render
from django.http import HttpResponse
import sys

sys.path.append("../../FinalProject")

import testing

#views in ui
#1) select from and to languages
#(once this is done, opencv will capture real-time data)
#2) display result

# Create your views here.

def index(request):
    return HttpResponse("Home page")

def home_page(request):
    # test_asl(request.GET)
    return render(request,"home.html")

def test_asl(request):
   return testing.executable('asl')

def test_isl(request):
    return testing.executable('isl')

def test_bsl(request):
    return testing.executable('bsl')
    