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
    return render(request,"home.html")

def test_asl(request):
   op_lang = request.GET['output']
   print(op_lang)
   return testing.executable('asl',op_lang)

def test_isl(request):
    op_lang = request.GET['output']
    return testing.executable('isl',op_lang)

def test_bsl(request):
    op_lang = request.GET['output']
    return testing.executable('bsl',op_lang)
    # return testing.executable('bsl')

def test_fsl(request):
    op_lang = request.GET['output']
    return testing.executable('fsl',op_lang)
    