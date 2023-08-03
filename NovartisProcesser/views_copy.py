from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
import datetime
import json
import sys

sys.path.append('NovartisProcesser')

from CriteriaExtraction import CriteriaExtraction
from CriteriaNormalization import CriteriaNormalization
from CriteriaGeneration import CriteriaGeneration
import argparse

# Create your views here.
def time(request):
    now = datetime.datetime.now()
    html = 'Time now is {}'.format(now)
    return HttpResponse(html)

def index(request):
    if request.method == 'GET':
        return render(request, 'index.html')

def extraction(request):
    keyword_name = "Novartis"
    names = ['Field 1', 'Field 2']
    if request.method == 'POST':
        keyword = CriteriaExtraction(keyword_name, is_train = False, is_output = False, cuda = True)
        string = request.POST.get('text')
        extractionitems = keyword.run(string)
        request.session['extractionitems'] = extractionitems
        print(extractionitems)
        return JsonResponse({'head' : names, 'body' : extractionitems})

def normalization(request):
    normalization_condition_file = "data/knowledgegraph_item.txt"
    normalization_unit_file = "data/knowledgegraph_unit.txt"
    normalization_name = "Novartis"
    names = ['English name', 'Criteria', 'Information']
    if request.method == 'POST':
        extractionitems = request.session['extractionitems']
        normalization = CriteriaNormalization(normalization_name, normalization_condition_file, normalization_unit_file)
        normalizationitems = normalization.run(extractionitems)
        request.session['normalizationitems'] = normalizationitems
        return JsonResponse({'head' : names, 'body' : normalizationitems})

def generation(request):
    print(request, '111')
    generation_condition_file = "data/database_link.txt"
    generation_name = "Novartis"
    names = ['SQL sequence', 'Information']
    if request.method == 'POST':
        normalizationitems = request.session['normalizationitems']
        generation = CriteriaGeneration(generation_name, generation_condition_file)
        generationitems = generation.run(normalizationitems)
        return JsonResponse({'head' : names,'body' : generationitems})
