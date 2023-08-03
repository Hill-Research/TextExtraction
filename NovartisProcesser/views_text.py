from django.shortcuts import render
from django.shortcuts import redirect
from django.http import HttpResponse
from django.http import JsonResponse
import datetime
import json
import sys

#sys.path.append('NovartisProcesser')

from NovartisProcesser.CriteriaExtraction import CriteriaExtraction
from NovartisProcesser.CriteriaNormalization import CriteriaNormalization
from NovartisProcesser.CriteriaGeneration import CriteriaGeneration
import argparse

# Create your views here.
class PatientSelection:
    @classmethod
    def text(cls, request):
        keyword_name = "Novartis"
        normalization_name = "Novartis"
        generation_name = "Novartis"
    
        normalization_condition_file = "NovartisProcesser/data/knowledgegraph_item.txt"
        normalization_unit_file = "NovartisProcesser/data/knowledgegraph_unit.txt"
    
        generation_condition_file = "NovartisProcesser/data/database_link.txt"
        if request.method == 'GET':
            cls.keyword = CriteriaExtraction(keyword_name, is_train = False, is_output = False, cuda = True)
            cls.normalization = CriteriaNormalization(normalization_name, normalization_condition_file, normalization_unit_file)
            cls.generation = CriteriaGeneration(generation_name, generation_condition_file)
            return render(request, 'text.html')
    
    @classmethod
    def extraction(cls, request):
        names = ['Field 1', 'Field 2']
        if request.method == 'POST':
            string = request.POST.get('text')
            extractionitems = cls.keyword.run(string)
            cls.extractionitems = extractionitems
            return JsonResponse({'head' : names, 'body' : extractionitems})
    
    @classmethod
    def normalization(cls, request):
        names = ['English name', 'Criteria', 'Information']
        if request.method == 'POST':
            extractionitems = cls.extractionitems
            normalizationitems = cls.normalization.run(extractionitems)
            cls.normalizationitems = normalizationitems
            return JsonResponse({'head' : names, 'body' : normalizationitems})
    
    @classmethod
    def generation(cls, request):
        names = ['SQL sequence', 'Information']
        if request.method == 'POST':
            normalizationitems = cls.normalizationitems
            generationitems = cls.generation.run(normalizationitems)
            return JsonResponse({'head' : names,'body' : generationitems})
