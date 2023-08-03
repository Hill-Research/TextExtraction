# -*- coding: utf-8 -*-
import re
import argparse
from tqdm import tqdm

class CriteriaGeneration(object):
    def __init__(self, name, condition_file):
        option = argparse.Namespace()
        option.type = '{}_criteria_generation'.format(name)
        option.condition_file = condition_file
        self.option = option
        print("\033[0;37;35m{}\033[0m: Loading parameters for SQL generation model.".format(self.option.type))
    
    def getnumber(self, string):
        Number = re.findall("\d+\.\d+[x]\d+\.\d+|\d+\.\d+[x]\d+|\d+[x]\d+\.\d+|\d+[x]\d+|\d+\.\d+|\d+", string)
        if(Number != []):
            standardNumberValue = float(eval(Number[0].replace('x', '*')))
        else:
            standardNumberValue = None
        return standardNumberValue
    
    def getsymbol(self, string):
        if('>' in string):
            return '>'
        if('≥' in string or '>=' in string):
            return '>='
        if('<' in string):
            return '<'
        if('≤' in string or '<=' in string):
            return '<='
    
    def getformula(self, string):
        Items = list()
        count = 0
        is_current = True
        current_item = ""
        for i in string:
            if(i == "$" and count == 0):
                count = 1
                is_current = False
            if(i == "$" and count == 1 and is_current):
                Items.append(current_item)
                current_item = ""
                count = 0
            if(count == 1 and is_current):
                current_item += i
            is_current = True
        return Items
    
    def getname(self, string):
        if(not hasattr(self, 'dict')):
            self.dict = dict()
            with open(self.option.condition_file, 'r', encoding = 'utf-8') as f:
                for line in f.readlines():
                    if(len(line.strip()) > 0):
                        line = line.strip()
                        item1, item2 = line.split('\t')
                        self.dict[item1] = item2
        if(string in self.dict):
            return self.dict[string]
        else:
            return None
    
    def run(self, items):
        print("\033[0;37;35m{}\033[0m: Generating SQL sequence for dataset link {}".format(self.option.type, self.option.condition_file))
        criterias = list()
        for item in tqdm(items):
            if('ULN' in item['Criteria'] and (('Info' not in item) or ('ULN' not in item['Info']))):
                continue
            if(item['Criteria'] == None):
                string = None
            else:
                if('ULN' in item['Criteria']):
                    string = item['Criteria'].replace('ULN', item['Info']['ULN'])
                else:
                    string = item['Criteria']
            
            standardname = self.getname(item['English name'])
            if(standardname == None and (('Info' not in item) or ('Formula' not in item['Info']))):
                continue
            
            if(standardname != None):
                if(string != None):
                    standardnumber = self.getnumber(string)
                    standardsymbol = self.getsymbol(string)
                    criterias.append({'SQL' : 'SELECT ID FROM UKBiobank WHERE {} {} {}'.format(standardname, standardsymbol, standardnumber), 'Info' : item['English name']})
                else:
                    criterias.append({'SQL' : 'SELECT ID, {} FROM UKBiobank'.format(standardname), 'Info' : item['English name']})
            
            if(('Info' in item) and ('Formula' in item['Info'])):
                total_criteria = item['Info']['Formula']
                standardnumber = self.getnumber(string)
                standardsymbol = self.getsymbol(string)
                standardformulas = self.getformula(item['Info']['Formula'])
                for formula in standardformulas:
                    standardformulaname = self.getname(formula)
                    total_criteria = total_criteria.replace('${}$'.format(formula), '${}$'.format(standardformulaname))
                    if(standardformulaname != None):
                        criterias.append({'SQL' : 'SELECT ID,{} FROM UKBiobank'.format(standardformulaname), 'Info' : formula})
                criterias.append({'SQL' : '{} {} {}'.format(total_criteria, standardsymbol, standardnumber), 'Info' : item['English name']})
        return criterias
