# -*- coding: utf-8 -*-
"""

Author: Qitong Hu, Shanghai Jiao Tong University

"""

import json

from .NormalizationModel import NormalizationModel

class NormalizationInterface:
    """
    Interface for Normalization.
    
    Args:
        option: The parameters for main model. option.type -- the name of model is used here.
    """
    
    @classmethod
    def init_parameters(cls, option):
        """
        Initialization for parameters.
        
        Args:
            option: The parameter of main model.
        """
        option.logger.info("\033[0;37;31m{}\033[0m: Loading parameters for normalization model.".format(option.type))
        cls.option = option
        
        option.logger.info("\033[0;37;31m{}\033[0m: Loading normalized model.".format(option.type))
        cls.model = model = NormalizationModel(cls.option)
        
        option.logger.info("\033[0;37;31m{}\033[0m: Loading normalized keywords.".format(option.type))
        keywords = list()
        add_info = dict()
        with open(cls.option.condition_file, "r", encoding = "utf-8") as f:
            for line in f.readlines():
                if(len(line.strip())>0):
                    key = line.strip().split("\t")[0].strip()
                    if(len(line.strip().split("\t")) > 1):
                        add_info[key] = line.strip().split("\t")[1]
                    keywords.append(key)
        cls.add_info = add_info
        
        if(cls.option.unit_file != None):
            unit = list()
            with open(cls.option.unit_file, "r", encoding = "utf-8") as f:
                for line in f.readlines():
                    if(len(line.strip())>0):
                        item = line.strip()
                        unit.append(item)
            cls.unit = unit
        else:
            cls.unit = None
        
        for keyword in keywords:
            if (not model.exists(keyword)):
                model.insert(keyword)
    
    @classmethod
    def run_name(cls, string):
        """
        Main Interface for normalization.
        
        Args:
            string: Input string.

        Returns:
            normalizedstring: Normalized string.

        """
        model = cls.model
        bestcenter = model.bestcenter(string)
        if (bestcenter == None):
            model.insert(string)
            normalizedstring = string
            normalizedinfo = None
        else:
            normalizedenglishname = model.get(bestcenter)
            if(normalizedenglishname in cls.add_info):
                normalizedinfo = eval(cls.add_info[normalizedenglishname])
            else:
                normalizedinfo = None
        return normalizedenglishname, normalizedinfo

    @classmethod
    def is_number(cls, string):
        if(string.strip() == '.'):
            return False
        for i in string:
            if(i.isdigit() or i == '.'):
                continue
            else:
                return False
        return True
    
    @classmethod
    def run_criteria(cls, string):
        string = string.strip()
        string1 = string.split('@')[0]
        string2 = string.split('@')[-1]
        if cls.unit != None:
            for item in string2.strip().split(' '):
                if(item in cls.unit or cls.is_number(item)):
                    string1 += ' {}'.format(item)
                else:
                    break
            else:
                string1 = string1
        return string1.replace(' ','')
