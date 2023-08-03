# -*- coding: utf-8 -*-

import re
import argparse
import torch
import logging
import sys
from tqdm import tqdm

#sys.path.append('../')
from KnowledgeGraphCoreLibrary.ClusteringBasedModel import NormalizationInterface

class CriteriaNormalization:
    """
    Interface for criteria normalization.
    
    Args:
        name: The only name of the generated model.
        condition_file: The standardized disease documentation.
        is_train: Whether to force retraining, True represent force retraining.
        cuda: Whether to use gpu, cuda=True mean yes.
    """
    def __init__(self, name, condition_file, unit_file, is_train = False, cuda = False):
        option = argparse.Namespace()
        option.is_train = is_train
        option.cuda = cuda
        option.model = "first_last_avg-whitening"#"SentenceTransformer"#"first_last_avg-whitening"
        option.type = "{}_criteria_normalization".format(name)
        option.path = ".model/{}".format(option.type)
        option.condition_file = condition_file
        option.unit_file = unit_file
        option.bertmodel = "Bio_clinicalBERT" #"medical_sentence_tokenizer" # "bert_chinese_base"
        option.modelpath = ".model"
        option.device = torch.device("cuda" if (option.cuda and torch.cuda.is_available()) else "cpu")
        option.alpha = 0.75
        
        logging.basicConfig(level=logging.DEBUG, 
                            filename="../.log/{}.hqt".format(option.type), 
                            format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s", 
                            datefmt="%Y-%m-%d %H:%M:%S")
        option.logger = logging.getLogger(__name__)
        self.option = option
        
        option.logger.info("\033[0;37;34m{}\033[0m: Loading extraction model.".format(self.option.type))
        print("\033[0;37;34m{}\033[0m: Loading parameters for normalization model.".format(self.option.type))
        NormalizationInterface.init_parameters(option)
        self.normalization_name_interface = lambda x : NormalizationInterface.run_name(x)
        self.normalization_criteria_interface = lambda x: NormalizationInterface.run_criteria(x)
    
    def run(self, items):
        """
        Interface to run.
        
        Args:
            string: input string or words.
            
        Returns:
            normalizationstring: Normalized string.
        """
        print("\033[0;37;34m{}\033[0m: Loading knowledge graph stored in {}".format(self.option.type, self.option.condition_file))
        normalizationitems = list()
        for name in tqdm(items):
            new_name = re.sub("\s", "", name)
        
            normalizationenglishname, normalizationinfo = self.normalization_name_interface(new_name)
            normalizationcriteria = self.normalization_criteria_interface(items[name])
            if(normalizationinfo != None):
                normalizationitems.append({'English name' : normalizationenglishname, \
                                           'Criteria' :  normalizationcriteria, 'Info' : normalizationinfo})
            else:
                normalizationitems.append({'English name' : normalizationenglishname, \
                                           'Criteria' : normalizationcriteria})
        
        return normalizationitems
