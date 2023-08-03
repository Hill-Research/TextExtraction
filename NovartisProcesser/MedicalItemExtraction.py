# -*- coding: utf-8 -*-

import os
import argparse
import torch
import logging
import torchmetrics.classification as clc
from sklearn.model_selection import train_test_split
import sys

sys.path.append('../')
import numpy as np
from NLPCoreLibrary.CRFExtractionModel import ExtractionDataLoader
from NLPCoreLibrary.CRFExtractionModel import ExtractionInterface
from NLPCoreLibrary.CRFExtractionModel import ExtractionTrainer
from NLPCoreLibrary.CRFExtractionModel import ExtractionIterator


class MedicalItemExtraction(object):
    """
    Interface for keyword extraction.
    
    Args:
        name: The only name of the generated model.
        is_train: Whether to force retraining, True represent force retraining.
        cuda: Whether to use gpu, cuda=True mean yes.
    """
    def __init__(self, name, is_train = False, is_output = False, cuda = False):
        option_extraction = argparse.Namespace()
        option_extraction.is_train = is_train
        option_extraction.is_output = is_output
        option_extraction.cuda = cuda
        option_extraction.optimizer = "Adam"
        option_extraction.metric = ["F1Score", "Precision", "Recall", "Accuracy", "TP", "FN"]
        option_extraction.dropout = 0.1
        option_extraction.train_step = 300
        option_extraction.batch_size = 64
        option_extraction.n_layers = 8
        option_extraction.bert_size = 768
        option_extraction.hidden_size = 512
        option_extraction.filter_size = 1024
        option_extraction.learning_rate = 0.01
        option_extraction.rate_scale = {'bert': 1, 'lstm' : 5, 'linear': 5, 'crf': 20}
        option_extraction.epsilon = 0
        option_extraction.weight_decay = 0.1
        option_extraction.tag2id = {"ELSE" : 0, "NAME" : 1, "[CLS]" : 2, "[SEP]" : 3}
        option_extraction.id2tag = ["ELSE", "NAME", "[CLS]", "[SEP]"]
        option_extraction.type = "{}_item_extraction".format(name)
        option_extraction.path = "../.model/{}".format(option_extraction.type)
        option_extraction.bertmodel = "Bio_clinicalBERT" #"medical_sentence_tokenizer" # "bert_chinese_base"
        option_extraction.modelpath = "../.model"
        option_extraction.device = torch.device("cuda" if (option_extraction.cuda and torch.cuda.is_available()) else "cpu")
        print("\033[0;37;31m{}\033[0m: Testing device:{}".format(option_extraction.type, option_extraction.device))
        logging.basicConfig(level=logging.DEBUG, 
                            filename="../.log/{}.hqt".format(option_extraction.type), 
                            format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s", 
                            datefmt="%Y-%m-%d %H:%M:%S")
        option_extraction.logger = logging.getLogger(__name__)
        self.option = option_extraction
        ExtractionDataLoader.init_parameters(self.option)
        option_extraction.logger.info("\033[0;37;31m{}\033[0m: Loading extraction model.".format(self.option.type))
        self.extraction_interface = lambda x : ExtractionInterface.run(x)
        print("\033[0;37;31m{}\033[0m: Loading parameters for extraction model.".format(self.option.type))
    
    def train_valid(self, train_data_dir, validation_data_dir = None):
        """
        Interface to train.
        
        Args:
            train_data_dir: Position of train data.
        """
        option = self.option
        print("\033[0;37;31m{}\033[0m: Reading test data in {}.".format(self.option.type, train_data_dir))
        if(validation_data_dir == None):
            data_paths = [os.path.join(train_data_dir, name) for name in os.listdir(train_data_dir)]
            train_data_paths, validation_data_paths = train_test_split(data_paths, test_size = 0.2)
        else:
            train_data_paths = [os.path.join(train_data_dir, name) for name in os.listdir(train_data_dir)]
            validation_data_paths = [os.path.join(validation_data_dir, name) for name in os.listdir(validation_data_dir)]

        if (not os.path.exists("{}/best_model.hqt".format(option.path)) or option.is_train == True):
            self.train(train_data_paths, validation_data_paths)
        self.valid(validation_data_paths)
        print("\033[0;37;31m{}\033[0m: Interface for data extraction is ready!".format(self.option.type))
        ExtractionInterface.init_parameters(self.option)

    def train(self, train_data, validation_data_paths):
        if(type(train_data) == list):
            train_data_paths = train_data
        else:
            train_data_paths = [os.path.join(train_data, name) for name in train_data]
        ExtractionTrainer.init_parameters(self.option)
        ExtractionTrainer.train(train_data_paths)
        
    def valid(self, validation_data):
        option = self.option
        if(type(validation_data) == list):
            validation_data_paths = validation_data
        else:
            validation_data_paths = [os.path.join(validation_data, name) for name in validation_data]
        ExtractionIterator.init_parameters(self.option)
        validation_iterator = ExtractionIterator.run(validation_data_paths, is_valid = True)
        self.validation_length = len(validation_data_paths)
        val_metric = self.metric(validation_iterator)
        metric_string = ",".join(["{}: {} ".format(key, val_metric[key]) for key in val_metric])
        self.option.logger.info("\033[0;37;31m{}\033[0m: The metric are {}".format(option.type, metric_string))
        print("\033[0;37;31m{}\033[0m: The metric are {}".format(option.type, metric_string))
        if(option.is_output):
            self.output(val_metric, ['txt'])

    def transform(self, chars, tags):
        name_tmp = ""
        name_list_tmp = list()
        number_tmp = ""
        number_list_tmp = list()
        name_items=list()
        number_items = list()
        seg_list = [0, ]
        
        for (i, (c, t)) in enumerate(zip(chars, tags)):
            t = self.option.id2tag[t]
            if(t == "NAME" and c not in ["[CLS]", "[PAD]", "[SEP]"]):
                name_tmp = name_tmp + " "+ c
                name_list_tmp.append(i)
            else:
                if(len(name_tmp) > 0):
                    name_items.append({'word': name_tmp, 'locate' : name_list_tmp})
                    name_tmp = ""
                    name_list_tmp = list()
                if(c == "[CLS]" or c == "[SEP]"):
                    continue
            if(t == "NUMBER" and c not in ["[CLS]", "[PAD]", "[SEP]"]):
                number_tmp = number_tmp + " " + c
                number_list_tmp.append(i)
            else:
                if(len(number_tmp) > 0):
                    number_items.append({'word': number_tmp, 'locate' : number_list_tmp})
                    number_tmp = ""
                    number_list_tmp = list()
                if(c == "[CLS]" or c == "[SEP]"):
                    continue
            if(c in [';', '.', ',']):
                seg_list.append(i)
            
            if(c in ["[CLS]", "[PAD]", "[SEP]"]):
                continue
        if(len(chars) not in seg_list):
            seg_list.append(len(chars))
        
        seg_list = np.array(seg_list)
        extractionitems = dict()
        number_words = dict()
        number_locates = list()
        for item in number_items:
            number_word = item['word']
            number_locate = [item['locate'][0], item['locate'][-1]]
            number_locates.append(number_locate)
            number_words[number_locate[0]] = number_word
        for item in name_items:
            name_word = item['word']
            # name_word = re.sub("\s", "", name_word)
            
            normalization_name_word = name_word
            name_locate = item['locate'][0]
            right_index = np.where(seg_list >= name_locate)[0][0]
            left_index = right_index - 1
            suitable_indexs = [[i,j] for (i,j) in number_locates if(i >= seg_list[left_index] and i <= seg_list[right_index])]
            if(suitable_indexs == []):
                extractionitems[normalization_name_word] = None
            else:
                distances = np.array([i - name_locate for (i,j) in suitable_indexs])
                selected_index = np.where(distances == np.min(distances))[0][0]
                number_locate = suitable_indexs[selected_index][0]
                tail_locate = suitable_indexs[selected_index][1]
                number_word = number_words[number_locate] + '@' + ' '.join(chars[tail_locate + 1: min(tail_locate + 5, len(chars))])
                extractionitems[normalization_name_word] = number_word
        selectedextractionitems = dict()
        for key in extractionitems:
            value = extractionitems[key]
            selectedextractionitems[key] = value
        return selectedextractionitems
    
    def run(self, string, type_ = 0):
        """
        Interface to run.
        
        Args:
            string: input string or words.
            type_: dir or string.
        Returns:
            extractionstring: Processed string.
        """
        if(os.path.exists("{}/best_model.hqt".format(self.option.path))):
            ExtractionInterface.init_parameters(self.option)
        else:
            return None
        if(type_ == 1):
            test_data_paths = [os.path.join(string, name) for name in string]
            ExtractionIterator.init_parameters(self.option)
            test_iterator = ExtractionIterator.run(test_data_paths, is_valid = True)
            self.test_length = len(test_data_paths)
            extractionstrings = dict()
            for (chars, tags, _) in test_iterator:
                extractionchars, extractiontags = self.test_iterator(string)
                extractionstring = self.transform(extractionchars, extractiontags)
                extractionstrings.update(extractionstring)
            return extractionstrings
        else:
            original_sentence, extractionchars, extractiontags = self.extraction_interface(string)
            extractiontags = extractiontags.squeeze(0).flatten()
            extractionstring = self.transform(original_sentence, extractiontags)
            return extractionstring

    def metric(self, validation_iterator):
        """
        Loss function.
        
        Args:
            validation_iterator: The validation set interface.
            Model: AI model.
            
        Returns:
            val_loss: Current loss.
        """
        option = self.option
        total_metric, total_count = dict(), dict()
        metric_funs = dict()
        if("F1Score" in option.metric):
            f1score = clc.F1Score(task = "multiclass", num_classes = len(option.id2tag), average = 'macro').to(self.option.device)
            metric_funs['F1Score'] = f1score
        if("Precision" in option.metric):
            precision = clc.Precision(task = "multiclass", num_classes = len(option.id2tag), average = 'macro').to(self.option.device)
            metric_funs['Precision'] = precision
        if("Recall" in option.metric):
            recall = clc.Recall(task = "multiclass", num_classes = len(option.id2tag), average = 'macro').to(self.option.device)
            metric_funs['Recall'] = recall
        if("Accuracy" in option.metric):
            accuracy = clc.Accuracy(task = "multiclass", num_classes = len(option.id2tag), average = 'macro').to(self.option.device)
            metric_funs['Accuracy'] = accuracy
        if("TP" in option.metric):
            tp = lambda x,y: torch.sum(x[y > 0] == y[y > 0]) / len(x[y > 0])
            metric_funs['TP'] = tp
        if("FN" in option.metric):
            fn = lambda x,y: torch.sum(x[x > 0] == y[x > 0]) / len(y[x > 0])
            metric_funs['FN'] = fn
        for (original_sentences, chars, tag, output) in validation_iterator:
            tag = tag.flatten()
            output = output.flatten()
#            print(self.transform(original_sentences, tag))
            for metric_key in metric_funs:
                metric_fun = metric_funs[metric_key]
                metric = metric_fun(output, tag)
                if(metric_key not in total_metric):
                    total_metric[metric_key] = (metric.item() if metric.item() <= 1 else 0)
                else:
                    total_metric[metric_key] += (metric.item() if metric.item() <= 1 else 0)
                if(metric_key not in total_count):
                    total_count[metric_key] = 1
                else:
                    total_count[metric_key] += 1
        mean_metric = dict()
        for metric_key in metric_funs:
            mean_metric[metric_key] = total_metric[metric_key] / total_count[metric_key]
        return mean_metric

    def output(self, val_metric, type_):
        option = self.option
        with open('{}/score.txt'.format(option.path), 'w+', encoding = 'utf-8') as f:
            for metric_key in val_metric:
                f.write('\t{}: {}\n'.format(metric_key, val_metric[metric_key]))
