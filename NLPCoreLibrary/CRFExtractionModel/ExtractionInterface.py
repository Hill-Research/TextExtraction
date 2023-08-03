# -*- coding: utf-8 -*-
"""

Author: Qitong Hu, Shanghai Jiao Tong University

"""

import os
import re
import torch
import jieba
from transformers import logging

from .ExtractionDataLoader import ExtractionDataLoader
from .ExtractionModel import ExtractionModel

logging.set_verbosity_error()

class ExtractionInterface:
    @classmethod
    def init_parameters(cls, option):
        """
        Initialization for parameters.
        
        Args:
            option: The parameter of main model.
        """
        option.logger.info("\033[0;37;31m{}\033[0m: Loading parameters for extraction model.".format(option.type))
        cls.option = option
        option.logger.info("\033[0;37;31m{}\033[0m: Testing device is {}.".format(option.type, option.device))

        option.logger.info("\033[0;37;31m{}\033[0m: Loading parameters for extraction data loader.".format(option.type))
        ExtractionDataLoader.init_parameters(cls.option)

        option.logger.info("\033[0;37;31m{}\033[0m: Loading model from checkpoint file.".format(option.type))
        model = ExtractionModel(cls.option)
        model.load_state_dict(cls.load_checkpoint())
        cls.model = model.to(cls.option.device)

        option.logger.info("\033[0;37;31m{}\033[0m: Loading tokenizer for extraction model.".format(option.type))
        cls.tokenizer = ExtractionDataLoader.loadTokenizer()
        option.logger.info("\033[0;37;31m{}\033[0m: Calculating the result.".format(option.type))

    
    @classmethod
    def load_chinese_inputstring(cls, string):
        """
        Load encoding of test data.
        
        Args:
            string: A sentence or words.
        
        Returns:
            sentences_tensor: Sentence encoding.
            mask: The mask matrix.
        """
        string = string.strip()
        string = string.replace('\n', '.').replace('\t', '.').replace('\r', '.')
        
        length = len(string)
        for i in range(length // 500):
            substring = string[500 * i : 500 * (i+1)]
            sentence = ["[CLS]"] + list(substring) + ["[SEP]"]
            original_sentence = sentence
            sentences_id = cls.tokenizer.convert_tokens_to_ids(sentence)
            sentences_tensor = torch.LongTensor(sentences_id).unsqueeze(0).to(cls.option.device)
            mask = (sentences_tensor > 0)
            yield original_sentence, sentences_tensor, mask

    @classmethod
    def load_english_inputstring(cls, string):
        """
        Load encoding of test data.
        
        Args:
            string: A sentence or words.
        
        Returns:
            sentences_tensor: Sentence encoding.
            mask: The mask matrix.
        """
        string = string.strip()
        string = string.replace('\n', '.')
        seg_list=[item for item in jieba.cut(string, cut_all = False) if item not in [' ', '\n', '\t', '\r']]
        
        length = len(seg_list)
        for i in range(max(1, length // 500)):
            sub_list = seg_list[500 * i : 500 * (i+1)]
            sentence = ["[CLS]"] + sub_list + ["[SEP]"]
            original_sentence = sentence
            sentences_id = cls.tokenizer.convert_tokens_to_ids(sentence)
            sentences_tensor = torch.LongTensor(sentences_id).unsqueeze(0).to(cls.option.device)
            mask = (sentences_tensor > 0)
            yield original_sentence, sentences_tensor, mask
    
    @classmethod
    def run(cls, string):
        """
        Extraction encryption of input string, return as a coded string

        Args:
            string: Input string.

        Returns:
            chars: Chars of inputs.
            tags: Tags of inputs.
        """
        option = cls.option
        torch.manual_seed(1026)
        
        total_original_sentence = None #torch.LongTensor([]).to(option.device)
        total_chars = None #torch.LongTensor([]).to(option.device)
        total_tag = None #torch.LongTensor([]).to(option.device)
        
        loadinputstring = cls.load_chinese_inputstring if(len(re.findall(u'[\u4e00-\u9fa5]', string)) > 0) else cls.load_english_inputstring
        with torch.no_grad():
            for (original_sentence, sentence, mask) in loadinputstring(string):
                output = cls.model(sentence, None, mask)
                output = torch.Tensor(output)[0].int().to(option.device)
                chars = cls.tokenizer.convert_ids_to_tokens([c.item() for c in sentence[0, :]])
                tag = output
                if(total_original_sentence == None):
                    total_original_sentence = original_sentence
                else:
                    total_original_sentence.extend(original_sentence)
                if(total_chars == None):
                    total_chars = chars
                else:
                    total_chars.extend(chars)
                if(total_tag == None):
                    total_tag = tag
                else:
                    total_tag = torch.cat((total_tag, tag), 0)
        return total_original_sentence, total_chars, total_tag

    @classmethod
    def load_checkpoint(cls):
        """
        Load trained model.

        Returns:
            model_state: Model parameters which have been trained.
        """
        option = cls.option
        if(os.path.exists("{}/best_model.hqt".format(option.path))):
            model_state = torch.load("{}/best_model.hqt".format(option.path))
        else:
            model_state = torch.load("{}/last_model.hqt".format(option.path))
        return model_state
