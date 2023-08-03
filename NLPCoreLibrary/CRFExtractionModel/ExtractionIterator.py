# -*- coding: utf-8 -*-

import os
import torch
from transformers import logging
from tqdm import tqdm

from .ExtractionDataLoader import ExtractionDataLoader
from .ExtractionModel import ExtractionModel
logging.set_verbosity_error()

class ExtractionIterator:
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
    def run(cls, test_data_paths, is_valid = True):
        option = cls.option
        torch.manual_seed(1026)
        
        cls.validation_length = len(test_data_paths)
        if(is_valid != True):
            test_iterator = ExtractionDataLoader.loadTestData(test_data_paths)
            with tqdm(total = cls.validation_length) as bar:
                with torch.no_grad():
                    for setp, (original_sentences, sentences, mask, data_paths) in enumerate(test_iterator):
                        batch_size = mask.size(0)
                        outputs = cls.model(sentences, None, mask)
                        batch_length = len(outputs)
                        for b in range(batch_length):
                            original_sentence = original_sentences[b]
                            chars = cls.tokenizer.convert_ids_to_tokens([c.item() for c in sentences[b, :]])
                            tag = torch.Tensor(outputs[b]).int().to(option.device)
                            data_path = data_paths[b]
                            yield original_sentence, chars, tag, data_path
                        bar.update(batch_size)
        else:
            validation_iterator = ExtractionDataLoader.loadTrainData(test_data_paths)
            with tqdm(total = cls.validation_length) as bar:
                with torch.no_grad():
                    for setp, (original_sentences, sentences, tags, mask) in enumerate(validation_iterator):
                        batch_size = tags.size(0)
                        outputs = cls.model(sentences, None, mask)
                        batch_length = len(outputs)
                        for b in range(batch_length):
                            original_sentence = original_sentences[b]
                            chars = cls.tokenizer.convert_ids_to_tokens([c.item() for c in sentences[b, :]])
                            output = torch.Tensor(outputs[b]).int().to(option.device)
                            tag = tags[b, :].squeeze(0)[0 : output.shape[0]]
                            yield original_sentence, chars, tag, output
                        bar.update(batch_size)
    
    @classmethod
    def load_checkpoint(cls):
        option = cls.option
        if(os.path.exists("{}/best_model.hqt".format(option.path))):
            model_state = torch.load("{}/best_model.hqt".format(option.path))
        else:
            model_state = torch.load("{}/last_model.hqt".format(option.path))
        return model_state
