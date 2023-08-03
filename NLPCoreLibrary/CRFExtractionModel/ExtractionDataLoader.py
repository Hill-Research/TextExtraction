# -*- coding: utf-8 -*-

import os
import torch
from tqdm import tqdm
from torch.utils import data
from transformers import BertTokenizer

class ExtractionTrainDataset(data.Dataset):
    """
    Inherit from data.dataset, Data Extraction.
    
    Args:
        data_paths: All paths for the training set.
        tokenizer: Letter transcoder.
        option: The main model parameter.
    """
    def __init__(self, data_paths, tokenizer, option):
        super(ExtractionTrainDataset, self).__init__()
        self.sentences = list()
        self.tags = list()
        self.tokenizer = tokenizer
        self.tag2id = option.tag2id
        
        for data_path in data_paths:
            with open(data_path, "r", encoding = "UTF-8") as f:
                lines = [line.split("\n")[0] for line in f.readlines() if (len(line.strip())>0)]
            
            sentence = ["[CLS]"] + [line.split("\t")[0] for line in lines] + ["[SEP]"]
            tag = ["[CLS]"] + [line.split("\t")[1] for line in lines] + ["[SEP]"]
            sentence = [c for (i, c) in enumerate(sentence) if i < 512]
            tag = [t for (i, t) in enumerate(tag) if i < 512]
            for (i, item) in enumerate(tag):
                tag[i] = tag[i] if (item in self.tag2id) else "ELSE"
            self.sentences.append(sentence)
            self.tags.append(tag)
    
    def __getitem__(self, index):
        """
        Return each item in the data set.

        Returns:
            sentences_ids: The transcoding of the sentence.
            tag_ids: The transcoding of label.
            sentence_length: Length of sentence.
        """
        sentence = self.sentences[index]
        original_sentence = sentence
        tag = self.tags[index]
        sentences_ids = self.tokenizer.convert_tokens_to_ids(sentence)
        tag_ids = [self.tag2id[t] for t in tag]
        sentence_length = len(sentences_ids)
        return original_sentence, sentences_ids, tag_ids, sentence_length
    
    def __len__(self):
        """
        Return the length of the data set.

        Returns:
            length: The length of the data set.
        """
        length = len(self.sentences)
        return length

class ExtractionTestDataset(data.Dataset):
    """
    Inherit from data.dataset, Data Extraction.
    
    Args:
        data_paths: All paths for the training set.
        tokenizer: Letter transcoder.
        option: The main model parameter.
    """
    def __init__(self, data_paths, tokenizer, option):
        super(ExtractionTestDataset, self).__init__()
        self.sentences = list()
        self.data_paths = data_paths
        self.tokenizer = tokenizer
        
        for data_path in tqdm(data_paths):
            with open(data_path, "r", encoding = "UTF-8") as f:
                lines = [line.strip().replace("\\s", "") for line in f.readlines() if (len(line.strip())>0)]
            sentence = ["[CLS]"] + [t for t in "".join(lines)] + ["[SEP]"]
            sentence = [c for (i, c) in enumerate(sentence) if i < 512]
            self.sentences.append(sentence)
    
    def __getitem__(self, index):
        """
        Return each item in the data set.

        Returns:
            sentences_ids: The transcoding of the sentence.
            tag_ids: The transcoding of label.
            sentence_length: Length of sentence.
        """
        sentence = self.sentences[index]
        original_sentence = sentence
        sentences_ids = self.tokenizer.convert_tokens_to_ids(sentence)
        data_path = self.data_paths[index]
        data_path = data_path.split("\\")[-1].split(".")[0]
        sentence_length = len(sentences_ids)
        return original_sentence, sentences_ids, sentence_length, data_path
    
    def __len__(self):
        """
        Return the length of the data set.

        Returns:
            length: The length of the data set.
        """
        return len(self.sentences)

def patchTrainBatch(batch, option):
    """
    Set the data set to patch, training as a faster speed.

    Parameters:
        batch: The size of each package.
        option: Parmeters of main model.

    Returns:
        sentences_tensors: The sentence encoding.
        tags_tensors: The label encoding.
        mask: The mask matrix.
    """
    maxlen = max([t[3] for t in batch])
    original_sentences = [i[0] for i in batch]
    sentences_tensors = torch.LongTensor([i[1] + [0] * (maxlen - len(i[1])) for i in batch]).to(option.device)
    tags_tensors = torch.LongTensor([i[2] + [0] * (maxlen - len(i[2])) for i in batch]).to(option.device)
    mask = (sentences_tensors > 0)
    return original_sentences, sentences_tensors, tags_tensors, mask

def patchTestBatch(batch, option):
    """
    Set the data set to patch, training as a faster speed.

    Parameters:
        batch: The size of each package.
        option: Parmeters of main model.

    Returns:
        sentences_tensors: The sentence encoding.
        mask: The mask matrix.
        data_paths: The path of test datas.
    """
    maxlen = max([t[1] for t in batch])
    original_sentences = [i[0] for i in batch]
    sentences_tensors = torch.LongTensor([i[0] + [0] * (maxlen - len(i[0])) for i in batch]).to(option.device)
    mask = (sentences_tensors > 0)
    data_paths = [t[2] for t in batch]
    return original_sentences, sentences_tensors, mask, data_paths

class ExtractionDataLoader:
    """
    Load the main model of the dataset.
    
    Args:
        option: The parameter of main model.
    """
    @classmethod
    def init_parameters(cls, option):
        """
        Initialization for parameters.
        
        Args:
            option: The parameter of main model.
        """
        cls.option = option
    
    @classmethod
    def loadTokenizer(cls):
        """
        Load tokenizer, an already trained model is used here, which depends 
            on option.bertmodel.
        
        Returns:
            tokenizer: A tokenizer.
        """
        tokenizer = BertTokenizer.from_pretrained(os.path.join(cls.option.modelpath, cls.option.bertmodel))
        cls.option.logger.info("\033[0;37;31m{}\033[0m: Sucessfully loading the tokenizer.".format(cls.option.type))
        cls.tokenizer = tokenizer
        return tokenizer
    
    @classmethod
    def loadTrainData(cls, train_data_paths):
        """
        Load training data interface.
        
        Args:
            train_data_dir: Position of training set.
            
        Returns:
            train_iterator: The training set interface.
            validation_iterator: The validation set interface.
        """
        option = cls.option

        train_dataset = ExtractionTrainDataset(train_data_paths, cls.tokenizer, option)
        cls.option.logger.info("\033[0;37;31m{}\033[0m: Reading training data.".format(option.type))
        train_iterator = data.DataLoader(dataset = train_dataset,
                                         batch_size = option.batch_size,
                                         shuffle = True,
                                         collate_fn = lambda x : patchTrainBatch(x, option))
        return train_iterator
    
    @classmethod
    def loadTestData(cls, test_data_paths):
        """
        Load training data interface.
        
        Args:
            train_data_dir: Position of training set.
            
        Returns:
            train_iterator: The training set interface.
            validation_iterator: The validation set interface.
        """
        option = cls.option
        test_dataset = ExtractionTestDataset(test_data_paths, cls.tokenizer, option)
        
        cls.option.logger.info("\033[0;37;31m{}\033[0m: Reading testing data.".format(option.type))
        test_iterator = data.DataLoader(dataset = test_dataset,
                                        batch_size = option.batch_size,
                                        shuffle = True,
                                        collate_fn = lambda x : patchTestBatch(x, option))
        
        return test_iterator
