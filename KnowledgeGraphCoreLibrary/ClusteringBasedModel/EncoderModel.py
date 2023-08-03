# -*- coding: utf-8 -*-
"""

Author: Qitong Hu, Shanghai Jiao Tong University

"""

import os
import torch
import warnings

from transformers import logging
from transformers import BertModel
from transformers import BertConfig
from transformers import BertTokenizer
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

class EncoderModel:
    """
    This is an implementation of Bert-whitening-first_last_avg algorithm[SJL2021].
    
    Args:
        option: The parameter of main model. Here we need option.model, 
                the potential alternative parameters including SentenceTransformer, 
                first_last_avg, first_avg, last_avg, first_last_avg-whitening, 
                first_avg-whitening, last_avg-whitening, and option.bertmodel is 
                the selected bert pre-training model
        
    [SJL2021] Su, Jianlin et al. “Whitening Sentence Representations for Better 
                Semantics and Faster Retrieval.” ArXiv abs/2103.15316 (2021): n. pag.
    """
    @classmethod
    def init_parameters(cls, option):
        """
        Initialization for parameters.
        
        Args:
            option: The parameter of main model.
        """
        cls.option = option
        if(option.model == "SentenceTransformer"):
            option.logger.info("\033[0;37;31m{}\033[0m: Selecing SentenceTransfomer as the encoder.".format(option.type))
            from sentence_transformers import SentenceTransformer
            cls.bert = SentenceTransformer(os.path.join(option.modelpath, option.bertmodel))
        else:
            option.logger.info("\033[0;37;31m{}\033[0m: Selecing bert as the encoder.".format(option.type))
            cls.config = BertConfig.from_pretrained(os.path.join(option.modelpath, option.bertmodel), output_hidden_states = True, output_attentions = True)
            cls.bert = BertModel.from_pretrained(os.path.join(option.modelpath, option.bertmodel), config = cls.config).to(option.device)
            cls.tokenizer = BertTokenizer.from_pretrained(os.path.join(option.modelpath, option.bertmodel))
            if("whitening" in option.model):
                option.logger.info("\033[0;37;31m{}\033[0m: Loading transformer matrix for PCA.".format(option.type))
                states = list()
                with open(option.condition_file, "r", encoding = "utf-8") as f:
                    keywords = [line.split("\n")[0].split("\t")[0].strip() for line in f.readlines() if (len(line.strip()) > 0)]
                for keyword in keywords:
                    states.append(cls.load_original_state(keyword))
                states = torch.cat(states, dim = 0)
                cls.kernel = cls.load_svd_params(states)
    
    @classmethod
    def load_state(cls, string):
        """
        Load the encoder state, and decide whether implemented by svd if option.model includes "whitening".

        Args:
            string: Input string.
            alpha: The ratio between the first state and the last state. The default is 0.5.

        Returns:
            state: The encoder state.

        """
        if(cls.option.model == "SentenceTransformer"):
            state = torch.Tensor(cls.bert.encode(string))
        else:
            state = cls.load_original_state(string)
            
            if("whitening" in cls.option.model):  
                state = cls.load_svd(state, cls.kernel).mean(dim = 0)
            else:
                state = state.mean(dim = 0)
        return state
    
    @classmethod
    def load_original_state(cls, string):
        """
        Load the state after bert initialization. The processing mode bases on the choise of option.model: 
            first_avg only select the first layer mask, last only select the last layer, first-last selection
            based on ratio by alpha.

        Args:
            string: Input string.
            alpha: The ratio between the first state and the last state. The default is 0.5.

        Returns:
            state: The encoder state.
        """
        alpha = cls.option.alpha
        sentencelist = list(string)
        sentencetensor = torch.LongTensor(cls.tokenizer.convert_tokens_to_ids(sentencelist)).unsqueeze(0).to(cls.option.device)
        hidden_states = cls.bert(sentencetensor)['hidden_states']
        first_state = hidden_states[0]
        last_state = hidden_states[-1]
        first_state = first_state.squeeze(0)
        last_state = last_state.squeeze(0)
        if("first_avg" in cls.option.model):
            state = first_state
        if("last_avg" in cls.option.model):
            state = last_state
        if("first_last_avg" in cls.option.model):
            state = alpha * first_state + (1 - alpha) * last_state
        return state
    
    @classmethod
    def load_svd_params(cls, inputs):
        """
        Load the parameter of svd.

        Args:
            inputs: The total original matrix.

        Returns:
            kernel: The transform kernel of total matrix.
        """
        cov = torch.cov(inputs.T)
        values, vectors = torch.linalg.eigh(cov)
        cls.option.logger.info("\033[0;37;31m{}\033[0m: Loading factor numbers for PCA.".format(cls.option.type))
        factor_number = cls.load_factor_number(values)
        kernel = vectors[:, -factor_number :]
        return kernel
    
    @classmethod
    def load_svd(cls, inputs, kernel):
        """
        Load the matrix after svd.

        Args:
            inputs: The original matrix.

        Returns:
            outputs: The matrix after svd.
        """
        outputs = inputs @ kernel
        # norms = torch.clip((outputs**2).sum(dim = 1) ** 0.5, 1e-8, torch.inf).unsqueeze(0).T
        # outputs = outputs / norms
        return outputs
    
    @classmethod
    def load_factor_number(cls, values, flag = 0.9):
        """
        Return number of features selected in svd.

        Args:
            values: Eigenvalues of covariance matrix.
            flag: Sum of the eigenvalues ends at flag.

        Returns:
            factor_number: Number of features selected in svd.

        """
        cum_values = torch.cumsum(values, dim = 0) / torch.sum(values)
        factor_number = values.size()[0] - torch.nonzero(cum_values < 1 - flag).max()
        factor_number = int(factor_number)
        return factor_number
