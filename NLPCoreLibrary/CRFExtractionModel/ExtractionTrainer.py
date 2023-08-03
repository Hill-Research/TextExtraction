# -*- coding: utf-8 -*-

#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor,
#  Boston, MA  02110-1301, USA.

import os
import time
import torch
import warnings
from tqdm import tqdm
from transformers import logging

from .ExtractionOptimizer import AdaBelief
from .ExtractionModel import ExtractionModel
from .ExtractionDataLoader import ExtractionDataLoader
from .ExtractionTrainerOutput import ExtractionTrainerOutput
from transformers import get_linear_schedule_with_warmup
#warnings.filterwarnings("ignore")
#logging.set_verbosity_error()

class ExtractionTrainer:
    """
    Trainer of extraction model.
    
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
        option.logger.info("\033[0;37;31m{}\033[0m: Loading parameters for extraction data loader.".format(option.type))
        ExtractionDataLoader.init_parameters(cls.option)
        option.logger.info("\033[0;37;31m{}\033[0m: Loading tokenizer for extraction model.".format(option.type))
        ExtractionDataLoader.loadTokenizer()
        
    @classmethod
    def train(cls, train_data_paths):
        """
        The interface for training.
        """
        option = cls.option
        cls.train_length = len(train_data_paths)
        torch.manual_seed(1026)
        
        cls.option.logger.info("\033[0;37;31m{}\033[0m: Loading parameters for training extraction model.".format(option.type))
        print("\033[0;37;31m{}\033[0m: Loading parameters for training extraction model.".format(option.type))
        
        cls.option.logger.info("\033[0;37;31m{}\033[0m: Training device is {}.".format(option.type, option.device))
        print("\033[0;37;31m{}\033[0m: Training device is {}.".format(option.type, option.device))
        
        if (not os.path.exists(option.path)):
            os.makedirs(os.path.join(option.path))
            
        if (os.path.exists("{}/last_model.hqt".format(option.path)) and (not option.is_train)):
            model = ExtractionModel(option)
            model.load_state_dict(cls.load_checkpoint())
            model = model.to(option.device)
            cls.option.logger.info("\033[0;37;31m{}\033[0m: Loading model from checkpoint file.".format(option.type))
            print("\033[0;37;31m{}\033[0m: Loading model from checkpoint file.".format(option.type))
        else:
            model = ExtractionModel(option)
            model = model.to(option.device)
            cls.option.logger.info("\033[0;37;31m{}\033[0m: Loading new model.".format(option.type))
            print("\033[0;37;31m{}\033[0m: Loading new model.".format(option.type))
        optimizer = cls.optim(model)
        cls.option.logger.info("\033[0;37;31m{}\033[0m: Optimizer is {}.".format(option.type, "AdaBelief"))
        print("\033[0;37;31m{}\033[0m: Optimizer is {}.".format(option.type, option.optimizer))
        
        scheduler = cls.warmup(optimizer)
        cls.option.logger.info("\033[0;37;31m{}\033[0m: Using warm up method.".format(option.type))
        print("\033[0;37;31m{}\033[0m: Using warm up method.".format(option.type))
        
        train_iterator = ExtractionDataLoader.loadTrainData(train_data_paths)
        
        best_loss = float("inf")
        pre_loss = -1
        cache = dict()
        for step in range(option.train_step):
            start_time = time.time()
            cls.option.logger.info("\033[0;37;31m{}\033[0m: Training Epoch {}".format(option.type, step))
            print("\033[0;37;31m{}\033[0m: Training Epoch {}".format(option.type, step))
            val_loss = cls.trainepoch(train_iterator, model, optimizer, scheduler)
            if(option.is_output):
                cache[step] = val_loss
            cls.option.logger.info("\033[0;37;31m{}\033[0m: Testing Epoch {}".format(option.type, step))
            print("\033[0;37;31m{}\033[0m: Testing Epoch {}".format(option.type, step))
            cls.store_checkpoint(model, val_loss < best_loss)
            best_loss = min(val_loss, best_loss)
            
            cls.option.logger.info("\033[0;37;31m{}\033[0m: Epoch: {}, loss: {}, time: {}".format(option.type, step, val_loss, time.time() - start_time))
            print("\033[0;37;31m{}\033[0m: Epoch: {}, loss: {}, time: {}".format(option.type, step, val_loss, time.time() - start_time))
            if(val_loss == 0 or abs(val_loss - pre_loss) / val_loss < option.epsilon):
                cls.option.logger.info("\033[0;37;31m{}\033[0m: The model has been trained.".format(option.type))
                print("\033[0;37;31m{}\033[0m: The model has been trained.".format(option.type))
                if(option.is_output):
                    ExtractionTrainerOutput.Output(cache, ['txt', 'pic'], option)
                break
            pre_loss = val_loss
    
    @classmethod
    def warmup(cls, optimizer):
        """
        Warmup technique, which can optimize training compensation.
        
        Args:
            optimizer: The selected optimize tool.
            
        Returns:
            scheduler: Warmuped optimizer.
        """
        option = cls.option
        warm_up_ratio = 0.015
        total_steps = (cls.train_length // option.batch_size) * option.train_step if cls.train_length % option.batch_size == 0 else (cls.train_length // option.batch_size + 1) * option.train_step
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warm_up_ratio * total_steps, num_training_steps = total_steps)
        return scheduler
    
    @classmethod
    def trainepoch(cls, train_iterator, model, optimizer, scheduler):
        """
        Word training function.
        
        Args:
            trainiterator: The training set interface.
            model: AI model.
            optimizer: Selected optimizer.
            scheduler: Warmuped step parameters.
        """
        total_loss = 0
        total_count = 0
        with tqdm(total = cls.train_length) as bar:
            for step, ( _ , sentences, tags, mask) in enumerate(train_iterator):
                # optimizer.zero_grad()
                batch_size, _ = tags.shape
                loss = model(sentences, tags, mask).to(cls.option.device)
                total_loss = total_loss + loss.item() * batch_size
                total_count = total_count + batch_size
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                bar.update(sentences.size(0))
        mean_loss = total_loss / total_count
        return mean_loss
    
    @classmethod
    def optim(cls, model):
        """
        Adjust the training compensation for different sub-modules in AI, 
        increase the training steps of crf. This provides a better way to 
        classification effect.
        
        Returns:
            optimizer: The training compensation optimizer that has been set up.
        """
        option = cls.option
        bert_optimizer = list(model.bert.named_parameters())
        lstm_optimizer = list(model.lstm.named_parameters())
        linear_optimizer = list(model.linear.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
             'lr': option.learning_rate * option.rate_scale['bert'], 'weight_decay': option.weight_decay},
            {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
             'lr': option.learning_rate * option.rate_scale['bert'], 'weight_decay': 0.0},
            {'params': [p for n, p in lstm_optimizer if not any(nd in n for nd in no_decay)],
             'lr': option.learning_rate * option.rate_scale['lstm'], 'weight_decay': option.weight_decay},
            {'params': [p for n, p in lstm_optimizer if any(nd in n for nd in no_decay)],
             'lr': option.learning_rate * option.rate_scale['lstm'], 'weight_decay': 0.0},
            {'params': [p for n, p in linear_optimizer if not any(nd in n for nd in no_decay)],
             'lr': option.learning_rate * option.rate_scale['linear'], 'weight_decay': option.weight_decay},
            {'params': [p for n, p in linear_optimizer if any(nd in n for nd in no_decay)],
             'lr': option.learning_rate * option.rate_scale['linear'], 'weight_decay': 0.0},
            {'params': model.crf.parameters(), 'lr': option.learning_rate * option.rate_scale['crf']}
        ]
        if(option.optimizer == "AdaBelief"):
            optimizer = AdaBelief(optimizer_grouped_parameters, lr = cls.option.learning_rate)
        if(option.optimizer == "Adam"):
            optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr = cls.option.learning_rate)
        if(option.optimizer == "SGD"):
            optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr = cls.option.learning_rate)
        return optimizer
    
    @classmethod
    def store_checkpoint(cls,model, is_best):
        """
        Store trained model.

        Args:
            model: AI model.
            is_best: Whether it is the current best model.
        """
        option = cls.option
        model_save_path = os.path.join(os.curdir, '{}/last_model.hqt'.format(option.path))
        torch.save(model.state_dict(), model_save_path)
        if (is_best):
            best_save_path = os.path.join(os.curdir, "{}/best_model.hqt".format(option.path))
            torch.save(model.state_dict(), best_save_path)
            
    @classmethod
    def load_checkpoint(cls):
        """
        Load trained model.

        Returns:
            model_state: Model parameters which have been trained.
        """
        option = cls.option
        if(os.path.exists(os.path.join(os.curdir, "{}/best_model.hqt".format(option.path)))):
            model_state = torch.load(os.path.join(os.curdir, "{}/best_model.hqt".format(option.path)))
        else:
            model_state = torch.load(os.path.join(os.curdir, "{}/last_model.hqt".format(option.path)))
        return model_state
