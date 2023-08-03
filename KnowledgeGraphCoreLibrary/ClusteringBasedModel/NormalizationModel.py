# -*- coding: utf-8 -*-
"""

Author: Qitong Hu, Shanghai Jiao Tong University

"""

import torch
import logging
logging.basicConfig(level=logging.ERROR)

from .EncoderModel import EncoderModel

class NormalizationModel(object):
    """
    The main model for clustering. Compare to words in the standard library 
    after mask processing by encodermodel, add as a new center if all 
    distances are large, otherwise pick the closest center.
    
    Args:
        option: The parameters for main model.
    """
    def __init__(self, option):
        self.option = option
        EncoderModel.init_parameters(option)
        self.dict = dict()
    
    def normalize(self, keyword):
        """
        Implement of mask operation with encodermodel

        Args:
            keyword: The input sentence or word.

        Returns:
            embeddings: Masked sentence or word.
        """
        with torch.no_grad():
            embeddingsoutput = EncoderModel.load_state(keyword)
        embeddings = embeddingsoutput.flatten()
        embeddings = torch.Tensor(embeddings)
        return embeddings
    
    def insert(self, keyword):
        """
        Add center to model.
        
        Args:
            keyword: The input sentence or word.
        """
        self.dict[keyword] = self.normalize(keyword)
    
    def exists(self, keyword):
        """
        Determine if there are any keyword in centers.
        
        Args:
            keyword: The input sentence or word.
        """
        flag = (keyword in self.centers())
        return flag
    
    def centers(self):
        """
        Return all centers of model.
        """
        centers = list(self.dict.keys())
        return centers
    
    def get(self, index):
        """
        Get the index'th center.
        
        Args:
            index: The index number.
        
        Returns:
            center: The selected center.
        """
        centers = self.centers()
        center = centers[index]
        return center
    
    def distance(self, embd1, embd2):
        """
        Return the distance between two masks.
        
        Args:
            embd1: Masked sequence 1.
            embd2: Masked sequence 2.
        
        Returns:
            distance: The cos_sim distance between two embedding sequences.
        """
        distance = embd1 @ embd2 / (torch.norm(embd1) * torch.norm(embd2))
        return distance
    
    def embeddings(self):
        """
        Implement mask operation with all centers. As center varies, 
        we needs to update regularly, and return the center and it's mask.
        
        Returns:
            centers: Centers in this model.
            centerembeddings: Embeddings for centers.
        """
        centers = list(self.dict.keys())
        centerembeddings = [self.dict[center] for center in centers]
        return centers, centerembeddings
    
    def bestcenter(self, keyword, flag = 0.5):
        """
        Return the most suitable clustering center. If each of these distances
        is very large, then return None.
        
        Args:
            keyword: The input sentence or word..
            flag: As the boundary. The default is 0.5, if all distances are 
                smaller than flag, then add as the new center.

        Returns:
            bestcenter: The best possible center.

        """
        centers, centerembeddings = self.embeddings()
        keyembedding = self.normalize(keyword)
        distances = list()
        for (center, centerembedding) in zip(centers, centerembeddings):
            distances.append(self.distance(keyembedding, centerembedding))
        distances = torch.Tensor(distances).to(self.option.device).view(-1)
        if (torch.max(distances).item() < flag):
            bestcenter = None
        else:
            bestcenter = torch.argmax(distances).item()
        return bestcenter