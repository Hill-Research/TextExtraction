# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class ExtractionTrainerOutput:
    @classmethod
    def Output(cls, dic, type_, option):
        cls.option = option
        cls.dic = dic
        if('txt' in type_):
            cls.OutputTxt()
        if('pic' in type_):
            cls.OutputPic()
    
    @classmethod
    def OutputTxt(cls):
        option = cls.option
        with open('{}/loss.txt'.format(option.path), 'w+', encoding = 'utf-8') as f:
            for key in cls.dic:
                loss = cls.dic[key]
                f.write('Epoch: {}\n'.format(key))
                f.write('\tLoss: {}\n'.format(loss))
    
    @classmethod
    def OutputPic(cls):
        epoch_value = list()
        loss_value = list()
        for key in cls.dic:
            epoch_value.append(int(key))
            loss = cls.dic[key]
            loss_value.append(loss)
        fig_loss=plt.figure(figsize=(6,6))
        ax_loss=fig_loss.add_subplot(111)
        ax_loss.plot(epoch_value, loss_value, c='#DC8910', alpha=1, marker='o')
        ax_loss.tick_params(axis='both',which='both',direction='out',width=1,length=10, labelsize=35)
    
        bwith=4
        ax_loss.spines['bottom'].set_linewidth(bwith)
        ax_loss.spines['left'].set_linewidth(bwith)
        ax_loss.spines['right'].set_visible(False)
        ax_loss.spines['top'].set_visible(False)
        plt.xlabel(r'Epoch',fontsize=30)
        plt.ylabel('$Loss$',fontsize=30)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=3))
        plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=3))
        plt.tight_layout()
        plt.savefig('{}/loss.jpg'.format(cls.option.path),dpi=300)
