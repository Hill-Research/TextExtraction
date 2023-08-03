# -*- coding: utf-8 -*-

from CriteriaExtraction import CriteriaExtraction
from CriteriaNormalization import CriteriaNormalization
from CriteriaGeneration import CriteriaGeneration
import FormatOutput
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--is_train", action='store_true', default = False)
parser.add_argument('--is_output', type = bool, default = True)
parser.add_argument('--cuda', type = str, default = True)
option = parser.parse_args()

keyword_train_data_dir = "../../data/Novartis/train_data"
keyword_name = "Novartis"
print("\033[0;37;33m{}\033[0m: Start data extraction.".format(keyword_name))
keyword = CriteriaExtraction(keyword_name, is_train = option.is_train, is_output = option.is_output, cuda = option.cuda)
keyword.train_valid(keyword_train_data_dir)
with open('../../data/Novartis/qianliexian.txt', 'r', encoding = 'utf-8') as f:
    string = f.read()
extractionitems = keyword.run(string)
print("\033[0;37;33m{}\033[0m: Extracted items are:".format(keyword_name))
FormatOutput.formatdic(extractionitems)

#normalization_condition_file = "../data/Novartis/knowledgegraph_item.txt"
#normalization_unit_file = "../data/Novartis/knowledgegraph_unit.txt"
#normalization_name = "Novartis"
#print("\033[0;37;33m{}\033[0m: Start data normalization.".format(normalization_name))
#normalization = CriteriaNormalization(normalization_name, normalization_condition_file, normalization_unit_file)
#normalizationitems = normalization.run(extractionitems)
#print("\033[0;37;33m{}\033[0m: Normalized items are:".format(keyword_name))
#FormatOutput.formatlist1(normalizationitems)

#generation_condition_file = "../data/Novartis/database_link.txt"
#generation_name = "Novartis"
#print("\033[0;37;33m{}\033[0m: Start criteria gernation.".format(generation_name))
#generation = CriteriaGeneration(generation_name, generation_condition_file)
#generationitems = generation.run(normalizationitems)
#print("\033[0;37;33m{}\033[0m: Criteria are:".format(generation_name))
#FormatOutput.formatlist2(generationitems)
