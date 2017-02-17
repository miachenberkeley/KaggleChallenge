import operator
import pandas as pd
from collections import Counter
from library import clean_text_simple, terms_to_graph, unweighted_k_core
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import csv
import numpy as np

path_to_data = './'

training_info = pd.read_csv(path_to_data + 'training_info.csv', sep=',', header=0)
stpwds = stopwords.words('english')

#extract key words
key_words_gow = {}
counter = 0

##########
# gow #
##########

for mid, info in zip(training_info['mid'], training_info['body']):
    # pre-process document
    try :
        my_tokens = clean_text_simple(info)
    except :
        print "info", info
        print "my_tokens", my_tokens
        key_words_gow[mid] = "erreur, a corriger"


    if len(my_tokens) == 0:
        pass
    elif len(my_tokens) == 1:
        keywords = my_tokens
        key_words_gow[mid] = keywords
    else :
        try:
            w = min(len(my_tokens),4)
            #print "w", w
            g = terms_to_graph(my_tokens, w)

            # decompose graph-of-words
            core_numbers = dict(zip(g.vs['name'], g.coreness()))
            #print "core_numbers", core_numbers

            max_c_n = max(core_numbers.values())
            keywords = [kwd for kwd, c_n in core_numbers.iteritems() if c_n == max_c_n]
            #print(keywords)
            key_words_gow[mid] = keywords
            # save results
        except:
            print "id", mid
            print "core_numbers", core_numbers
    counter += 1
    if counter % 100 == 0:
        print counter, 'body processed'

gow_np = np.array(key_words_gow)
np.save('gow_np', gow_np)
