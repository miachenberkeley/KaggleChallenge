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

key_words_pr = {}
counter = 0
#########
#page rank
#########


for mid, info in zip(training_info['mid'], training_info['body']):
    # pre-process document
    try :
        my_tokens = clean_text_simple(info)
    except :
        print "info", info
        print "my_tokens", my_tokens
        key_words_pr[mid] = "erreur, a corriger"

    if len(my_tokens) == 0:
        pass
    elif len(my_tokens) == 1:
        keywords = my_tokens
        key_words_pr[mid] = keywords
    else :

        w = min(len(my_tokens), 4)
        # create graph-of-words
        g = terms_to_graph(my_tokens, w)
        # compute PageRank scores
        pr_scores = zip(g.vs['name'], g.pagerank())
        # rank in decreasing order
        pr_scores = sorted(pr_scores, key=operator.itemgetter(1), reverse=True)
        # retain top 33% words as keywords
        numb_to_retain = int(round(len(pr_scores) / 3))
        keywords = [tuple[0] for tuple in pr_scores[:numb_to_retain]]
        key_words_pr[mid] = keywords


    counter += 1
    if counter % 100 == 0:
        print counter, 'bodys processed'

pr_np = np.array(key_words_pr)
np.save('page_rank_key_words', pr_np)
