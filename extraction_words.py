import operator
import pandas as pd
from collections import Counter
from library import clean_text_simple, terms_to_graph, unweighted_k_core
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

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

    if len(my_tokens) == 0:
        pass
    elif len(my_tokens) == 1:
        keywords = my_tokens
        key_words_pr[mid] = keywords
    else :
        w = min(len(my_tokens), 4)
        # create graph-of-words
        g = terms_to_graph(my_tokens, w=4)
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




key_words_tf_idf = {}
counter = 0
########
#tf_idf#
########

bodys_cleaned_strings = [' '.join(elt) for elt in training_info['body']]

tfidf_vectorizer = TfidfVectorizer(stop_words=stpwds)
doc_term_matrix = tfidf_vectorizer.fit_transform(bodys_cleaned_strings)
terms = tfidf_vectorizer.get_feature_names()
vectors_list = doc_term_matrix.todense().tolist()

keywords_tfidf = []
counter = 0

for vector in vectors_list:

    # bow feature vector as list of tuples
    terms_weights = zip(terms, vector)
    # keep only non zero values (the words in the document)
    nonzero = [tuple for tuple in terms_weights if tuple[1] != 0]
    # rank by decreasing weights
    nonzero = sorted(nonzero, key=operator.itemgetter(1), reverse=True)
    # retain top 33% words as keywords
    numb_to_retain = int(round(len(nonzero) / 3))
    keywords = [tuple[0] for tuple in nonzero[:numb_to_retain]]

    keywords_tfidf.append(keywords)

    counter += 1
    if counter % 100 == 0:
        print counter, 'vectors processed'