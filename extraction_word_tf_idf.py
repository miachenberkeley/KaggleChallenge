import operator
import pandas as pd
from collections import Counter
from library import clean_text_simple, terms_to_graph, unweighted_k_core
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import numpy as np

path_to_data = './'

training_info = pd.read_csv(path_to_data + 'training_info.csv', sep=',', header=0)
stpwds = stopwords.words('english')


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

tfidf_np = np.array(keywords_tfidf)
np.save('tf_idf_key_words', tfidf_np)
