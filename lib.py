import itertools
import operator
import igraph
import re
from string import punctuation
from collections import Counter
from nltk.corpus import stopwords
from nltk import pos_tag, stem

keeped_tag = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJS', 'JJR']  # WHAT ABOUT VERBS ??
stpwds = stopwords.words('english')
punct = punctuation + '\t'
# punct = punctuation.replace('-', '')


def clean_text(text, remove_stopwords=True, pos_filtering=True, stemming=True):
    text = text.lower()
    text = ''.join(l if l not in punct else ' ' for l in text)
    tokens = text.split()
    if pos_filtering:
        tagged_tokens = pos_tag(tokens)
        tokens = [word for word, tag in tagged_tokens if tag in keeped_tag]
    if remove_stopwords:
        tokens = [word for word in tokens if word not in stpwds]
    if stemming:
        stemmer = stem.PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
    return(tokens)


def terms_to_graph(terms, w):
    from_to = {}
    w = min(len(terms), w)

    terms_temp = terms[0:w]
    indexes = list(itertools.combinations(range(w), r=2))
    new_edges = [tuple([terms_temp[i] for i in my_tuple]) for my_tuple in indexes]

    for new_edge in new_edges:
        if new_edge in from_to:
            from_to[new_edge] += 1
        else:
            from_to[new_edge] = 1

    for i in xrange(w, len(terms)):
        considered_term = terms[i]
        terms_temp = terms[(i-w+1):(i+1)]
        for term_temp in terms_temp[:-1]:
            try_edge = (term_temp, considered_term)
            if try_edge[1] != try_edge[0]:
                if try_edge in from_to:
                    from_to[try_edge] += 1
                else:
                    from_to[try_edge] = 1

    # create empty graph
    g = igraph.Graph(directed=True)
    g.add_vertices(sorted(set(terms)))
    g.add_edges(from_to.keys())

    # set edge and vertice weights
    g.es['weight'] = from_to.values()  # based on co-occurence within sliding window
    g.vs['weight'] = g.strength(weights=from_to.values())  # weighted degree

    return(g)
