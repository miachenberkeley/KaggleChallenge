# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from lib import clean_text, terms_to_graph
from gensim.models import Word2Vec
import sys


# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def get_clean_bodies(file_info, filename):
    mid_index = file_info.columns.tolist().index('mid')
    body_index = file_info.columns.tolist().index('body')

    body_per_mail_id = {}

    try:
        sys.stdout.write('Trying to load %d emails... ' % len(file_info.values))
        sys.stdout.flush()
        body_per_mail_id = np.load(filename).item()
        sys.stdout.write('OK\n')
    except:
        sys.stdout.write('FAIL\n')
        print("Cleaning %d emails..." % len(file_info.values))
        i = 0
        for index, series in file_info.iterrows():
            row = series.tolist()
            mid = row[mid_index]
            body = row[body_index]
            body_per_mail_id[mid] = clean_text(body)
            if i % 100 == 0:
                print_progress(i, len(file_info.values), bar_length=40)
            i += 1
            np.save(filename, body_per_mail_id)
        print_progress(len(file_info.values), len(file_info.values), bar_length=40)
    return body_per_mail_id


def learn_word_representation(sentences, filename):
    try:
        sys.stdout.write('Trying to load word representations... ')
        sys.stdout.flush()
        wordsVectors = np.load(filename).item()
        sys.stdout.write('OK\n')
    except:
        sys.stdout.write('FAIL\n')
        print("Learning model...")
        model = Word2Vec(sentences, size=100, min_count=5, workers=4)
        model.init_sims(replace=True)
        wordsVectors = {word: model[word] for word in model.vocab.keys()}
        print("Saving model...")
        np.save(filename, wordsVectors)
    return wordsVectors


def extract_keywords_from_tokens(tokens):
    if len(tokens) <= 1:
        keywords = tokens
    else:
        g = terms_to_graph(tokens, w=4)
        core_numbers = dict(zip(g.vs['name'], g.coreness()))
        max_c_n = max(core_numbers.values())
        keywords = [kwd for kwd, c_n in core_numbers.iteritems() if c_n == max_c_n]
    return keywords


def extract_keywords(body_per_mail_id):
    print("Extracting keywords...")
    key_words_gow = {}
    counter = 0

    for mid, tokens in body_per_mail_id.iteritems():
        if len(tokens) > 0:
            key_words_gow[mid] = extract_keywords_from_tokens(tokens)

        counter += 1
        if counter % 2000 == 0:
            sys.stdout.write('\r%d emails processed' % counter)
            sys.stdout.flush()
    sys.stdout.write('\r%d emails processed\n' % counter)
    sys.stdout.flush()
    return key_words_gow


def build_vectors_for_mails(words_per_mail_id, vectors_for_words):
    vectors_for_mails = {}
    for mid, words in words_per_mail_id.iteritems():
        try:
            vectors_for_mails[mid] = np.mean(operator.itemgetter(*words)(vectors_for_words))
        except:
            vectors_for_mails[mid] = np.zeros(100)
            lgth = 0
            for word in words:
                if word in vectors_for_words:
                    vectors_for_mails[mid] += vectors_for_words[word]
                    lgth += 1
            if lgth > 0:
                vectors_for_mails[mid] /= lgth
            else:
                del vectors_for_mails[mid]
    return vectors_for_mails


if __name__ == '__main__':
    path_to_data = "./data/"
    training_info = pd.read_csv(path_to_data + 'training_info.csv', sep=',', header=0)

    body_per_mail_id = get_clean_bodies(training_info, filename="body_per_mail_id.npy")

    sentences = body_per_mail_id.values()
    learn_word_representation(sentences, filename="wordsVectors.npy")

    extract_keywords(body_per_mail_id)
