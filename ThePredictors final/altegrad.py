# -*- coding: utf-8 -*-
"""
@author: louati
"""

import pandas as pd
from collections import Counter
import operator
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from stop_words import get_stop_words
from library_for_altegrad import *

path_to_data = './'

##############
# load files #
##############

training = pd.read_csv(path_to_data + 'training_set.csv', sep=',', header=0)
training_info = pd.read_csv(path_to_data + 'training_info.csv', sep=',', header=0)
test = pd.read_csv(path_to_data + 'test_set.csv', sep=',', header=0)
test_info = pd.read_csv(path_to_data + 'test_info.csv',sep=',', header=0)

################################
# create some handy structures #
################################

# convert training set to dictionary
emails_ids_per_sender = {}
for index, series in training.iterrows():
    row = series.tolist()
    sender = row[0]
    ids = row[1:][0].split(' ')
    emails_ids_per_sender[sender] = ids

# save all unique sender names
all_senders = emails_ids_per_sender.keys()

# create address book with frequency information for each user
address_books = {}
i = 0

for sender, ids in emails_ids_per_sender.iteritems():
    recs_temp = []
    for my_id in ids:
        recipients = training_info[training_info['mid'] == int(my_id)]['recipients'].tolist()
        recipients = recipients[0].split(' ')
        # keep only legitimate email addresses
        recipients = [rec for rec in recipients if '@' in rec]
        recs_temp.append(recipients)
    # flatten
    recs_temp = [elt for sublist in recs_temp for elt in sublist]
    # compute recipient counts
    rec_occ = dict(Counter(recs_temp))
    # order by frequency
    sorted_rec_occ = sorted(rec_occ.items(), key=operator.itemgetter(1), reverse=True)
    # save
    address_books[sender] = sorted_rec_occ

    if i % 10 == 0:
        print i
    i += 1

# save all unique recipient names
all_recs = list(set([elt[0] for sublist in address_books.values() for elt in sublist]))

# save all unique user names
all_users = []
all_users.extend(all_senders)
all_users.extend(all_recs)
all_users = list(set(all_users))

##########################################################
# Data preprocessings: add sender column for info files  #
##########################################################
i=0
training_info['sender']=training_info['mid']    
for row in training.iterrows():
    mids = row[1]['mids'].split(' ')
    sender = row[1]['sender']
    for mid in mids :
        training_info.loc[training_info['mid'] == int(mid),'sender'] = sender
    i += 1
    if i % 10 ==0 :
        print i

test_info['sender']=test_info['mid']     
for row in test.iterrows():
    mids = row[1]['mids'].split(' ')
    sender = row[1]['sender']
    for mid in mids :
        test_info.loc[test_info['mid'] == int(mid),'sender']= sender

########################################
# Data preprocessings: Date Correction #
########################################
for row in training_info.sort(['date']).iterrows():
    date = row[1]['date']
    if date[:3] == '000':
        date = '2' + date[1:]        
    training_info.loc[row[0], 'date'] = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')

for row in test_info.sort(['date']).iterrows():
    date = row[1]['date']        
    test_info.loc[row[0], 'date'] = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')

#################################################
# create graph of senders and recipients #
#################################################

import networkx as nx
g=nx.DiGraph()
g.add_nodes_from(all_users)
for sender in address_books.keys():
    g.add_weighted_edges_from([(sender,address_books[sender][j][0],address_books[sender][j][1])
                                    for j in range(len(address_books[sender]))])

#########################################################
# First frequency feature : outgoing message percentage #
#########################################################        
OMP = outgoing_message_percentage(all_senders, g)
#########################################################
# Second frequency feature :incoming message percentage #
#########################################################
IMP = incoming_message_percentage(all_recs, all_senders, g)
         

new_train_info = training_info.sort_values(by='date') 
new_test_info = test_info
########################
# Performing embedding #
########################
#global embedding_array
stop_words = get_stop_words('english')
tfidf = TfidfVectorizer(stop_words = stop_words)
embedding_array = tfidf.fit_transform(np.concatenate((new_train_info['body'].values,new_test_info['body'].values)))
embedding_array = embedding_array[:new_train_info.shape[0]]

####################################################################
# This function calculates the different features for a given mail # 
####################################################################           
def compute_features_for_mail(mail_tfidf, mail_date, ground_truth, sender, number_of_similar_messages, number_of_recency_messages, mail_header, OMP, IMP):
    
    info_sender = new_train_info[new_train_info.sender == sender]
    index_sender = info_sender.index.values
    info_sender.index = range(info_sender.shape[0])
    centroid_for_sender = embedding_array[index_sender]
    similar_ids, similarity_scores = cosine_similarities(centroid_for_sender, mail_tfidf, number_of_similar_messages)
    dic_recency = compute_recency_features(info_sender, mail_date, number_of_recency_messages)
    recipients_dict = get_recipients(similar_ids, info_sender, similarity_scores)
    
    features_for_mail = np.zeros((len(recipients_dict), 5))
    labels_for_mail = np.zeros((len(recipients_dict), 1))    
    
    index = 0
    for k,v in recipients_dict.iteritems():
        KNN_Score = v
        OMP_value = OMP[sender][k]
        IMP_value = 0
        if sender in IMP.keys():
            IMP_value = IMP[sender].get(k, 0)
        recency = 0
        if k in dic_recency.keys():
            recency = dic_recency[k]
        if ground_truth != None:
            if k in ground_truth:
                labels_for_mail[index, :] = 1                
        greeting_score = int(greeting(mail_header, k))
        features_for_mail[index, :] = [KNN_Score, OMP_value, IMP_value, recency, greeting_score]
        index +=1
    return features_for_mail, labels_for_mail, recipients_dict    

######################################################################
# We developped this function in the following way in order to be    #
# able to use it for both training and generating submission for       #
# testing                                                            #
######################################################################           
def compute_features(data, true_recipients , predict, submission ,tfidf, OMP, IMP): 
    
    features = np.zeros((0,5))
    labels = np.zeros((0,1))
    
    number_of_similar_messages = 50
    number_of_recency_messages = 100

    data.index = range(data.shape[0])
    results = pd.DataFrame(columns=['recipients'], index=data['mid'])

    for query_id in data.index.values:       
    # pre processing for the given mail
        mail = data['body'][query_id]
        mid = data.loc[query_id]['mid']
        sender= data.loc[query_id]['sender']
        mail_tfidf = tfidf.transform([mail])
        mail_date = data['date'][query_id]
        if true_recipients == True:
            ground_truth = data['recipients'][query_id].split()    
        else: 
            ground_truth = None
        # Compute Features For this email
        features_per_mail, labels_per_mail, recipients_dict = compute_features_for_mail(mail_tfidf, mail_date, ground_truth, sender, number_of_similar_messages, number_of_recency_messages, mail[:20], OMP , IMP)               
        # Add to global features for training:
        features = np.concatenate((features, features_per_mail))
        labels = np.concatenate((labels, labels_per_mail))        
        # Predicting recipients for prediction step:
        if predict :    
            order = clf.predict_proba(features_per_mail)[:,1].argsort()[::-1]
            recipients = np.array(recipients_dict.keys())
            predicted_recipients = recipients[order][:10] 
        # prepare results for submission    
        if submission:
            predicted_recipients_string = ''
            for k in predicted_recipients:
                predicted_recipients_string += k + ' '    
            results.loc[ mid, 'recipients'] = predicted_recipients_string
    if submission:        
        return(results)    
    else :
        return (features, labels)
########################
# Calculate features   #
########################
truth = True ; predict = False ; submission = False ; 
[features,labels]= compute_features(new_train_info, truth , predict , submission ,tfidf , OMP, IMP)    
########################
# Training the model   #
########################
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(features, labels)
########################
# Predicting results   #
########################
truth = False ; predict = True ; submission = True ;
results = compute_features(new_test_info, truth , predict, submission ,tfidf, OMP , IMP)            
########################
# Save predictions     #
########################
path_to_results='./'
with open(path_to_results + 'predictions.txt', 'wb') as my_file:
    my_file.write('mid,recipients' + '\n')
    for i in range(len(results)):
        mid = results.index[i]
        preds = results['recipients'][mid]
        my_file.write(str(mid) + ',' + preds + '\n')