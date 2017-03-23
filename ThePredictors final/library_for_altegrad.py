# -*- coding: utf-8 -*-
"""
@author: louati
"""
from sklearn.metrics.pairwise import cosine_similarity

#########################################################################
# Textual feature : cosine similarity between the centroid and the mail #
#########################################################################
def cosine_similarities(centroid, mail, n):    
    scores = cosine_similarity(centroid, mail)
    if int(round(sorted(scores[:,0], reverse=True)[0])) == 1:
        similar_ids = scores[:,0].argsort()[::-1][1:]
    else:
        similar_ids = scores[:,0].argsort()[::-1]    
    return similar_ids[:n], scores
    
#############################################
# Get recipients from most similar messages #
#############################################    
def get_recipients(ids, data, scores):
    recs_dict = {}
    for idx in ids:
        recipients = data.loc[idx,'recipients'].split()
        for recipient in recipients:
            if '@' in recipient:
                recs_dict[recipient] = recs_dict.get(recipient, 0) + scores[idx][0]
    return recs_dict

#########################################################
# First frequency feature : outgoing message percentage #
#########################################################
def outgoing_message_percentage(all_senders, g):
    # Initialization
    OMP = {}
    for sender in all_senders:
        OMP[sender]=dict()
    
    # Compute OMP feature:
     
    for sender in all_senders:
        norm_sender = g.out_degree(sender)        
        for recipient in g.neighbors(sender):
            OMP[sender][recipient]=float(g[sender][recipient]['weight'])/norm_sender    
    return OMP

#########################################################
# Second frequency feature :incoming message percentage #
#########################################################
def incoming_message_percentage(all_recs, all_senders, g):
    # Initialization
    IMP = {}
    for recipient in all_recs:
        IMP[recipient]=dict()
    
    # Compute OMP feature:
    
    for sender in all_senders:
        norm_sender = g.in_degree(sender) 
        for recipient in all_recs:
            if g.has_edge(recipient, sender):
                IMP[recipient][sender]= float(g[recipient][sender]['weight'])/norm_sender 
    return IMP

###########################################################
# Recency features :extract the normalized sent frequency #
# of all users in the training set based on a given       # 
# parameter that indicates the number of messages to take #
# in consideration for computing this feature             #
###########################################################
def compute_recency_features(data_sender, mail_date, number_of_recency_features):    
    recency_feature = {}
    recent_sent_emails = data_sender[data_sender.date <= mail_date].sort_values(by = 'date', ascending = False)[:number_of_recency_features]
    for row in recent_sent_emails.iterrows():
        recipients = row[1]['recipients'].split()
        for recipient in recipients:
            if '@' in recipient:
                recency_feature[recipient] = recency_feature.get(recipient, 0) + 1
    # Normalization: 
    norm = sum(recency_feature.values())
    for k,v in recency_feature.iteritems():
        recency_feature[k] = float(v)/norm
    
    return recency_feature
    
#####################################################
# Greeting feature : Try to capture the presence of #
# recipients'names in the header of the mail        #
#####################################################
def greeting(text, address):
    header = text[:30].lower()
    names = address[:address.index('@')].split('.')
    for name in names:
        if len(name) >= 3:
            if name in header:
                return True
    return False  
  

                
     