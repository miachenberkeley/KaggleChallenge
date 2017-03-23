import random
import operator
import pandas as pd
from collections import Counter
from buildModel import *
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense


path_to_data = "./data/"

##########################
# load some of the files #
##########################

training = pd.read_csv(path_to_data + 'training_set.csv', sep=',', header=0)
training_info = pd.read_csv(path_to_data + 'training_info.csv', sep=',', header=0)
test = pd.read_csv(path_to_data + 'test_set.csv', sep=',', header=0)
test_info = pd.read_csv(path_to_data + 'test_info.csv', sep=',', header=0)


################################
# create some handy structures #
################################

# info_per_mail_id

# convert training set to dictionary
emails_ids_per_sender = {}
sender_per_mail_ids = {}
for index, series in training.iterrows():
    row = series.tolist()
    sender = row[0]
    ids = row[1].split(' ')
    emails_ids_per_sender[sender] = ids
    for mid in ids:
        sender_per_mail_ids[int(mid)] = sender

rcps_per_mail_ids = {}
for index, series in training_info.iterrows():
    row = series.tolist()
    mid = row[0]
    rcps = row[3].split(' ')
    rcps_per_mail_ids[mid] = rcps

# save all unique sender names
all_senders = emails_ids_per_sender.keys()

# create address book with frequency information for each user
address_books = {}
i = 0

for sender, ids in emails_ids_per_sender.iteritems():
    recs_temp = training_info[np.in1d(training_info['mid'], ids)]['recipients'].tolist()
    recs_temp = [rcp.split() for rcp in recs_temp]
    # compute recipient counts and order by frequency
    recs_temp = [elt for sublist in recs_temp for elt in sublist]
    recs_temp = [rec for rec in recs_temp if '@' in rec]
    rec_occ = dict(Counter(recs_temp))
    sorted_rec_occ = sorted(rec_occ.items(), key=operator.itemgetter(1), reverse=True)
    address_books[sender] = sorted_rec_occ

# save all unique recipient names
all_recs = list(set([elt[0] for sublist in address_books.values() for elt in sublist]))

# save all unique user names
all_users = []
all_users.extend(all_senders)
all_users.extend(all_recs)
all_users = list(set(all_users))


body_per_mail_id = get_clean_bodies(training_info, filename="body_per_mail_id.npy")  # Peut-etre a ameliorer en gardant les verbes aussi ?
keywords_per_mail_id = extract_keywords(body_per_mail_id)
vectors_for_words = learn_word_representation(body_per_mail_id.values(), filename="wordsVectors.npy")  # Peut-etre a ameliorer pour splitter sur les points !
vectors_for_mails = build_vectors_for_mails(keywords_per_mail_id, vectors_for_words)
vocabulary = vectors_for_words.keys()
considered_mails_ids = vectors_for_mails.keys()


####################
#   Basic Classes  #
####################

def oneHiddenLayerClf(inp, mid, outp):
    clf = Sequential()
    clf.add(Dense(mid, input_dim=inp, init='uniform', activation='relu'))
    clf.add(Dense(outp, init='uniform', activation='sigmoid'))
    clf.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return clf


####################
#   NN Method One  #
####################

def firstModel(sender, recipients):
    total_recipients = len(recipients)
    clf = oneHiddenLayerClf(100, (total_recipients + 100) / 2, total_recipients)

    considered_mails_ids_for_sender = set(considered_mails_ids).intersection(map(int, emails_ids_per_sender[sender]))
    train_in = [vectors_for_mails[mid] for mid in considered_mails_ids_for_sender]
    train_out = [map(int, np.in1d(recipients, rcps_per_mail_ids[mid]).tolist()) for mid in considered_mails_ids_for_sender]

    return clf, np.asarray(train_in), np.asarray(train_out)


####################
#   NN Method Two  #
####################

def secondModel(sender, recipients):
    considered_mails_ids_for_sender = set(considered_mails_ids).intersection(map(int, emails_ids_per_sender[sender]))
    total_emails_sender = len(considered_mails_ids_for_sender)
    total_recipients = len(recipients)

    clf = oneHiddenLayerClf(total_emails_sender, (total_recipients + total_emails_sender) / 2, total_recipients)

    mail_matrix = np.array([vectors_for_mails[mid] for mid in considered_mails_ids_for_sender])
    train_in = [np.dot(mail_matrix, vectors_for_mails[mid]) for mid in considered_mails_ids_for_sender]
    train_out = [map(int, np.in1d(recipients, rcps_per_mail_ids[mid]).tolist()) for mid in considered_mails_ids_for_sender]

    return clf, np.asarray(train_in), np.asarray(train_out)


#######################
# Fit data with Model #
#######################

def fitModel(model, nb_epoch=20, batch_size=20):
    clf_per_sender = {}
    for i, sender in enumerate(all_senders):
        print i+1, '/', len(all_senders), 'senders'
        recipients = map(lambda x: x[0], address_books[sender])
        total_recipients = len(recipients)

        if total_recipients <= 1:
            continue

        clf, train_in, train_out = model(sender, recipients)

        clf.fit(train_in, train_out, nb_epoch=nb_epoch, batch_size=batch_size)
        clf_per_sender[sender] = clf

    return clf_per_sender


useModel = 0

models = [firstModel, secondModel]

print("Learning with model %d" % useModel)
clf_per_sender = fitModel(models[useModel])

# DEFINE


##############
# Predicting #
##############

print("Predicting for test set")

body_per_mail_id_test = get_clean_bodies(test_info, filename="body_per_mail_id_test.npy")
keywords_per_mail_id_test = extract_keywords(body_per_mail_id_test)
vectors_for_mails_test = build_vectors_for_mails(keywords_per_mail_id_test, vectors_for_words)


# will contain email ids, predictions for random baseline, and predictions for frequency baseline
predictions_per_sender = {}

# number of recipients to predict
k = 10

for index, row in test.iterrows():
    name_ids = row.tolist()
    sender = name_ids[0]
    recipients = map(lambda x: x[0], address_books[sender])
    # IF WE HAVE A PREDICTOR FOR THIS SENDER, WE USE IT
    # OTHERWISE WE WILL USE THE FREQUENCIES
    try:
        clf = clf_per_sender[sender]
    except:
        clf = None
    ids_predict = map(int, name_ids[1].split(' '))
    freq_preds = []
    nn_preds = []
    k_most = [elt[0] for elt in address_books[sender][:k]]
    #
    considered_mails_ids_for_sender = set(considered_mails_ids).intersection(map(int, emails_ids_per_sender[sender]))
    mail_matrix_for_sender = np.array([vectors_for_mails[mid] for mid in considered_mails_ids_for_sender])  # TO SCALE..
    #
    for id_predict in ids_predict:
        if (clf is not None and id_predict in vectors_for_mails_test):
            # IF METHOD NN 1
            k_most_nn = np.argsort(clf.predict(np.atleast_2d(vectors_for_mails_test[id_predict])))[0][:-k-1:-1]
            # IF METHOD NN 2
            # k_most_nn = np.argsort(clf.predict(np.atleast_2d(np.dot(mail_matrix_for_sender, vectors_for_mails_test[id_predict]))))[0][:-k-1:-1]
            k_most_nn = [recipients[i] for i in k_most_nn]
        else:
            k_most_nn = k_most
        nn_preds.append(k_most_nn)
        #
        freq_preds.append(k_most)
    predictions_per_sender[sender] = [ids_predict, freq_preds, nn_preds]


######################
#  TEST IT ON TRAIN  #
######################

print("Predicting for train set")

# number of recipients to predict
k = 10

# FREQUENCY BASED PREDICTIONS
predictions_per_sender_train = {}

for index, row in training.iterrows():
    name_ids = row.tolist()
    sender = name_ids[0]
    recipients = map(lambda x: x[0], address_books[sender])

    try:
        clf = clf_per_sender[sender]
    except:
        clf = None

    ids_predict = map(int, name_ids[1].split(' '))
    freq_preds = []
    nn_preds = []
    k_most = [elt[0] for elt in address_books[sender][:k]]
    #
    considered_mails_ids_for_sender = set(considered_mails_ids).intersection(map(int, emails_ids_per_sender[sender]))
    mail_matrix_for_sender = np.array([vectors_for_mails[mid] for mid in considered_mails_ids_for_sender])  # TO SCALE
    #
    for id_predict in ids_predict:
        if (clf is not None and id_predict in vectors_for_mails):
            k_most_nn = np.argsort(clf.predict(np.atleast_2d(vectors_for_mails[id_predict])))[0][:-k-1:-1]
            # k_most_nn = np.argsort(clf.predict(np.atleast_2d(np.dot(mail_matrix_for_sender, vectors_for_mails_test[id_predict]))))[0][:-k-1:-1]
            k_most_nn = [recipients[i] for i in k_most_nn]
        else:
            k_most_nn = k_most
        nn_preds.append(k_most_nn)
        #
        freq_preds.append(k_most)
    predictions_per_sender_train[sender] = [ids_predict, freq_preds, nn_preds]


## PROBLEM : WE EVALUATE ON ALL TRAIN SET
print("Evaluating on train set")

average_precisions = np.array([0., 0.])
for sender, preds in predictions_per_sender_train.iteritems():
    mids = preds[0]
    for j, predicted_rcps_for_mids in enumerate(preds[1:]):
        for mid, pred_rcps in zip(mids, predicted_rcps_for_mids):
            true_rcps = np.array(rcps_per_mail_ids[mid]).copy().tolist()
            running_correct_count = 0
            running_score = 0
            total_correct = len(true_rcps)
            for i in range(min(k, len(pred_rcps))):
                if pred_rcps[i] in true_rcps:
                    true_rcps.remove(pred_rcps[i])
                    running_correct_count += 1
                    running_score += float(running_correct_count) / (i+1)
            average_precisions[j] += running_score / min(k, total_correct)
average_precisions /= len(rcps_per_mail_ids)

print "MAP@10 FREQ:", average_precisions[0]
print "MAP@10 CLF:", average_precisions[1]


#################################################
# write predictions in proper format for Kaggle #
#################################################

path_to_results = './results/'

with open(path_to_results + 'predictions_NN_1.txt', 'wb') as my_file:
    my_file.write('mid,recipients' + '\n')
    for sender, preds in predictions_per_sender.iteritems():
        ids = preds[0]
        freq_preds = preds[1]
        for index, my_preds in enumerate(freq_preds):
            my_file.write(str(ids[index]) + ',' + ' '.join(my_preds) + '\n')
