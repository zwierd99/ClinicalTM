from __future__ import absolute_import
from __future__ import print_function

import argparse
import datetime
import json
import shutil

import yaml
import os
import re
import pickle
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import time

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier

from mimic3benchmark.mimic3csv import *
from mimic3benchmark.preprocessing import add_hcup_ccs_2015_groups, make_phenotype_label_matrix
from mimic3benchmark.util import *


def create_feature_vector(notes_df, force_create=False):
    if not os.path.exists(os.path.join('data/root', 'tf_idf_vector.pkl')) or force_create:
        starttime = time.time()
        # test = notes_df.head(1000)
        vectorizer = TfidfVectorizer(max_features=2000)
        corpus = notes_df['TEXT'].tolist()
        fvector = vectorizer.fit_transform(corpus)
        # featurenames = vectorizer.get_feature_names()
        endtime = time.time()
        print(f"Time elapsed: {endtime-starttime}")
        vectorizer.stop_words_ = {}
        pickle.dump((vectorizer, fvector), open(os.path.join('data/root', 'tf_idf_vector.pkl'), "wb"))
        # fvector.to_pickle(os.path.join('data/root', 'tf_idf_vector.pkl'))
    else:
        vectorizer, fvector = pickle.load(open(os.path.join('data/root', 'tf_idf_vector.pkl'), "rb"))

    fvector = pd.DataFrame(fvector.toarray())

    fvector['ICUSTAY_ID'] = notes_df.reset_index(inplace=False)['ICUSTAY_ID']
    fvector.set_index('ICUSTAY_ID', inplace=True)
    return vectorizer, fvector

def merge_features_with_labels(fvector,df_notes):
    mortality_df = pd.read_csv(os.path.join('data/root', 'all_stays.csv')).sort_values(by=['ICUSTAY_ID'])
    mortality_df.set_index('ICUSTAY_ID', inplace=True)
    # mortality_only_df = mortality_df['MORTALITY_INHOSPITAL']
    # fvector.reset_index(inplace=True)
    fvector.index = fvector.index.astype('int64')
    df_notes.index = df_notes.index.astype('int64')
    mortality_df2 = pd.merge(left=mortality_df, right=fvector, left_index=True, right_index=True, how='right')
    stay_notes_merge = pd.merge(left=mortality_df2, right=df_notes, left_index=True, right_index=True, on=['SUBJECT_ID', 'HADM_ID'], how='left')
    return stay_notes_merge


def export_train_test_split(list, tm_split_size, flush):
    if not os.path.exists('data/root/data_splits') or flush:
        try:
            shutil.rmtree('data/root/data_splits')
        except FileNotFoundError:
            pass
        os.mkdir('data/root/data_splits')
        namelist = ["X_tm_train", "X_tm_test", "y_tm_train", "y_tm_test", "X_pm_train",
                   "y_pm_train", "X_test", "y_test"]

        for index, df in enumerate(list):
            df.to_csv(os.path.join('data/root/data_splits', f'{namelist[index]}_{tm_split_size}.csv'), index=False)


def copy_train_test_split(df, tm_split_size, flush):
    X = df.iloc[:, 19:2019]
    y = df.iloc[:, 18]
    train_icustay_list = []
    test_icustay_list = []
    if not os.path.exists(os.path.join('data/root', 'train_test_icustay_id.pkl')):
        for partition in ('train', 'test'):

            patients = list(filter(str.isdigit, os.listdir(os.path.join('data/root', partition))))
            for (patient_index, patient) in enumerate(patients):
                patient_folder = os.path.join('data/root', partition, patient)
                patient_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder)))

                for ts_filename in patient_ts_files:
                    with open(os.path.join(patient_folder, ts_filename)) as tsfile:
                        lb_filename = ts_filename.replace("_timeseries", "")
                        label_df = pd.read_csv(os.path.join(patient_folder, lb_filename))
                        if label_df.shape[0] == 0:
                            continue
                        elif partition == 'train':
                            train_icustay_list.append(label_df['Icustay'][0])
                        elif partition == 'test':
                            test_icustay_list.append(label_df['Icustay'][0])
        pickle.dump((train_icustay_list, test_icustay_list), open(os.path.join('data/root', 'train_test_icustay_id.pkl'), "wb"))
    else:
        (train_icustay_list, test_icustay_list) = pickle.load(open(os.path.join('data/root', 'train_test_icustay_id.pkl'), "rb"))
    # df = pd.DataFrame(train_icustay_list, columns=['ICUSTAY_ID'])
    # df2 = X.index.values
    train_icustay_list = pd.merge(pd.DataFrame(train_icustay_list, columns=['ICUSTAY_ID']), X, right_index=True,
                                  left_on='ICUSTAY_ID')
    train_icustay_list = train_icustay_list['ICUSTAY_ID'].tolist()
    test_icustay_list = pd.merge(pd.DataFrame(test_icustay_list, columns=['ICUSTAY_ID']), X, right_index=True,
                                  left_on='ICUSTAY_ID')
    test_icustay_list = test_icustay_list['ICUSTAY_ID'].tolist()
    X_train = X.loc[train_icustay_list, :]
    X_test = X.loc[test_icustay_list, :]
    y_train = y.loc[train_icustay_list]
    y_test = y.loc[test_icustay_list]
    # We only use X_train and X_test for tf_idf development, since test set is reserved for prediction
    # model development. We split the entirety of the data into a text mining part, a prediction model
    # part and a test part. The size of the tm train set is determined by tm_split_size
    X_tm, X_pm_train, y_tm, y_pm_train = train_test_split(X_train, y_train, train_size=0.5, random_state=42)
    X_tm_train, X_tm_test, y_tm_train, y_tm_test = train_test_split(X_tm, y_tm, train_size=0.8,
                                                                    random_state=42)
    X_tm_train, X_tm_train_unused, y_tm_train, y_tm_train_unused = train_test_split(X_tm_train, y_tm_train, train_size=tm_split_size,
                                                                    random_state=42)
    # X_pm_train, X_pm_val, y_pm_train, y_pm_val = train_test_split(X_pm, y_pm, train_size=pm_split_size,
    #                                                               random_state=42)
    export_train_test_split([X_tm_train, X_tm_test, y_tm_train, y_tm_test, X_pm_train,
                   y_pm_train, X_test, y_test], tm_split_size, flush)

    return X_tm_train, X_tm_test, y_tm_train, y_tm_test, X_pm_train, y_pm_train, X_tm_train_unused, y_tm_train_unused


def train_model(X_train, y_train, tm_split_size, flush):
    if not os.path.exists(os.path.join(f'data/root/tf_idf/ss{tm_split_size}', 'tf_idf_model.pkl')) or flush:
        start = time.time()
        now = datetime.datetime.now()
        print(f'Start training the model at {now.hour}:{now.minute}:{now.second}')
        parameters = {'solver': ['lbfgs'], 'alpha': [1e-7, 1e-5, 1e-3], 'hidden_layer_sizes': [(10,2), (5,2), (10,5),]}
        parameters = {'solver': ['lbfgs'], 'alpha': [1e-5],
                      'hidden_layer_sizes': [(10, 2)], 'max_iter': [1000]}
        # clf = GridSearchCV(MLPClassifier(), parameters, verbose=3, n_jobs=5)
        # C_param_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        clf = LogisticRegression(penalty='l2', C=1, random_state=42)
        # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 2), random_state=42)
        clf.fit(X_train, y_train)

        end = time.time()
        print(f'Total model training time {end-start}')
        pickle.dump(clf, open(os.path.join(f'data/root/tf_idf/ss{tm_split_size}', 'tf_idf_model.pkl'), "wb"))
    else:
        clf = pickle.load(open(os.path.join(f'data/root/tf_idf/ss{tm_split_size}', 'tf_idf_model.pkl'), "rb"))
    # start = time.time()
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,2))
    # clf.fit(X_train, y_train)
    #
    # end = time.time()
    # print(f'Total model training time {end - start}')
    return clf


def create_new_labels(df, tm_split_size, threshold_ranges, flush):
    split = copy_train_test_split(df, tm_split_size, flush)
    X_tm_train, X_tm_test, y_tm_train, y_tm_test, X_pm_train, y_pm_train, X_tm_train_unused, y_tm_train_unused = split
    clf = train_model(X_tm_train, y_tm_train, tm_split_size, flush)
    print(f'Split size: {tm_split_size}')
    for th in threshold_ranges:
        y_pm_train_pred_df = pd.DataFrame(clf.predict_proba(X_pm_train)[:, 1])
        y_tm_test_pred_df = pd.DataFrame(clf.predict_proba(X_tm_test)[:, 1])
        y_pm_train_pred_df = y_pm_train_pred_df.applymap(lambda x: 1 if x > th else 0)
        y_tm_test_pred_df = y_tm_test_pred_df.applymap(lambda x: 1 if x > th else 0)
        y_pm_train_pred_df = pd.DataFrame(y_pm_train_pred_df).set_index(X_pm_train.index).sort_index()
        # results = pd.DataFrame(clf.cv_results_)

        perf1 = metrics.confusion_matrix(y_tm_test, y_tm_test_pred_df)
        perf2 = metrics.classification_report(y_tm_test, y_tm_test_pred_df)
        # print(performance)
        # print(metrics.classification_report(y_tm_test, y_tm_test_pred))
        # if not os.path.exists(os.path.join('data/root', f'tf_idf_performance_{tm_split_size}.pkl')) or flush:
        perf_dict = {'Confusion Matrix': perf1,
                    'Classification Report': perf2}
        pickle.dump(perf_dict, open(os.path.join(f'data/root/tf_idf/ss{tm_split_size}/th{th}', f'tf_idf_performance.pkl'), "wb"))
        export_labels(y_pm_train_pred_df, tm_split_size, th, flush)
    return

def export_labels(new_labels, tm_split_size, th, flush):
    # if not os.path.exists(os.path.join(f'data/root/tf_idf/ss{tm_split_size}/th{th}', f'tf_idf_labels.pkl')) or flush:
    pickle.dump(new_labels, open(os.path.join(f'data/root/tf_idf/ss{tm_split_size}/th{th}', f'tf_idf_labels.pkl'), "wb"))
    return

def preprocess_notes(df_notes: pd.DataFrame):
    if not os.path.exists(os.path.join('data/root', 'notes_preprocessed.pkl')):
        stop_words = stopwords.words('english')
        stopwords_dict = Counter(stop_words)
        lemmatizer = WordNetLemmatizer()
        notelist = []
        # test = df_notes.head(100)
        for i, row in enumerate(df_notes['TEXT']):
            # icustay_id, note = df_notes.ix[i, ['ICUSTAY_ID','TEXT']]
            note = row.lower()
            note = re.sub(r'[^a-zA-Z]', ' ', note)
            note = ' '.join([lemmatizer.lemmatize(word) for word in note.split() if word not in stopwords_dict])


            note = " ".join(note.split())
            notelist.append(note)
            print(i)
        df_notes['TEXT'] = notelist
        df_notes.to_pickle(os.path.join('data/root', 'notes_preprocessed.pkl'))
        print('DONE!')
    else:
        df_notes = pd.read_pickle(os.path.join('data/root', 'notes_preprocessed.pkl'))
        print('Processed notes imported from notes_preprocessed.pkl!')
    return df_notes


def run_all(tm_split_size, threshold_ranges = [0.5], flush=False):
    df_notes = pd.read_pickle(os.path.join('data/root', 'notes.pkl'))
    preprocessed_notes = preprocess_notes(df_notes)
    vectorizer, feature_vector = create_feature_vector(preprocessed_notes)
    notes_per_icustay = merge_features_with_labels(feature_vector, preprocessed_notes)
    create_new_labels(notes_per_icustay, tm_split_size, threshold_ranges, flush)
    return


def main():
    run_all(tm_split_size=1, flush=False)


if __name__ == '__main__':
    main()
