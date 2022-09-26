import json
import os
import time

import numpy as np

import extract_subjects
import validate_events
import extract_episodes_from_subjects
import extract_notes
import split_train_and_test
import create_in_hospital_mortality
from mimic3benchmark.scripts import tf_idf, visualisation
from mimic3models import split_train_val
from mimic3models.in_hospital_mortality.FFNN import main as ffnn
from mimic3models.in_hospital_mortality.logistic import main as logreg
import matplotlib.pylab as plt


def tf_idf_pipeline():
    # splits = np.arange(55, 100, 5)/1000
    splits = np.arange(50, 1000, 50)/1000
    # splits = [0.25]
    for split_size in splits:
        print(f'\n\n\nRunning model for split: {split_size}')
        # if not os.path.exists(os.path.join('data/root', 'tf_idf_labels.pkl')):
        print('Creating tf_idf model and outcome vector')
        tf_idf.run_all(split_size, flush=False)
        # if not os.path.exists('data/in-hospital-mortality-tf_idf'):
        print('Creating in-hospital mortality dataset for tf_idf')
        create_in_hospital_mortality.create_folder(tf_idf=True,tm_split_size=split_size, flush=False)
        split_train_val.split(f'data/in-hospital-mortality-tf_idf-{split_size}')
        print('Training prediction models using tf_idf data')
        train_models(tm_split_size=split_size, tf_idf_bool=True)

def regular_data_pipeline():
    if not os.path.exists('data/in-hospital-mortality'):
        create_in_hospital_mortality.create_folder(tf_idf=False, threshold=0.5)
        split_train_val.split('data/in-hospital-mortality')
    train_models(tf_idf_bool=False)

def setup_data():
    if not os.path.exists('data/root'):
        extract_subjects.main()
        validate_events.main()

        extract_episodes_from_subjects.main()
        split_train_and_test.main()
    if not os.path.exists(os.path.join('data/root', 'notes.pkl')):
        extract_notes.main()

def tf_idf_threshold_pipeline():
    # splits = np.arange(50, 150, 10)/1000
    # splits = np.arange(50, 150, 10) / 1000
    splits = np.arange(50, 1000, 50)/1000
    # splits = np.arange(100, 1000, 100) / 1000
    # splits = [0.25]
    threshold_ranges = np.arange(100, 1000, 100)/1000
    # threshold_ranges = [0.3]
    for split_size in splits:
        print(f'\n\n\nRunning model for split: {split_size}')
        # if not os.path.exists(os.path.join('data/root', 'tf_idf_labels.pkl')):
        for th in threshold_ranges:
            if not os.path.exists(f'data/root/tf_idf/ss{split_size}/th{th}'):
                os.makedirs(f'data/root/tf_idf/ss{split_size}/th{th}')
        print('Creating tf_idf model and outcome vector')
        tf_idf.run_all(split_size, threshold_ranges, flush=False)

        for threshold in threshold_ranges:
            # if not os.path.exists('data/in-hospital-mortality-tf_idf'):
            print(f"Split size: {split_size}, Decision boundary: {threshold}")
            print('Creating in-hospital mortality dataset for tf_idf')
            create_in_hospital_mortality.create_folder(threshold, tf_idf=True,tm_split_size=split_size, flush=False)
            split_train_val.split(f'data/in-hospital-mortality-tf_idf-{split_size}/{threshold}')
            print('Training prediction models using tf_idf data')
            train_models(tm_split_size=split_size, tf_idf_bool=True, threshold=threshold)



def train_models(tf_idf_bool, tm_split_size=0.0, threshold= 0.5):
    print('Training logistic regression prediction model')
    logreg.train_model(C=0.001, l2=True, period='all', features='all', tf_idf=tf_idf_bool, flush=False, tm_split_size= tm_split_size, threshold=threshold)
    print('Training feed forward neural network prediction model')
    ffnn.train_model(tf_idf=tf_idf_bool, period='all', features='all', tm_split_size=tm_split_size, flush=False, threshold=threshold)



def main():
    start = time.time()
    # setup_data()
    regular_data_pipeline()
    # threshold_plot()
    tf_idf_threshold_pipeline()
    visualisation.main()
    end = time.time()
    # print('\007')
    print(f'Total time taken: {end-start}')


def threshold_plot():
    # splits = np.arange(100, 1000, 100) / 1000
    splits = np.arange(50, 1000, 100) / 1000
    # splits = [0.5]
    recall_range = np.arange(100, 1000, 100) / 1000
    threshold_dict = tf_idf.determine_thresholds(splits, recall_range)
    for split_size in splits:
        for threshold in threshold_dict[split_size]:
            # if not os.path.exists('data/in-hospital-mortality-tf_idf'):
            print(f"Split size: {split_size}, Decision boundary: {threshold}")
            print('Creating in-hospital mortality dataset for tf_idf')
            create_in_hospital_mortality.create_folder(threshold, tf_idf=True, tm_split_size=split_size, flush=False)
            split_train_val.split(f'data/in-hospital-mortality-tf_idf-{split_size}/{threshold}')
            print('Training prediction models using tf_idf data')
            train_models(tm_split_size=split_size, tf_idf_bool=True, threshold=threshold)

if '__main__' == __name__:
    main()

