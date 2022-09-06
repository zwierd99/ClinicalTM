from __future__ import absolute_import
from __future__ import print_function

import pickle

# from sklearn.calibration import calibration_curve
from mimic3benchmark.scripts.calibration_copy import calibration_curve

from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3benchmark.scripts import visualisation
from mimic3benchmark.scripts.CalibrationSlopeIntercept import calibration_slope_intercept_inthelarge
from mimic3models import common_utils
from mimic3models.metrics import print_metrics_binary
from mimic3models.in_hospital_mortality.utils import save_results
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.linear_model import LogisticRegression

import os
import numpy as np
import argparse
import json


def read_and_extract_features(reader, period, features):
    ret = common_utils.read_chunk(reader, reader.get_number_of_examples())
    # ret = common_utils.read_chunk(reader, 100)
    X = common_utils.extract_features_from_rawdata(ret['X'], ret['header'], period, features)
    return (X, ret['y'], ret['name'])




def train_model(C, l2, period, features, tf_idf, tm_split_size=0, threshold=0.5, flush=False):
    if tf_idf:
        data = f'data/in-hospital-mortality-tf_idf-{tm_split_size}/{threshold}/'
        output_dir = f'mimic3models/in_hospital_mortality/logistic/tm/{tm_split_size}'
    else:
        data = f'data/in-hospital-mortality/'
        output_dir = 'mimic3models/in_hospital_mortality/logistic/norm'

    train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(data, 'train'),
                                             listfile=os.path.join(data, 'train_listfile.csv'),
                                             period_length=48.0)

    val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(data, 'train'),
                                           listfile=os.path.join(data, 'val_listfile.csv'),
                                           period_length=48.0)

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(data, 'test'),
                                            listfile=os.path.join(data, 'test_listfile.csv'),
                                            period_length=48.0)

    datatype = 'tm' if tf_idf else 'norm'
    if tf_idf:
        pkl_path = f"data/root/tf_idf/ss{tm_split_size}/th{threshold}"
    else:
        pkl_path = f"data/root/"
    if not os.path.exists(os.path.join(pkl_path, "scaled_data.pkl")) or flush:
        print('Reading data and extracting features ...')
        (train_X, train_y, train_names) = read_and_extract_features(train_reader, period, features)
        (val_X, val_y, val_names) = read_and_extract_features(val_reader, period, features)
        (test_X, test_y, test_names) = read_and_extract_features(test_reader, period, features)
        print('  train data shape = {}'.format(train_X.shape))
        print('  validation data shape = {}'.format(val_X.shape))
        print('  test data shape = {}'.format(test_X.shape))

        print('Imputing missing values ...')
        imputer = Imputer(missing_values=np.nan, strategy='mean', axis=0, verbose=0, copy=True)
        imputer.fit(train_X)
        train_X = np.array(imputer.transform(train_X), dtype=np.float32)
        val_X = np.array(imputer.transform(val_X), dtype=np.float32)
        test_X = np.array(imputer.transform(test_X), dtype=np.float32)

        print('Normalizing the data to have zero mean and unit variance ...')
        scaler = StandardScaler()
        scaler.fit(train_X)
        train_X = scaler.transform(train_X)
        val_X = scaler.transform(val_X)
        test_X = scaler.transform(test_X)

        saved_values = (train_X, train_y, train_names, val_X, val_y, val_names, test_X, test_y, test_names)
        pickle.dump(saved_values, open(os.path.join(pkl_path, "scaled_data.pkl"), "wb"))
    train_X, train_y, train_names, val_X, val_y, val_names, test_X, test_y, test_names = pickle.load(
        open(os.path.join(pkl_path, "scaled_data.pkl"), "rb"))

    penalty = ('l2' if l2 else 'l1')
    if tf_idf:
        file_name = '{}.{}.{}.C{}.ss{}.th{}'.format(period, features, penalty, C, tm_split_size, threshold)
    else:
        file_name = '{}.{}.{}.C{}'.format(period, features, penalty, C)
    train_contains_only_0 = (np.sum(train_y) == 0)
    val_contains_only_0 = (np.sum(val_y) == 0)
    if not train_contains_only_0:
        logreg = LogisticRegression(penalty=penalty, C=C, random_state=42)
        logreg.fit(train_X, train_y)

    result_dir = os.path.join(output_dir, 'results')
    common_utils.create_directory(result_dir)

    if train_contains_only_0:
        prediction = np.zeros(test_X.shape[0])
    else:
        prediction = logreg.predict_proba(train_X)
        # visualisation.plot_calibration(train_y, prediction[:,1], tm_split_size, threshold, "Logistic Regression training")
        # visualisation.save_calibration_metrics(*calibration_slope_intercept_inthelarge(prediction[:, 1], train_y), "logreg",
        #                                        0, 0.5)
        with open(os.path.join(result_dir, 'train_{}.json'.format(file_name)), 'w') as res_file:
            ret = print_metrics_binary(train_y, prediction, verbose=False)
            ret = {k: float(v) for k, v in ret.items()}
            json.dump(ret, res_file)
        if not val_contains_only_0:
            prediction = logreg.predict_proba(val_X)
            with open(os.path.join(result_dir, 'val_{}.json'.format(file_name)), 'w') as res_file:
                ret = print_metrics_binary(val_y, prediction, verbose=False)
                ret = {k: float(v) for k, v in ret.items()}
                json.dump(ret, res_file)

        prediction = logreg.predict_proba(test_X)[:, 1]
    with open(os.path.join(result_dir, 'test_{}.json'.format(file_name)), 'w') as res_file:
        ret = print_metrics_binary(test_y, prediction)
        ret = {k: float(v) for k, v in ret.items()}
        json.dump(ret, res_file)

        # calibration = calibration_curve(test_y, prediction, n_bins=10)
    visualisation.plot_calibration(test_y, prediction, tm_split_size, threshold, "Logistic Regression")
    visualisation.save_calibration_metrics(*calibration_slope_intercept_inthelarge(prediction, test_y), "logreg",
                                               tm_split_size, threshold)

    save_results(test_names, prediction, test_y,
                 os.path.join(output_dir, 'predictions',
                              file_name + '.csv'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--C', type=float, default=1.0, help='inverse of L1 / L2 regularization')
    parser.add_argument('--l1', dest='l2', action='store_false')
    parser.add_argument('--l2', dest='l2', action='store_true')
    parser.set_defaults(l2=True)
    parser.add_argument('--period', type=str, default='all', help='specifies which period extract features from',
                        choices=['first4days', 'first8days', 'last12hours', 'first25percent', 'first50percent', 'all'])
    parser.add_argument('--features', type=str, default='all', help='specifies what features to extract',
                        choices=['all', 'len', 'all_but_len'])
    # parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
    #                     default=os.path.join(os.path.dirname(__file__), '../../../data/in-hospital-mortality/'))
    # parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
    #                     default='.')
    parser.add_argument('--tf_idf', default=False, type=bool, help="Whether tf_idf is enabled")
    parser.add_argument('--tm_split_size', default=0, type=float,
                        help="The amount of data used to train the text mining model (value between 0, 1)")
    args = parser.parse_args()
    print(args)
    train_model(args.C, args.l2, args.period, args.features, args.tf_idf, args.tm_split_size)




if __name__ == '__main__':
    main()
