import json
import os
import pickle
import re
import numpy as np
from matplotlib import pyplot as plt


def get_auroc(path ='FFNN/tm/results'):
    filepath = os.path.join('mimic3models/in_hospital_mortality/', path)
    split_auroc_dict = {}
    for filename in os.listdir(filepath):
        with open(os.path.join(filepath, filename), 'r') as f:
            if filename[:4] == 'test':
                eval = json.load(f)
                # test['splitsize']
                no_json = filename[:filename.rindex('.')]
                dot_index = no_json.rindex('.')
                eval['splitsize'] = float(no_json[dot_index - 1:])
                split_auroc_dict[eval['splitsize']] = eval['auroc']
    lists = sorted(split_auroc_dict.items())
    x, y = zip(*lists)
    return x, y


def create_AUROC_plots():
    split_auroc_dict = {}
    path_ffnn_tm = 'FFNN/tm/results'
    path_logreg_tm = 'logistic/tm/results'
    ffnn_x, ffnn_y = get_auroc(path_ffnn_tm)
    logreg_x, logreg_y = get_auroc(path_logreg_tm)

    split_auroc_dict3 = {}
    split_auroc_dict4 = {}
    path_ffnn_norm = os.path.join('mimic3models/in_hospital_mortality/', 'FFNN/norm/results')
    path_logreg_norm = os.path.join('mimic3models/in_hospital_mortality/', 'logistic/norm/results')
    for filename in os.listdir(path_logreg_norm):
        if filename[:4] == 'test':
            with open(os.path.join(path_logreg_norm, filename), 'r') as f:
                eval = json.load(f)
                split_auroc_dict3[0] = eval['auroc']
    for filename in os.listdir(path_ffnn_norm):
        if filename[:4] == 'test':
            with open(os.path.join(path_ffnn_norm, filename), 'r') as f:
                eval = json.load(f)
                split_auroc_dict4[0] = eval['auroc']

    # lists = sorted(split_auroc_dict3.items())

    # logreg_norm_x, logreg_norm_y = zip(*lists)


    plt.plot(ffnn_x, ffnn_y, label="FFNN TM")
    plt.plot(logreg_x, logreg_y, label="LogReg TM")
    # plt.plot(logreg_norm_x, logreg_norm_y, marker="o")
    plt.axhline(split_auroc_dict3[0], color='r', linestyle=":", label="LogReg Non-TM")
    plt.axhline(split_auroc_dict4[0], color='b', linestyle=":", label="FFNN Non-TM")
    plt.xticks(np.arange(0, 1, .05))
    plt.xticks(rotation=315)
    plt.ylabel('AUROC')
    plt.xlabel('Training data size')
    plt.tight_layout()
    plt.legend()
    plt.show()


def get_f1(path):
    f1_dict = {}
    for filename in os.listdir(path):
        if re.search("performance", filename):
            with open(os.path.join(path, filename), 'r') as f:
                perf = pickle.load(open(os.path.join('data/root', filename), "rb"))
                clasrep = perf['Classification Report'].split()
                f1_score = clasrep[12]
                split_size = float("0." + re.split(r'[. _]', filename)[4])
                f1_dict[split_size] = float(f1_score)
    lists = sorted(f1_dict.items())
    x, y = zip(*lists)
    return x, y


def create_f1score_plot(path):
    x, y = get_f1(path)
    plt.plot(x, y, label="tf-idf")
    plt.xticks(np.arange(0, 1, .05))
    plt.yticks(np.arange(0, 1.1, .1))
    plt.xticks(rotation=315)
    plt.ylabel(r'F1 score of label $\bf{1}$')
    plt.xlabel('Training data size')
    plt.tight_layout()
    plt.legend()
    plt.show()

def plot_calibration(calibration, tm_split_size, model_type):
    x, y = calibration
    plt.plot(y, x, label= f"{model_type}, {tm_split_size}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xticks(np.arange(0, 1, .1))
    plt.yticks(np.arange(0, 1.1, .1))
    plt.xticks(rotation=315)
    plt.ylabel(r'Fraction of positives')
    plt.xlabel('Mean predicted probability (class 1)')
    plt.tight_layout()
    plt.legend()
    if not os.path.exists('data/root/calibration_plots'):
        os.mkdir('data/root/calibration_plots')
    plt.savefig(f'data/root/calibration_plots/{model_type}_{tm_split_size}.png')
    plt.show()

def plot_calibration_metrics():
    with open("data/calibration_metrics/metrics.json", "r") as file:
        metrics = json.load(file)
    FFNN_list = []
    logreg_list = []
    FFNN_splitsizes = []
    logreg_splitsizes = []
    for key, value in metrics.items():
        if re.search("FFNN", key):
            FFNN_splitsizes.append(float(re.sub('FFNN_', '', key)))
            FFNN_list.append(value)
        if re.search("logreg", key):
            logreg_splitsizes.append(float(re.sub('logreg_', '', key)))
            logreg_list.append(value)
    for tuple in [(FFNN_list, FFNN_splitsizes, "FFNN"),
                  (logreg_list, logreg_splitsizes, "LogReg")]:
        single_metric_plot(*tuple)


def single_metric_plot(metric_list, splitsize_list, model):
    slopes = []
    intercepts = []
    citls = []
    for row in metric_list:
        row = row.split(", ")
        slope = row[0]
        intercept = row[1]
        citl = row[2]
        slopes.append(float(slope))
        intercepts.append(float(intercept))
        citls.append(float(citl))
    for list, name in [(slopes, "Slope"), (intercepts, "Intercept")
        , (citls, "CITL")]:
        plt.plot(splitsize_list, list, label=f"{name}")
        # plt.xticks(np.arange(0, 1, .1))
        # plt.yticks(np.arange(0, 1.1, .1))
        # plt.xticks(rotation=315)
        # plt.ylabel(r'F1 score of label $\bf{1}$')
    plt.axhline(0, color='orange', linestyle=":", label="Perfect Intercept, CITL")
    plt.axhline(1, color='b', linestyle=":", label="Perfect Slope")
    plt.xlabel('Training data size')
    plt.title(f'Calibration metrics for {model}')
    plt.tight_layout()
    plt.legend()
    plt.show()




def save_calibration_metrics(slope, intercept, CITL, model_type, tm_split_size):
    if not os.path.exists('data/calibration_metrics'):
        os.mkdir('data/calibration_metrics')
    try:
        in_file = open("data/calibration_metrics/metrics.json", "r")
    except: #If file doesn't exist yet
        metrics = {
            f'{model_type}_{tm_split_size}': f"{slope}, {intercept}, {CITL}"
        }
        out_file = open("data/calibration_metrics/metrics.json", "w")
        json.dump(metrics, out_file, indent=2)
        out_file.close()

    else:#If file already exists we update it
        old = json.load(in_file)
        metrics = {
            f'{model_type}_{tm_split_size}':f"{slope}, {intercept}, {CITL}"
        }
        old.update(metrics)
        out_file = open("data/calibration_metrics/metrics.json", "w")
        json.dump(old, out_file, sort_keys=True, indent=2)
        out_file.close()

    # with open('test.txt', 'a') as f:
    #     print('appended text', file=f)
def plot_calibration_slope_intercept_ITL(slope, intercept, CITL):
    pass

def create_f1_auroc_plot():
    path_ffnn_tm = 'FFNN/tm/results'
    path_logreg_tm = 'logistic/tm/results'
    ffnn_x, ffnn_y = get_auroc(path_ffnn_tm)
    logreg_x, logreg_y = get_auroc(path_logreg_tm)
    tm_x, tm_y = get_f1('data/root')
    ffnn_x = tm_y
    logreg_x = tm_y

    plt.plot(ffnn_x, ffnn_y, label="FFNN TM")
    plt.plot(logreg_x, logreg_y, label="LogReg TM")
    # plt.plot(logreg_norm_x, logreg_norm_y, marker="o")
    # plt.axhline(split_auroc_dict3[0], color='r', linestyle=":", label="LogReg Non-TM")
    # plt.axhline(split_auroc_dict4[0], color='b', linestyle=":", label="FFNN Non-TM")
    # plt.xticks(np.arange(0, 1, .05))
    plt.xticks(rotation=315)
    plt.ylabel('AUROC')
    plt.xlabel('F1 score of tf-idf text mining algorithm')
    plt.tight_layout()
    plt.legend()
    plt.show()

def main():
    create_AUROC_plots()
    create_f1score_plot('data/root')
    create_f1_auroc_plot()
    plot_calibration_metrics()

if __name__ == '__main__':
    main()


