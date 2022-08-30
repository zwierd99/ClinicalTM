import json
import os
import pickle
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mimic3benchmark.scripts.calibration_copy import calibration_curve
from mpl_toolkits import mplot3d


def get_auroc(path ='FFNN/tm/0.5/results'):
    filepath = os.path.join('mimic3models/in_hospital_mortality/', path)
    split_auroc_dict = {}
    for root, dirs, files in os.walk(filepath):
        for filename in files:
            if filename[:4] == 'test':
                th = filename[-8:-5]
                if th == "0.5":
                    with open(os.path.join(root, filename), 'r') as f:
                        eval = json.load(f)
                        # test['splitsize']
                        no_json = filename[:filename.rindex('.')]
                        eval['splitsize'] = float(no_json[-9:-6])
                        split_auroc_dict[eval['splitsize']] = eval['auroc']
    lists = sorted(split_auroc_dict.items())
    x, y = zip(*lists)
    return x, y


def create_AUROC_plots():
    path_ffnn_tm = 'FFNN/tm/'
    path_logreg_tm = 'logistic/tm/'
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


    plt.plot(ffnn_x, ffnn_y,marker='o', label="FFNN TM")
    plt.plot(logreg_x, logreg_y, marker='o',label="LogReg TM")
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
    for root, dirs, files in os.walk(path):
        th = root[-3:]
        if th == '0.5':
            for filename in files:
                if re.search("performance", filename):
                    with open(os.path.join(root, filename), 'rb') as f:
                        perf = pickle.load(f)
                        clasrep = perf['Classification Report'].split()
                        f1_score = clasrep[12]
                        split_size = float(root[-9:-6])
                        f1_dict[split_size] = float(f1_score)
    lists = sorted(f1_dict.items())
    x, y = zip(*lists)
    return x, y


def create_f1score_plot(path):
    x, y = get_f1(path)
    plt.plot(x, y,marker='o', label="tf-idf")
    plt.xticks(np.arange(0, 1, .05))
    plt.yticks(np.arange(0, 1.1, .1))
    plt.xticks(rotation=315)
    plt.ylabel(r'F1 score of label $\bf{1}$')
    plt.xlabel('Training data size')
    plt.tight_layout()
    plt.legend()
    plt.show()

def plot_calibration(y_test, prediction, tm_split_size, threshold, model_type):
    # plt.plot(y, x, marker='o',label= f"{model_type}, SS:{tm_split_size}, TH:{threshold}")
    # plt.plot([0, 1], [0, 1], linestyle='--')
    # plt.xticks(np.arange(0, 1, .1))
    # plt.yticks(np.arange(0, 1.1, .1))
    # plt.xticks(rotation=315)
    # plt.ylabel(r'Fraction of positives')
    # plt.xlabel('Mean predicted probability (class 1)')
    # plt.tight_layout()
    # plt.legend()


    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")


    prob_pos = prediction
    fraction_of_positives, mean_predicted_value = \
        calibration_curve(y_test, prob_pos, n_bins=10)

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s" % (model_type,))

    ax2.hist(prob_pos, range=(0, 1), bins=10, label=model_type,
             histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plot  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    plt.show()

    if not os.path.exists('data/root/calibration_plots'):
        os.mkdir('data/root/calibration_plots')
    plt.savefig(f'data/root/calibration_plots/{model_type}_ss{tm_split_size}_th{threshold}.png')
    plt.show()

def find_precision(line):
    list =  re.split('\n', line["Classification Report"])
    row = list[3].split()
    precision = float(row[1])
    return precision

def find_recall(line):
    list =  re.split('\n', line["Classification Report"])
    row = list[3].split()
    recall = float(row[2])
    return recall

def plot_precision_recall_auroc():
    dir_tf_idf = "data/root/tf_idf"
    evals = []

    #NOG OMSCHRIJVEN!!!
    for root, dirs, files in os.walk(dir_tf_idf):
        for parent in dirs:
            for root2, dirs2, _ in os.walk(os.path.join(root, parent)):
                for leaf in dirs2:
                    for root3, dirs, files in os.walk(os.path.join(root2, leaf)):
                        for filename in files:
                            if re.search("performance", filename):
                                with open(os.path.join(root3, filename), 'rb') as f:
                                    eval = (parent[-3:],leaf[-3:], pickle.load(f))
                                evals.append(eval)
    evals = pd.DataFrame(evals, columns=['ss', 'th', 'metrics'])
    evals['precision'] = evals['metrics'].apply(find_precision)
    evals['recall'] = evals['metrics'].apply(find_recall)
    evals = evals.drop('metrics', axis=1)

    dir_auroc = 'mimic3models/in_hospital_mortality/logistic/tm/'
    aucs = []
    for root, dirs, files in os.walk(dir_auroc):
        for filename in files:
            if re.search("test", filename):
                with open(os.path.join(root, filename)) as file:
                    aucs.append(json.load(file)['auroc'])
    evals['auroc'] = aucs

    x = evals['precision'].values
    y = evals['recall'].values
    z = evals['auroc'].values
    plt.scatter(x, y, c=z)
    plt.colorbar(label='AUROC')
    # plt.xticks(np.arange(0, 1, .05))
    # plt.yticks(np.arange(0, 1.1, .1))
    plt.xticks(rotation=315)
    plt.xlabel('Precision')
    plt.ylabel(r'Recall')
    plt.tight_layout()
    plt.legend()
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z, c=z, cmap='Greens', depthshade=False);
    fig.show()


def plot_calibration_metrics():
    with open("data/calibration_metrics/metrics.json", "r") as file:
        metrics = json.load(file)
    FFNN_list = []
    logreg_list = []
    FFNN_splitsizes = []
    logreg_splitsizes = []
    for key, value in metrics.items():
        if re.search("FFNN", key):
            if key[-3:] == '0.5':
                key = re.sub('FFNN_', '', key)
                key = key.split("_")[0]
                FFNN_splitsizes.append(float(key))
                FFNN_list.append(value)
        if re.search("logreg", key):
            if key[-3:] == '0.5':
                key = re.sub('logreg_', '', key)
                key = key.split("_")[0]
                logreg_splitsizes.append(float(key))
                logreg_list.append(value)
    FFNN_sorted = sorted(zip(FFNN_splitsizes, FFNN_list))
    logreg_sorted = sorted(zip(logreg_splitsizes, logreg_list))
    for tuple in [(FFNN_sorted, "FFNN"),
                  (logreg_sorted, "LogReg")]:
        single_metric_plot(*tuple)


def single_metric_plot(metric_tuples, model):
    slopes = []
    intercepts = []
    citls = []
    for ss, metrics in metric_tuples:
        row = metrics.split(", ")
        slope = row[0]
        intercept = row[1]
        citl = row[2]
        slopes.append(float(slope))
        intercepts.append(float(intercept))
        citls.append(float(citl))
    for list, name in [(slopes, "Slope"), (intercepts, "Intercept")
        , (citls, "CITL")]:
        plt.plot([x[0] for x in metric_tuples], list,marker='o', label=f"{name}")
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




def save_calibration_metrics(slope, intercept, CITL, model_type, tm_split_size, threshold):
    metrics = {
        f'{model_type}_{tm_split_size}_{threshold}': f"{slope}, {intercept}, {CITL}"
    }
    if not os.path.exists('data/calibration_metrics'):
        os.mkdir('data/calibration_metrics')
    try:
        in_file = open("data/calibration_metrics/metrics.json", "r")
    except: #If file doesn't exist yet
        out_file = open("data/calibration_metrics/metrics.json", "w")
        json.dump(metrics, out_file, indent=2)
        out_file.close()

    else:#If file already exists we update it
        old = json.load(in_file)
        old.update(metrics)
        out_file = open("data/calibration_metrics/metrics.json", "w")
        json.dump(old, out_file, sort_keys=True, indent=2)
        out_file.close()

    # with open('test.txt', 'a') as f:
    #     print('appended text', file=f)

def create_f1_auroc_plot():
    path_ffnn_tm = 'FFNN/tm/'
    path_logreg_tm = 'logistic/tm/'
    ffnn_x, ffnn_y = get_auroc(path_ffnn_tm)
    logreg_x, logreg_y = get_auroc(path_logreg_tm)
    tm_x, tm_y = get_f1('data/root')
    ffnn_x = tm_y
    logreg_x = tm_y

    plt.plot(ffnn_x, ffnn_y,marker='o', label="FFNN TM")
    plt.plot(logreg_x, logreg_y, marker='o', label="LogReg TM")
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
    create_f1score_plot('data/root/tf_idf/')
    create_f1_auroc_plot()
    plot_calibration_metrics()
    plot_precision_recall_auroc()
    return

if __name__ == '__main__':
    main()


