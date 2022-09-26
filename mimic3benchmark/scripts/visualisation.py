import json
import os
import pickle
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, colors
from matplotlib.colors import LinearSegmentedColormap

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


    plt.plot(ffnn_x, ffnn_y,marker='o', ms = 4, label="FFNN TM")
    plt.plot(logreg_x, logreg_y, marker='o', ms = 4,label="LogReg TM")
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


def get_f1(path='data/root'):
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

def get_precision(path='data/root'):
    precision_dict = {}
    for root, dirs, files in os.walk(path):
        th = root[-3:]
        if th == '0.5':
            for filename in files:
                if re.search("performance", filename):
                    with open(os.path.join(root, filename), 'rb') as f:
                        perf = pickle.load(f)
                        clasrep = perf['Classification Report'].split()
                        precision = clasrep[10]
                        split_size = float(root[-9:-6])
                        precision_dict[split_size] = float(precision)
    lists = sorted(precision_dict.items())
    x, y = zip(*lists)
    return x, y

def get_recall(path='data/root'):
    recall_dict = {}
    for root, dirs, files in os.walk(path):
        th = root[-3:]
        if th == '0.5':
            for filename in files:
                if re.search("performance", filename):
                    with open(os.path.join(root, filename), 'rb') as f:
                        perf = pickle.load(f)
                        clasrep = perf['Classification Report'].split()
                        recall = clasrep[11]
                        split_size = float(root[-9:-6])
                        recall_dict[split_size] = float(recall)
    lists = sorted(recall_dict.items())
    x, y = zip(*lists)
    return x, y

def create_f1score_plot(path):
    x, y = get_f1(path)
    plt.plot(x, y,marker='o', ms = 4, label="tf-idf")
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

    ax2.hist(prob_pos, range=(0, 1), bins=30, label=f"{model_type} {tm_split_size} {threshold}",
             histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plot  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    # plt.show()

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

def find_f1(line):
    list =  re.split('\n', line["Classification Report"])
    row = list[3].split()
    f1 = float(row[3])
    return f1

def plot_precision_recall_calibration():
    with open("data/calibration_metrics/metrics.json", "r") as file:
        metrics = json.load(file)
    logreg_list = []
    logreg_splitsizes = []
    logreg_ths = []
    for key, value in metrics.items():
            if re.search("logreg", key):
                # if key[-3:] == '0.5':
                key = re.sub('logreg_', '', key)
                ss = key.split("_")[0]
                th = key.split("_")[1]
                logreg_splitsizes.append(float(ss))
                logreg_ths.append(float(th))
                logreg_list.append(value)
    logreg_df = pd.DataFrame(list(zip(logreg_splitsizes, logreg_ths, logreg_list)), columns=['ss', 'th', 'calibration'])
    dir_tf_idf = "data/root/tf_idf"
    evals = []

    # NOG OMSCHRIJVEN!!!
    for root, dirs, files in os.walk(dir_tf_idf):
        for parent in dirs:
            for root2, dirs2, _ in os.walk(os.path.join(root, parent)):
                for leaf in dirs2:
                    for root3, dirs, files in os.walk(os.path.join(root2, leaf)):
                        for filename in files:
                            if re.search("performance", filename):
                                with open(os.path.join(root3, filename), 'rb') as f:
                                    eval = (float(parent[-3:]), float(leaf[-3:]), pickle.load(f))
                                evals.append(eval)
    evals = pd.DataFrame(evals, columns=['ss', 'th', 'metrics'])
    evals['precision'] = evals['metrics'].apply(find_precision)
    evals['recall'] = evals['metrics'].apply(find_recall)
    evals['f1'] = evals['metrics'].apply(find_f1)
    evals = evals.drop('metrics', axis=1)
    dir_auroc = 'mimic3models/in_hospital_mortality/logistic/tm/'
    aucs = []
    split_th_list = []
    for root, dirs, files in os.walk(dir_auroc):
        for filename in files:
            if re.search("test", filename) and re.search("C0.001", filename):
                with open(os.path.join(root, filename)) as file:
                    digits = re.findall(r'\d+', filename)  # ['2', '1', '0', '3', '0', '25']
                    split_th_list.append((float(f"0.{digits[4]}"), float(f"0.{digits[6]}")))
                    # test = json.load(file)['auroc']
                    # if test < 0.5:
                    #     print()
                    aucs.append(json.load(file)['auroc'])
    evals['auroc'] = aucs

    merged_df = evals.merge(logreg_df, how="inner", on=['ss', 'th'])
    merged_df['slope'] = [float(x.split(', ')[0]) for x in merged_df['calibration']]
    merged_df['intercept'] = [float(x.split(', ')[1]) for x in merged_df['calibration']]
    merged_df['CITL'] = [float(x.split(', ')[2]) for x in merged_df['calibration']]
    # merged_df = merged_df[merged_df['precision'] != 0]
    # merged_df = merged_df[merged_df['slope'] <2]
    # merged_df = merged_df[merged_df['intercept'] < 2]
    # merged_df = merged_df[merged_df['th'] == 0.5]

    x = merged_df['precision'].values
    # x2 = evals['f1'].values
    y = merged_df['recall'].values
    z = merged_df['slope'].values
    # divnorm = colors.TwoSlopeNorm(vmin=-0.1, vcenter=1, vmax=0.5)
    plt.scatter(x, y, c=z, s=12, cmap="Blues_r")
    plt.colorbar(label='Calibration Slope')
    # plt.xticks(np.arange(0, 1, .05))
    # plt.yticks(np.arange(0, 1.1, .1))
    plt.xticks(rotation=315)
    plt.xlabel('Precision')
    plt.ylabel(r'Recall')
    plt.tight_layout()
    # plt.legend()
    plt.show()

    x = merged_df['th'].values
    # x2 = evals['f1'].values
    y = merged_df['auroc'].values
    # z = merged_df['auroc'].values
    plt.scatter(x, y, s=12)
    # plt.colorbar(label='AUROC')
    plt.xticks(np.arange(0, 1, .1))
    # plt.yticks(np.arange(0, 1.1, .1))
    plt.xticks(rotation=315)
    plt.xlabel('Threshold')
    plt.ylabel(r'AUROC')
    plt.tight_layout()
    # plt.legend()
    plt.show()

    x = merged_df['precision'].values
    # x2 = evals['f1'].values
    y = merged_df['recall'].values
    z = merged_df['CITL'].values
    divnorm = colors.TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.1)
    plt.scatter(x, y, c=z, s=12, cmap="coolwarm", norm=divnorm)
    plt.colorbar(label='CITL')
    # plt.xticks(np.arange(0, 1, .05))
    # plt.yticks(np.arange(0, 1.1, .1))
    plt.xticks(rotation=315)
    plt.xlabel('Precision')
    plt.ylabel(r'Recall')
    plt.tight_layout()
    # plt.legend()
    plt.show()

    x = merged_df['precision'].values
    # x2 = evals['f1'].values
    y = merged_df['recall'].values
    z = merged_df['intercept'].values
    divnorm = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=0.1)
    plt.scatter(x, y, c=z, s=12, cmap="Blues_r")
    plt.colorbar(label='Calibration Intercept')
    # plt.xticks(np.arange(0, 1, .05))
    # plt.yticks(np.arange(0, 1.1, .1))
    plt.xticks(rotation=315)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.tight_layout()
    # plt.legend()
    plt.show()

    x = merged_df['precision'].values
    # x2 = evals['f1'].values
    y = merged_df['recall'].values
    z = merged_df['CITL'].values
    plt.scatter(x, y, c=z, s=12)
    plt.colorbar(label='Calibration in the Large')
    # plt.xticks(np.arange(0, 1, .05))
    # plt.yticks(np.arange(0, 1.1, .1))
    plt.xticks(rotation=315)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.tight_layout()
    # plt.legend()
    plt.show()

    x = merged_df['th'].values
    # x2 = evals['f1'].values
    y = merged_df['f1'].values

    # z = np.polyfit(x, y, 4)
    # p = np.poly1d(z)
    #
    # line = p(x)
    #
    plt.scatter(x, y, s=12)
    # plt.plot(x, line)
    # plt.colorbar(label='Calibration in the Large')
    plt.xticks(np.arange(0, 1, .1))
    # plt.yticks(np.arange(0, 1.1, .1))
    plt.xticks(rotation=315)
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.tight_layout()
    # plt.legend()
    plt.show()

    x = merged_df['th'].values
    # x2 = evals['f1'].values
    y = merged_df['precision'].values

    # z = np.polyfit(x, y, 4)
    # p = np.poly1d(z)
    #
    # line = p(x)
    #
    plt.scatter(x, y, s=12)
    # plt.plot(x, line)
    # plt.colorbar(label='Calibration in the Large')
    plt.xticks(np.arange(0, 1, .1))
    # plt.yticks(np.arange(0, 1.1, .1))
    plt.xticks(rotation=315)
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    plt.legend()
    plt.tight_layout()
    # plt.legend()
    plt.show()

    x = merged_df['th'].values
    # x2 = evals['f1'].values
    y = merged_df['recall'].values
    z = merged_df['precision'].values
    # z = np.polyfit(x, y, 1)
    # p = np.poly1d(z)
    # line = p(x)

    plt.scatter(x, y, s=12)
    # plt.scatter(x, z, s=12, color="red")
    # plt.plot(x,line, "--", label=f"Slope: {np.round(p.coef[0],2)}, Intercept: {np.round(p.coef[1])}")
    # plt.colorbar(label='Calibration in the Large')
    plt.xticks(np.arange(0, 1, .1))
    # plt.yticks(np.arange(0, 1.1, .1))
    plt.xticks(rotation=315)
    plt.xlabel('Threshold')
    plt.ylabel('Recall')
    plt.tight_layout()
    plt.legend()
    # plt.legend()
    plt.show()


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
    evals['f1'] = evals['metrics'].apply(find_f1)
    evals = evals.drop('metrics', axis=1)

    dir_auroc = 'mimic3models/in_hospital_mortality/logistic/tm/'
    aucs = []
    split_th_list = []
    for root, dirs, files in os.walk(dir_auroc):
        for filename in files:
            if re.search("test", filename) and re.search("C0.001", filename):
                with open(os.path.join(root, filename)) as file:
                    digits = re.findall(r'\d+', filename) #['2', '1', '0', '3', '0', '25']
                    split_th_list.append((float(f"0.{digits[4]}"), float(f"0.{digits[6]}")))
                    aucs.append(json.load(file)['auroc'])
    evals['auroc'] = aucs
    # evals

    x = evals['precision'].values
    x2 = evals['f1'].values
    y = evals['recall'].values
    z = evals['auroc'].values
    plt.scatter(x, y, c=z, s=6)
    plt.colorbar(label='AUROC')
    # plt.xticks(np.arange(0, 1, .05))
    # plt.yticks(np.arange(0, 1.1, .1))
    plt.xticks(rotation=315)
    plt.xlabel('Precision')
    plt.ylabel(r'Recall')
    plt.tight_layout()
    # plt.legend()
    plt.show()

    plt.scatter(x2, y, c=z, s=6)
    plt.colorbar(label='Logistic Regression AUROC')
    # plt.xticks(np.arange(0, 1, .05))
    # plt.yticks(np.arange(0, 1.1, .1))
    plt.xticks(rotation=315)
    plt.xlabel('F1 score')
    plt.ylabel(r'Recall')
    plt.tight_layout()
    plt.show()

    plt.scatter(x2, x, c=z, s=6)
    plt.colorbar(label='Logistic Regression AUROC')
    # plt.xticks(np.arange(0, 1, .05))
    # plt.yticks(np.arange(0, 1.1, .1))
    plt.xticks(rotation=315)
    plt.xlabel('F1 score')
    plt.ylabel(r'Precision')
    plt.tight_layout()
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z, depthshade=False, s=6) #cmap='Greens'
    ax.set_xlabel('Precision')
    ax.set_ylabel('Recall')
    ax.set_zlabel('AUROC')
    fig.show()


def plot_calibration_metrics_splitsize():
    with open("data/calibration_metrics/metrics.json", "r") as file:
        metrics = json.load(file)
    FFNN_list = []
    logreg_list = []
    FFNN_splitsizes = []
    logreg_splitsizes = []
    for key, value in metrics.items():
        # if re.search("None", key):
        #     single_metric_plot([(0.0, value)], "Baseline LogReg")
        # else:
            if re.search("FFNN", key):
                if key[-3:] == '0.5':
                    key = re.sub('FFNN_', '', key)
                    key = key.split("_")[0]
                    # if float(value.split(", ")[0]) < 2:
                    FFNN_splitsizes.append(float(key))
                    FFNN_list.append(value)
            if re.search("logreg", key):
                if key[-3:] == '0.5':
                    key = re.sub('logreg_', '', key)
                    key = key.split("_")[0]
                    # if float(value.split(", ")[0]) < 2:
                    logreg_splitsizes.append(float(key))
                    logreg_list.append(value)
    FFNN_sorted = sorted(zip(FFNN_splitsizes, FFNN_list))
    logreg_sorted = sorted(zip(logreg_splitsizes, logreg_list))
    for tuple in [(FFNN_sorted, "FFNN"),
                  (logreg_sorted, "LogReg")]:
        single_metric_plot(*tuple)

def plot_calibration_metrics_f1_score():
    with open("data/calibration_metrics/metrics.json", "r") as file:
        metrics = json.load(file)
    FFNN_list = []
    logreg_list = []
    FFNN_splitsizes = []
    logreg_splitsizes = []

    f1_tuples = get_f1()
    f1_splits = list(f1_tuples[0])
    f1s = list(f1_tuples[1])
    f1s.insert(0,0)
    for key, value in metrics.items():

        # if re.search("None", key):
        #     single_metric_plot([(0.0, value)], "Baseline LogReg")
        # else:
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
                    # if float(value.split(", ")[0]) < 2:
                    logreg_splitsizes.append(float(key))
                    logreg_list.append(value)
    FFNN_sorted = sorted(zip(FFNN_splitsizes, FFNN_list))
    logreg_sorted = sorted(zip(logreg_splitsizes, logreg_list))
    f1_ffnn_sorted = list(zip(f1s, [x[1] for x in FFNN_sorted]))
    f1_logreg_sorted = list(zip(f1s, [x[1] for x in logreg_sorted]))
    for tuple in [(f1_ffnn_sorted, "FFNN"),
                  (f1_logreg_sorted, "LogReg")]:
        single_metric_plot(*tuple, x_label="F1 Score")


def single_metric_plot(metric_tuples, model, x_label="Training data size"):
    slopes = []
    intercepts = []
    citls = []
    for ss, metrics in metric_tuples:
        row = metrics.split(", ")
        slope = row[0]
        intercept = row[1]
        citl = row[2]
        if ss == 0.0:
            plt.axhline(float(intercept), color='orange', linestyle="dashed", label=f"{model} Non-TM Intercept")
            plt.axhline(float(slope), color='blue', linestyle="dashed", label=f"{model} Non-TM Slope")
            plt.axhline(float(citl), color='green', linestyle="dashed", label=f"{model} Non-TM CITL")
        else:
            slopes.append(float(slope))
            intercepts.append(float(intercept))
            citls.append(float(citl))
    for list, name in [(slopes, "Slope"), (intercepts, "Intercept")
        , (citls, "CITL")]:
        plt.plot([x[0] for x in metric_tuples[1:]], list,marker='o', ms = 4, label=f"{name}")
    plt.xticks(np.arange(0, 1, .1))
    # plt.yticks(np.arange(0, 1.1, .1))
    plt.xticks(rotation=315)
        # plt.ylabel(r'F1 score of label $\bf{1}$')
    plt.axhline(0, color='orange', linestyle=":", label="Perfect Intercept, CITL")
    plt.axhline(1, color='b', linestyle=":", label="Perfect Slope")
    plt.xlabel(x_label)
    plt.title(f'Calibration metrics for {model}')
    if model == "LogReg":
        plt.ylim([-1, 1.2])
    else: plt.ylim([-0.6, 1.1])
    plt.legend(loc='center left', fontsize='x-small', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    # plt.legend()
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
    _, ffnn_y = get_auroc(path_ffnn_tm)
    _, logreg_y = get_auroc(path_logreg_tm)
    tm_x, tm_y = get_f1('data/root')

    plt.plot(tm_y, ffnn_y, marker='o', ms = 4, label="FFNN TM")
    plt.plot(tm_y, logreg_y, marker='o',ms = 4, label="LogReg TM")

    logreg_norm = {}
    ffnn_norm = {}
    path_ffnn_norm = os.path.join('mimic3models/in_hospital_mortality/', 'FFNN/norm/results')
    path_logreg_norm = os.path.join('mimic3models/in_hospital_mortality/', 'logistic/norm/results')
    for filename in os.listdir(path_logreg_norm):
        if filename == r"test_all.all.l2.C1.json":
            with open(os.path.join(path_logreg_norm, filename), 'r') as f:
                eval = json.load(f)
                logreg_norm[0] = eval['auroc']
    for filename in os.listdir(path_ffnn_norm):
        if filename[:4] == 'test':
            with open(os.path.join(path_ffnn_norm, filename), 'r') as f:
                eval = json.load(f)
                ffnn_norm[0] = eval['auroc']
    # plt.plot(logreg_norm_x, logreg_norm_y, marker="o")
    plt.axhline(logreg_norm[0], color='r', linestyle=":", label="LogReg Non-TM")
    plt.axhline(ffnn_norm[0], color='b', linestyle=":", label="FFNN Non-TM")
    # plt.xticks(np.arange(0, 1, .05))
    plt.xticks(rotation=315)
    plt.ylabel('AUROC')
    plt.xlabel('F1 score of tf-idf text mining algorithm')
    plt.tight_layout()
    plt.legend()
    plt.show()

def create_precision_auroc_plot():
    path_ffnn_tm = 'FFNN/tm/'
    path_logreg_tm = 'logistic/tm/'
    _, ffnn_y = get_auroc(path_ffnn_tm)
    _, logreg_y = get_auroc(path_logreg_tm)
    tm_x, tm_y = get_precision('data/root')

    plt.scatter(tm_y, ffnn_y, marker='o', label="FFNN TM")
    plt.scatter(tm_y, logreg_y, marker='o', label="LogReg TM")

    logreg_norm = {}
    ffnn_norm = {}
    path_ffnn_norm = os.path.join('mimic3models/in_hospital_mortality/', 'FFNN/norm/results')
    path_logreg_norm = os.path.join('mimic3models/in_hospital_mortality/', 'logistic/norm/results')
    for filename in os.listdir(path_logreg_norm):
        if filename[:4] == 'test':
            with open(os.path.join(path_logreg_norm, filename), 'r') as f:
                eval = json.load(f)
                logreg_norm[0] = eval['auroc']
    for filename in os.listdir(path_ffnn_norm):
        if filename[:4] == 'test':
            with open(os.path.join(path_ffnn_norm, filename), 'r') as f:
                eval = json.load(f)
                ffnn_norm[0] = eval['auroc']
    # plt.plot(logreg_norm_x, logreg_norm_y, marker="o")
    plt.axhline(logreg_norm[0], color='r', linestyle=":", label="LogReg Non-TM")
    plt.axhline(ffnn_norm[0], color='b', linestyle=":", label="FFNN Non-TM")
    # plt.xticks(np.arange(0, 1, .05))
    plt.xticks(rotation=315)
    plt.ylabel('AUROC')
    plt.xlabel('Precision of tf-idf text mining algorithm')
    plt.tight_layout()
    plt.legend()
    plt.show()

def create_recall_auroc_plot():
    path_ffnn_tm = 'FFNN/tm/'
    path_logreg_tm = 'logistic/tm/'
    _, ffnn_y = get_auroc(path_ffnn_tm)
    _, logreg_y = get_auroc(path_logreg_tm)
    tm_x, tm_y = get_recall('data/root')

    plt.plot(tm_y, ffnn_y, marker='o', ms = 4, label="FFNN TM")
    plt.plot(tm_y, logreg_y, marker='o',ms = 4, label="LogReg TM")

    logreg_norm = {}
    ffnn_norm = {}
    path_ffnn_norm = os.path.join('mimic3models/in_hospital_mortality/', 'FFNN/norm/results')
    path_logreg_norm = os.path.join('mimic3models/in_hospital_mortality/', 'logistic/norm/results')
    for filename in os.listdir(path_logreg_norm):
        if filename[:4] == 'test':
            with open(os.path.join(path_logreg_norm, filename), 'r') as f:
                eval = json.load(f)
                logreg_norm[0] = eval['auroc']
    for filename in os.listdir(path_ffnn_norm):
        if filename[:4] == 'test':
            with open(os.path.join(path_ffnn_norm, filename), 'r') as f:
                eval = json.load(f)
                ffnn_norm[0] = eval['auroc']
    # plt.plot(logreg_norm_x, logreg_norm_y, marker="o")
    plt.axhline(logreg_norm[0], color='r', linestyle=":", label="LogReg Non-TM")
    plt.axhline(ffnn_norm[0], color='b', linestyle=":", label="FFNN Non-TM")
    # plt.xticks(np.arange(0, 1, .05))
    plt.xticks(rotation=315)
    plt.ylabel('AUROC')
    plt.xlabel('Recall of tf-idf text mining algorithm')
    plt.tight_layout()
    plt.legend()
    plt.show()


def combine_all_data(model='logreg'):
    with open("data/calibration_metrics/metrics.json", "r") as file:
        metrics = json.load(file)
    metrics_list = []
    splitsizes_list = []
    th_list = []
    for key, value in metrics.items():
        if re.search(model, key):
            # if key[-3:] == '0.5':
            key = re.sub(f'{model}_', '', key)
            ss = key.split("_")[0]
            th = key.split("_")[1]
            splitsizes_list.append(float(ss))
            th_list.append(float(th))
            metrics_list.append(value)
    df = pd.DataFrame(list(zip(splitsizes_list, th_list, metrics_list)), columns=['Split size', 'Decision boundary', 'calibration'])
    dir_tf_idf = "data/root/tf_idf"
    evals = []

    # NOG OMSCHRIJVEN?
    for root, dirs, files in os.walk(dir_tf_idf):
        for parent in dirs:
            for root2, dirs2, _ in os.walk(os.path.join(root, parent)):
                for leaf in dirs2:
                    for root3, dirs, files in os.walk(os.path.join(root2, leaf)):
                        for filename in files:
                            if re.search("performance", filename):
                                with open(os.path.join(root3, filename), 'rb') as f:
                                    eval = (float(parent[-3:]), float(leaf[-3:]), pickle.load(f))
                                evals.append(eval)
    evals = pd.DataFrame(evals, columns=['Split size', 'Decision boundary', 'metrics'])
    if model=='FFNN':
        path_norm = 'mimic3models/in_hospital_mortality/FFNN/norm/results'

    evals['Precision'] = evals['metrics'].apply(find_precision)
    evals['Recall'] = evals['metrics'].apply(find_recall)
    evals['F1 score'] = evals['metrics'].apply(find_f1)
    evals = evals.drop('metrics', axis=1)
    if model=='logreg':
        dir_auroc = 'mimic3models/in_hospital_mortality/logistic/tm/'
        path_norm = 'mimic3models/in_hospital_mortality/logistic/norm/results'
        aucs = []
        split_th_list = []
        for filename in os.listdir(path_norm):
            if filename[:4] == 'test':
                with open(os.path.join(path_norm, filename), 'r') as f:
                    eval = json.load(f)
                    reference_auroc = eval['auroc']
        for root, dirs, files in os.walk(dir_auroc):
            for filename in files:
                if re.search("test", filename) and re.search("C0.001", filename):
                    with open(os.path.join(root, filename)) as file:
                        digits = re.findall(r'\d+', filename)  # ['2', '1', '0', '3', '0', '25']
                        split_th_list.append((float(f"0.{digits[4]}"), float(f"0.{digits[6]}")))
                        # test = json.load(file)['auroc']
                        # if test < 0.5:
                        #     print()
                        aucs.append(json.load(file)['auroc'])

    if model=='FFNN':
        dir_auroc = 'mimic3models/in_hospital_mortality/FFNN/tm/'
        aucs = []
        path_norm = 'mimic3models/in_hospital_mortality/FFNN/norm/results'
        split_th_list = []
        for filename in os.listdir(path_norm):
            if filename[:4] == 'test':
                with open(os.path.join(path_norm, filename), 'r') as f:
                    eval = json.load(f)
                    reference_auroc = eval['auroc']
        for root, dirs, files in os.walk(dir_auroc):
            for filename in files:
                if re.search("test", filename):
                    with open(os.path.join(root, filename)) as file:
                        digits = re.findall(r'\d+', filename)  # ['2', '1', '0', '3', '0', '25']
                        split_th_list.append((float(f"0.{digits[1]}"), float(f"0.{digits[3]}")))
                        # test = json.load(file)['auroc']
                        # if test < 0.5:
                        #     print()
                        aucs.append(json.load(file)['auroc'])
    splits, ths = zip(*split_th_list)
    auc_df = pd.DataFrame()
    auc_df['Split size'] = splits
    auc_df['Decision boundary'] = ths
    auc_df['AUROC'] = aucs
    pm_df = df.merge(auc_df, how="left", on=['Split size', 'Decision boundary'], sort='Split Size')

    merged_df = pm_df.merge(evals, how="left", on=['Split size', 'Decision boundary'])
    merged_df.at[0, 'AUROC'] = reference_auroc
    merged_df['Slope'] = [float(x.split(', ')[0]) for x in merged_df['calibration']]
    merged_df['Intercept'] = [float(x.split(', ')[1]) for x in merged_df['calibration']]
    merged_df['CITL'] = [float(x.split(', ')[2]) for x in merged_df['calibration']]
    merged_df.drop('calibration',axis=1, inplace=True)
    if model == "logreg":
        reference_tuple = (f"Logistic Regression Non-TM", reference_auroc)
    if model == "FFNN":
        reference_tuple = (f"FFNN Non-TM", reference_auroc)
    return(merged_df)


def plot_columns_line(df, x, ys, z=None, lines=[]):
    for y in ys:
        label = y
        if y == "Decision boundary":
            label = "Decision threshold"

        if len(ys) == 1:
            label = None
            plt.ylabel(y, size=14)
        plt.plot(df[x], df[y], marker='o',c=z, ms=4, label=label)
    for tuple in lines:
        label, height = tuple
        plt.axhline(height, label=label, linestyle='--')
    plt.xticks(np.arange(0, 1.1, .2), size=14, rotation=315)
    # plt.yticks(np.arange(0, 1.1, .2), size=14)

    if x == "Decision boundary":
        x = "Decision threshold"
    plt.xlabel(x, size=16)
    # plt.margins(0.1, 0.15)
    plt.tight_layout()
    plt.legend(fontsize=14)
    plt.show()

def plot_columns_scatter(df, x, ys, z=None, lines=[], no_line=False, model=None):
    for y in ys:
        label = y
        if len(ys) == 1:
            label = None
            plt.ylabel(y)
        sub_df = df[df['Decision boundary'] == 0.5]
        my_cmap = LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#0034FF-049585-08FF00-C7030D-FF0700
    (0.000, (0.000, 0.204, 1.000)),
    (0.250, (0.016, 0.584, 0.522)),
    (0.500, (0.031, 1.000, 0.000)),
    (0.750, (0.780, 0.012, 0.051)),
    (1.000, (1.000, 0.027, 0.000))))
        if z:
            if z=='AUROC':
                divnorm = colors.TwoSlopeNorm(vmin=0.80, vcenter=0.85, vmax=0.85001)
                plt.scatter(df[x], df[y], c=df[z].values, cmap=my_cmap, marker='o', s=5, norm=divnorm)
                plt.colorbar(label=z, extend="min")
            if z == 'F1 score':
                divnorm = colors.TwoSlopeNorm(vmin=0.5, vcenter=1, vmax=1.0001)
                plt.scatter(df[x], df[y], c=df[z].values, cmap=my_cmap, marker='o', s=5, norm=divnorm)
                plt.colorbar(label=z)
            # plt.clim(0.80, 0.85)
            if z == 'Slope':
                divnorm = colors.TwoSlopeNorm(vmin=0.1, vcenter=1., vmax=4)
                plt.scatter(df[x], df[y], c=df[z].values, cmap=my_cmap, marker='o', s=5, norm=divnorm)
                plt.colorbar(label=z, extend="both")
            if z == 'Intercept':
                divnorm = colors.TwoSlopeNorm(vmin=-1, vcenter=0., vmax=1)
                plt.scatter(df[x], df[y], c=df[z].values, cmap=my_cmap, marker='o', s=5, norm=divnorm)
                plt.colorbar(label=z, extend="both")
            if z == 'CITL':
                divnorm = colors.TwoSlopeNorm(vmin=-0.3, vcenter=0, vmax=0.1)
                plt.scatter(df[x], df[y], c=df[z].values, cmap=my_cmap, marker='o', s=5, label=label, norm=divnorm)
                plt.colorbar(label=z, extend="both")

            # else:
            #     plt.scatter(df[x], df[y], c=df[z].values, cmap='Spectral', marker='o', s=5, label=label)
            #     # plt.colorbar(label=z)

        else:
            plt.scatter(df[x], df[y], marker='o', s=4, label=label)

        if not no_line:
            plt.plot(sub_df[x], sub_df[y], ms=4, label=f"0.5 decision threshold")



    for tuple in lines:
        label, height = tuple
        plt.axhline(height, label=label, linestyle='--')

    plt.xticks(rotation=315)
    # if ys.length == 1:
    #     plt.ylabel()
    plt.xlabel(x)
    if len(ys) > 1 or len(lines)>0:
        plt.legend(loc='lower left', bbox_to_anchor=(0, 1))
    plt.tight_layout()
    plt.show()

def calibration_plots(df, x, ys, z=None, lines=[], model=None):
    for y in ys:
        label = y
        if len(ys) == 1:
            label = None
            plt.ylabel(y)

        sub_df = df[df['Decision boundary'] == 0.5]
        plt.scatter(df[x], df[y],c=z, marker='o', s=4, label=label)
        plt.plot(sub_df[x], sub_df[y], c=z, ms=4, label=f"0.5 decision threshold")
    i = 0
    colors = ['blue', 'orange', 'green']
    for tuple in lines:
        label, height = tuple
        plt.axhline(height, label=label,color=colors[i], linestyle='--')
        i+=1
    plt.xticks(rotation=315)
    if ys[0] == "Slope" and len(ys)==1 and model=='logreg':
        plt.ylim(0.5,4)
    if ys[0] == "Intercept" and len(ys)==1 and model=='logreg':
        plt.ylim(-1,2)
    if ys[0] == "CITL" and len(ys)==1 and model=='logreg':
        plt.ylim(-0.3,0)
    # if ys.length == 1:
    #     plt.ylabel()
    plt.xlabel(x)
    plt.legend(loc='lower left', fontsize='x-small', bbox_to_anchor=(0, 1.02), ncol=3)
    plt.tight_layout()
    plt.show()

def tm_plots_th(df):
    sub_df = df[df['Split size'] == 0.5]
    plot_columns_line(sub_df, 'Decision boundary', ['Precision', 'Recall', 'F1 score'])

def tm_plots_ss(df):
    sub_df = df[df['Decision boundary'] == 0.5]
    plot_columns_line(sub_df, 'Split size', ['Precision', 'Recall', 'F1 score'])


def tm_plots_curve(df):
    plot_columns_scatter(df, 'Precision', ['Recall'],'F1 score', no_line=True)


def text_mining_plots(df):
    tm_plots_th(df)
    tm_plots_ss(df)
    tm_plots_curve(df)


def tm_pm_plots(df, ref_df, model):
    # AUROC PLOTS
    plot_columns_scatter(df, 'F1 score', ['AUROC'], lines=[("Reference model", ref_df['AUROC'].values[0])])
    sub_df = df[df['Split size'] == 0.5]
    plot_columns_line(sub_df, 'Decision boundary', ['AUROC'])
    plot_columns_scatter(df, 'Precision', ['Recall'],'AUROC', no_line=True)
    plot_columns_scatter(df, 'Precision', ['Recall'], 'Slope', no_line=True, model=model)
    plot_columns_scatter(df, 'Precision', ['Recall'], 'Intercept', no_line=True, model=model)
    plot_columns_scatter(df, 'Precision', ['Recall'], 'CITL', no_line=True, model=model)
    # plot_columns_scatter(df, 'Recall', ['AUROC'], lines=[("Reference model", ref_df['AUROC'].values[0])])

    # Calibration plots

    # calibration_plots(df, 'F1 score', ['Slope', 'Intercept', 'CITL'], lines=[("Reference model Slope", ref_df['Slope'].values[0]),
    #                                                                                     ("Reference model Intercept", ref_df['Intercept'].values[0]),
    #                                                                                     ("Reference model CITL", ref_df['CITL'].values[0])])
    # calibration_plots(df, 'Precision', ['Slope', 'Intercept', 'CITL'], lines=[("Reference model Slope", ref_df['Slope'].values[0]),
    #                                                                                     ("Reference model Intercept", ref_df['Intercept'].values[0]),
    #                                                                                     ("Reference model CITL", ref_df['CITL'].values[0])])
    calibration_plots(df, 'F1 score', ['Slope'],
                      lines=[("Reference model Slope", ref_df['Slope'].values[0])], model=model)

    calibration_plots(df, 'F1 score', ['Intercept'],
                      lines=[("Reference model Slope", ref_df['Intercept'].values[0])], model=model)

    calibration_plots(df, 'F1 score', ['CITL'],
                      lines=[("Reference model Slope", ref_df['CITL'].values[0])], model=model)

    # calibration_plots(df, 'Decision boundary', ['Slope', 'Intercept', 'CITL'],
    #                   lines=[("Reference model Slope", ref_df['Slope'].values[0]),
    #                          ("Reference model Intercept", ref_df['Intercept'].values[0]),
    #                          ("Reference model CITL", ref_df['CITL'].values[0])])
    #
    sub_df = df[df['Split size'] == 0.5]
    plot_columns_line(sub_df, 'Decision boundary', ['Slope'])
    plot_columns_line(sub_df, 'Decision boundary', ['Intercept'])
    plot_columns_line(sub_df, 'Decision boundary', ['CITL'])
    # sub_df = df[df['Split size'] == 0.5]
    plot_columns_line(sub_df, 'Decision boundary', ['F1 score'])


def main():
    # Model can be FFNN or logreg
    font = {'size': 12}

    plt.rc('font', **font)

    model = "logreg"
    df = combine_all_data(model=model)
    tm_df = df[df['Split size'] != 0]
    ref_df = df[df['Split size'] == 0]
    text_mining_plots(tm_df)
    tm_pm_plots(df, ref_df, model)



    # create_AUROC_plots()
    # create_f1score_plot('data/root/tf_idf/')
    # create_f1_auroc_plot()
    # create_precision_auroc_plot()
    # create_recall_auroc_plot()
    # plot_calibration_metrics_splitsize()
    # plot_calibration_metrics_f1_score()
    # plot_precision_recall_auroc()
    # plot_precision_recall_calibration()
    return

if __name__ == '__main__':
    main()


