from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import pickle
import shutil
import pandas as pd
import random
random.seed(49297)

def process_partition(root_path, output_path, tf_idf, partition, tm_split_size=0, threshold=0.5, flush=False, eps=1e-6, n_hours=48):
    if tf_idf:
        mined_mortality = pickle.load(open(os.path.join(f'data/root/tf_idf/ss{tm_split_size}/th{threshold}', f'tf_idf_labels.pkl'), "rb"))
    print(partition)
    output_dir = os.path.join(output_path, partition)
    if flush:
        try:
            shutil.rmtree(output_dir)
        except FileNotFoundError:
            pass
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    xy_pairs = []
    patients = list(filter(str.isdigit, os.listdir(os.path.join(root_path, partition))))
    difference_count=0
    for (patient_index, patient) in enumerate(patients):
        patient_folder = os.path.join(root_path, partition, patient)
        patient_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder)))

        for ts_filename in patient_ts_files:
            with open(os.path.join(patient_folder, ts_filename)) as tsfile:
                lb_filename = ts_filename.replace("_timeseries", "")
                label_df = pd.read_csv(os.path.join(patient_folder, lb_filename))

                # empty label file
                if label_df.shape[0] == 0:
                    continue
                if tf_idf and partition == 'train':

                    icustay_id = label_df['Icustay'][0]
                    try:
                        tm_label = mined_mortality.loc[icustay_id][0]
                        real_label = int(label_df.iloc[0]["Mortality"])
                        if tm_label != real_label:
                            difference_count += 1
                    except:
                        # print("\n\t(No notes for ICU stay, or this sample is not part of the tm training set.)", patient, ts_filename)
                        continue
                    mortality = tm_label
                # elif not tf_idf and partition == 'train':
                #     icustay_id = label_df['Icustay'][0]
                #     try:
                #         check = mined_mortality.loc[icustay_id][0]
                #         mortality = int(label_df.iloc[0]["Mortality"])
                #     except:
                #         continue
                elif partition == 'test' or not tf_idf:
                    mortality = int(label_df.iloc[0]["Mortality"])

                los = 24.0 * label_df.iloc[0]['Length of Stay']  # in hours
                if pd.isnull(los):
                    # print("\n\t(length of stay is missing)", patient, ts_filename)
                    continue

                if los < n_hours - eps:
                    continue

                ts_lines = tsfile.readlines()
                header = ts_lines[0]
                ts_lines = ts_lines[1:]
                event_times = [float(line.split(',')[0]) for line in ts_lines]

                ts_lines = [line for (line, t) in zip(ts_lines, event_times)
                            if -eps < t < n_hours + eps]

                # no measurements in ICU
                if len(ts_lines) == 0:
                    # print("\n\t(no events in ICU) ", patient, ts_filename)
                    continue

                output_ts_filename = patient + "_" + ts_filename
                with open(os.path.join(output_dir, output_ts_filename), "w") as outfile:
                    outfile.write(header)
                    for line in ts_lines:
                        outfile.write(line)

                xy_pairs.append((output_ts_filename, mortality))

        # if (patient_index + 1) % 100 == 0:
            # print("processed {} / {} patients".format(patient_index + 1, len(patients)), end='\r')
    print(f'Number of differences between true and text mined label: {difference_count}')

    print("\n", len(xy_pairs))
    if partition == "train":
        random.shuffle(xy_pairs)
    if partition == "test":
        xy_pairs = sorted(xy_pairs)

    with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
        listfile.write('stay,y_true\n')
        for (x, y) in xy_pairs:
            listfile.write('{},{:d}\n'.format(x, y))


def main():
    parser = argparse.ArgumentParser(description="Create data for in-hospital mortality prediction task.")
    parser.add_argument('--root_path', default='data/root', type=str, help="Path to root folder containing train and test sets.")
    parser.add_argument('--tf_idf', default=False, type=bool, help="Whether tf_idf is enabled")
    parser.add_argument('--tm_split_size', default=0, type=float, help="The amount of data used to train the text mining model (value between 0, 1)")
    args, _ = parser.parse_known_args()
    create_folder(args.root_path, args.tf_idf, args.tm_split_size)
    
def create_folder(threshold, tf_idf, root_path='data/root', flush=False, tm_split_size=0):
    if tf_idf:
        output_path = f'data/in-hospital-mortality-tf_idf-{tm_split_size}/{threshold}'
    else:
        output_path = 'data/in-hospital-mortality/'

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        process_partition(root_path, output_path, tf_idf, "test", tm_split_size=tm_split_size, threshold=threshold, flush=flush)
        process_partition(root_path, output_path, tf_idf, "train", tm_split_size=tm_split_size, threshold=threshold, flush=flush)


if __name__ == '__main__':
    main()
