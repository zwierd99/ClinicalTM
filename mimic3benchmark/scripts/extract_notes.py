from __future__ import absolute_import
from __future__ import print_function

import argparse
import yaml
import os

from mimic3benchmark.mimic3csv import *
from mimic3benchmark.preprocessing import add_hcup_ccs_2015_groups, make_phenotype_label_matrix
from mimic3benchmark.util import *
# stays = dataframe_from_csv(os.path.join('data/root/test/6', 'stays.csv'))
# print()

def main():
    if not os.path.exists(os.path.join('data/root', 'all_stays_notes.csv')):
        stays = dataframe_from_csv(os.path.join('data/root', 'all_stays.csv'))
        mimic3_path = "data\MIMIC"
        notes = dataframe_from_csv(os.path.join(mimic3_path, 'NOTEEVENTS.csv'))
        print("Total rows of notes: " + str(notes.shape[0]))
        # errors = notes[notes['ISERROR'] == 1]   # Contains 886 samples that are identified by a physician as an error,
                                                # so we remove them.
        notes = notes[notes['ISERROR'] != 1].drop(['ISERROR'], axis=1)


            # df['bar'] = df.bar.map(str) + " is " + df.foo
            # subjects.loc[stays.SUBJECT_ID == subject_id][''] =
        stay_notes_merge = pd.merge(left=stays, right=notes,on='HADM_ID').sort_values(by=['HADM_ID'])
        stay_notes_merge.to_csv(os.path.join('data/root', 'all_stays_notes.csv'), index=False)

    if not os.path.exists(os.path.join('data/root', 'notes.pkl')):
        stays_notes_merge = pd.read_csv(os.path.join('data/root', 'all_stays_notes.csv')
                                        # , nrows=10000
                                        )
        stay_ids = stays_notes_merge['HADM_ID'].unique()
        # note_only_df = pd.DataFrame(columns=np.append(stays_notes_merge.columns[0:20].values,('TEXT')))
        note_df = pd.DataFrame(columns=['SUBJECT_ID', 'ICUSTAY_ID', 'HADM_ID', 'TEXT'])
        for i, stay_id in enumerate(stay_ids):
            notesforstay = stays_notes_merge[stays_notes_merge['HADM_ID'] == stay_id]
            notesconcat = " ".join(notesforstay['TEXT'])
            notesconcat = " ".join(notesconcat.split())
            # newrow = notesforstay.iloc[0,:20].tolist() # Take the first 20 columns of the notesforsubject class (IDs, gender, dob, etc.
            #                                       and copy the first row.
            newrow = notesforstay[['SUBJECT_ID','ICUSTAY_ID', 'HADM_ID']].head(1).values
            newrow = np.append(newrow, notesconcat)        # Append the concatenated notes as text
            note_df.loc[len(note_df)] = newrow
        note_df.sort_values(by=['ICUSTAY_ID'])
        # note_only_df.to_csv(os.path.join('data/root', 'notes.csv'), index=False)
        note_df = note_df.set_index('ICUSTAY_ID')
        note_df.to_pickle(os.path.join('data/root', 'notes.pkl'))

    note_df = pd.read_pickle(os.path.join('data/root', 'notes.pkl')
                                      # , nrows=18109

                               )


    # break_up_stays_by_subject(note_per_stay, 'data/test',verbose=1, sortby='HADM_ID')
    print()
if __name__ == '__main__':
    main()