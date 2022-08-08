ClinicalTM
=========================

This project is built upon the reference model for in-hospital mortality, logistic regression created by *Hrayr Harutyunyan, Hrant Khachatrian, David C. Kale, and Aram Galstyan. Multitask Learning and Benchmarking with Clinical Time Series Data. arXiv:1703.07771* which is now available [on arXiv](https://arxiv.org/abs/1703.07771), using the data gathered in the [MIMIC-III paper](http://www.nature.com/articles/sdata201635).** Note that not all models and files in the original project are used, but as of yet they have not been removed.

In this project, we aim to gain insights into how accurate text mining should be to create a valuable clinical prediction model.

## To run

Ideally, running tf_idf_pipeline.py should perform the actions below. However, I need to check this without preprocessed data, so for my own record keeping I will enumerate the scripts that need to be ran.

To run the code, running the tf_idf_pipeline.py file produces the plots that are used in my thesis. For first time usage, the data/MIMIC directory should be filled with the csv files. Then, the following commands should be ran from the terminal.

       python -m mimic3benchmark.scripts.extract_subjects {PATH TO MIMIC-III CSVs} data/root/
       python -m mimic3benchmark.scripts.validate_events data/root/
       python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root/
       python -m mimic3benchmark.scripts.split_train_and_test data/root/
       python -m mimic3benchmark.scripts.tf_idf
       python -m mimic3benchmark.scripts.create_in_hospital_mortality data/root/ data/in-hospital-mortality/
       python -m mimic3models.split_train_val data/in-hospital-mortality
       python -um mimic3models.in_hospital_mortality.logistic.main --l2 --C 0.001 --output_dir mimic3models/in_hospital_mortality/logistic
       python -um mimic3models.in_hospital_mortality.FFNN.main
