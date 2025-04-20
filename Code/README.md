# ClearCRC

This is the repository of the replication package for paper "Understanding Practitioners’ Expectations on Clear Code Review Comments".


### Table of Contents
- `Code/` contains the code of our study
- `Data/` contains our manually labelled data, the five-fold data for experiments, and extra dataset
- `Others/` contains the sample survey and the paper list we collected


### Code
1. Model Set 1: `lstm_classifier.py` and `ml_classifier.py`
2. Model Set 2: `run_evaluator.py` with `run.sh`
3. Model Set 3: `llm.py` and `llm_compute_metrics.py`
4. `augmentor.py`: the data augmentation code


### Data
- `crc_labelled_data.csv`: the labelled dataset for our main experiments, processed from the [CodeReviewer dataset](https://github.com/microsoft/CodeBERT/blob/master/CodeReviewer/README.md)
- `codereviewer_new_test.csv`: the manually labelled subset of the [CodeReviewer_New dataset](https://sites.google.com/view/chatgptcodereview)
- `five_fold_data`: the up sampling data for five-fold cross validation


### Others
- `Survey Sample.pdf`: the sample survey we used in our study
- `Code Review Paper List`: 47 papers we collected related to CRCs in our study