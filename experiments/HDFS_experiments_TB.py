#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
import pandas as pd
from loglizer.models import *
from loglizer.models import SVMDecisionTreeVC
from loglizer import dataloader, preprocessing
import matplotlib.pyplot as plt
import numpy as np


run_models = ['PCA', 'InvariantsMiner', 'LogClustering', 'IsolationForest', 'LR', 'SVM', 'DecisionTree', 'DecisionTree+SVM']

#run_models = ['IsolationForest']

#struct_log = '../data/HDFS/HDFS.npz' # The benchmark dataset

#struct_log = '../data/HDFS/HDFS_100k.log_structured.csv'

event_traces = '../data/HDFS/Event_traces.csv'

#struct_log = '../data/HDFS/HDFS_100k.log_structured.csv' # The structured log file

struct_log = '../data/HDFS/hdfs/HDFS.log_structured.csv'

label_file = '../data/HDFS/anomaly_label.csv' # The anomaly label file

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test), data_df = dataloader.load_HDFS(struct_log,
                                                                event_traces,
                                                                label_file=label_file,
                                                                window='session', 
                                                                train_ratio=0.5,
                                                                split_type='sequential',
                                                                TB=True)
    
    first_elements_x_train = np.empty(len(x_train), dtype=object)
    i=0
    while i<len(x_train):
        first_elements_x_train[i] = x_train[i][0]
        i += 1
    first_elements_x_test = np.empty(len(x_test), dtype=object)
    i=0
    while i<len(x_test):
        first_elements_x_test[i] = x_test[i][0]
        i += 1    
    
    feature_extractor = preprocessing.FeatureExtractor()
    first_elements_x_train = feature_extractor.fit_transform(first_elements_x_train, term_weighting='tf-idf')
    first_elements_x_test = feature_extractor.transform(first_elements_x_test)
    
    nfirst_elements_x_train = np.empty(len(x_train), dtype=object)
    nfirst_elements_x_test = np.empty(len(x_test), dtype=object)
    i=0
    while i<len(x_train):
        nfirst_elements_x_train[i] = np.append(first_elements_x_train[i], x_train[i][1])
        #nfirst_elements_x_train[i] = np.append(first_elements_x_train[i], x_train[i][2])
        i += 1
    i=0
    while i<len(x_test):
        nfirst_elements_x_test[i] = np.append(first_elements_x_test[i], x_test[i][1]) 
        #nfirst_elements_x_test[i] = np.append(first_elements_x_test[i], x_test[i][2])
        i += 1
    
    """
    print(type(x_train), type(y_train))
    print(x_train.shape, y_train.shape)
    """
    nfirst_elements_x_train = np.vstack(nfirst_elements_x_train)  # Stack vertically into a single table
    nfirst_elements_x_test = np.vstack(nfirst_elements_x_test)
    
    
    benchmark_results = []
    for _model in run_models:
        print('Evaluating {} on HDFS:'.format(_model))
        if _model == 'PCA':
            x_train = nfirst_elements_x_train
            
            model = PCA()
            model.fit(x_train)
        
        elif _model == 'InvariantsMiner':
            x_train = nfirst_elements_x_train
            
            model = InvariantsMiner(epsilon=0.5)
            model.fit(x_train)

        elif _model == 'LogClustering':
            x_train = nfirst_elements_x_train
            model = LogClustering(max_dist=0.3, anomaly_threshold=0.3)
            model.fit(x_train[y_train == 0, :]) # Use only normal samples for training

        elif _model == 'IsolationForest':
            x_train = nfirst_elements_x_train
            #model = IsolationForest(random_state=2019, max_samples=0.9999, contamination=0.03, n_jobs=4)
            model = IsolationForest(contamination=0.03)
            model.fit(x_train)

        elif _model == 'LR':
            x_train = nfirst_elements_x_train
            model = LR()
            model.fit(x_train, y_train)

        elif _model == 'SVM':
            x_train = nfirst_elements_x_train
            model = SVM()
            model.fit(x_train, y_train)

        elif _model == 'DecisionTree':
            x_train = nfirst_elements_x_train
            model = DecisionTree()
            model.fit(x_train, y_train)
        elif _model == 'DecisionTree+SVM':    
            x_train = nfirst_elements_x_train
            model = SVMDecisionTreeVC.SVMDT()
            model.fit(x_train, y_train)    
        
        x_test = nfirst_elements_x_test
        print('Train accuracy:')
        precision, recall, f1 = model.evaluate(x_train, y_train)
        benchmark_results.append([_model + '-train', precision, recall, f1])
        print('Test accuracy:')
        precision, recall, f1 = model.evaluate(x_test, y_test)
        benchmark_results.append([_model + '-test', precision, recall, f1])

    pd.DataFrame(benchmark_results, columns=['Model', 'Precision', 'Recall', 'F1']) \
      .to_csv('experiment_result_TB.csv', index=False)
    
    
    # Read into DataFrame
    df = pd.read_csv('experiment_result_TB.csv')

    # Plot
    labels = df['Model']
    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(figsize=(14, 6))
    rects1 = ax.bar(x - width, df['Precision'], width, label='Precision')
    rects2 = ax.bar(x, df['Recall'], width, label='Recall')
    rects3 = ax.bar(x + width, df['F1'], width, label='F1 Score')

    # Add labels, title and custom x-axis tick labels
    ax.set_ylabel('Scores')
    ax.set_title('Model Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()

    fig.tight_layout()
    plt.show()
