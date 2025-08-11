#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
from loglizer.models import IsolationForest
from loglizer import dataloader, preprocessing, utils

#import shap

#struct_log = '../data/HDFS/HDFS_100k.log_structured.csv' # The structured log file
label_file = '../data/HDFS/anomaly_label.csv' # The anomaly label file

#struct_log = '../data/HDFS/HDFS.npz'

struct_log = '../data/HDFS/hdfs/HDFS.log_structured.csv'

event_traces = '../data/HDFS/Event_traces.csv'

anomaly_ratio = 0.03 # Estimate the ratio of anomaly samples in the data

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test), data_df = dataloader.load_HDFS(struct_log,
                                                                event_traces, 
                                                                label_file=label_file,
                                                                window='session', 
                                                                train_ratio=0.9,
                                                                split_type='uniform')
    
    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train)
    x_test = feature_extractor.transform(x_test)

    import numpy as np
    n_anomalies = np.sum(y_train == 1)
    contamination_ratio = n_anomalies / len(y_train)
    
    n_runs = 10
    precisions, recalls, f1s = [], [], []
    for _ in range(n_runs):

        model = IsolationForest(contamination=anomaly_ratio)
        model.fit(x_train)

        #print('Train validation:')
        #precision, recall, f1 = model.evaluate(x_train, y_train)

        print('Test validation:')
        precision, recall, f1 = model.evaluate(x_test, y_test)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    mean_p, std_p = np.mean(precisions), np.std(precisions, ddof=1)
    mean_r, std_r = np.mean(recalls), np.std(recalls, ddof=1)
    mean_f1, std_f1 = np.mean(f1s), np.std(f1s, ddof=1)

    # Print and store
    print(f"  Precision: {mean_p:.3f} ± {std_p:.3f}")
    print(f"  Recall:    {mean_r:.3f} ± {std_r:.3f}")
    print(f"  F1-score:  {mean_f1:.3f} ± {std_f1:.3f}")

    #X_background = shap.sample(x_train, 5)
    
    #utils.explain(X_background, model, x_test, y_test)   

    #utils.explainAnomalies(X_background, model, x_test, y_test)
