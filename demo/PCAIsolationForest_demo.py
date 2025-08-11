#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
from sklearn.decomposition import PCA
from loglizer.models import IsolationForest
from loglizer import dataloader, preprocessing

import numpy as np

# struct_log = '../data/HDFS/HDFS_100k.log_structured.csv' # The structured log file
# label_file = '../data/HDFS/anomaly_label.csv' # The anomaly label file

#struct_log = '../data/HDFS/HDFS_100k.log_structured.csv' # The structured log file
label_file = '../data/HDFS/anomaly_label.csv' # The anomaly label file

struct_log = '../data/HDFS/HDFS.npz'

event_traces = '../data/HDFS/Event_traces.csv'


#pkl_path = "../../proceeded_data/BGL"


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test), data_df = dataloader.load_HDFS(struct_log,
                                                                event_traces, 
                                                                label_file=label_file,
                                                                window='session', 
                                                                train_ratio=0.5,
                                                                split_type='uniform')
    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf', 
                                              normalization='zero-mean')
    x_test = feature_extractor.transform(x_test)

    # Step 1: Apply PCA
    pca = PCA(n_components=6)  # Adjust based on variance explained
    X_pca = pca.fit_transform(x_train)
    
    X_pcaTest = pca.fit_transform(x_test)
    X_pcaTest_full = np.hstack((x_test, X_pcaTest))
    
    # Step 2: Train Isolation Forest
    iso_forest = IsolationForest(contamination=0.03, random_state=42)
    
    X_pcaTrain_full = np.hstack((x_train, X_pca))
    iso_forest.fit(X_pcaTrain_full)

    print('Train validation:')
    precision, recall, f1 = iso_forest.evaluate(X_pcaTrain_full, y_train)
    
    print('Test validation:')
    precision, recall, f1 = iso_forest.evaluate(X_pcaTest_full, y_test)
