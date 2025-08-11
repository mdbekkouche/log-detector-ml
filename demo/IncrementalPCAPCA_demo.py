#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
from loglizer.models import PCA
from loglizer import dataloader, preprocessing

#struct_log = '../data/HDFS/HDFS_100k.log_structured.csv' # The structured log file
label_file = '../data/HDFS/anomaly_label.csv' # The anomaly label file

struct_log = '../data/HDFS/hdfs/HDFS.log_structured.csv'

#struct_log = '../data/HDFS/HDFS.npz'

event_traces = '../data/HDFS/Event_traces.csv'

#pkl_path = "../../proceeded_data/BGL"

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test), data_df = dataloader.load_HDFS(struct_log,
                                                                event_traces, 
                                                                label_file=label_file,
                                                                window='session', 
                                                                train_ratio=0.5,
                                                                split_type='sequential')
    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf', 
                                              normalization='zero-mean')
    x_test = feature_extractor.transform(x_test)

    from sklearn.decomposition import IncrementalPCA
    ipca = IncrementalPCA(n_components=10, batch_size=200)
    X_reduced = ipca.fit_transform(x_train)
    
    model = PCA()
    model.fit(X_reduced)

    print('Train validation:')
    precision, recall, f1 = model.evaluate(X_reduced, y_train)
    ipca = IncrementalPCA(n_components=10, batch_size=200)
    X_reduced = ipca.fit_transform(x_test)
    print('Test validation:')
    precision, recall, f1 = model.evaluate(X_reduced, y_test)
