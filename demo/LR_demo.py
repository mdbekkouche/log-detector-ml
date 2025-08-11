#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
from loglizer.models import LR
from loglizer import dataloader, preprocessing, utils

import shap

#struct_log = '../data/HDFS/HDFS_100k.log_structured.csv' # The structured log file

struct_log = '../data/HDFS/hdfs/HDFS.log_structured.csv'

label_file = '../data/HDFS/anomaly_label.csv' # The anomaly label file

#struct_log = '../data/HDFS/HDFS.npz'

event_traces = '../data/HDFS/Event_traces.csv'

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test), data_df = dataloader.load_HDFS(struct_log,
                                                                event_traces,
                                                                label_file=label_file,
                                                                window='session', 
                                                                train_ratio=0.5,
                                                                split_type='sequential')
    
    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
    x_test = feature_extractor.transform(x_test)
    
    model = LR()
    model.fit(x_train, y_train)

    print('Train validation:')
    precision, recall, f1 = model.evaluate(x_train, y_train)

    print('Test validation:')
    precision, recall, f1 = model.evaluate(x_test, y_test)
    
    #y_test = model.predict(x_test)
    
    #X_background = shap.sample(x_train,20)
    
    #utils.explain(X_background, model, x_test, y_test)   

    #utils.explainAnomalies(X_background, model, x_test, y_test)