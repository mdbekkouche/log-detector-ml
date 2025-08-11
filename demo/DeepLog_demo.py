#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
from loglizer import dataloader
from loglizer.models import DeepLog
from loglizer.preprocessing import Vectorizer, Iterator


batch_size = 2048
hidden_size = 64
num_directions = 2
topk = 9
train_ratio = 0.2
window_size = 10
epoches = 50
num_workers = 2
device = 0 

#struct_log = '../data/HDFS/HDFS_100k.log_structured.csv' # The structured log file
label_file = '../data/HDFS/anomaly_label.csv' # The anomaly label file

struct_log = '../data/HDFS/HDFS.npz'

event_traces = '../data/HDFS/Event_traces.csv'

if __name__ == '__main__':
    (x_train, window_y_train, y_train), (x_test, window_y_test, y_test) = dataloader.load_HDFS(struct_log, event_traces, label_file=label_file, window='session', window_size=window_size, train_ratio=train_ratio, split_type='sequential')
    
    feature_extractor = Vectorizer()
    train_dataset = feature_extractor.fit_transform(x_train, window_y_train, y_train)
    test_dataset = feature_extractor.transform(x_test, window_y_test, y_test)
    
    train_loader = Iterator(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers).iter
    test_loader = Iterator(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers).iter

    model = DeepLog.DL(num_labels=feature_extractor.num_labels, hidden_size=hidden_size, num_directions=num_directions, topk=topk, device=device)
    model.fit(train_loader, epoches)
    
    print('Train validation:')
    metrics = model.evaluate(train_loader)

    print('Test validation:')
    metrics = model.evaluate(test_loader)
     