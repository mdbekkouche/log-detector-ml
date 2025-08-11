#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as p
import re
sys.path.append('../')
from loglizer.models import SVM
from loglizer import dataloader, preprocessing, utils
import shap
from collections import OrderedDict

struct_log = '../data/HDFS/HDFS_100k.log_structured.csv' # The structured log file

#struct_log = '../data/HDFS/HDFS.npz'

event_traces = '../data/HDFS/Event_traces.csv'

label_file = '../data/HDFS/anomaly_label.csv' # The anomaly label file

if __name__ == '__main__':
    df = p.read_csv(struct_log, engine='c',
                na_filter=False, memory_map=True)
    
    label_data = p.read_csv(label_file, engine='c',
                na_filter=False, memory_map=True)
    
    log_traces = p.read_csv(event_traces)
    
    df['timestamp'] = p.to_datetime(df['Time'].astype(str), format='%H%M%S')
    
    df = df.sort_values(by='timestamp')
    
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
    
    df['Content'] = df['Content'].astype(str)
    
    #df['block_id'] = re.findall(r'(blk_-?\d+)', df['Content'])
    
    #df['block_id'] = df['Content'].apply(lambda x: re.findall(r'(blk_-?\d+)', x))
    
    df['block_id'] = df['Content'].str.extract(r'(blk_-?\d+)', expand=False)
    
    #df['block_id'] = df['block_id'].apply(tuple)
    
    df['session_id'] = df['block_id'].astype('category').cat.codes  # Convert block ID to numeric
    #print(df['session_id'])
    
    df['session_time_diff'] = df.groupby('session_id')['timestamp'].diff().dt.total_seconds().fillna(0)
    
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    #print(df['hour'])
    
    df['rolling_time_mean'] = df['time_diff'].rolling(window=10, min_periods=1).mean()
    
    from sklearn.preprocessing import MinMaxScaler

    print(df[['time_diff', 'session_time_diff', 'rolling_time_mean']])
    
    
    scaler = MinMaxScaler()
    
    df[['time_diff', 'session_time_diff', 'rolling_time_mean']] = scaler.fit_transform(df[['time_diff', 'session_time_diff', 'rolling_time_mean']])
    
    import numpy as np

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    data_dict = OrderedDict()
    data_dict2 = OrderedDict()
    for idx, row in df.iterrows():
        blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])

        blkId_set = set(blkId_list)
        mean=0
        i=0
        for blk_Id in blkId_set:
            if not blk_Id in data_dict:
                data_dict[blk_Id] = []
                data_dict2[blk_Id] = []
            data_dict[blk_Id].append(row['EventId']) # A sequence is all EventIds with the same blk_Id
            mean = mean + row['time_diff']
            i += 1
        data_dict2[blk_Id]=mean/i
    
    data_df = p.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])
     
    df['EventSequence'] = df['block_id'].map(data_dict)
    df['MeanTimeDiffSequence'] = df['block_id'].map(data_dict2)
    
    #print("data_df['EventSequence'].values:",df['EventSequence'].values)
    
    feature_extractor = preprocessing.FeatureExtractor()
    #print(df['EventSequence'][0:num_train])
    
    df_u = df.groupby('block_id').first().reset_index()
    print(df_u)
    
    features = ['MeanTimeDiffSequence']#, 'time_diff', 'session_time_diff', 'rolling_time_mean', 'hour_sin', 'hour_cos']
    X = df_u[features]
    num_train = int(0.5 * X.shape[0])
    
    x_esTrain = feature_extractor.fit_transform(df_u['EventSequence'][0:num_train].values, term_weighting='tf-idf')
    x_esTest = feature_extractor.transform(df_u['EventSequence'][num_train:].values)
    
    
    label_data = label_data.set_index('BlockId')
    label_dict = label_data['Label'].to_dict()
    df_u['Label'] = df_u['block_id'].apply(lambda x: 1 if label_dict[x] == 'Anomaly' else 0)
    
    x_esTrain = p.DataFrame(x_esTrain)
    x_esTest = p.DataFrame(x_esTest)
    
    x_es = p.concat([x_esTrain, x_esTest], axis=0)
    #print('len(x_es):',len(x_es))
    #print('len(x):',len(X))
    #print("x_es shape:", x_es.shape)
    #print("X shape:", X.shape)
    
    X = X.reset_index(drop=True)
    x_es = x_es.reset_index(drop=True)
    X = p.concat([X, x_es], axis=1)
    
    print(X)
    
    num_train = int(0.5 * X.shape[0])
    x_train = X[0:num_train]
    x_test = X[num_train:]
    
    
    print(df_u['Label'])
    num_ones = df_u['Label'].sum()
    print(num_ones)    
    y = df_u['Label']  
    print(len(y))
    
    """
    i=0
    while i<len(df['Label']):
        if df['Label'][i]==1:
            print(df['block_id'][i])
            
        i += 1
    """
    
    
    y_train = y[0:num_train]
    y_test = y[num_train:]
    
    #from sklearn.ensemble import RandomForestClassifier
    from loglizer.models import DecisionTree
    
    #model = RandomForestClassifier()
    
    model = DecisionTree()
    
    model.fit(x_train, y_train)
    #model.fit(x_es, y)
    
    """
    df["Anomaly"] = model.predict(x_es)
    num_ones = df["Anomaly"].sum()
    print(num_ones)
    """
    
    
    print('Train validation:')
    y_pred = model.predict(x_train)
    precision, recall, f1 = utils.metrics(y_pred, y_train)
    print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))

    print('Test validation:')
    y_pred = model.predict(x_test)
    precision, recall, f1 = utils.metrics(y_pred, y_test)
    print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))
    
    """
    from sklearn.metrics import precision_score, recall_score, f1_score

    # Compute Precision, Recall, F1-score
    precision = precision_score(df["Label"], df["Anomaly"])
    recall = recall_score(df["Label"], df["Anomaly"])
    f1 = f1_score(df["Label"], df["Anomaly"])

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    """