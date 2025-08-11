#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
from loglizer.models import IsolationForest
from river import anomaly
from loglizer import dataloader, preprocessing, utils
from sklearn.metrics import precision_score, recall_score, f1_score

import shap

#struct_log = '../data/HDFS/HDFS_100k.log_structured.csv' # The structured log file
label_file = '../data/HDFS/anomaly_label.csv' # The anomaly label file

#struct_log = '../data/HDFS/HDFS.npz'

struct_log = '../data/HDFS/hdfs/HDFS.log_structured.csv'

event_traces = '../data/HDFS/Event_traces.csv'


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test), data_df = dataloader.load_HDFS(struct_log,
                                                                event_traces, 
                                                                label_file=label_file,
                                                                window='session', 
                                                                train_ratio=0.5,
                                                                split_type='sequential')
    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train)
    x_test = feature_extractor.transform(x_test)
    
    from sklearn.cluster import KMeans
    
    # Step : KMeans clustering
    n_clusters = 2
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(x_train)
    
    # Step : Anomaly detection per cluster using Isolation Forest
    anomalies = []
    from sklearn.cluster import MiniBatchKMeans

    model = MiniBatchKMeans(n_clusters=2, random_state=42)
    for cluster_id in range(n_clusters):
        # Extract data for current cluster
        cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
        cluster_data = x_train[cluster_indices]
        a = y_train[cluster_indices]
        
        print(len(cluster_data))
        
        a = y_train[cluster_indices] 
        import numpy as np
        n_anomalies = np.sum(a == 1)
        contamination_ratio = n_anomalies / len(a)
        
        print(contamination_ratio)
        
        model.partial_fit(cluster_data)
       
        # Predict on the test set
        y_pred = model.predict(x_test)

        # Compute evaluation metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Print results
        print(f"Precision: {precision:.2f}")
        print(f"Recall:    {recall:.2f}")
        print(f"F1 Score:  {f1:.2f}")
        
    # Predict on the test set
    y_pred = model.predict(x_test)
    
    # Compute evaluation metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print results
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F1 Score:  {f1:.2f}")    
    '''   
        #cluster_indices = [i for i, c in enumerate(c) if c == 1]
        
        #print(cluster_indices)
        print('Train validation:')
        precision, recall, f1 = model.evaluate(cluster_data, a)
        
        print('Test validation:')
        precision, recall, f1 = model.evaluate(x_test, y_test)
           
        for i, pred in zip(cluster_indices, y_pred):
            if pred == 1:
                anomalies.append(i)

    # Step : Display results
    print("Detected Anomalous Sessions:")
    for i in anomalies:
        print(f"- Session {i}: {x_train[i]}")

   
    model = IsolationForest(contamination=anomaly_ratio)
    model.fit(x_train)
   
    
    print('Train validation:')
    precision, recall, f1 = model.evaluate(x_train, y_train)
    
    print('Test validation:')
    precision, recall, f1 = model.evaluate(x_test, y_test)
    '''
    #X_background = shap.sample(x_train, 5)
    
    #utils.explain(X_background, model, x_test, y_test)   

    #utils.explainAnomalies(X_background, model, x_test, y_test)
