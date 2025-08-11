#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
from loglizer.models import IsolationForest
from loglizer import dataloader, preprocessing, utils
from sklearn.metrics import precision_score, recall_score, f1_score

import shap

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
                                                                split_type='sequential')
    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train)
    x_test = feature_extractor.transform(x_test)
    
    from sklearn.cluster import KMeans
    #from sklearn.cluster import SpectralClustering
    #from sklearn.cluster import DBSCAN
    
    # Step : KMeans clustering
    n_clusters = 2
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(x_train)
    #from sklearn.cluster import SpectralClustering
    #n_clusters = 5  # or any value you want to test
    #spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', assign_labels='kmeans', random_state=42)
    #clusters = spectral.fit_predict(x_train)
    #from sklearn.cluster import DBSCAN
    #dbscan = DBSCAN(eps=0.5, min_samples=5)  # tune eps, min_samples
    #clusters = dbscan.fit_predict(x_train)
        
    cluster_records = []

    # Step : Anomaly detection per cluster using Isolation Forest
    anomalies = []
    #for cluster_id in range(n_clusters):
     
    n_runs = 1  # Number of repetitions
    cluster_results = {}  # To store metrics per cluster
    
    for cluster_id in set(clusters):
        if cluster_id == -1:
            # DBSCAN noise points are labeled -1
            continue
        # Extract data for current cluster
        cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
        cluster_data = x_train[cluster_indices]
        
        cluster_records.append((cluster_id, cluster_data))  # Save cluster_id for tracking 
        
        print(len(cluster_data))
        
        a = y_train[cluster_indices] 
        import numpy as np
        n_anomalies = np.sum(a == 1)
        contamination_ratio = n_anomalies / len(a)
        
        print(contamination_ratio)
        
        precisions, recalls, f1s = [], [], []
        
        for _ in range(n_runs):
            # Apply Isolation Forest
            model = IsolationForest(contamination=contamination_ratio)
            model.fit(cluster_data)
            c = model.predict(cluster_data)  # 1 = anomaly
            """
            model.partial_fit(cluster_data)

            #cluster_indices = [i for i, c in enumerate(c) if c == 1]

            #print(cluster_indices)
            """

            # Predict on the test set
            y_pred = model.predict(x_test)
            
            precisions.append(precision_score(y_test, y_pred, zero_division=0))
            recalls.append(recall_score(y_test, y_pred, zero_division=0))
            f1s.append(f1_score(y_test, y_pred, zero_division=0))
            
        mean_p, std_p = np.mean(precisions), np.std(precisions, ddof=1)
        mean_r, std_r = np.mean(recalls), np.std(recalls, ddof=1)
        mean_f1, std_f1 = np.mean(f1s), np.std(f1s, ddof=1)

        # Print and store
        print(f"\nCluster {cluster_id} (size: {len(cluster_data)}, anomalies: {n_anomalies})")
        print(f"  Precision: {mean_p:.3f} ± {std_p:.3f}")
        print(f"  Recall:    {mean_r:.3f} ± {std_r:.3f}")
        print(f"  F1-score:  {mean_f1:.3f} ± {std_f1:.3f}")

        cluster_results[cluster_id] = {
            "size": len(cluster_data),
            "n_anomalies": int(n_anomalies),
            "precision": (mean_p, std_p),
            "recall": (mean_r, std_r),
            "f1_score": (mean_f1, std_f1)
        }
        '''
            # Compute evaluation metrics
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Print results
            print(f"Precision: {precision:.3f}")
            print(f"Recall:    {recall:.3f}")
            print(f"F1 Score:  {f1:.3f}")
        '''
    '''    
    # Compute reconstruction error (used as anomaly score)
    X_proj = model.transform(x_test)
    X_reconstructed = model.inverse_transform(X_proj)
    reconstruction_error = np.mean((x_test - X_reconstructed) ** 2, axis=1)

    # Set threshold (e.g., 95th percentile)
    threshold = np.percentile(reconstruction_error, 95)
    y_pred = (reconstruction_error > threshold).astype(int)
    
    # Metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    '''
    '''       
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
