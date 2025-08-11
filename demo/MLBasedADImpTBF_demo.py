#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
from loglizer import dataloader, preprocessing, utils
import shap
import numpy as np

struct_log = '../data/HDFS/HDFS_100k.log_structured.csv' # The structured log file
label_file = '../data/HDFS/anomaly_label.csv' # The anomaly label file

#struct_log = '../data/HDFS/hdfs/HDFS.log_structured.csv'

#struct_log = '../data/HDFS/HDFS.npz'

event_traces = '../data/HDFS/Event_traces.csv'

def create_sequences_with_labels(data, labels, seq_len):
    sequences = []
    seq_labels = []
    for i in range(len(data) - seq_len + 1):
        seq = data[i:i+seq_len]
        sequences.append(seq)
        k = i
        while (k < len(labels) and k <= i+seq_len and labels[k]==0):
            k+=1
        if (k > i+seq_len or k == len(labels)):
            seq_labels.append(0)  # label for the whole sequence
        else:
            seq_labels.append(1)  # label for the whole sequence
            
    return np.stack(sequences), np.array(seq_labels)

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
    
    '''
    from imblearn.over_sampling import SMOTE
    smote_enn = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote_enn.fit_resample(nfirst_elements_x_train, y_train)
    '''
    
    #Supervied 
    #from loglizer.models import LR
    #from loglizer.models import DecisionTree
    #from loglizer.models import SVM
    #from loglizer.models import RandomForest
    from loglizer.models import SVMDecisionTreeVC
    
    #Unsupervied
    #from loglizer.models import PCA
    #from loglizer.models import IncrementalPCA
    #from loglizer.models import IsolationForest
    #from loglizer.models import InvariantsMiner
    #from loglizer.models import OneClassSVM
    #import xgboost as xgb # not complete
    #from loglizer.models import AutoencoderClustering
    #from loglizer.models import LSTMAutoencoder
    
    
    import numpy as np
    n_runs = 10
    precisions, recalls, f1s = [], [], []
    for _ in range(n_runs):
        #model = LR()

        #model = DecisionTree()

        #model = SVM()

        model = SVMDecisionTreeVC.SVMDT() # Using Voting Classifier (Simple Ensemble)

        #model = PCA()

        #model = IncrementalPCA.IncPCA(n_components=5)

        #model = IsolationForest(contamination=0.03)

        #model = InvariantsMiner(epsilon=0.5)

        #model = RandomForest.RF()

        #model = OneClassSVM.OCSVM()

        '''   
        input_dim = nfirst_elements_x_train[y_train == 0, :].shape[1]
        model = AutoencoderClustering.DeepAutoencoder(input_dim)
        #model = AutoencoderClustering.train_autoencoder(model, nfirst_elements_x_train[y_train == 0, :])
        model = AutoencoderClustering.train_autoencoder_incremental(model, nfirst_elements_x_train[y_train == 0, :])
        '''   

        '''
        from torch.utils.data import TensorDataset, DataLoader
        import torch

        seq_len = 4

        # Filter only normal sessions
        normal_train_data = nfirst_elements_x_train
        normal_test_data = nfirst_elements_x_test

        # Create 3D sequences
        train_seq, y_seq = create_sequences_with_labels(normal_train_data[y_train == 0, :], y_train[y_train == 0], seq_len)

        train_seq1, y_seq1 = create_sequences_with_labels(normal_train_data, y_train, seq_len)

        test_seq, y_seq_test = create_sequences_with_labels(normal_test_data, y_test, seq_len)

        print(len(y_seq))
        #test_seq = create_sequences(normal_test_data, seq_len)

        train_dataset = TensorDataset(torch.tensor(train_seq, dtype=torch.float))
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        #test_dataset = torch.tensor(test_seq, dtype=torch.float)  # no DataLoader needed for inference

        input_dim = nfirst_elements_x_train.shape[1]
        #model = LSTMAutoencoder.LSTMAutoencoder(input_dim=input_dim, hidden_dim=64)
        model = LSTMAutoencoder.StackedLSTMAutoencoder(input_dim=input_dim)
        model = LSTMAutoencoder.train_lstm_autoencoder(model, train_loader, n_epochs=10, lr=1e-3)
        '''

        model.fit(nfirst_elements_x_train, y_train)

        #model.fit(X_train_resampled, y_train_resampled)
        
        #model.fit(nfirst_elements_x_train)

        '''
        #predictions, mse, latents, threshold = AutoencoderClustering.detect_anomalies(model, nfirst_elements_x_train)
        #print(f"Autoencoder threshold: {threshold:.4f}")

        #final_predictions = AutoencoderClustering.refine_with_clustering(latents, predictions)
        #AutoencoderClustering.evaluate(final_predictions, y_train)
        #####
        predictions1, mse1, latents1, threshold1 = AutoencoderClustering.detect_anomalies(model, nfirst_elements_x_test)
        print(f"Autoencoder threshold: {threshold1:.4f}")

        final_predictions1 = AutoencoderClustering.refine_with_clustering(latents1, predictions1)
        precision, recall, f1 = AutoencoderClustering.evaluate(final_predictions1, y_test)
        '''

        
        #predictions, latents, threshold = LSTMAutoencoder.detect_anomalies(model, train_seq1)
        #print(predictions, latents, threshold)
        #np.set_printoptions(threshold=np.inf)
        #print(predictions)
        #print("----")
        #print(y_seq1)
        #LSTMAutoencoder.evaluate(predictions, y_seq1)

        '''
        predictions, latents, threshold = LSTMAutoencoder.detect_anomalies(model, test_seq)
        precision, recall, f1 = LSTMAutoencoder.evaluate(predictions, y_seq_test)
        '''
        
        
        #print('Train validation:')
        #precision, recall, f1 = model.evaluate(nfirst_elements_x_train, y_train)

        print('Test validation:')
        precision, recall, f1 = model.evaluate(nfirst_elements_x_test, y_test)   
        
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

    
    """
    #y_pred = model.predict(nfirst_elements_x_test)
    y_pred = iso_forest.predict(X_pcaTest_full)
    
    y_pred = np.where(y_pred == -1, 1, 0)  # Convert -1 (anomaly) to 1, normal to 0
       
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    # Compute Precision, Recall, F1-score
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    """
    
    #X_background = shap.sample(x_train, 20)
    
    #y_test = model.predict(x_test) 
    
    #utils.explain(X_background, model, x_test, y_test)  

    #utils.explainAnomalies(X_background, model, x_test, y_test)
    