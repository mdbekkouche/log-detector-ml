#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')

import numpy as np
from loglizer import dataloader, preprocessing, utils
from loglizer.models import LSTMAutoencoder

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


# ----- Example Usage -----
if __name__ == "__main__":
    # Dummy load of preprocessed HDFS data (replace with real loading)
    # Each row is a log sequence represented as an event count vector
    # data.npy should be shape (n_samples, n_features)
    #data = np.load("data.npy")
    #labels = np.load("labels.npy")  # 0: normal, 1: anomaly
    
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
    
    from torch.utils.data import TensorDataset, DataLoader
    import torch
    
    seq_len = 4

    # Create 3D sequences
    train_seq, y_seq = create_sequences_with_labels(x_train[y_train == 0, :], y_train[y_train == 0], seq_len)
    
    train_seq1, y_seq1 = create_sequences_with_labels(x_train, y_train, seq_len)
    
    test_seq, y_seq_test = create_sequences_with_labels(x_test, y_test, seq_len)
    
    #print(len(y_seq))
    #test_seq = create_sequences(normal_test_data, seq_len)
    
    train_dataset = TensorDataset(torch.tensor(train_seq, dtype=torch.float))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    #test_dataset = torch.tensor(test_seq, dtype=torch.float)  # no DataLoader needed for inference
    
    import numpy as np
    n_runs = 10
    precisions, recalls, f1s = [], [], []
    for _ in range(n_runs):
        input_dim = x_train.shape[1]
        #model = LSTMAutoencoder.LSTMAutoencoder(input_dim=input_dim, hidden_dim=64)
        model = LSTMAutoencoder.StackedLSTMAutoencoder(input_dim=input_dim)
        model = LSTMAutoencoder.train_lstm_autoencoder(model, train_loader, n_epochs=10, lr=1e-3)
        
        '''
        predictions, latents, threshold = LSTMAutoencoder.detect_anomalies(model, train_seq1)
        #print(predictions, latents, threshold)
        #np.set_printoptions(threshold=np.inf)
        #print(predictions)
        #print("----")
        #print(y_seq1)
        LSTMAutoencoder.evaluate(predictions, y_seq1)
        '''

        predictions, latents, threshold = LSTMAutoencoder.detect_anomalies(model, test_seq)
        precision, recall, f1 = LSTMAutoencoder.evaluate(predictions, y_seq_test)
        
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