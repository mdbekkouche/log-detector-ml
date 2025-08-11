#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')

import numpy as np
from loglizer import dataloader, preprocessing, utils
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Files
struct_log = '../data/SPIRIT/Spirit5M.log_structured.csv'
#struct_log = '../data/HDFS/HDFS_100k.log_structured.csv'
#struct_log = '../data/HDFS/hdfs/HDFS.log_structured.csv'
label_file = '../data/HDFS/anomaly_label.csv'
event_traces = '../data/HDFS/Event_traces.csv'

def find_best_threshold(y_true, y_prob, metric=f1_score):
    best_thresh = 0.0
    best_score = 0.0
    for t in np.arange(0.001, 1.0, 0.001):
        preds = (y_prob > t).astype(int)
        score = metric(y_true, preds)
        if score > best_score:
            best_score = score
            best_thresh = t
    return best_thresh, best_score

# Hyperparameters
SEQUENCE_LENGTH = 1  # TF-IDF produces flat vectors, not sequences
HIDDEN_DIM = 64
EPOCHS = 10
BATCH_SIZE = 64
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LSTM Model
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (n, _) = self.lstm(x)
        out = self.dropout(n[-1])  # use last hidden state
        out = self.fc(out)
        return self.sigmoid(out)

# ----- Main -----
if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = dataloader.load_SPIRIT(struct_log,
                                                                event_traces,         
                                                                label_file=label_file,
                                                                window='session', 
                                                                train_ratio=0.8,
                                                                split_type='uniform')
    
    '''
    (x_train, y_train), (x_test, y_test), data_df = dataloader.load_HDFS(
        struct_log,
        event_traces,
        label_file=label_file,
        window='session',
        train_ratio=0.8,
        split_type='uniform'
    )
    '''
    
    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf', normalization='zero-mean')
    x_test = feature_extractor.transform(x_test)

    # Add sequence dimension for LSTM: (samples, seq_len=1, features)
    x_train = x_train[:, np.newaxis, :]
    x_test = x_test[:, np.newaxis, :]

    # Convert to PyTorch
    X_train_tensor = torch.tensor(x_train, dtype=torch.float)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float).unsqueeze(1)
    X_test_tensor = torch.tensor(x_test, dtype=torch.float)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    n_runs = 1
    precisions, recalls, f1s = [], [], []

    for run in range(n_runs):
        model = LSTMClassifier(input_dim=x_train.shape[2], hidden_dim=HIDDEN_DIM, num_layers=2).to(DEVICE)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        # Train
        model.train()
        for epoch in range(EPOCHS):
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # Eval
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(DEVICE)
                outputs = model(batch_X).cpu().numpy()
                #####
                #print(len(batch_X))
                #print(len(batch_y))
                #print(len(outputs))
                #print(batch_y)
                best_thresh, best_f1 = find_best_threshold(batch_y, outputs)
                preds = (outputs > best_thresh).astype(int)
                print(f"Best threshold: {best_thresh:.2f}, F1: {best_f1:.4f}")
                
                #preds = (outputs > 0.5).astype(int)
                #####
                all_preds.extend(preds.flatten())
                all_labels.extend(batch_y.numpy().flatten())

        # Metrics
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    mean_p, std_p = np.mean(precisions), np.std(precisions, ddof=1)
    mean_r, std_r = np.mean(recalls), np.std(recalls, ddof=1)
    mean_f1, std_f1 = np.mean(f1s), np.std(f1s, ddof=1)

    print(f"  Precision: {mean_p:.3f} ± {std_p:.3f}")
    print(f"  Recall:    {mean_r:.3f} ± {std_r:.3f}")
    print(f"  F1-score:  {mean_f1:.3f} ± {std_f1:.3f}")
