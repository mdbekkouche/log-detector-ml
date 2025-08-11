#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')

import numpy as np
from loglizer import dataloader, preprocessing, utils
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Files
struct_log = '../data/SPIRIT/Spirit5M.log_structured.csv'
#struct_log = '../data/HDFS/HDFS_100k.log_structured.csv'
#struct_log = '../data/HDFS/hdfs/HDFS.log_structured.csv'
label_file = '../data/HDFS/anomaly_label.csv'
event_traces = '../data/HDFS/Event_traces.csv'

# ----------------------------
# 1. Example LSTM Model
# ----------------------------
class LSTMFeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMFeatureExtractor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    
    def forward(self, x):
        # LSTM outputs (batch, seq_len, hidden_size)
        out, (hn, cn) = self.lstm(x)
        # Use last hidden state as feature vector
        return hn[-1]  # shape: (batch, hidden_size)

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

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(x_train)
X_test_tensor = torch.tensor(x_test)

X_train_tensor = X_train_tensor.double()
X_test_tensor = X_test_tensor.double()

# ----------------------------
# 3. Train LSTM as Feature Extractor
# ----------------------------
input_size = x_train.shape[2]  # 50
hidden_size = 64
num_layers = 1
num_epochs = 10
learning_rate = 0.001

lstm_model = LSTMFeatureExtractor(input_size, hidden_size, num_layers)
lstm_model = lstm_model.double()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)

# Temporary classifier head for LSTM training
classifier_head = nn.Linear(hidden_size, 2)
classifier_head = classifier_head.double()

for epoch in range(num_epochs):
    lstm_model.train()
    optimizer.zero_grad()
    features = lstm_model(X_train_tensor)
    outputs = classifier_head(features)
    loss = criterion(outputs, torch.tensor(y_train))
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# ----------------------------
# 4. Extract Features from LSTM
# ----------------------------
lstm_model.eval()
with torch.no_grad():
    train_features = lstm_model(X_train_tensor).numpy()
    test_features = lstm_model(X_test_tensor).numpy()

# ----------------------------
# 5. Train XGBoost on LSTM Features
# ----------------------------
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)
xgb_model.fit(train_features, y_train)

# ----------------------------
# 6. Evaluate Hybrid Model
# ----------------------------
y_pred = xgb_model.predict(test_features)
print(classification_report(y_test, y_pred))
