"""
The implementation of LSTM Autoencoder model for anomaly detection.


"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')

import torch
import torch.nn as nn

from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from loglizer import dataloader, preprocessing, utils


#struct_log = '../data/HDFS/HDFS_100k.log_structured.csv' # The structured log file
label_file = '../data/HDFS/anomaly_label.csv' # The anomaly label file

#struct_log = '../data/HDFS/hdfs/HDFS.log_structured.csv'

struct_log = '../data/HDFS/HDFS.npz'

event_traces = '../data/HDFS/Event_traces.csv'

# ----- Autoencoder Architecture -----
'''
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        # Repeat hidden across time steps to feed into decoder
        decoder_input = hidden.repeat(x.size(1), 1, 1).permute(1, 0, 2)
        reconstructed, _ = self.decoder(decoder_input)
        return reconstructed
'''
class StackedLSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=32, num_layers=2, dropout=0.2):
        super(StackedLSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                               num_layers=num_layers,dropout=dropout, batch_first=True)
        self.latent = nn.Linear(hidden_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(input_size=hidden_dim, hidden_size=input_dim,
                               num_layers=num_layers, dropout=dropout, batch_first=True)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        encoded, _ = self.encoder(x)  # shape: (batch, seq_len, hidden_dim)
        last_hidden = encoded[:, -1, :]  # shape: (batch, hidden_dim)
        latent = self.latent(last_hidden)  # shape: (batch, latent_dim)

        # Decode
        decoder_input = self.decoder_input(latent).unsqueeze(1)  # (batch, 1, hidden_dim)
        decoder_input = decoder_input.repeat(1, x.size(1), 1)  # (batch, seq_len, hidden_dim)
        decoded, _ = self.decoder(decoder_input)  # (batch, seq_len, input_dim)

        return decoded

    
def train_lstm_autoencoder(model, dataloader, n_epochs=10, lr=1e-4):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0
        for batch in dataloader:
            inputs = batch[0]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(dataloader):.4f}")
        
    return model

# ----- Anomaly Detection -----
def detect_anomalies(model, sequences, threshold=None, method='percentile'):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(sequences, dtype=torch.float)
        outputs = model(inputs)
        loss = torch.mean((outputs - inputs) ** 2, dim=(1, 2))  # one loss per sequence
    
    if method == 'percentile':
        # Convert loss tensor to NumPy array
        loss_np = loss.detach().cpu().numpy()
        if threshold is None:
            threshold = np.percentile(loss_np, 95)  # top 5% are anomalies
    else: 
        if threshold is None:
            threshold = torch.mean(loss) + 3 * torch.std(loss)  # 3σ rule

    anomalies = (loss > threshold).int()
    return anomalies.numpy(), loss.numpy(), threshold.item()

# ----- Evaluation -----
def evaluate(preds, labels):
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    return precision, recall, f1