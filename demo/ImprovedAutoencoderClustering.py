#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from loglizer import dataloader, preprocessing

# Data paths
struct_log = '../data/HDFS/HDFS_100k.log_structured.csv'
#struct_log = '../data/HDFS/HDFS.npz'
label_file = '../data/HDFS/anomaly_label.csv'
event_traces = '../data/HDFS/Event_traces.csv'

# ----- Deep Autoencoder Class -----
class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(DeepAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16)  # Latent dimension = 16
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

# ----- Train Function -----
def train_autoencoder(model, data, n_epochs=50, batch_size=512, lr=0.001):
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    
    dataset = torch.utils.data.TensorDataset(torch.tensor(data, dtype=torch.float32))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        for batch in loader:
            inputs = batch[0]
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss/len(loader):.6f}")
    return model

# ----- Anomaly Detection -----
def detect_anomalies(model, data, threshold=None):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(data, dtype=torch.float32)
        outputs, _ = model(inputs)
        mse = torch.mean((outputs - inputs) ** 2, dim=1).numpy()

    if threshold is None:
        threshold = np.percentile(mse, 95)  # Top 5% as anomalies

    preds = (mse > threshold).astype(int)
    return preds, mse, threshold

# ----- Evaluation -----
def evaluate(preds, labels):
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# ----- Main Script -----
if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test), data_df = dataloader.load_HDFS(
        struct_log,
        event_traces,
        label_file=label_file,
        window='session',
        train_ratio=0.5,
        split_type='sequential'
    )
    
    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf', normalization='zero-mean')
    x_test = feature_extractor.transform(x_test)

    print("Training data shape:", x_train.shape)
    input_dim = x_train.shape[1]
    model = DeepAutoencoder(input_dim)
    model = train_autoencoder(model, x_train)

    print("Evaluating on training data:")
    preds_train, _, th_train = detect_anomalies(model, x_train)
    evaluate(preds_train, y_train)

    print("Evaluating on test data:")
    preds_test, _, th_test = detect_anomalies(model, x_test, threshold=th_train)
    evaluate(preds_test, y_test)
