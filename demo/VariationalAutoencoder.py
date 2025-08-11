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
#struct_log = '../data/HDFS/HDFS_100k.log_structured.csv'
struct_log = '../data/HDFS/HDFS.npz'
label_file = '../data/HDFS/anomaly_label.csv'
event_traces = '../data/HDFS/Event_traces.csv'

# ----- Variational Autoencoder Class -----
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

# ----- VAE Loss -----
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div

# ----- Train Function -----
def train_vae(model, data, n_epochs=50, batch_size=512, lr=0.001):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataset = torch.utils.data.TensorDataset(torch.tensor(data, dtype=torch.float32))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):
        epoch_loss = 0
        for batch in loader:
            x = batch[0]
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = vae_loss(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss / len(loader):.2f}")
    return model

# ----- Anomaly Detection -----
def detect_anomalies(model, data, threshold=None):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(data, dtype=torch.float32)
        recon_x, mu, logvar = model(x)
        mse = torch.mean((recon_x - x) ** 2, dim=1).numpy()

    if threshold is None:
        threshold = np.percentile(mse, 95)

    preds = (mse > threshold).astype(int)
    return preds, mse, threshold

# ----- Evaluation -----
def evaluate(preds, labels):
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# ----- Main -----
if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test), data_df = dataloader.load_HDFS(
        struct_log, event_traces, label_file=label_file,
        window='session', train_ratio=0.5, split_type='sequential'
    )

    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf', normalization='zero-mean')
    x_test = feature_extractor.transform(x_test)

    input_dim = x_train.shape[1]
    model = VAE(input_dim, latent_dim=16)
    model = train_vae(model, x_train)

    print("Evaluating on training set:")
    preds_train, _, threshold = detect_anomalies(model, x_train)
    evaluate(preds_train, y_train)

    print("Evaluating on test set:")
    preds_test, _, _ = detect_anomalies(model, x_test, threshold=threshold)
    evaluate(preds_test, y_test)
