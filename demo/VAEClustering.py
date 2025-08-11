#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.cluster import DBSCAN
import torch
import torch.nn as nn
import torch.optim as optim
from loglizer import dataloader, preprocessing

# VAE model definition
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def detect_anomalies(model, data, threshold=None):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(data, dtype=torch.float32)
        recon_x, mu, logvar = model(x)
        mse = torch.mean((recon_x - x) ** 2, dim=1).numpy()
        latents = mu.numpy()

    if threshold is None:
        threshold = np.percentile(mse, 95)

    preds = (mse > threshold).astype(int)
    return preds, mse, threshold, latents

def refine_with_clustering(latents, predictions, eps=0.5, min_samples=5):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(latents)
    cluster_labels = clustering.labels_
    final_preds = predictions.copy()

    for cluster_id in set(cluster_labels):
        if cluster_id == -1:
            continue
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_preds = predictions[cluster_indices]
        if np.mean(cluster_preds) > 0.8:
            final_preds[cluster_indices] = 1
        else:
            final_preds[cluster_indices] = 0
    return final_preds
'''
# ----- Clustering Refinement -----
def refine_with_clustering(latents, predictions):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    latents_scaled = scaler.fit_transform(latents)
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=4)
    latents_reduced = pca.fit_transform(latents_scaled)
    
    from sklearn.cluster import MiniBatchKMeans
    clustering = MiniBatchKMeans(n_clusters=10, batch_size=1000).fit(latents_reduced)
    #clustering = DBSCAN(eps=0.5, min_samples=5).fit(latents_reduced)
    labels = clustering.labels_

    # Mark points labeled -1 (outliers) as anomalies
    clustered_preds = np.where(labels == -1, 1, 0)

    
    from sklearn.metrics import pairwise_distances_argmin_min

    cluster_ids, distances = pairwise_distances_argmin_min(latents_reduced, clustering.cluster_centers_)
    threshold = np.percentile(distances, 95)  # Top 5% farthest points = anomalies
    clustered_preds = (distances > threshold).astype(int)
    
    # Combine reconstruction error and clustering
    final_preds = np.logical_or(predictions, clustered_preds).astype(int)
    return final_preds
'''

def evaluate(y_pred, y_true):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

if __name__ == '__main__':
    struct_log = '../data/HDFS/HDFS_100k.log_structured.csv'
    #struct_log = '../data/HDFS/HDFS.npz'
    event_traces = '../data/HDFS/Event_traces.csv'
    label_file = '../data/HDFS/anomaly_label.csv'

    (x_train, y_train), (x_test, y_test), data_df = dataloader.load_HDFS(
        struct_log, event_traces, label_file=label_file,
        window='session', train_ratio=0.5, split_type='sequential')

    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf', normalization='zero-mean')
    x_test = feature_extractor.transform(x_test)

    input_dim = x_train.shape[1]
    model = VAE(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training
    model.train()
    epochs = 200
    for epoch in range(epochs):
        x = torch.tensor(x_train, dtype=torch.float32)
        recon_x, mu, logvar = model(x)
        loss = vae_loss(recon_x, x, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.2f}")

    # Detection and Evaluation
    print("Evaluating on test set:")
    preds, mse, threshold, latents = detect_anomalies(model, x_test)
    final_predictions = refine_with_clustering(latents, preds)
    evaluate(final_predictions, y_test)