"""
The implementation of Autoencoder Clustering model for anomaly detection.


"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from loglizer import dataloader, preprocessing, utils


#struct_log = '../data/HDFS/HDFS_100k.log_structured.csv' # The structured log file
label_file = '../data/HDFS/anomaly_label.csv' # The anomaly label file

#struct_log = '../data/HDFS/hdfs/HDFS.log_structured.csv'

struct_log = '../data/HDFS/HDFS.npz'

event_traces = '../data/HDFS/Event_traces.csv'

#Modified DeepAutoencoder with Dropout and Batch Normalization
class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(DeepAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(16, 8)  # Latent space (no activation)
        )

        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(16, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, input_dim)  # Output layer (no activation)
        )
        
         # Apply weight initialization
        #self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
'''
class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(DeepAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 8)  # Smaller latent space
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
'''        
'''    
# ----- Autoencoder Architecture -----
class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(DeepAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),  # latent space
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )
                  
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
'''    
'''    
# ----- Train Function -----
def train_autoencoder(model, data, n_epochs=50, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        inputs = torch.tensor(data, dtype=torch.float)
        outputs, _ = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    return model
'''

from torch.utils.data import DataLoader, TensorDataset

# ----- Incremental Train Function -----
def train_autoencoder_incremental(model, data, batch_size=512, n_epochs=10, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Create a DataLoader for mini-batch training
    tensor_data = torch.tensor(data, dtype=torch.float)
    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for batch in dataloader:
            inputs = batch[0]  # batch is a tuple (input,)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader):.4f}")
    
    return model

'''
from torch.utils.data import DataLoader, TensorDataset
import copy

def train_autoencoder(model, train_data, val_data=None, n_epochs=200, lr=0.001, 
                      batch_size=128, patience=10):
    train_tensor = torch.tensor(train_data, dtype=torch.float)
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)

    if val_data is not None:
        val_tensor = torch.tensor(val_data, dtype=torch.float)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    best_model = None
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs = batch[0]
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)

        avg_train_loss = total_loss / len(train_tensor)

        # Validation
        if val_data is not None:
            model.eval()
            with torch.no_grad():
                val_outputs, _ = model(val_tensor)
                val_loss = criterion(val_outputs, val_tensor).item()

            print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                best_model = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    model.load_state_dict(best_model)
                    break
        else:
            print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

    return model
'''
# ----- Anomaly Detection -----
def detect_anomalies(model, data, threshold=None):
    model.eval()
    inputs = torch.tensor(data, dtype=torch.float)
    with torch.no_grad():
        outputs, latents = model(inputs)
        mse = torch.mean((outputs - inputs) ** 2, dim=1).numpy()

    # Threshold can be dynamically set or fixed
    if threshold is None:
        threshold = np.percentile(mse, 90)  # Top 5% as anomalies

    predictions = (mse > threshold).astype(int)
    return predictions, mse, latents.numpy(), threshold

# ----- Clustering Refinement -----
def refine_with_clustering(latents, predictions):
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
    '''
    from loglizer.models import IncrementalPCA
    n_components = 4
    model = IncrementalPCA.IncPCA(n_components=n_components)
    model.fit(latents)
    clustered_preds = model.predict(latents)
    '''
    # Combine reconstruction error and clustering
    final_preds = np.logical_or(predictions, clustered_preds).astype(int)
    return final_preds

# ----- Evaluation -----
def evaluate(preds, labels):
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    return precision, recall, f1