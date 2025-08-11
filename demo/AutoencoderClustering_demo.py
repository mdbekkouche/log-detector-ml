#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')

import numpy as np
from loglizer import dataloader, preprocessing, utils
from loglizer.models import AutoencoderClustering

#struct_log = '../data/HDFS/HDFS_100k.log_structured.csv' # The structured log file
label_file = '../data/HDFS/anomaly_label.csv' # The anomaly label file

struct_log = '../data/HDFS/hdfs/HDFS.log_structured.csv'

#struct_log = '../data/HDFS/HDFS.npz'

event_traces = '../data/HDFS/Event_traces.csv'

"""
# ----- Autoencoder Architecture -----
class DeepAutoencoder(nn.Module):
    '''
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        #####
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )
        ####
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),  # optional: increase width too
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),  # latent size = 16
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    '''
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
    
# ----- Train Function -----
def train_autoencoder(model, data, n_epochs=270, lr=0.001):
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
        threshold = np.percentile(mse, 95)  # Top 5% as anomalies
    
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
    
    # Combine reconstruction error and clustering
    final_preds = np.logical_or(predictions, clustered_preds).astype(int)
    return final_preds

# ----- Evaluation -----
def evaluate(preds, labels):
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

"""

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
                                                                train_ratio=0.9,
                                                                split_type='uniform')
    print(x_train)
    
    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf', 
                                              normalization='zero-mean')
    
    print(x_train)
    
    x_test = feature_extractor.transform(x_test)
    
    print("Shape of data:", x_train.shape)
    print("Data type:", x_train.dtype)
    input_dim = x_train[y_train == 0, :].shape[1]
    
    
    import numpy as np
    n_runs = 10
    precisions, recalls, f1s = [], [], []
    for _ in range(n_runs):
        #model = Autoencoder(input_dim)
        model = AutoencoderClustering.DeepAutoencoder(input_dim)
      
        #model = AutoencoderClustering.train_autoencoder(model, x_train[y_train == 0, :])
        model = AutoencoderClustering.train_autoencoder_incremental(model, x_train[y_train == 0, :])
        #model = train_autoencoder(model, x_train, val_data=x_test)
        '''
        predictions, mse, latents, threshold = AutoencoderClustering.detect_anomalies(model, x_train)
        print(f"Autoencoder threshold: {threshold:.4f}")

        final_predictions = AutoencoderClustering.refine_with_clustering(latents, predictions)
        AutoencoderClustering.evaluate(final_predictions, y_train)
        '''
        #####
        predictions1, mse1, latents1, threshold1 = AutoencoderClustering.detect_anomalies(model, x_test)
        print(f"Autoencoder threshold: {threshold1:.4f}")

        final_predictions1 = AutoencoderClustering.refine_with_clustering(latents1, predictions1)
        precision, recall, f1 = AutoencoderClustering.evaluate(final_predictions1, y_test)
        #precision, recall, f1 = AutoencoderClustering.evaluate(predictions1, y_test)
        
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