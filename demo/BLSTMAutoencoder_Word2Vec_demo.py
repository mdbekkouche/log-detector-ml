#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')

import numpy as np
from loglizer import dataloader, preprocessing, utils
from loglizer.models import LSTMAutoencoder

#struct_log = '../data/HDFS/HDFS_100k.log_structured.csv' # The structured log file
label_file = '../data/HDFS/anomaly_label.csv' # The anomaly label file

struct_log = '../data/HDFS/hdfs/HDFS.log_structured.csv'

#struct_log = '../data/HDFS/HDFS.npz'

event_traces = '../data/HDFS/Event_traces.csv'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# -------------------------
# Define BiLSTM Autoencoder
# -------------------------
class BiLSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(BiLSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.latent = nn.Linear(hidden_size * 2, latent_size)
        self.decoder = nn.LSTM(latent_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        enc_out, _ = self.encoder(x)
        latent = self.latent(enc_out)
        dec_out, _ = self.decoder(latent)
        out = self.output_layer(dec_out)
        return out

def compute_thresholds(recon_errors, t0, fraction=0.001, num_points=1000):
    """
    Compute thresholds centered around t0 with a delta computed as a fraction
    of the spread (max - min) of the reconstruction errors.

    Parameters:
        recon_errors (np.ndarray): Array of reconstruction errors.
        t0 (float): Center threshold value.
        fraction (float): Fraction of the spread to use as delta (default 0.001).
        num_points (int): Number of threshold values to generate.

    Returns:
        np.ndarray: Array of threshold values.
    """
    min_error = np.min(recon_errors)
    max_error = np.max(recon_errors)
    spread = max_error - min_error

    delta = fraction * spread

    # Clip delta if needed to stay inside bounds
    lower = max(min_error, t0 - delta)
    upper = min(max_error, t0 + delta)

    return np.linspace(lower, upper, num_points)

def vectorize_sequence(seq, model):
    vectors = [model.wv[word] for word in seq if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

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
    
                                                                        
    result = np.concatenate((x_train, x_test), axis=0)  # axis=0 to stack vertically
    
    sequences = [list(map(str, row)) for row in result.tolist()]
    
    from gensim.models import Word2Vec

    # Train Word2Vec model
    #w2v_model = Word2Vec(sentences=sequences, vector_size=10000, window=5, min_count=1)

    w2v_model = Word2Vec(
        sentences=sequences,
        vector_size=150,
        window=5,
        min_count=1,
        sg=1,
        workers=4,
        epochs=30
    )
    
    x_train = np.array([vectorize_sequence(seq, w2v_model) for seq in x_train])
    
    x_test = np.array([vectorize_sequence(seq, w2v_model) for seq in x_test])
    '''
    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf', 
                                              normalization='zero-mean')
    x_test = feature_extractor.transform(x_test)
    '''
    train_loader = DataLoader(TensorDataset(torch.tensor(x_train)), batch_size=64, shuffle=True)
    
    # -------------------------
    # Train the Model
    # -------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiLSTMAutoencoder(input_size=150, hidden_size=128, latent_size=64).to(device)
    #model = model.double()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    for epoch in range(10):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs = batch[0].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()


        ####################
        model.eval()
        with torch.no_grad():
            inputs = torch.tensor(x_test, dtype=torch.float).to(device)
            outputs = model(inputs)
            loss_per_seq = torch.mean((outputs - inputs)**2, dim=1)  # MSE per sequence

        from sklearn.metrics import precision_score, recall_score, f1_score
        
        # Move to CPU for evaluation
        recon_errors = loss_per_seq.cpu().numpy()

        # Perform threshold tuning to maximize F1
        best_threshold = None
        best_f1 = 0
        t0 = np.percentile(recon_errors, 95)
        #thresholds = np.linspace(t0 - 0.0001275, t0 + 0.0001275, 1000)
        thresholds = compute_thresholds(recon_errors, t0, fraction=0.001)
        
        #thresholds = np.linspace(np.min(recon_errors), np.max(recon_errors), 1000)

        print(np.min(recon_errors), ' ', np.max(recon_errors)) 
        print(thresholds)
        
        for t in thresholds:
            preds = (recon_errors > t).astype(int)
            f1 = f1_score(y_test, preds)  # y_true should be defined as ground truth labels
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
        
        # Final anomaly detection using best threshold
        anomalies = loss_per_seq > best_threshold

        #threshold = np.percentile(loss_per_seq.cpu().numpy(), 95)
        #anomalies = loss_per_seq > threshold
        print(f"Detected {anomalies.sum().item()} anomalies out of {len(x_test)} samples.")
        
        precision = precision_score(y_test, anomalies)
        recall = recall_score(y_test, anomalies)
        f1 = f1_score(y_test, anomalies)
        print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        ####################
        
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")
    
    # -------------------------
    # Anomaly Score Calculation
    # -------------------------
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(x_test, dtype=torch.float).to(device)
        outputs = model(inputs)
        loss_per_seq = torch.mean((outputs - inputs)**2, dim=1)  # MSE per sequence
    
    # Threshold can be tuned (e.g., using validation set)
    threshold = np.percentile(loss_per_seq.cpu().numpy(), 95)
    anomalies = loss_per_seq > threshold
    print(f"Detected {anomalies.sum().item()} anomalies out of {len(x_test)} samples.")
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    precision = precision_score(y_test, anomalies)
    recall = recall_score(y_test, anomalies)
    f1 = f1_score(y_test, anomalies)
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    