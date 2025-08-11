"""
The implementation of Incremental PCA model for anomaly detection.

"""

import numpy as np
from ..utils import metrics
from sklearn.decomposition import IncrementalPCA


class IncPCA(object):

    def __init__(self, n_components=2, batch_size=500):
        #self.n_components = n_components
        #self.batch_size = batch_size
        self.model = IncrementalPCA(n_components=n_components,batch_size = batch_size)


    def fit(self, X):
        """
        Auguments
        ---------
            X: ndarray, the event count matrix of shape num_instances-by-num_events
        """

        print('====== Model summary ======')
        self.model.partial_fit(X)
        
    def predict(self, X):
        # Compute reconstruction error (used as anomaly score)
        X_proj = self.model.transform(X)
        X_reconstructed = self.model.inverse_transform(X_proj)
        reconstruction_error = np.mean((X - X_reconstructed) ** 2, axis=1)

        # Set threshold (e.g., 95th percentile)
        threshold = np.percentile(reconstruction_error, 95)
        y_pred = (reconstruction_error > threshold).astype(int)
        
        return y_pred

    def evaluate(self, X, y_true):
        print('====== Evaluation summary ======')
        y_pred = self.predict(X)
        precision, recall, f1 = metrics(y_pred, y_true)
        print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))
        return precision, recall, f1

