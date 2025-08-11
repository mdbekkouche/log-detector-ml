"""
The implementation of One-ClassSVM model for anomaly detection.

"""

import numpy as np
from sklearn.svm import OneClassSVM
from ..utils import metrics

class OCSVM(object):

    def __init__(self, kernel='rbf', gamma='auto', nu=0.1):
        """
        Attributes
        ----------
            classifier: object, the classifier for anomaly detection
        """
        self.classifier = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)

    def fit(self, X):
        """
        Arguments
        ---------
            X: ndarray, the event count matrix of shape num_instances-by-num_events
        """
        print('====== Model summary ======')
        self.classifier.fit(X)

    def predict(self, X):
        """ Predict anomalies with mined invariants

        Arguments
        ---------
            X: the input event count matrix

        Returns
        -------
            y_pred: ndarray, the predicted label vector of shape (num_instances,)
        """
        y_pred = self.classifier.predict(X)
        return y_pred

    def predict_proba(self, X):
        """ Predict anomalies with mined invariants

        Arguments
        ---------
            X: the input event count matrix

        Returns
        -------
            y_pred: ndarray, the predicted label vector of shape (num_instances,)
        """
        y_pred = self.classifier.predict_proba(X)
        return y_pred

    def evaluate(self, X, y_true):
        print('====== Evaluation summary ======')
        y_pred = self.predict(X)
        y_pred = np.where(y_pred == -1, 1, 0)  # Convert -1 (anomaly) to 1, normal to 0
        precision, recall, f1 = metrics(y_pred, y_true)
        print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))
        return precision, recall, f1