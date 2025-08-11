"""
The implementation of the Random Forest model for anomaly detection.

"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ..utils import metrics

class RF(object):

    def __init__(self, n_estimators=100, random_state=42):
        """ The Invariants Mining model for anomaly detection
        Arguments
        ---------
        See SVM API: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
        
        Attributes
        ----------
            classifier: object, the classifier for anomaly detection

        """
        self.classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def fit(self, X, y):
        """
        Arguments
        ---------
            X: ndarray, the event count matrix of shape num_instances-by-num_events
        """
        print('====== Model summary ======')
        self.classifier.fit(X, y)

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

    def evalu(self, x):
        y_pred = self.predict(x)
        return y_pred
    
    def evaluate(self, X, y_true):
        print('====== Evaluation summary ======')
        y_pred = self.predict(X)
        precision, recall, f1 = metrics(y_pred, y_true)
        print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))
        return precision, recall, f1
    def predict_proba(self, X):  # Add this method for SHAP
        return self.classifier.predict_proba(X)