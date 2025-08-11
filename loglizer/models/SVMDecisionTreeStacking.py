"""
The implementation of the SVM+Decision tree combined model usinf stacking for anomaly detection.

"""

import numpy as np
from sklearn import svm
from sklearn import tree
from ..utils import metrics
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier

class SVMDT(object):
    def __init__(self, penalty='l1', tol=0.1, C=1, dual=False,
                 max_iter=500, criterion='gini', max_depth=None, max_features=None, class_weight=None):
        """
        Arguments
        ---------
        See SVM API: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
        
        Attributes
        ----------
            classifier: object, the classifier for anomaly detection

        """
        svm_model = svm.SVC(tol=tol, C=C, class_weight=class_weight, max_iter=max_iter,probability=True)
        #self.classifier = svm.LinearSVC(penalty=penalty, tol=tol, C=C, dual=dual, class_weight=class_weight, max_iter=max_iter)
        
        tree_model = tree.DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,
                          max_features=max_features, class_weight=class_weight)
        
        # Define base models
        base_models = [
            ('svm', svm_model),
            ('tree', tree_model)
        ]

        # Meta-model
        meta_model = RandomForestClassifier(n_estimators=100, random_state=42)  # Stronger meta-learner

        # Stacking Classifier
        self.classifier = StackingClassifier(estimators=base_models, final_estimator=meta_model)
        
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
         