"""
The utility functions of log-detector-ml

"""

from sklearn.metrics import precision_recall_fscore_support
import numpy as np
#import shap

def metrics(y_pred, y_true):
    """ Calucate evaluation metrics for precision, recall, and f1.

    Arguments
    ---------
        y_pred: ndarry, the predicted result list
        y_true: ndarray, the ground truth label list

    Returns
    -------
        precision: float, precision value
        recall: float, recall value
        f1: float, f1 measure value
    """
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return precision, recall, f1

def summarize_shap_values(shap_values):
    """
    Summarizes SHAP values by computing the average absolute impact of each feature.

    Parameters:
    - shap_values: List of NumPy arrays (one per class) or a single NumPy array (for binary classification).
    - feature_names: List of feature names.

    Returns:
    - summary_dict: Dictionary with feature importance scores.
    """
    summary_dict = {}

    if isinstance(shap_values, list):  # Multiclass case
        num_classes = len(shap_values)
        #avg_shap = np.mean([np.abs(values).mean(axis=0) for values in shap_values], axis=0)
        avg_shap = np.mean([values.mean(axis=0) for values in shap_values], axis=0)
    else:  # Binary classification case
        #avg_shap = np.abs(shap_values).mean(axis=0)
        avg_shap = shap_values.mean(axis=1)

    """
    # Create dictionary mapping feature names to average SHAP impact
    for i, feature in enumerate(feature_names):
        summary_dict[feature] = avg_shap[i]

    # Sort features by importance
    summary_dict = dict(sorted(summary_dict.items(), key=lambda item: item[1], reverse=True))
    """
    return avg_shap
#############
def explain(X_background, model, x_test, y_test):
    # Use K-Means background for SHAP
    explainer = shap.KernelExplainer(model.predict_proba, X_background)
    shap_values = explainer.shap_values(x_test)
    
    print(len(shap_values[0]))
    """
    i=0
    mean0=0
    mean1=0
    while i<3971:
        mean0 = mean0 + abs(shap_values[0][i][4])
        mean1 = mean1 + abs(shap_values[1][i][4])
        i += 1
    print("Mean shape for feature 6 is (Class 0): ",mean0/i)
    print("Mean shape for feature 6 is (Class 1): ",mean1/i)
    """
    shap.summary_plot(shap_values, x_test)
    
    #shap.force_plot(explainer.expected_value, shap_values[0], x_test[0])  
    
    #shap.force_plot(explainer.expected_value[0], shap_values[0])
    
    #print(len(shap_values[0]))
    #print(shap_values)
    #summary = utils.summarize_shap_values(shap_values)
    #print(summary)
    #shap.summary_plot(shap_values[0], x_test)
    
    #shap.summary_plot(shap_values[1], x_test)
    
    #print(shap_values[1])
    #print(x_test[0]) 
def classFeatures(mean):
    sorted_indicesMean = np.argsort(mean)[::-1]
    print("The order of features by importance in anomalies:", sorted_indicesMean)

def explainAnomalies(X_background, model, x_test, y_test):
    # Use K-Means background for SHAP
    explainer = shap.KernelExplainer(model.predict_proba, X_background)
    #explainer = shap.KernelExplainer(model.decision_function, X_background)  # for Isolation Forest approach
    shap_values = explainer.shap_values(x_test)
    i=0
    cpt=0
    mean = [0 for _ in range(14)]
    while i<3971:
        if (y_test[i] == 1) and (model.predict(x_test[i].reshape(1, -1))==1):
            for k in range(14):  
                mean[k] += abs(shap_values[0][i][k])
                #mean[k] += abs(shap_values[i][k]) # for Isolation Forest approach
            cpt += 1
            print(shap_values[0][i])
            #print(shap_values[i]) # for Isolation Forest approach
        i += 1       
    print(cpt)
    for k in range(14):
        mean[k] = mean[k]/i
    
    print("Absolut shap mean values for anomalies:", mean)
    
    classFeatures(mean)
    
    return shap_values
        