#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
from loglizer.models import DecisionTree
from loglizer import dataloader, preprocessing, utils
#import shap

#struct_log = '../data/HDFS/HDFS_100k.log_structured.csv' # The structured log file

struct_log = '../data/SPIRIT/Spirit5M.log_structured.csv'

#struct_log = '../data/HDFS/hdfs/HDFS.log_structured.csv'

label_file = '../data/HDFS/anomaly_label.csv' # The anomaly label file

#struct_log = '../data/HDFS/HDFS.npz'

event_traces = '../data/HDFS/Event_traces.csv'

def vectorize_sequence(seq, model):
    vectors = [model.wv[word] for word in seq if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

if __name__ == '__main__':
    '''
    (x_train, y_train), (x_test, y_test), data_df = dataloader.load_HDFS(struct_log,
                                                                event_traces,
                                                                label_file=label_file,
                                                                window='session', 
                                                                train_ratio=0.5,
                                                                split_type='uniform')
    '''
    (x_train, y_train), (x_test, y_test) = dataloader.load_SPIRIT(struct_log,
                                                                event_traces,         
                                                                label_file=label_file,
                                                                window='session', 
                                                                train_ratio=0.8,
                                                                split_type='uniform', WV=True)

    import numpy as np
                                                                         
    result = np.concatenate((x_train, x_test), axis=0)  # axis=0 to stack vertically
    
    sequences = [list(map(str, row)) for row in result.tolist()]
    
    from gensim.models import Word2Vec

    # Train Word2Vec model
    w2v_model = Word2Vec(sentences=sequences, vector_size=500, window=5, min_count=1)
    
    
    X_train = np.array([vectorize_sequence(seq, w2v_model) for seq in x_train])
    
    X_test = np.array([vectorize_sequence(seq, w2v_model) for seq in x_test])
    
    '''
    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
    x_test = feature_extractor.transform(x_test)
    '''
    import numpy as np
    n_runs = 10
    precisions, recalls, f1s = [], [], []
    for _ in range(n_runs):
        model = DecisionTree()
        model.fit(X_train, y_train)

        #print('Train validation:')
        #precision, recall, f1 = model.evaluate(x_train, y_train)

        print('Test validation:')
        precision, recall, f1 = model.evaluate(X_test, y_test)
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
        
        #X_background = shap.sample(x_train, 20)

        #y_test = model.predict(x_test) 

        #utils.explain(X_background, model, x_test, y_test)  

        #utils.explainAnomalies(X_background, model, x_test, y_test)
