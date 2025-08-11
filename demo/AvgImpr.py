'''
import sys
sys.path.append('../')
import numpy as np
import pickle

"""
# Load the NPZ file with allow_pickle=True
data = np.load("../data/HDFS/HDFS.npz", allow_pickle=True)

# List available keys
print("Available keys:", data.files)

# Extract x_data and y_data
x_data = data["x_data"]
y_data = data["y_data"]

# Print first 5 rows of each dataset
print("First 5 rows of x_data:\n", x_data[:5])
print("First 5 rows of y_data:\n", y_data[:5])

print(f"Number of rows in x_data : {len(data['x_data'])}")
print(f"Number of rows in y_data : {len(data['y_data'])}")
"""      

with open("../data/HDFS/session_train.pkl", "rb") as f:
    data = pickle.load(f)  # Désérialiser le fichier
    print(len(data))  # Afficher le contenu      
    
with open("../data/HDFS/session_trainComplet.pkl", "rb") as f:
    data = pickle.load(f)  # Désérialiser le fichier
    print(len(data))  # Afficher le contenu        
    
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Sample log sequences
log_sequences = [
    "ERROR Disk failure detected", 
    "INFO System started", 
    "WARNING High CPU usage detected", 
    "ERROR Network timeout", 
    "INFO User login successful"
]

# Convert log sequences into an event count matrix
vectorizer = CountVectorizer()
event_matrix = vectorizer.fit_transform(log_sequences)

# Convert to DataFrame for better visualization
df = pd.DataFrame(event_matrix.toarray(), columns=vectorizer.get_feature_names())
print(df)
"""
'''

import sys
sys.path.append('../')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

struct_log_job = '../data/CevitalDataSet/JOB (intervention).csv'
'''
struct_log_work_order = '../data/CevitalDataSet/work_order.csv'

struct_work_orders_jobs = '../data/CevitalDataSet/work_orders_jobs.csv'

# Charger les fichiers CSV
work_orders = pd.read_csv(struct_log_work_order, error_bad_lines=False, encoding="ISO-8859-1")
jobs = pd.read_csv(struct_log_job, error_bad_lines=False, encoding="ISO-8859-1")
print("Colonnes de jobs.csv :", jobs.columns.tolist())
# Fusionner les deux DataFrames sur les colonnes appropriées
merged_df = pd.merge(jobs, work_orders,  left_on="MDJB_CODE", right_on="WOWO_JOB", how="inner")

# Afficher les premières lignes du résultat
print(merged_df.head())

# Sauvegarder le résultat dans un nouveau fichier CSV si besoin
merged_df.to_csv(struct_work_orders_jobs, index=False)
'''
'''
# Load CSV into a DataFrame
df = pd.read_csv(struct_log, error_bad_lines=False, encoding="ISO-8859-1")

# Drop unnecessary columns (keeping relevant features)
df = df[['MDJB_DESCRIPTION', 'MDJB_JOB_TYPE']]

# Drop rows with missing labels
df.dropna(subset=['MDJB_JOB_TYPE'], inplace=True)

# Convert Text Features into Numerical Data
label_encoder = LabelEncoder()
df['MDJB_JOB_TYPE'] = label_encoder.fit_transform(df['MDJB_JOB_TYPE'])  # Encode Target Variable

# Convert job descriptions into numerical features using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['MDJB_DESCRIPTION'])  # Text Features
y = df['MDJB_JOB_TYPE']  # Target (Classification Labels)

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluate Model Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Test on New Data
new_jobs = ["Nettoyage du site", "Remplacement du moteur", "Maintenance préventive annuelle"]
new_jobs_vectorized = vectorizer.transform(new_jobs)
predictions = clf.predict(new_jobs_vectorized)

# Decode Predictions
decoded_predictions = label_encoder.inverse_transform(predictions)
for job, category in zip(new_jobs, decoded_predictions):
    print(f"Job: {job} -> Predicted Category: {category}")
'''
    

'''
# Drop rows with missing labels
df.dropna(subset=['MDJB_JOB_TYPE'], inplace=True)

# Convert log messages into numerical features using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['MDJB_DESCRIPTION'])  # Log messages as features

# Apply K-Means Clustering
num_clusters = 3  # Define the number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X)

# Display clustered log messages
for cluster in range(num_clusters):
    print(f"\nCluster {cluster}:")
    print(df[df['cluster'] == cluster][['MDJB_DESCRIPTION']])
'''    
'''
import numpy as np
precisions, recalls, f1s = [0.743,0.743,0.744,0.743,0.742,0.744,0.743,0.744,0.743,0.743], [0.690,0.690,0.691,0.690,0.690,0.691,0.690,0.691,0.690,0.685], [0.716,0.716,0.716,0.716,0.715,0.717,0.716,0.716,0.716,0.712]
mean_p, std_p = np.mean(precisions), np.std(precisions, ddof=1)
mean_r, std_r = np.mean(recalls), np.std(recalls, ddof=1)
mean_f1, std_f1 = np.mean(f1s), np.std(f1s, ddof=1)
# Print and store
print(f"  Precision: {mean_p:.3f} ± {std_p:.3f}")
print(f"  Recall:    {mean_r:.3f} ± {std_r:.3f}")
print(f"  F1-score:  {mean_f1:.3f} ± {std_f1:.3f}")
'''

import os
import pkg_resources
from datetime import datetime

for dist in pkg_resources.working_set:
    try:
        # Get the metadata directory path
        dist_info_path = dist.egg_info
        # Get the modification time
        mtime = os.path.getmtime(dist_info_path)
        install_date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{dist.project_name}=={dist.version} | Installed on: {install_date}")
    except Exception as e:
        print(f"{dist.project_name}=={dist.version} | Installation date unknown: {e}")

'''
def compute_average_improvement(recall_A, recall_B):
    improvements = []
    a = sum(recall_A) / len(recall_A)
    b = sum(recall_B) / len(recall_B)
    improvement = ((a - b) / b) * 100
    return improvement

# Example data
recall_A = [0.625, 0.602]
recall_B = [0.620, 0.441, 0.596, 0.558]

avg_improvement = compute_average_improvement(recall_A, recall_B)
print(f"Average Improvement in Recall: {avg_improvement:.2f}%")
'''
'''
def compute_average_improvement(recall_A, recall_B):
    if len(recall_A) != len(recall_B):
        raise ValueError("Both recall lists must have the same length.")

    improvements = []
    for a, b in zip(recall_A, recall_B):
        if b == 0:
            raise ValueError("Recall value for approach B is zero, cannot compute relative improvement.")
        improvement = ((a - b) / b) * 100
        improvements.append(improvement)

    average_improvement = sum(improvements) / len(improvements)
    return average_improvement

# Example data
recall_A = [0.624, 0.510, 0.624, 0.650, 0.591, 0.580, 0.590, 0.612]

recall_B = [0.620, 0.441, 0.625, 0.620, 0.596, 0.558, 0.602, 0.596]
avg_improvement = compute_average_improvement(recall_A, recall_B)
print(f"Average Improvement in Recall: {avg_improvement:.2f}%")
'''