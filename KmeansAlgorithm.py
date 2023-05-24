import pandas as pd
import numpy as np
import warnings

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")


def compute_metrics(confMatrix, class_index):
    true_positive = confMatrix[class_index, class_index]
    false_positive = np.sum(confMatrix[:, class_index]) - true_positive
    false_negative = np.sum(confMatrix[class_index, :]) - true_positive

    return false_negative, false_positive, true_positive


def compute_purity(predicted_labels, true_labels):
    clusters = np.unique(predicted_labels)
    purityy = 0

    for cluster in clusters:
        mask = predicted_labels == cluster
        cluster_labels = true_labels[mask]
        counts = np.bincount(cluster_labels)
        majority_count = np.max(counts)
        cluster_purity = majority_count / len(cluster_labels)
        purityy += cluster_purity * len(cluster_labels)

    purityy /= len(predicted_labels)

    return purityy


print('-Kmeans Algorithm-')

trainingData = pd.read_csv('train.csv')
target_col = 'price_range'
y = trainingData[target_col]
x = trainingData.drop(target_col, axis=1)

X, X_, Y, Y_ = train_test_split(x, y, test_size=0.2, random_state=42)
X = X.to_numpy()
Y = Y.to_numpy()

k_values = [2, 4, 6, 8, 10]

totalFMeasures = []
print('-----------------------')
for k in k_values:
    print(f"K = {k}")
    purity_scores = []
    fMeasures_scores = []
    cluster_data = defaultdict(list)
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(X):
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = Y[train_index], Y[test_index]
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(X_train)
        labels = kmeans.predict(X_val)
        for i in range(len(X_val)):
            data_point = X_val[i]
            cluster_label = labels[i]
            cluster_data[cluster_label].append(data_point)
        cluster_majority = {}
        for cluster_label, points in cluster_data.items():
            true_labels = [y_val[i] for i in range(len(X_val)) if labels[i] == cluster_label]
            majority_class = max(set(true_labels), key=true_labels.count)
            cluster_majority[cluster_label] = majority_class
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for i in range(len(X_val)):
            cluster_label = labels[i]
            true_label = y_val[i]
            if true_label == cluster_majority[cluster_label]:
                true_positives += 1
            else:
                if true_label in cluster_majority.values():
                    false_positives += 1
                else:
                    false_negatives += 1
        purity = compute_purity(labels, y_val)
        purity_scores.append(purity)
        precision = true_positives/(true_positives + false_positives)
        recall = true_positives/(true_positives + false_negatives)
        fMeasure = 2 / ((1 / precision) + (1 / recall))
        if not np.isnan(fMeasure):
            fMeasures_scores.append(fMeasure)
    mean_purity = np.mean(purity_scores)
    mean_fMeasure = np.mean(fMeasures_scores)
    if not np.isnan(mean_fMeasure):
        totalFMeasures.append(mean_fMeasure)

    print('F-Measure: ', mean_fMeasure)
    print('Purity: ', mean_purity)
    print('-----------------------')

print('\nTotal F-Measure: ', sum(totalFMeasures))
print('Mean Purity: ', np.mean(purity_scores))
