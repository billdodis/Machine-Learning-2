import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.model_selection import KFold


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

input_dim = x.shape[1]
encoding_dim = [2, 10, 50]

k_values = [2, 4, 6, 8, 10]

totalFMeasures = []
for m in encoding_dim:
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(100, activation='relu')(input_layer)
    encoder = Dense(m, activation='relu')(encoder)

    decoder = Dense(100, activation='relu')(encoder)
    decoder = Dense(input_dim, activation='sigmoid')(decoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)

    encoder_model = Model(inputs=input_layer, outputs=encoder)

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    autoencoder.fit(x, x, epochs=10, batch_size=32)

    encoded_data = encoder_model.predict(x)

    if m == 2:
        plt.scatter(encoded_data[:, 0], encoded_data[:, 1])
        plt.xlabel('Encoded Dimension 1')
        plt.ylabel('Encoded Dimension 2')
        plt.title('Autoencoder Transformed Data')
        plt.show()
    X, X_, Y, Y_ = train_test_split(encoded_data, y, test_size=0.2, random_state=42)
    Y = Y.to_numpy()
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
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
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
