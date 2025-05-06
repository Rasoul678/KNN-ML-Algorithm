import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k: int = 3):
        """
        Initialize KNN classifier with a default of 3 neighbors.

        Parameters:
        k (int): Number of neighbors to consider for voting
        """
        self.k = k

    def fit(self, X, y):
        """
        Store the training data. KNN doesn't actually "train" - it just memorizes the data.

        Parameters:
        X (numpy.ndarray): Training features of shape (n_samples, n_features)
        y (numpy.ndarray): Training labels of shape (n_samples)
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predict the class labels for the provided data.

        Parameters:
        X (numpy.ndarray): Test samples of shape (n_samples, n_features)

        Returns:
        numpy.ndarray: Predicted class labels for each test sample
        """
        predicted_labels = [self.__predict(x) for x in X]
        return np.array(predicted_labels)

    def __predict(self, x):
        """
        Helper method to predict the label for a single sample.

        Parameters:
        x (numpy.ndarray): A single test sample of shape (n_features)

        Returns:
            int: Predicted class label
        """

        # Step 1: Compute distances between x and all examples in the training set
        distances = [self.__euclidean_distance(x, x_train) for x_train in self.X_train]

        # Step 2: Sort by distance and get indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]

        # Step 3: Extract the labels of the k nearest neighbors
        k_nearest_label = [self.y_train[i] for i in k_indices]

        # Step 4: Return the most common class label (majority vote)
        most_common = Counter(k_nearest_label).most_common()

        return most_common[0][0]

    @staticmethod
    def __euclidean_distance(x1, x2):
        """
        Compute the Euclidean distance between two vectors.

        Parameters:
        x1 (numpy.ndarray): First vector
        x2 (numpy.ndarray): Second vector

        Returns:
        float: Euclidean distance between x1 and x2
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))
