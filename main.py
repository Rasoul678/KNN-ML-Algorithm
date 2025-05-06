import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from KNN import KNN

cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])


def main():
    # Load dataset
    iris = load_iris()
    print(iris.DESCR)

    X = iris.data
    print("Feature names:", iris.feature_names)
    print("First 5 rows of data:\n", X[:5])
    print(X.shape)

    y = iris.target
    print("Target names:", iris.target_names)
    print("First 5 targets:", y[:5])
    print(y.shape)


    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    print(f"Training set: {X_train[:5]}")
    print(f"Training set size: {len(X_train)}")
    print(f"Test set: {X_test[:5]}")
    print(f"Test set size: {len(X_test)}")

    plt.figure()
    plt.scatter(X[:,2],X[:,3], c=y, cmap=cmap, edgecolor="k", s=20)
    # plt.savefig('filename.png')

    # Create and train classifier
    clf = KNN(k=5)
    clf.fit(X_train, y_train)

    # Make predictions
    predictions = clf.predict(X_test)
    print(f"Predictions Result: {predictions}")
    print(f"y_test:             {y_test}")

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.2f}")

    acc = np.sum(predictions == y_test) / len(y_test)
    print(acc)

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)
    predictions2 = clf.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, predictions2)
    print(f"Accuracy2: {accuracy:.2f}")


if __name__ == "__main__":
    main()
