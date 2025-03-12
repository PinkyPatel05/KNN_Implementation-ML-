import numpy as np
import pytest
from src.knn import KNearestNeighbors

def test_knn_initialization():
    knn = KNearestNeighbors(k=3)
    assert knn.k == 3
    assert knn.distance_metric == 'euclidean'

def test_knn_fit():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 2])
    knn = KNearestNeighbors()
    knn.fit(X, y)
    
    assert np.array_equal(knn.X_train, X)
    assert np.array_equal(knn.y_train, y)

def test_knn_predict():
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([0, 1, 2])
    X_test = np.array([[2, 3], [4, 5]])
    
    knn = KNearestNeighbors(k=3)
    knn.fit(X_train, y_train)
    
    predictions = knn.predict(X_test)
    assert len(predictions) == len(X_test)
