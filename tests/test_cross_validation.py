import numpy as np
import pytest
from src.knn import KNearestNeighbors
from src.cross_validation import k_fold_cross_validation

def test_k_fold_cross_validation():
    # Create sample data
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 3, 100)
    
    knn = KNearestNeighbors(k=3)
    scores = k_fold_cross_validation(X, y, k=5, model=knn)
    
    assert len(scores) == 5
    for score in scores:
        assert 0 <= score <= 1
