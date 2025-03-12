import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

def ensure_numpy(X):
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        return X.to_numpy()
    return X

def k_fold_cross_validation(X, y, k=10, model=None, random_state=42):
    np.random.seed(random_state)
    
    is_pandas_X = isinstance(X, pd.DataFrame)
    is_pandas_y = isinstance(y, pd.Series) or isinstance(y, pd.DataFrame)
    
    X_np = ensure_numpy(X)
    y_np = ensure_numpy(y)
    n_samples = len(y_np)

    indic = np.random.permutation(n_samples)
    
    # k-fold size calculation
    fold_size = n_samples // k
    scores = []
    
    # k-fold cross-validation implementation
    for i in range(k):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < k - 1 else n_samples
        
        test_indic = indic[start_idx:end_idx]
        train_indic = np.array([idx for idx in indic if idx not in test_indic])
        
        # split data into train and test sets 
        if is_pandas_X:
            X_train = X.iloc[train_indic]
            X_test = X.iloc[test_indic]
        else:
            X_train = X_np[train_indic]
            X_test = X_np[test_indic]
            
        if is_pandas_y:
            y_train = y.iloc[train_indic]
            y_test = y.iloc[test_indic]
        else:
            y_train = y_np[train_indic]
            y_test = y_np[test_indic]
        
        # train the model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        scores.append(acc)
    
    return scores