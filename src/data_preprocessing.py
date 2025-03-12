import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

def ensure_numpy(X):
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        return X.to_numpy()
    return X

def preprocess_data(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    print(data.head()) 
    
    if 'target' not in data.columns:
        feature_cols = data.columns[:-1]
        target_col = data.columns[-1]
    else:
        feature_cols = [col for col in data.columns if col != 'target']
        target_col = 'target'
    
    cat_cols = []
    numerical_cols = []
    
    for col in feature_cols:
        if data[col].dtype == 'object' or data[col].nunique() < 10:
            cat_cols.append(col)
        else:
            numerical_cols.append(col)
    print("\nColumns identification:")
    print(f"Categorical columns identified: {cat_cols}")
    print(f"Numerical columns identified: {numerical_cols}")
    
    data_process = data.copy()
    
    if numerical_cols:
        imputer_num = SimpleImputer(strategy='mean')
        data_process[numerical_cols] = imputer_num.fit_transform(data_process[numerical_cols])
    
    if cat_cols:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        data_process[cat_cols] = imputer_cat.fit_transform(data_process[cat_cols])
        
        for col in cat_cols:
            le = LabelEncoder()
            data_process[col] = le.fit_transform(data_process[col])
    else:
        print("No categorical columns found!")
    
    if numerical_cols:
        scaler = StandardScaler()
        data_process[numerical_cols] = scaler.fit_transform(data_process[numerical_cols])
    
    # Prepare features and target
    scaler = MinMaxScaler()
    X = scaler.fit_transform(data_process[feature_cols]) 
    y = data_process[target_col].to_numpy()
    
    print(f"\nData shape after preprocessing: X={X.shape}, y={y.shape}")
    
    return X, y

class KNearestNeighbors:
    def __init__(self, n_neighbors=3, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = ensure_numpy(X)
        self.y_train = ensure_numpy(y)
        return self
    
    def _calculate_distance(self, x1, x2):
        # calculate distance between two points
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def predict(self, X):
        # convert to numpy array if needed
        X = ensure_numpy(X)
        
        y_pred = np.zeros(X.shape[0], dtype=self.y_train.dtype)
        
        for i, x_test in enumerate(X):
            dist = np.array([self._calculate_distance(x_test, x_train) for x_train in self.X_train])
            
            k_indices = np.argsort(dist)[:self.n_neighbors]
            
            k_nearest_labels = self.y_train[k_indices]
            most_common = np.bincount(k_nearest_labels).argmax()
            
            y_pred[i] = most_common
            
        return y_pred
    
    def score(self, X, y_true):
        X = ensure_numpy(X)
        y_true = ensure_numpy(y_true)
        
        y_pred = self.predict(X)
        
        correct_predict = sum(y_pred == y_true)
        total_predict = len(y_true)
        
        return correct_predict / total_predict