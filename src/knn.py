import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os
import warnings
import scipy.stats as stats

def ensure_numpy(X):
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        return X.to_numpy()
    return X

class KNearestNeighbors:
    def __init__(self, k=5):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X_test):
        if hasattr(X_test, 'values'):
            X_test = X_test.values
        else:
            X_test = np.asarray(X_test)
    
        if hasattr(self.y_train, 'iloc'):
            first_y = self.y_train.iloc[0] if len(self.y_train) > 0 else 0
            y_dtype = type(first_y)
        else:
            first_y = self.y_train[0] if len(self.y_train) > 0 else 0
            y_dtype = type(first_y)
        
        y_pred = np.empty(X_test.shape[0], dtype=y_dtype)
        
        if hasattr(self.y_train, 'values'):
            y_train_values = self.y_train.values
        else:
            y_train_values = np.asarray(self.y_train)
        
        if hasattr(self.X_train, 'values'):
            X_train_values = self.X_train.values
        else:
            X_train_values = np.asarray(self.X_train)
        
        for i, x in enumerate(X_test):
            distances = []
            for idx, x_train in enumerate(X_train_values):
                dist = np.sqrt(np.sum((x - x_train) ** 2))
                distances.append((dist, y_train_values[idx]))
            
            distances.sort(key=lambda x: x[0])
            k_nearest_labels = [label for _, label in distances[:self.k]]
            
            if k_nearest_labels:
                from collections import Counter
                y_pred[i] = Counter(k_nearest_labels).most_common(1)[0][0]
            else:
                from collections import Counter
                y_pred[i] = Counter(y_train_values).most_common(1)[0][0]
        
        return y_pred
    def score(self, X, y_true):
        
        y_pred = self.predict(X)
        y_true_array = ensure_numpy(y_true)
        
        correct_predictions = sum(y_pred == y_true_array)
        total_predictions = len(y_true_array)
        
        return correct_predictions / total_predictions    

def preprocess_data(df, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Categorical columns identified: {categorical_cols}")
    print(f"Numerical columns identified: {numerical_cols}")
    
    if not numerical_cols and categorical_cols:
        print("Only categorical data found - applying one-hot encoding without normalization")
        X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
        return X_encoded, y

    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
    
    if numerical_cols:
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    return X, y

def evaluate_model(model, X, y, dataset_name, n_splits=10):
    try:
        # handle small datasets
        class_counts = dict(Counter(y))
        min_samples = min(class_counts.values())
        
        if min_samples < n_splits:
            adjusted_splits = max(2, min(5, min_samples))
            warnings.warn(f"The least populated class has only {min_samples} samples. "
                         f"Reducing n_splits from {n_splits} to {adjusted_splits}.")
            n_splits = adjusted_splits
        
        # use stratified K-fold to preserve class distribution
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        custom_scores = []
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            custom_scores.append(accuracy)
        
        return np.mean(custom_scores), np.std(custom_scores)
    
    except Exception as e:
        print(f"Error evaluating model on {dataset_name}: {str(e)}")
        return None, None

def process_dataset(filepath, k=5):
    try:
        print(f"\nAnalyzing Dataset: {filepath}")
        
        df = pd.read_csv(filepath)
        print(df.head())
        
        if 'target' in df.columns:
            target_column = 'target'
        else:
            target_column = df.columns[-1]
        
        X, y = preprocess_data(df, target_column)
        print(f"Data shape after preprocessing: X={X.shape}, y={y.shape}")
        
        custom_knn = KNearestNeighbors(k=k)  
        sklearn_knn = KNeighborsClassifier(n_neighbors=k)
        
        custom_acc, custom_std = evaluate_model(custom_knn, X, y, filepath, n_splits=5)
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        sklearn_scores = cross_val_score(sklearn_knn, X, y, cv=kf)
        sklearn_acc, sklearn_std = sklearn_scores.mean(), sklearn_scores.std()

        if custom_acc is not None and sklearn_acc is not None:
            print(f"Custom KNN Accuracy: {custom_acc:.4f} ± {custom_std:.4f}")
            print(f"Scikit-learn KNN Accuracy: {sklearn_acc:.4f} ± {sklearn_std:.4f}")
            
            # Statistical comparison
            t_stat, p_val = stats.ttest_rel(sklearn_scores, [custom_acc] * len(sklearn_scores))
            print(f"T-statistic: {t_stat}, p-value: {p_val}")
        
        print("-" * 50)
        return True
        
    except Exception as e:
        print(f"Error processing dataset {filepath}: {str(e)}")
        print("-" * 50)
        return False

def main(data_dir='data', k=5):
    os.makedirs(data_dir, exist_ok=True)

    csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}.")
        return
    
    successful = 0
    for filepath in csv_files:
        success = process_dataset(filepath, k)
        if success:
            successful += 1
    
    print(f"Successfully processed {successful}/{len(csv_files)} datasets.")

if __name__ == "__main__":
    main()