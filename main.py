import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from scipy import stats
import os

from src.knn import KNearestNeighbors
from src.cross_validation import k_fold_cross_validation
from src.data_preprocessing import preprocess_data

def ensure_numpy(X):
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        return X.to_numpy()
    return X

def load_dataset(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading dataset {filepath}: {e}")
        return None

def statistical_comparison(custom_scores, sklearn_scores):
    t_statistic, p_value = stats.ttest_rel(custom_scores, sklearn_scores)
    return t_statistic, p_value

def main():
    # datasets to process
    datasets = [
        'data/hayes_roth.csv',
        'data/car_evaluation.csv',
        'data/breast_cancer.csv'
    ]
    
    for dataset_path in datasets:
        print(f"\nAnalyzing Dataset: {dataset_path}")
        
        try:
            # load data
            data = load_dataset(dataset_path)
            if data is None:
                print(f"Skipping dataset: {dataset_path}")
                continue
                
            # preprocess data
            X, y = preprocess_data(data)
            smote = SMOTE()
            X, y = smote.fit_resample(X, y)
            
            if not isinstance(X, pd.DataFrame):
                feature_count = X.shape[1] if len(X.shape) > 1 else 1
                column_names = [f'feature_{i}' for i in range(feature_count)]
                X = pd.DataFrame(X, columns=column_names)

            if not isinstance(y, pd.Series):
                y = pd.Series(y)

            print(f"Final Data shape after preprocessing: X={X.shape}, y={y.shape}")

            # custom KNN Implementation
            cv_folds = min(2, len(set(y)), len(y))

            best_k = 3
            best_accuracy = 0

            # try different k values
            for k in [1, 3, 5, 7, 9]:
                custom_knn = KNearestNeighbors(k=k)
                custom_scores = k_fold_cross_validation(X, y, k=cv_folds, model=custom_knn)
                mean_accuracy = np.mean(custom_scores)
                
                print(f"K={k}, Custom KNN Accuracy: {mean_accuracy:.4f}")
                
                if mean_accuracy > best_accuracy:
                    best_accuracy = mean_accuracy
                    best_k = k

            print(f"Best k value found: {best_k}")
            custom_knn = KNearestNeighbors(k=best_k)  # Use best k
            custom_scores = k_fold_cross_validation(X, y, k=cv_folds, model=custom_knn)
                        

            # scikit-learn KNN Implementation
            sklearn_knn = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='manhattan')
            cv_folds = min(2, len(set(y))) 
            skf = StratifiedKFold(n_splits=cv_folds)
            sklearn_scores = cross_val_score(sklearn_knn, ensure_numpy(X), ensure_numpy(y), cv=skf)

            # statistical Comparison for t_stat & p_val
            t_stat, p_val = statistical_comparison(custom_scores, sklearn_scores)

            custom_knn_mean_acc = np.mean(custom_scores)
            sklearn_knn_mean_acc = np.mean(sklearn_scores)

            print(f"\nCustom KNN Mean Accuracy: {custom_knn_mean_acc:.4f}")
            print(f"Scikit-Learn KNN Mean Accuracy: {sklearn_knn_mean_acc:.4f}")
            t_stat, p_val = stats.ttest_rel(custom_scores, sklearn_scores)
            print(f"\nT-statistic: {t_stat:.10f}")
            print(f"P-value: {p_val:.10f}")

            if p_val < 0.05:
                print("There is a significant difference between the models.")
            else:
                print("No significant difference.")
            
        except Exception as e:
            print(f"Error processing dataset {dataset_path}: {str(e)}")
            import traceback
            print(traceback.format_exc())  

        print("-" * 50)

if __name__ == "__main__":
    main()