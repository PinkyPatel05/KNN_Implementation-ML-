# K-Nearest Neighbors Implementation and K-Fold Cross_Validation

This project implements the K-Nearest Neighbors (KNN) algorithm from scratch, along with k-fold cross-validation functionality. The implementation is compared against scikit-learn's KNN implementation using hypothesis testing on three different datasets.

## File Structure

100225188_ML_ProgramingAssignment/
│
├── data/
│   ├── hayes_roth.csv  --ayes-Roth Dataset
│   ├── car_evaluation.csv  --Car Evaluation Dataset
│   └── breast_cancer.csv  --Breast Cancer Dataset
│
├── src/
│   ├── __init__.py
│   ├── knn.py  --Custom KNN implementation
│   ├── cross_validation.py  --Custom k-fold cross-validation implementation
│   └── data_preprocessing.py  --Functions for preprocessing datasets
│
├── tests/
│   ├── __init__.py
│   ├── test_knn.py
│   └── test_cross_validation.py
│
├── requirements.txt
├── README.md
└── main.py  --Main execution script that processes datasets, runs comparisons, and displays results

## Requirements

- Python 3.7+
- NumPy
- Pandas
- scikit-learn
- imbalanced-learn (for SMOTE)
- SciPy (for statistical tests)

Install the required packages using:

pip install numpy pandas scikit-learn imbalanced-learn scipy

## How to Run

1. Make sure all required datasets are present in the `data/` directory:
   - `hayes_roth.csv`
   - `car_evaluation.csv`
   - `breast_cancer.csv`

2. Run the main script:

python3 main.py


The program will:
1. Load each dataset
2. Preprocess the data (handle categorical features, missing values, scaling)
3. Apply SMOTE to balance class distributions
4. Implement KNN with different k values (1, 3, 5, 7, 9) to find the optimal k
5. Perform k-fold cross-validation with both custom KNN and scikit-learn's KNN
6. Compare results using statistical hypothesis testing (t-test)
7. Report accuracy metrics and whether there's a significant difference between implementations

## Implementation Details

### Custom KNN Algorithm

- Euclidean distance metric to find nearest neighbors
- Support for both regression and classification tasks
- Proper handling of categorical features through preprocessing

### Cross-Validation

- Implementation of stratified k-fold cross-validation
- Support for different random seeds
- Proper handling of smaller datasets with few samples per class

### Data Preprocessing

- Automatic detection of categorical and numerical features
- Imputation of missing values
- Standardization of numerical features
- One-hot encoding of categorical features
- Class balancing using SMOTE

## Extensions Implemented

The KNN implementation includes several enhancements beyond the basic algorithm:

1. Support for different distance metrics (Euclidean and Manhattan)
2. Optimized k value selection through cross-validation
3. Data preprocessing pipeline with automatic feature type detection
4. Statistical comparison using paired t-tests for hypothesis testing
5. Class imbalance handling through SMOTE

## Results

The program outputs:
- Accuracy scores for both custom KNN and scikit-learn's KNN implementation
- Statistical comparison results including t-statistic and p-value
- Determination of whether there is a significant difference between implementations
