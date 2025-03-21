import pandas as pd
import numpy as np
from collections import Counter
from scipy.sparse import coo_matrix
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB, ComplementNB
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from scipy.sparse import issparse


# Configuration and constants
NUM_FEATURES = 100001
CV_SPLITS = 5
K_BEST_VALUES = range(1, 10000, 1)
SEED = 42
SAMPLING_STRATEGY = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
K_NEIGHBORS = [5, 10, 20, 30, 40, 50 ,60, 70, 77]
ALPHA = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0] 

def data_loader(file_path, has_labels=True, num_features=NUM_FEATURES):
    """
    Loads data from file, parses features and labels, and returns a sparse matrix.
    """
    df = pd.read_csv(file_path, header=None, names=['line'])
    df['tokens'] = df['line'].str.strip().str.split()
    df['label'] = df['tokens'].apply(lambda x: int(x[0]) if has_labels else None)
    df['features'] = df['tokens'].apply(lambda x: list(set(map(int, x[1:] if has_labels else x))))

    data, rows, columns = [], [], []
    for i, feature_list in enumerate(df['features']):
        data.extend([1] * len(feature_list))
        rows.extend([i] * len(feature_list))
        columns.extend(feature_list)

    X_sparse = coo_matrix((data, (rows, columns)), shape=(len(df), num_features))
    labels = df['label'].to_numpy() if has_labels else None
    return X_sparse, labels

def feature_selection(X_train, X_test, threshold=0.01):
    """
    Applies VarianceThreshold feature selection.
    """
    VT_selector = VarianceThreshold(threshold=threshold)
    X_train_selected = VT_selector.fit_transform(X_train)
    X_test_selected = VT_selector.transform(X_test)
    return X_train_selected, X_test_selected

def evaluate_models(X, y, X_test, resampling_methods, models, k_best_values):
    """
    Evaluates models with various resampling methods and SelectKBest feature selection.
    """
    results = []
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)
    
    for resampler_name, resampler in resampling_methods.items():
        X_resampled, y_resampled = resampler.fit_resample(X, y)
        
        for k in k_best_values:
            selector_kbest = SelectKBest(chi2, k=k)
            X_kbest = selector_kbest.fit_transform(X_resampled, y_resampled)
            X_test_kbest = selector_kbest.transform(X_test)

            for model_name, model in models.items():
                # Convert to dense if using GaussianNB
                if isinstance(model, GaussianNB) and issparse(X_kbest):
                    X_kbest = X_kbest.toarray()
                    X_test_kbest = X_test_kbest.toarray()
                
                cv_scores = cross_val_score(model, X_kbest, y_resampled, cv=cv, scoring='f1_macro')
                results.append({
                    'resampler': resampler_name,
                    'model': model_name,
                    'k_best': k,
                    'cv_f1_score': np.mean(cv_scores)
                })

    return pd.DataFrame(results)


def resample_and_select(X, y, k):
    """
    Resamples the dataset and selects top k features using SelectKBest.
    """
    sampler = RandomUnderSampler(random_state=SEED)
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    selector_kbest = SelectKBest(chi2, k=k)
    X_train_kbest = selector_kbest.fit_transform(X_resampled, y_resampled)
    return X_train_kbest, y_resampled, selector_kbest

def main():
    # Load and preprocess data
    X_train, y_train = data_loader("/scratch/mbiswas2/CS 584/Assignment 02/train.txt", has_labels=True, num_features=NUM_FEATURES)
    X_test, _ = data_loader("/scratch/mbiswas2/CS 584/Assignment 02/test.txt", has_labels=False, num_features=NUM_FEATURES)
    X_train_selected, X_test_selected = feature_selection(X_train, X_test)

    # Define resampling techniques and models
    resampling_methods = {
        'RandomUnderSampler': RandomUnderSampler(random_state=SEED),
        'SMOTEENN': SMOTEENN(random_state=SEED),
        'SMOTE': SMOTE(random_state=SEED)
    }
    models = {
        'BernoulliNB': BernoulliNB(),
        'MultinomialNB': MultinomialNB(),
        'Decision Tree': DecisionTreeClassifier(class_weight='balanced'),
        'ComplementNB':ComplementNB(),
        'GaussianNB':GaussianNB()
    }

    # # Evaluate models
    # results = evaluate_models(X_train_selected, y_train, X_test_selected, resampling_methods, models, K_BEST_VALUES)
    # best_result = results.loc[results['cv_f1_score'].idxmax()]
    # print("\nBest Parameter Combination:")
    # print(best_result)

    # Further processing with BernoulliNB on the selected top k features
    X_train_kbest, y_train_resampled, selector_kbest = resample_and_select(X_train_selected, y_train, k=613)
    X_test_kbest = selector_kbest.transform(X_test_selected)

    # # Train classifier and evaluate
    classifier = BernoulliNB()
    # cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)
    # cv_scores = cross_val_score(classifier, X_train_kbest, y_train_resampled, cv=cv, scoring='accuracy')
    # print("Cross-validation accuracy scores:", cv_scores)
    # print("Average cross-validation accuracy:", np.mean(cv_scores))

    # Fit classifier and make predictions
    classifier.fit(X_train_kbest, y_train_resampled)
    test_predictions = classifier.predict(X_test_kbest)

    # Save predictions to file
    with open('result_nb_chi_updated.dat', 'w') as prediction_file:
        for prediction in test_predictions:
            prediction_file.write(f"{prediction}\n")
    print("Predictions saved to result_nb_chi_updated.dat.")

if __name__ == "__main__":
    main()
