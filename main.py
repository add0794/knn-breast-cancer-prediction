from breast_cancer_classifier import BreastCancerClassifier
import kagglehub
import os
import pandas as pd

# Download dataset
path = kagglehub.dataset_download("rahmasleam/breast-cancer")
dataset_path = os.path.join(path, 'breast-cancer.csv')

# Create and run classifier
classifier = BreastCancerClassifier(dataset_path, k=3)
classifier.load_data()
classifier.split_and_scale_data()

df_acc = classifier.evaluate_knn('accuracy', 50)
print(df_acc)
best_acc_k = classifier.find_best_k(df_acc, df_acc['Train Score'])
print(f'Using differences in accuracy between training and test sets, the optimal k value is {best_acc_k}')

df_f1 = classifier.evaluate_knn('f1', 50)
print(df_f1)
best_f1_k = classifier.find_best_k(df_f1, df_f1['Test Score'])
print(f'Using differences in FI score between training and test sets, the optimal k value is {best_f1_k}')

# Cross-validation
k_range = range(1, 50)
cv_scores = classifier.cross_validate(k_range)

# Make regular predictions
y_pred_series = classifier.make_predictions()
print("\nRegular Predictions:")
print(y_pred_series.head())

# Make simulated predictions
y_pred_series_sim, features_sim = classifier.simulate_predictions(n_samples=310, n_splits=10)
print("\nSimulated Features:")
print(features_sim.head())
print("\nSimulated Predictions:")
print(y_pred_series_sim.head())

print(classifier.metrics())
print(classifier.conf_matrix())

classifier.plot_results(df_acc['differences'], cv_scores, k_range)