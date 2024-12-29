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

# Run analysis on accuracy and f1 scores
train_acc, test_acc = classifier.evaluate_knn(metric='accuracy', max_k=50)
best_k_acc, differences_acc = classifier.find_best_k(train_acc, test_acc)
acc = pd.DataFrame()
acc['train_acc'] = train_acc
acc['test_acc'] = test_acc
acc['differences_acc'] = differences_acc
print(acc.head())

train_f1, test_f1 = classifier.evaluate_knn(metric='f1', max_k=50)
best_k_f1, differences_f1 = classifier.find_best_k(train_f1, test_f1)
f1 = pd.DataFrame()
f1['train_f1'] = train_f1
f1['test_f1'] = test_f1
f1['differences_f1'] = differences_f1
print(f1.head())

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

classifier.plot_results(differences_acc, cv_scores, k_range)