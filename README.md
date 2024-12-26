# Breast Cancer Prediction with KNN

This project aims to make predicting breast cancer easier. Unlike a typical [approach](https://pmc.ncbi.nlm.nih.gov/articles/PMC4916348/) to choosing for the $k$ in K-nearest neighbors, the hyperparameter is derived by finding the $k$ with the smallest difference in accuracy (and F1 score!) between the training and test set. Not surprisingly, that value is equal to the one derived by cross validation. KNN is then used to predict future values, and an evaluation (e.g. precision, recall, confusion matrix) is done. 
