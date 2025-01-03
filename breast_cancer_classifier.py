import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    classification_report
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

class BreastCancerClassifier:
    def __init__(self, dataset_path, k):
        self.path = dataset_path
        self.raw_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.pred = None
        self.k = k
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and preprocess the dataset"""
        self.raw_data = pd.read_csv(self.path)
        self.raw_data['diagnosis'] = self.raw_data['diagnosis'].map({'M': 1, 'B': 0})
        self.raw_data.dropna(inplace=True)
        self.raw_data.drop_duplicates(inplace=True)
        
    def split_and_scale_data(self, test_size=0.2, random_state=10):
        """Split data into train/test sets and scale features"""
        features = self.raw_data.drop(['diagnosis'], axis=1)
        label = self.raw_data['diagnosis']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features, label, test_size=test_size, random_state=random_state
        )
        
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
    def evaluate_knn(self, metric, max_k):
        """Evaluate KNN with different k values using specified metric"""
        train_scores = []
        test_scores = []
        
        for k in range(1, max_k + 1):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(self.X_train, self.y_train)
            
            train_pred = knn.predict(self.X_train)
            test_pred = knn.predict(self.X_test)
            
            if metric == 'accuracy':
                score_func = accuracy_score
            elif metric == 'f1':
                score_func = f1_score
            
            train_scores.append(score_func(self.y_train, train_pred))
            test_scores.append(score_func(self.y_test, test_pred))
            self.differences = [abs(train - test) for train, test in zip(train_scores, test_scores)]
            
        df = pd.DataFrame({'Train Score': train_scores, 'Test Score': test_scores})
        df['differences'] = [abs(train - test) for train, test in zip(train_scores, test_scores)]       

        return df
    
    def find_best_k(self, train_scores, test_scores):
        """Find best k value based on smallest difference between train and test scores"""
        best_k = self.differences.index(min(self.differences)) + 1
        return best_k

    def cross_validate(self, k_range):
        """Perform cross-validation for different k values"""
        cv_scores = []
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, self.X_train, self.y_train, cv=10, scoring='accuracy')
            cv_scores.append(scores.mean())
        return cv_scores

    def plot_results(self, differences, cv_scores, k_range):
        """Plot evaluation results and keep the plots open"""
        plt.figure(figsize=(15, 5))
        
        # Plot differences
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(differences) + 1), differences, marker='o')
        plt.xlabel('Number of Neighbors (k)')
        plt.ylabel('Difference between Train and Test')
        plt.title('Train-Test Differences vs. k')
        
        # Plot cross-validation scores
        plt.subplot(1, 2, 2)
        plt.plot(k_range, cv_scores, marker='o')
        plt.xlabel('Number of Neighbors (k)')
        plt.ylabel('Cross-Validation Accuracy')
        plt.title('Cross-Validation Scores vs. k')
        
        plt.tight_layout()
        plt.show()

    def make_predictions(self):
        """Make predictions using specified k value"""
        self.knn = KNeighborsClassifier(n_neighbors=self.k)
        self.knn.fit(self.X_train, self.y_train)
        self.y_pred = self.knn.predict(self.X_test)
        return pd.Series(self.y_pred)
    
    def simulate_predictions(self, n_samples, n_splits):
        """Create and predict on simulated data"""
        # Create random features
        x = np.random.rand(n_samples)
        array = np.array_split(x, n_splits)
        
        # Get feature columns from original data
        feature_columns = self.raw_data.drop(['diagnosis'], axis=1).columns
        
        # Create DataFrame with same columns as original features
        features_sim = pd.DataFrame(array, columns=feature_columns)
        
        # Make predictions on simulated data
        y_pred_sim = self.knn.predict(self.X_test)  # Using previously fitted model
        return pd.Series(y_pred_sim), features_sim
    
    def metrics(self):
        """Evaluate how well the KNN model performs using specified k value"""
        accuracy = round(accuracy_score(self.y_test, self.y_pred), 3)
        precision = round(precision_score(self.y_test, self.y_pred, average='macro'), 3)
        recall = round(recall_score(self.y_test, self.y_pred, average='macro'), 3)
        f1 = round(f1_score(self.y_test, self.y_pred, average='macro'), 3)

        series = pd.Series([accuracy, precision, recall, f1])
        series.index = ['accuracy', 'precision', 'recall', 'F1 score'] 

        return series

    def conf_matrix(self):
        "Provide confusion matrix with the specified k value"
        conf_matrix = confusion_matrix(self.y_test, self.y_pred)
        return conf_matrix