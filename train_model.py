
# #working
# import joblib
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.compose import make_column_transformer
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
# from imblearn.over_sampling import SMOTE
# from scipy.stats import randint
# import time

# # Load the synthetic data
# df = pd.read_csv('synthetic_data_encoded.csv')
# X = df.drop(['abuseConfidenceScore', 'abuseConfidenceClass'], axis=1)
# y = df['abuseConfidenceClass']

# # Apply SMOTE to handle class imbalance
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X, y)

# # Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# # Define the preprocessor
# preprocessor = make_column_transformer(
#     (StandardScaler(), X.columns),
#     remainder='passthrough'
# )

# # Define a pipeline with the preprocessor and the RandomForest Classifier
# pipeline = Pipeline([
#     ('preprocessor', preprocessor),  # Preprocess the features
#     ('classifier', RandomForestClassifier())
# ])

# # Hyperparameter tuning for RandomForestClassifier using RandomizedSearchCV
# param_dist = {
#     'classifier__n_estimators': randint(100, 150),
#     'classifier__max_depth': randint(3, 5),
#     'classifier__min_samples_split': randint(2, 5),
#     'classifier__min_samples_leaf': randint(1, 5)
# }

# # Use StratifiedKFold for cross-validation
# cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# # Measure the start time
# start_time = time.time()

# # Perform RandomizedSearchCV
# random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=5, cv=cv, scoring='f1_weighted', n_jobs=-1, random_state=42)
# random_search.fit(X_train, y_train)
# best_model = random_search.best_estimator_

# # Measure the end time
# end_time = time.time()
# training_time = end_time - start_time
# print(f"Training Time: {training_time:.2f} seconds")

# # Predictions on the test set
# y_pred = best_model.predict(X_test)

# # Calculate metrics
# accuracy = accuracy_score(y_test, y_pred) * 100
# precision = precision_score(y_test, y_pred, average='weighted') * 100
# recall = recall_score(y_test, y_pred, average='weighted') * 100
# f1 = f1_score(y_test, y_pred, average='weighted') * 100
# roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]) * 100

# print(f"Accuracy: {accuracy:.2f}%")
# print(f"Precision: {precision:.2f}%")
# print(f"Recall: {recall:.2f}%")
# print(f"F1 Score: {f1:.2f}%")
# print(f"ROC AUC Score: {roc_auc:.2f}%")

# # Print classification report
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred, target_names=['Non-Abusive', 'Abusive']))

# # Save the model and preprocessor
# joblib.dump(best_model, 'model/risk_model.pkl')
# print("Model and preprocessor trained and saved as 'risk_model.pkl'")


#!/usr/bin/env python
#!/usr/bin/env python
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report, matthews_corrcoef)
from imblearn.over_sampling import SMOTE
# import random

# # Set random seeds for reproducibility
# np.random.seed(42)
# random.seed(42)

# # Load the synthetic encoded data
# df = pd.read_csv('synthetic_data_encoded.csv')

# # Features: drop the target columns.
# X = df.drop(['abuseConfidenceScore', 'abuseConfidenceClass'], axis=1)
# y = df['abuseConfidenceClass']

# # Apply SMOTE to handle any imbalance (even though we manually balanced earlier)
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X, y)

# # Split the data into training and test sets.
# X_train, X_test, y_train, y_test = train_test_split(
#     X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
# )

# # Define the preprocessor; here we standardize all features.
# preprocessor = make_column_transformer(
#     (StandardScaler(), X.columns),
#     remainder='passthrough'
# )

# # Define a pipeline with the preprocessor and the RandomForest Classifier.
# pipeline = Pipeline([
#     ('preprocessor', preprocessor),
#     ('classifier', RandomForestClassifier(random_state=42))
# ])

# # Hyperparameter tuning for RandomForestClassifier using GridSearchCV.
# param_grid = {
#     'classifier__n_estimators': [100, 120, 150],
#     'classifier__max_depth': [3, 4, 5],
#     'classifier__min_samples_split': [2, 3, 4],
#     'classifier__min_samples_leaf': [1, 2, 3]
# }

# # Use StratifiedKFold for cross-validation.
# cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# # Perform GridSearchCV.
# grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
# grid_search.fit(X_train, y_train)
# best_model = grid_search.best_estimator_

# print("Best Hyperparameters:", grid_search.best_params_)

# # Predictions on the test set.
# y_pred = best_model.predict(X_test)

# # Calculate and print metrics.
# accuracy = accuracy_score(y_test, y_pred) * 100
# precision = precision_score(y_test, y_pred, average='weighted') * 100
# recall = recall_score(y_test, y_pred, average='weighted') * 100
# f1 = f1_score(y_test, y_pred, average='weighted') * 100
# roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]) * 100
# mcc = matthews_corrcoef(y_test, y_pred) * 100

# print(f"Accuracy: {accuracy:.2f}%")
# print(f"Precision: {precision:.2f}%")
# print(f"Recall: {recall:.2f}%")
# print(f"F1 Score: {f1:.2f}%")
# print(f"ROC AUC Score: {roc_auc:.2f}%")
# print(f"Matthews Correlation Coefficient: {mcc:.2f}%")

# print("\nClassification Report:")
# print(classification_report(y_test, y_pred, target_names=['Non-Abusive', 'Abusive']))

# # Ensure the model directory exists.
# os.makedirs("model", exist_ok=True)

# # Save the complete pipeline (including the preprocessor).
# joblib.dump(best_model, 'model/risk_model.pkl')
# print("Model and preprocessor trained and saved as 'model/risk_model.pkl'")


import os
import random
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    classification_report
)

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Load the synthetic encoded data
df = pd.read_csv('synthetic_data_encoded.csv')

# Features: drop the target columns.
X = df.drop(['abuseConfidenceScore', 'abuseConfidenceClass'], axis=1)
y = df['abuseConfidenceClass']

# Apply SMOTE to handle any imbalance.
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the data into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Define the preprocessor; here we standardize all features.
preprocessor = make_column_transformer(
    (StandardScaler(), X.columns),
    remainder='passthrough'
)

# Define a pipeline with the preprocessor and the RandomForest Classifier.
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Hyperparameter tuning for RandomForestClassifier using GridSearchCV.
param_grid = {
    'classifier__n_estimators': [100, 120, 150],
    'classifier__max_depth': [3, 4, 5],
    'classifier__min_samples_split': [2, 3, 4],
    'classifier__min_samples_leaf': [1, 2, 3]
}

# Use StratifiedKFold for cross-validation.
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Perform GridSearchCV.
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print("Best Hyperparameters:", grid_search.best_params_)

# Predictions on the test set.
y_pred = best_model.predict(X_test)

# Calculate and print metrics.
accuracy = accuracy_score(y_test, y_pred) * 100
precision = precision_score(y_test, y_pred, average='weighted') * 100
recall = recall_score(y_test, y_pred, average='weighted') * 100
f1 = f1_score(y_test, y_pred, average='weighted') * 100
roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]) * 100
mcc = matthews_corrcoef(y_test, y_pred) * 100

print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"F1 Score: {f1:.2f}%")
print(f"ROC AUC Score: {roc_auc:.2f}%")
print(f"Matthews Correlation Coefficient: {mcc:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Non-Abusive', 'Abusive']))

# Ensure the model directory exists.
os.makedirs("model", exist_ok=True)

# Save the complete pipeline (including the preprocessor).
joblib.dump(best_model, 'model/risk_model.pkl')
print("Model and preprocessor trained and saved as 'model/risk_model.pkl'")
