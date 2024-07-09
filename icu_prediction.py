import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve, auc, classification_report
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from xgboost import XGBClassifier
import joblib
import mongodb_helper as mh

# Load the dataset
dataset = pd.read_csv('C:/Users/User/.spyder-py3/Kaggle_Sirio_preprocessed.csv')

# Check the shape and columns of the data
print(dataset.shape)
print(dataset.columns)

# Define independent and dependent variables
X = dataset.drop(columns=['ICU']).values
y = dataset['ICU'].values

# Check unique values in the target variable
print("Unique values in the target variable:", np.unique(y))

# Assuming the target variable needs to be discretized or encoded
# Let's use LabelEncoder to encode the target variable
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

# Check the unique values after encoding
print("Unique values in the encoded target variable:", np.unique(y_encoded))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.20, stratify=y_encoded, random_state=42)

# Scale the features
feature_scaler = StandardScaler()
X_train = feature_scaler.fit_transform(np.nan_to_num(X_train))
X_test = feature_scaler.transform(np.nan_to_num(X_test))

# Train a Random Forest Classifier
random_forests_classifier = RandomForestClassifier(n_jobs=-1, n_estimators=200, criterion='entropy', oob_score=True, random_state=42)
random_forests_classifier.fit(X_train, y_train)
random_forests_predictions = random_forests_classifier.predict(X_test)
random_forests_accuracy = metrics.accuracy_score(y_test, random_forests_predictions)

print('Random Forests accuracy:', random_forests_accuracy)

# Use classification_report to get detailed metrics
#print(classification_report(y_test, random_forests_predictions))

# Calculate ROC AUC for each class
roc_auc = dict()
fpr = dict()
tpr = dict()
for i in range(len(np.unique(y_encoded))):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, random_forests_predictions == i)
    roc_auc[i] = auc(fpr[i], tpr[i])
print('AUC:', roc_auc)

# XGBoost model tuning
xgboost_params = {
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'max_depth': [3, 4, 5]
}

folds = 3
param_comb = 5
xgboost_model = XGBClassifier(n_estimators=100, random_state=42)
stratified_kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(xgboost_model, param_distributions=xgboost_params, n_iter=param_comb,
                                   scoring='roc_auc_ovr', n_jobs=-1, cv=stratified_kfold.split(X, y_encoded), verbose=3, random_state=42)
random_search.fit(X, y_encoded)

# Train the final XGBoost model
best_xgboost_model = random_search.best_estimator_
training_start = time.perf_counter()
best_xgboost_model.fit(X_train, y_train)
training_end = time.perf_counter()
prediction_start = time.perf_counter()
xgboost_predictions = best_xgboost_model.predict(X_test)
prediction_end = time.perf_counter()

# Calculate accuracy and timing
xgboost_accuracy = metrics.accuracy_score(y_test, xgboost_predictions)
xgboost_train_time = training_end - training_start
xgboost_prediction_time = prediction_end - prediction_start

print("XGBoost's prediction accuracy is: %3.2f" % (xgboost_accuracy))
print("Time consumed for training: %4.3f seconds" % (xgboost_train_time))
print("Time consumed for prediction: %6.5f seconds" % (xgboost_prediction_time))

# Save the final XGBoost model and scaler
joblib.dump(best_xgboost_model, 'xgboost_icu_prediction.pkl')
joblib.dump(feature_scaler, 'scaler.pkl')
joblib.dump(dataset.drop(columns=['ICU']).columns, 'feature_columns.pkl')  # Save the feature columns

mh.save_model_to_db(best_xgboost_model, 'predict_icu_xgboost', 'xgboost_icu_prediction.pkl')
mh.save_model_to_db(feature_scaler, 'predict_icu_xgboost', 'scaler.pkl')
mh.save_model_to_db(dataset.drop(columns=['ICU']).columns, 'predict_icu_xgboost', 'feature_columns.pkl')
