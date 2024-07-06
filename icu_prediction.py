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

# Load the dataset
data = pd.read_csv('C:/Users/User/.spyder-py3/Kaggle_Sirio_preprocessed.csv')

# Check the shape and columns of the data
print(data.shape)
print(data.columns)

# Define independent and dependent variables
X = data[list(data.columns)[:-1]].values
y = data[data.columns[-1]].values

# Check unique values in the target variable
print("Unique values in the target variable:", np.unique(y))

# Assuming the target variable needs to be discretized or encoded
# Let's use LabelEncoder to encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Check the unique values after encoding
print("Unique values in the encoded target variable:", np.unique(y_encoded))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.20, stratify=y_encoded, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(np.nan_to_num(X_train))
X_test = scaler.transform(np.nan_to_num(X_test))

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_jobs=-1, n_estimators=200, criterion='entropy', oob_score=True, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
acc_rf = metrics.accuracy_score(y_test, y_pred_rf)

print('Random Forest accuracy:', acc_rf)

# Use classification_report to get detailed metrics
#print(classification_report(y_test, y_pred_rf, zero_division=0))

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(np.unique(y_encoded))):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_pred_rf == i)
    roc_auc[i] = auc(fpr[i], tpr[i])
print('AUC:', roc_auc)

# XGBoost model tuning
params = {
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'max_depth': [3, 4, 5]
}

folds = 3
param_comb = 5
xgb = XGBClassifier(n_estimators=100, random_state=42)
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb,
                                   scoring='roc_auc_ovr', n_jobs=-1, cv=skf.split(X, y_encoded), verbose=3, random_state=42)  # Use 'roc_auc_ovr' for multiclass
random_search.fit(X, y_encoded)

# Train the final XGBoost model
xgb_best = random_search.best_estimator_
training_start = time.perf_counter()
xgb_best.fit(X_train, y_train)
training_end = time.perf_counter()
prediction_start = time.perf_counter()
preds = xgb_best.predict(X_test)
prediction_end = time.perf_counter()

# Calculate accuracy and timing
acc_xgb = metrics.accuracy_score(y_test, preds)
xgb_train_time = training_end - training_start
xgb_prediction_time = prediction_end - prediction_start

print("XGBoost's prediction accuracy is: %3.2f" % (acc_xgb))
print("Time consumed for training: %4.3f seconds" % (xgb_train_time))
print("Time consumed for prediction: %6.5f seconds" % (xgb_prediction_time))

# Save the final XGBoost model and scaler
joblib.dump(xgb_best, 'C:/Users/User/.spyder-py3/xgb_icu_prediction.pkl')
joblib.dump(scaler, 'C:/Users/User/.spyder-py3/scaler.pkl')