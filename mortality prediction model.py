import pandas as pd
from sklearn.utils import resample
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay, classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib
from sklearn.calibration import CalibratedClassifierCV
import mongodb_helper as mh

# Load the dataset
file_path = 'C:/Users/shenhao/Downloads/Big_Data_Project-main/Big_Data_Project-main/Dataset/covidDataPreprocessed.csv'
df = pd.read_csv(file_path)

# Separate the majority and minority classes
df_majority = df[df.RESULT == 0]
df_minority = df[df.RESULT == 1]

# Undersample the majority class
df_majority_undersampled = resample(df_majority, 
                                    replace=False,    # sample without replacement
                                    n_samples=len(df_minority),  # to match minority class
                                    random_state=42)  # reproducible results

# Combine minority class with undersampled majority class
df_balanced = pd.concat([df_minority, df_majority_undersampled])

# Display the class distribution before and after balancing
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
df.RESULT.value_counts().plot(kind='bar', ax=ax[0], title='Before Balancing')
df_balanced.RESULT.value_counts().plot(kind='bar', ax=ax[1], title='After Balancing')
plt.show()

# Split the data into features and target variable
X = df_balanced.drop(columns=['RESULT'])
y = df_balanced['RESULT']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model with fewer boosting rounds and limited depth
model = xgb.XGBClassifier(n_estimators=50, max_depth=3, eval_metric='logloss')
model.fit(X_train, y_train)

# Calibrate the model using Platt Scaling
calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
calibrated_model.fit(X_train, y_train)

# Make predictions
y_pred = calibrated_model.predict(X_test)
y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Model performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

print("Model Performance Metrics with All Features:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Specificity: {specificity:.4f}")

# Classification report
report = classification_report(y_test, y_pred)
print("\nClassification Report with All Features:")
print(report)

# Export the model
joblib_file = 'xgboost_covid_model.pkl'

mh.save_model_to_db(calibrated_model, 'predict_mortality_xgb', 'xgboost_covid_model.pkl')
