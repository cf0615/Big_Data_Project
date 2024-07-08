import pandas as pd
from sklearn.utils import resample
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay, classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib
from sklearn.calibration import CalibratedClassifierCV

# Load the dataset
file_path = 'C:/Users/shenhao/OneDrive/Inti/Degree/Sem 6/Big Data/dataset/Preprocessed/covidDataPreprocessed.csv'
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

# Extract feature importances
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display feature importances
plt.figure(figsize=(10, 8))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
plt.gca().invert_yaxis()
plt.show()

# Select the most important features (e.g., top 10)
important_features = feature_importance_df['Feature'].head(10).tolist()

# Use only the important features
X_train_important = X_train[important_features]
X_test_important = X_test[important_features]

# Train a new XGBoost model with the selected important features
model_important = xgb.XGBClassifier(n_estimators=50, max_depth=3, eval_metric='logloss')
model_important.fit(X_train_important, y_train)

# Calibrate the model using Platt Scaling
calibrated_model_important = CalibratedClassifierCV(model_important, method='sigmoid', cv='prefit')
calibrated_model_important.fit(X_train_important, y_train)

# Make predictions
y_pred_important = calibrated_model_important.predict(X_test_important)
y_pred_proba_important = calibrated_model_important.predict_proba(X_test_important)[:, 1]

# Confusion matrix
cm_important = confusion_matrix(y_test, y_pred_important)
disp_important = ConfusionMatrixDisplay(confusion_matrix=cm_important)
disp_important.plot()

# ROC curve
fpr_important, tpr_important, _ = roc_curve(y_test, y_pred_proba_important)
roc_auc_important = auc(fpr_important, tpr_important)

plt.figure()
plt.plot(fpr_important, tpr_important, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_important)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Model performance metrics
accuracy_important = accuracy_score(y_test, y_pred_important)
precision_important = precision_score(y_test, y_pred_important)
recall_important = recall_score(y_test, y_pred_important)
f1_important = f1_score(y_test, y_pred_important)
specificity_important = cm_important[0, 0] / (cm_important[0, 0] + cm_important[0, 1])

print("Model Performance Metrics with Important Features:")
print(f"Accuracy: {accuracy_important:.4f}")
print(f"Precision: {precision_important:.4f}")
print(f"Recall: {recall_important:.4f}")
print(f"F1 Score: {f1_important:.4f}")
print(f"Specificity: {specificity_important:.4f}")

# Classification report
report_important = classification_report(y_test, y_pred_important)
print("\nClassification Report with Important Features:")
print(report_important)

# Export the model
joblib_file = 'C:/Users/shenhao/OneDrive/Inti/Degree/Sem 6/Big Data/xgboost_covid_model_important.pkl'
joblib.dump(calibrated_model_important, joblib_file)


# Function to compare a sample from the dataset with the calibrated model's prediction using a custom threshold
def verify_sample_with_threshold_calibrated_important(sample_index, threshold=0.5):
    sample = df.iloc[sample_index].drop('RESULT')
    actual_result = df.iloc[sample_index]['RESULT']
    
    # Use only the important features
    sample_important = sample[important_features]
    
    # Convert sample to DataFrame
    sample_df_important = pd.DataFrame([sample_important])
    
    # Make prediction with custom threshold using calibrated model
    prediction_proba = calibrated_model_important.predict_proba(sample_df_important)
    prediction = (prediction_proba[:, 1] >= threshold).astype(int)
    
    print(f"Sample Index: {sample_index}")
    print(f"Input Features:\n{sample_important}")
    print(f"Actual Result: {'Death' if actual_result == 1 else 'Survival'}")
    print(f"Calibrated Model Prediction with threshold {threshold}: {'Death' if prediction[0] == 1 else 'Survival'}")
    print(f"Prediction Probability: {prediction_proba[0][1] * 100:.2f}% chance of death")
    print()

# Verify a specific sample from the dataset with a custom threshold using the calibrated model
verify_sample_with_threshold_calibrated_important(35, threshold=0.3)
