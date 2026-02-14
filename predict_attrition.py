import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

'''
Notes: 
	-This script assumes your continuous dataset is already preprocessed and encoded exactly like you used for modeling.
	-Adjust the example input (new_employee) to match your actual feature set.
	-You can later export lgb_model using joblib if you want to reuse the trained model without retraining each time.
'''

# --- Load Data ---
df = pd.read_csv("your_cleaned_continuous_dataset.csv")  # <- Replace with your file path

# --- Features & Target ---
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# --- Split into train, val, test ---
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=42)

# --- Impute missing values ---
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)
X_test_imputed = imputer.transform(X_test)

# --- SMOTE resampling ---
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_imputed, y_train)

# --- Train LightGBM model ---
lgb_model = LGBMClassifier(
    random_state=42,
    n_estimators=200,
    max_depth=-1,
    learning_rate=0.1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8
)
lgb_model.fit(X_train_sm, y_train_sm)

# --- Threshold tuning on validation set ---
val_probs = lgb_model.predict_proba(X_val_imputed)[:, 1]
val_preds = (val_probs >= 0.4).astype(int)

print("Validation Results (Threshold = 0.4):")
print(classification_report(y_val, val_preds))
print(f"ROC-AUC: {roc_auc_score(y_val, val_probs):.4f}")

# --- Final evaluation on test set ---
test_probs = lgb_model.predict_proba(X_test_imputed)[:, 1]
test_preds = (test_probs >= 0.4).astype(int)

print("\nTest Results:")
print(classification_report(y_test, test_preds))
print(f"ROC-AUC: {roc_auc_score(y_test, test_probs):.4f}")

# --- Example: Predict on New Data ---
# Replace with a new employee row (matching feature structure of your dataset)
new_employee = pd.DataFrame([{
    "ProjectsAssigned": 10,
    "ProjectsCompleted": 9,
    "PerformanceRating": 4,
    "Salary": 72000,
    "YearsAtCompany": 5,
    "YearsInCurrentRole": 3,
    "OvertimeHours": 12,
    "Absences": 3,
    "EmployeeEngagementScore": 4,
    "TeamSize": 7,
    "PromotionCount": 1,
    "WorkFromHomeDays": 2,
    "JobSatisfactionScore": 4,
    "SkillsAssessmentScore": 88.0,
    "WorkHoursPerWeek": 40,
    "StressLevelScore": 2,
    "HireYear": 2018,
    "Age": 29,
    "ProjectsPerYear": 2.0,
    "ProjectCompletionRate": 0.9,
    # include your encoded categorical features as needed
}])

new_employee_imputed = imputer.transform(new_employee)
new_prob = lgb_model.predict_proba(new_employee_imputed)[:, 1][0]
new_prediction = int(new_prob >= 0.4)

print(f"\nNew Employee Attrition Probability: {new_prob:.3f}")
print(f"Prediction (1 = Will Leave): {new_prediction}")