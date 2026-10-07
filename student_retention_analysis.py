"""
Running analysis on student retention data taken from a community college. The data has the following fields: Term, Credit_Load, Full_Time, Term_GPA, Advising_Appointments, No_Shows, Alerts, Retained_Next_Semester, Major_Category, Cumul_GPA, Gender, Age
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
    ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
import joblib

#Load dataset
#dataset not included in repository due to student privacy considerations
df = pd.read_excel("Data.xlsx") 

#Correcting column names
df = df.rename(columns={"Major Category": "Major_Category", "Cumul GPA ":"Cumul_GPA", "Retained_Next_Semester":"Retained"})

#creating bins for term gpa, cumul gpa, and age
gpa_bins = [0.0, 0.49, 0.9, 1.49, 1.99, 2.49, 2.99, 3.49, 3.99, 4.0]


gpa_labels = [
    "0-0.49",
    "0.5-0.9",
    "1-1.49",
    "1.5-1.99",
    "2-2.49",
    "2.5-2.99",
    "3-3.49",
    "3.5-3.99",
    "4.0"
]

df["Cumul_GPA_bin"] = pd.cut(
    df["Cumul_GPA"],
    bins=gpa_bins,
    labels=gpa_labels,
    include_lowest=True
)
df["Term_GPA_bin"] = pd.cut(
    df["Term_GPA"],
    bins=gpa_bins,
    labels=gpa_labels,
    include_lowest=True
)

age_bins = [18, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74]


age_labels = [
    "18-19",
    "20-24",
    "25-29",
    "30-34",
    "35-39",
    "40-44",
    "45-49",
    "50-54",
    "55-59",
    "60-64",
    "65-69",
    "70-74"
]

df["Age_bin"] = pd.cut(
    df["Age"],
    bins=age_bins,
    labels=age_labels,
    include_lowest=True
)


df["Gender"] = df["Gender"].astype(str)
df["Age_bin"] = df["Age_bin"].astype(str)
df["Cumul_GPA_bin"] = df["Cumul_GPA_bin"].astype(str)
df["Term_GPA_bin"] = df["Term_GPA_bin"].astype(str)
df["Full_Time"] = df["Full_Time"].astype(str)
df["Major_Category"] = df["Major_Category"].astype(str)

#Define features and target
target = "Retained"
features = [
    "Cumul_GPA",
    "Term_GPA",
    "Gender",
    "Age_bin",
    "Cumul_GPA_bin",
    "Advising_Appointments",
    "Alerts",
    "Credit_Load",
    "Full_Time",
    "Term_GPA_bin",
    "No_Shows",
    "Major_Category"
]
df = df.dropna(subset=features)
X = df[features]
y = df[target]

#Create splits for training/testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

#Oversample training data only
ros = RandomOverSampler(random_state=42)
X_train, y_train = ros.fit_resample(X_train, y_train)

#Identify categorical columns
categorical_cols = ["Gender", "Age_bin", "Cumul_GPA_bin", "Term_GPA_bin", "Full_Time", "Major_Category"]
numeric_cols = [col for col in features if col not in categorical_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical_cols),
        ("num", "passthrough", numeric_cols)])

model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("logreg", LogisticRegression(max_iter=1000, C = 0.01))
])


model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    cmap="Blues"
    )
plt.title("Student Retention Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

print("ROC AUC Score:", roc_auc_score(y_test, y_prob))
RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("Student Retention ROC Curve")
plt.savefig("roc_curve.png")
plt.show()

#Extract logistic regression formula

# Get the trained logistic regression model inside the pipeline
logreg = model.named_steps["logreg"]

# Get the one-hot encoder to retrieve feature names
encoder = model.named_steps["preprocess"].named_transformers_["cat"]

# Get encoded categorical feature names
encoded_cat_features = encoder.get_feature_names_out(categorical_cols)

# Combine encoded categorical + numeric feature names
all_features = list(encoded_cat_features) + numeric_cols

# Extract coefficients and intercept
coefficients = logreg.coef_[0]
intercept = logreg.intercept_[0]

print("\n=== Logistic Regression Formula ===\n")
print(f"Intercept: {intercept}\n")

for feature, coef in zip(all_features, coefficients):
    print(f"{feature}: {coef}")

print("\nFull logistic regression equation:")
print("logit(p) = intercept + Σ(coef_i * x_i)")
print("p = 1 / (1 + exp(-logit(p)))")

joblib.dump(model, "retention_model.pkl")
