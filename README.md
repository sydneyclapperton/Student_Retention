# Student Retention Prediction Dashboard

This project builds a machine learning model to predict student retention using institutional data and presents the results through an interactive Streamlit dashboard. The app provides retention probability, risk factors, and protective factors based on GPA, alerts, advising engagement, credit load, and demographic information.

---

## Project Overview
Student retention is a critical metric for colleges and universities. This project uses logistic regression to identify patterns associated with retention and provides an explainable interface for advisors, faculty, and administrators.

The goal is to:
- Identify students who may be at risk of not returning
- Highlight actionable risk factors
- Support early intervention and advising strategies
- Demonstrate applied machine learning skills in a real-world context

The final model achieved:

- 72% Accuracy
- 0.764 ROC-AUC

These results demonstrate moderate predictive power for identifying students who may be at risk of not returning the following semester. Logistic Regression outperformed a Random Forest benchmark model during testing and was selected as the final model for deployment.

The model was deployed through a Streamlit application, allowing users to generate retention predictions through an interactive interface.

## Dataset
This project was built using 13,567 anonymized student-term observations collected from a Student Information System (SIS) and EAB Navigate.
To protect student privacy, all student identifiers and institutional identifiers were removed prior to analysis.
Variables included:
- Cumulative GPA
- Term GPA
- Credit Load
- Full-Time Status
- Advising Appointments
- No-Shows
- Alerts
- Age Group
- Gender
- Major Category
- Retained Next Semester

## Methodology

The final model pipeline consisted of:

1. Data cleaning and preprocessing
2. GPA and age-group feature engineering
3. One-hot encoding of categorical variables
4. Train/test split with stratification
5. Random oversampling of the minority class
6. Logistic regression model training
7. Evaluation using classification metrics, confusion matrix, and ROC-AUC

Alternative models including Random Forest were evaluated, with Logistic Regression demonstrating the strongest overall performance on the test dataset.

## Streamlit App
The interactive dashboard allows users to:
- Input student information through sidebar controls
- Generate a retention probability using a trained ML model
- View risk factors and protective factors in clean, advisor-friendly tables
- Understand *why* the model made its prediction

The app is designed for deployment on **Streamlit Cloud**.

[Launch the Streamlit App](https://studentretention-prh2cmrdhpqjjap9ng39cd.streamlit.app/) 
## Application Preview
### Student Retention Prediction Tool
![Student Retention Prediction App Preview](./images/Student_Retention_App_Preview.png)
---

## Model Details
A logistic regression model was trained using institutional student data.  
Key features include:

- Cumulative GPA
- Term GPA
- Advising Appointments
- Alerts
- No-Shows
- Credit Load
- Major Category
- Age Group (binned)
- Gender
- Full-Time Status (calculated internally from credit load)

The model is serialized using `joblib` and loaded directly inside the Streamlit app.

## Exploratory Data Analysis & Visualizations
The following visualizations were developed to explore factors associated with student retention and to identify patterns later incorporated into the predictive model.
### Overall Retention Rate
![Overall Retention](./images/Overall_Retention.png)

**Finding:** Approximately 66% of student-term observations were retained to the following primary semester.

---
### Retention by Cumulative GPA and Full-Time Status
![Retention by Cumuluative GPA and Full-time](./images/Retention_by_Cumul_GPA_and_FT.png)

**Finding:** Retention increased substantially as cumulative GPA increased. Full-time students consistently demonstrated higher retention rates than part-time students across nearly all GPA ranges.

---
### Retention by Credit Load
![Retention by Credit Load](./images/Retention_by_Credit_Load.png)

**Finding:** Students enrolled in higher credit loads generally demonstrated stronger retention outcomes.

---

##  Technologies Used
- Python
- pandas
- numpy
- scikit-learn
- Streamlit
- joblib
- Excel (data cleaning and preprocessing)
- Tableau (data visualization and dashboard development)

## Model Performance

The final logistic regression model was evaluated using an untouched test dataset after addressing class imbalance through RandomOverSampler.

### Performance Metrics

- Accuracy: 72%
- ROC-AUC: 0.764

Classification results indicate the model can moderately distinguish between retained and non-retained students and may be useful for identifying students who could benefit from proactive support interventions.

### Confusion Matrix

![Confusion Matrix](./images/confusion_matrix.png)

### ROC Curve

![ROC Curve](./images/roc_curve.png)
  
## Feature Analysis

To better understand the factors influencing retention predictions, logistic regression coefficients were examined.

![Feature Chart](./images/Feature_Importance.png)

### Key Findings

- Higher Term GPA was the strongest positive predictor of retention.
- Students who attended more advising appointments demonstrated higher retention likelihood.
- No-show behavior was among the strongest negative predictors of retention.
- Students without a declared academic program were less likely to be retained.
- Higher cumulative GPA and credit load were associated with increased retention probability.

These findings improve model interpretability and highlight student characteristics and behaviors most strongly associated with retention outcomes.

## Future Enhancements:
- Interactive Tableau dashboards
- Additional exploratory visualizations
- Enhanced project documentation

Screenshots and links to Tableau dashboards will be added as this project continues to evolve.

I will continue updating this repository as new features and visualizations are completed. 

## Environment
Python 3.11
Dependencies are listed in requirements.txt.

## Contact
Created by Sydney Clapperton  
For questions or collaboration, feel free to reach out via GitHub.
