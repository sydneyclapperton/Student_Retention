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


## Streamlit App
The interactive dashboard allows users to:
- Input student information through sidebar controls
- Generate a retention probability using a trained ML model
- View risk factors and protective factors in clean, advisor-friendly tables
- Understand *why* the model made its prediction

The app is designed for deployment on **Streamlit Cloud**.

https://studentretention-prh2cmrdhpqjjap9ng39cd.streamlit.app/

---

## 🧠 Model Details
A logistic regression model was trained using institutional student data.  
Key features include:

- **Cumulative GPA**
- **Term GPA**
- **Advising Appointments**
- **Alerts**
- **Credit Load**
- **Age Group (binned)**
- **Gender**
- **Full-Time Status (calculated internally from credit load)**

The model is serialized using `joblib` and loaded directly inside the Streamlit app.

##  Technologies Used
- Python
- pandas
- numpy
- scikit-learn
- Streamlit
- joblib
- Excel (data cleaning and preprocessing)
- Tableau (data visualization and dashboard development)

## Contact
Created by Sydney Clapperton  
For questions or collaboration, feel free to reach out via GitHub.
