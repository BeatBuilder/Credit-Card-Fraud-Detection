# Credit-Card-Fraud-Detection
Credit Card Fraud Detection with Logistic Regression

This repository contains Python code that implements a Logistic Regression model to detect credit card fraud using the creditcard.csv dataset.

Project Overview

The script utilizes pandas for data manipulation, NumPy for numerical operations, scikit-learn for machine learning tasks, and various functions for model evaluation.
It performs the following steps:
Loads the credit card fraud dataset.
Explores the data through methods like head(), info(), and describe().
Checks for missing values using isnull().sum().
Separates features (X) and the target variable (y).
Splits the data into training and testing sets using train_test_split with stratification to maintain class balance.
Scales the features using StandardScaler for better model performance.
Initializes and trains a Logistic Regression model.
Makes predictions on the testing set.
Calculates accuracy using accuracy_score.
Generates a confusion matrix using confusion_matrix.
Generates a classification report using classification_report.
