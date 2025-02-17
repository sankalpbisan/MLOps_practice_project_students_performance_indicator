# Student Performance Indicator

**Predicting Student Performance with Machine Learning**

This project uses machine learning to predict student performance based on various factors. It demonstrates an MLOps approach, from data preprocessing and model training to deployment.

---

The project is live and hosted on Streamlit cloud for making predictions.
The webpage can be accessed by [clicking here](https://sankalp-mlops-practice-project-student-performance-predictor.streamlit.app/)

---
## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Usage](#usage)
- [Model Performance](#evaluation)

---

## Introduction

Predicting student performance is crucial for educational institutions to identify at-risk students and provide timely interventions. This project aims to build a predictive model that estimates student performance based on demographics, academic history, and other relevant factors.  Accurate performance prediction can help educators personalize learning strategies and improve student outcomes.

---

## Features

* **Data Preprocessing:** Handles missing values, categorical encoding, and feature scaling.
* **Model Training:** Trains a model using various algorithms to predict student performance.
- **MLOps Integration**: Implements modular coding structure for model training and deployment.
- **Deployment**: Deployed the model on Streamlit cloud for real-time predictions.

---

## Usage

This section provides instructions on how to use the Student Performance Indicator project.

### 1. Setting up the Environment

Before running the code, ensure you have set up the required environment. This includes cloning the repository, creating a virtual environment, and installing the necessary dependencies using `requirements.txt`.

### 2. Executing the Repo

* **Running the app on the local system:**

    ```bash
    streamlit run app.py
    ```

    This will start the Streamlit application. By default, it runs on `http://127.0.0.1:8501` (Streamlit uses port 8501, not 5000).

* **Making Predictions:**

    You can make predictions by directly entering the respective values into each field in the Streamlit web application.  The application directly calls and executes the `predict_pipeline.py` script in the background.

    Here is an example of the values that can be entered:

    ```
    gender: male
    race/ethnicity: group C
    parental level of education: some college
    lunch: standard
    test preparation course: none
    math score: 70
    reading score: 80
    writing score: 75
    ```

    * **Input Fields:** The web application provides input fields for the following features.  **It is crucial to enter the data in the correct format and select from the allowed options for categorical features.**

        * `gender` (string): Student's gender ("male" or "female").
        * `race/ethnicity` (string): Student's race/ethnicity (e.g., "group A", "group B", "group C", "group D", "group E").
        * `parental level of education` (string): Parental level of education (e.g., "bachelor's degree", "some college", "master's degree", "associate's degree", "high school", "some high school"). 
        * `lunch` (string): Type of lunch ("standard" or "free/reduced").
        * `test preparation course` (string): Test preparation course status ("none" or "completed").
        * `math score` (integer): Math score (0-100).
        * `reading score` (integer): Reading score (0-100).
        * `writing score` (integer): Writing score (0-100).

    * **Response:** The system will display the predicted student performance level in a designated output area on the web page.  For example:

    ```
    Predicted marks : some integer
    ```

### 3. Training the Model

The project includes a training script that you can use to retrain the model.

* **Running the Training Script:**

    ```bash
    python src/pipeline/training_pipeline.py
    ```

    This script will train the model and save the trained model artifacts (including the data transformation pipeline) in the `artifacts` directory. The trained model is saved as a pickle file.  The script will also output a performance score (r2_score) after training. 

---

## Model Performance

The performance of the hospital charges prediction model was evaluated using R-squared `r2_square`.  These metrics were chosen because R2 measures the goodness of fit.

The model was trained and evaluated on a dataset, initially data is splitted into train and test before preprocessing. Before training train data is splitted into train and validation set and performed a 10-fold cross-validation to ensure model robustness.

---
