**M.Tech AIML Machine Learning - Assignment 2**\
**HEART FAILURE PREDICTION DATASET**\
**a. Problem statement**\
The objective of this assignment is to build, evaluate, and deploy five distinct machine learning classification models (Logistic Regression, Decision Tree, K-Nearest Neighbors, Naive Bayes, and Support Vector Machine) to predict clinical heart disease risk based on patient demographic and physiological diagnostic attributes.

**b. Dataset description**\
**Dataset Source:** Synthetic Clinical Heart Failure & Diagnostic dataset (derived from standard UCI/Kaggle medical classification benchmarks).
**Feature Size:** 14 input features (including clinical measurements like Age, Cholesterol, RestingBP, MaxHR, and encoded categorical markers).
**Instance Size:** 1000 instances (800 training, 200 test samples).
**Target Variable:** Target (Binary: 0 = Healthy, 1 = Heart Disease Risk Present).

**c. GitHub Repository Link:**
    [github](https://github.com/2026ak05560/ml_assignment_2)


**d. Models Used & Comparison Table***
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8850 | 0.9420 | 0.8738 | 0.9010 | 0.8872 | 0.7705 |
| **Decision Tree** | 0.8550 | 0.9110 | 0.8416 | 0.8713 | 0.8562 | 0.7108 |
| **KNN** | 0.8650 | 0.9250 | 0.8571 | 0.8762 | 0.8665 | 0.7303 |
| **Naive Bayes** | 0.8700 | 0.9310 | 0.8614 | 0.8812 | 0.8712 | 0.7404 |
| **Support Vector Machine** | 0.8900 | 0.9480 | 0.8812 | 0.9010 | 0.8910 | 0.7808 |

**Evaluation Metrics**\
    Accuracy: proportion of all predictions that are correct.\
    AUC: area under the ROC curve; measures discrimination between positive and negative classes across thresholds.\
    Precision: proportion of predicted positive cases that are truly positive.\
    Recall: proportion of actual positive cases correctly identified.\
    F1: harmonic mean of precision and recall.\
    MCC: Matthews Correlation Coefficient; summarizes agreement between predicted and actual classes using all four confusion-matrix outcomes. It ranges from -1 to +1.



**Observations on Model Performance:**\
**Logistic Regression:** Performed consistently with strong linear separation, achieving high accuracy and balanced precision-recall.\
**Decision Tree:** Prone to slight overfitting compared to ensemble/linear models, but captured split thresholds efficiently.\
**KNN:** Benefited significantly from feature scaling, capturing local neighborhood structures well.\
**Naive Bayes:** Fast and robust probabilistic baseline, handling conditional independence assumptions effectively.\
**Support Vector Machine (Winner):** Achieved the highest overall accuracy (89.00%), AUC (0.9480), and MCC (0.7808) due to its optimal margin hyperplane separation in scaled feature space.

**Overall Winner:** Support Vector Machine (SVM).

To run the application locally, install the dependencies from requirements.txt and run:

python -m pip install -r requirements.txt \
python -m streamlit run app.py

**Links** \
**GitHub Repository:** https://github.com/2026ak05560/ml_assignment_2 \
**Streamlit Application:** https://mlassignment2-fwiof8v7ypetezfxazywtk.streamlit.app/ \
**Dataset:** [https://archive.ics.uci.edu/dataset/45/heart+disease](https://archive.ics.uci.edu/dataset/45/heart+disease?hl=en-IN)
