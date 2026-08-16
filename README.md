# customer_churning
# Customer Churn Prediction using Machine Learning

A machine learning project for predicting whether a telecom customer is likely to churn based on customer demographics, services, contract information, and billing details.

## 📌 Project Overview

Customer churn is an important business problem for subscription-based companies. Identifying customers who are likely to leave can help businesses take preventive actions and improve customer retention.

This project uses the **Telco Customer Churn** dataset and develops a classification pipeline to predict customer churn using multiple machine learning algorithms.

The notebook covers:

* Data loading and understanding
* Exploratory Data Analysis (EDA)
* Data cleaning and preprocessing
* Categorical feature encoding
* Train/test splitting
* Handling class imbalance using SMOTE
* Training multiple classification models
* 5-fold cross-validation
* Model evaluation
* Saving and loading the trained model
* Building a simple predictive system

## 📊 Dataset

The project uses the **Telco Customer Churn** dataset:

`WA_Fn-UseC_-Telco-Customer-Churn.csv`

The dataset contains **7,043 customers and 21 columns** before preprocessing.

Important features include:

* `gender`
* `SeniorCitizen`
* `Partner`
* `Dependents`
* `tenure`
* `PhoneService`
* `MultipleLines`
* `InternetService`
* `OnlineSecurity`
* `OnlineBackup`
* `DeviceProtection`
* `TechSupport`
* `StreamingTV`
* `StreamingMovies`
* `Contract`
* `PaperlessBilling`
* `PaymentMethod`
* `MonthlyCharges`
* `TotalCharges`
* `Churn`

The `customerID` column is removed because it is not required for modeling.

The target variable is:

* `0` → No Churn
* `1` → Churn

The original target distribution contains:

* **5,174** customers who did not churn
* **1,869** customers who churned

This indicates a class imbalance in the target variable.

## 🔎 Exploratory Data Analysis

The notebook performs exploratory analysis of both numerical and categorical features.

The analysis includes:

* Distribution analysis of numerical features
* Box plots for numerical variables
* Correlation heatmap
* Count plots for categorical variables
* Examination of unique values
* Checking for missing values

The numerical variables analyzed include:

* `SeniorCitizen`
* `tenure`
* `MonthlyCharges`
* `TotalCharges`

The notebook also identifies blank values in `TotalCharges`. These values are associated with customers whose tenure is zero, and the notebook replaces the missing `TotalCharges` values with `0`.

## 🛠️ Data Preprocessing

The preprocessing pipeline consists of several steps.

### 1. Remove unnecessary columns

The `customerID` column is removed because it does not provide useful information for predicting churn.

### 2. Handle `TotalCharges`

`TotalCharges` is initially stored as an object/string column. Blank values are identified and replaced with `0`, after which the column is converted to a numerical representation.

### 3. Encode the target

The `Churn` column is converted from categorical values into binary values:

```text
Yes → 1
No  → 0
```

### 4. Encode categorical features

Categorical columns are transformed using `LabelEncoder`.

The encoders are stored in a dictionary and saved to:

```text
encoders.pkl
```

This allows the same encoding scheme to be reused when making predictions on new customer data.

### 5. Train/test split

The dataset is divided into training and testing sets using:

* Training data: 80%
* Test data: 20%
* `random_state=42`

The resulting training set contains **5,634 records**, while the test set contains **1,409 records**.

### 6. Handle class imbalance with SMOTE

The training data is balanced using **Synthetic Minority Oversampling Technique (SMOTE)**.

Before SMOTE:

```text
No Churn    4,138
Churn       1,496
```

After SMOTE:

```text
No Churn    4,138
Churn       4,138
```

SMOTE is applied only to the training data.

## 🤖 Machine Learning Models

Three classification algorithms are evaluated:

1. Decision Tree
2. Random Forest
3. XGBoost

The models are initially trained using their default hyperparameters.

### Cross-Validation Results

5-fold cross-validation is performed on the SMOTE-balanced training data.

| Model         | Cross-Validation Accuracy |
| ------------- | ------------------------: |
| Decision Tree |                      0.78 |
| Random Forest |                      0.84 |
| XGBoost       |                      0.83 |

Based on the cross-validation accuracy, **Random Forest** performs best among the three models tested.

## 🌲 Final Model

The final model selected in the notebook is:

```text
RandomForestClassifier(random_state=42)
```

The Random Forest model is trained using the SMOTE-balanced training dataset.

## 📈 Model Evaluation

The trained Random Forest model is evaluated against the original test set.

### Accuracy

```text
0.7786
```

Approximately **77.86% test accuracy**.

### Confusion Matrix

```text
[[878, 158],
 [154, 219]]
```

### Classification Report

| Class                | Precision | Recall | F1-Score |
| -------------------- | --------: | -----: | -------: |
| No Churn             |      0.85 |   0.85 |     0.85 |
| Churn                |      0.58 |   0.59 |     0.58 |
| **Overall Accuracy** |           |        | **0.78** |

The model performs better at identifying customers who do not churn than customers who do churn.

This difference is important because, in a real-world churn prediction application, improving recall for the churn class could be particularly valuable.

## 💾 Saved Model Artifacts

The notebook saves two pickle files.

### Trained model

```text
customer_churn_model.pkl
```

This file contains:

* The trained Random Forest model
* The feature names used during training

### Encoders

```text
encoders.pkl
```

This file contains the `LabelEncoder` objects used to transform categorical features.

Keeping the encoders is important because new prediction data must be transformed using the same encoding scheme as the training data.

## 🔮 Predictive System

The notebook demonstrates how to load the saved model and encoders and use them to predict churn for a new customer.

The example customer has characteristics such as:

* Female
* Senior citizen: No
* Partner: Yes
* Tenure: 1 month
* Internet service: DSL
* Month-to-month contract
* Electronic check payment
* Monthly charges: 29.85
* Total charges: 29.85

The example prediction produced:

```text
Prediction: No Churn
Prediction Probability: [[0.78 0.22]]
```

The model therefore predicted **No Churn**, with a predicted probability distribution of approximately 78% for no churn and 22% for churn.

## 🧰 Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn
* XGBoost
* Pickle

### Main Libraries

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

### 3. Add the dataset

Place the Telco Customer Churn CSV file in the location expected by the notebook:

```text
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The notebook currently loads it using:

```python
pd.read_csv("/content/WA_Fn-UseC_-Telco-Customer-Churn.csv")
```

If running locally, update the path accordingly.

### 4. Run the notebook

Open:

```text
Customer_Churn_Prediction_using_ML.ipynb
```

You can run the notebook using Jupyter Notebook, JupyterLab, or Google Colab.

## 📁 Suggested Repository Structure

```text
customer-churn-prediction/
│
├── Customer_Churn_Prediction_using_ML.ipynb
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── customer_churn_model.pkl
├── encoders.pkl
├── README.md
└── requirements.txt
```

A `requirements.txt` file can contain:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
```

## 🔄 Machine Learning Workflow

```text
Raw Customer Data
        ↓
Data Loading
        ↓
Data Understanding
        ↓
Remove Customer ID
        ↓
Handle TotalCharges
        ↓
Exploratory Data Analysis
        ↓
Encode Target & Categorical Features
        ↓
Train/Test Split
        ↓
SMOTE on Training Data
        ↓
Train Decision Tree / Random Forest / XGBoost
        ↓
5-Fold Cross-Validation
        ↓
Select Random Forest
        ↓
Evaluate on Test Data
        ↓
Save Model + Encoders
        ↓
Predict Churn for New Customers
```

## 📌 Current Limitations

The current notebook is a working baseline rather than a fully optimized production model.

The notebook itself identifies several areas for future improvement:

* Hyperparameter tuning
* Further model selection
* Downsampling experiments
* Addressing potential overfitting
* Stratified K-Fold cross-validation

The current final Random Forest model achieves strong overall accuracy, but its recall for the churn class is only **0.59**, leaving room for improvement in identifying customers who are actually at risk of churning.

## 🔮 Future Improvements

Potential next steps include:

1. Perform systematic hyperparameter tuning for Random Forest and XGBoost.
2. Compare additional classification algorithms.
3. Experiment with downsampling as an alternative to SMOTE.
4. Investigate and reduce model overfitting.
5. Use Stratified K-Fold cross-validation.
6. Evaluate additional metrics such as ROC-AUC, precision, recall, and F1-score.
7. Improve churn-class recall.
8. Build a reusable prediction script or web application around the saved model.
9. Create a proper preprocessing pipeline to make training and inference more consistent.

## 📜 License

This project is intended for educational and portfolio purposes. Add an appropriate license to the repository if you plan to distribute or modify the project publicly.
