# Customer Churn Prediction


> End-to-end machine learning pipeline for predicting customer churn using Scikit-learn, SMOTE, XGBoost, hyperparameter optimization, and SHAP explainability.


## Overview


This project implements a complete customer churn prediction workflow, from raw data exploration and preprocessing to model training, hyperparameter optimization, explainability, and inference on new customer records.


The main focus of the implementation is building a **reproducible machine learning pipeline** rather than manually preprocessing data at different stages.


The final workflow is:


```text
Raw Data
   │
   ▼
Data Cleaning
   │
   ├── Remove customerID
   ├── Convert TotalCharges → numeric
   ├── Handle missing values
   └── Encode Churn → 0 / 1
   │
   ▼
Train / Test Split
   │
   ▼
ColumnTransformer
   │
   ├── Numerical features
   │
   └── OneHotEncoder
   │
   ▼
SMOTE
   │
   ▼
Model
   │
   ├── Decision Tree
   ├── Random Forest
   └── XGBoost
   │
   ▼
Cross-Validation
   │
   ▼
Hyperparameter Tuning
   │
   ▼
Tuned XGBoost
   │
   ├── Evaluation
   ├── Feature Importance
   ├── SHAP
   └── Saved Prediction Pipeline

```
Project Goals

The implementation is designed to answer the following questions:

Can customer churn be predicted from customer demographics, services, contracts and billing information?
Which classification algorithm performs best?
How should categorical variables and class imbalance be handled?
Which features have the greatest influence on churn predictions?
Why does the model predict that an individual customer will churn?
Can the trained model be reused to make predictions on new customers?
Dataset```

The project uses the Telco Customer Churn dataset.

The target variable is:

Churn
├── 0 → No churn
└── 1 → Churn

The target is imbalanced, so the implementation does not rely on accuracy alone when evaluating models.

Repository Structure
customer_churning/
│
├── CODE.ipynb
├── README.md
├── Telco-Customer-Churn.csv
├── customer_churn_model.pkl
└── requirements.txt
CODE.ipynb

Contains the complete implementation:

1. Dependencies
2. Data Loading & Understanding
3. Data Cleaning
4. Exploratory Data Analysis
5. Data Preprocessing
6. Model Training
7. Random Forest Tuning
8. XGBoost Tuning
9. Feature Importance
10. SHAP Explainability
11. Model Serialization
12. Prediction System



 Data Cleaning and Processing 
Remove Identifier

customerID is removed because it uniquely identifies a customer but does not provide meaningful predictive information:

df = df.drop(
    columns=["customerID"]
)
Convert TotalCharges

TotalCharges is initially represented as a string/object column.

It is converted to numeric:

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

Any resulting missing values are replaced using the median:

df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)
Encode Target

The target is converted into a binary representation:

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

The resulting target is:

0 → No Churn
1 → Churn
4. Exploratory Data Analysis

The notebook performs exploratory analysis before model training.

Numerical Analysis

The numerical features are:

numerical_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

Reusable functions are created for visual analysis.

Histograms
def plot_histogram(df, column_name):
    plt.figure(figsize=(5, 3))
    sns.histplot(
        df[column_name],
        kde=True
    )
    ...

The function is then reused for:

plot_histogram(df, "tenure")
plot_histogram(df, "MonthlyCharges")
plot_histogram(df, "TotalCharges")
Box Plots

A reusable box-plot function is also defined:

def plot_boxplot(df, column_name):
    plt.figure(figsize=(5, 3))
    sns.boxplot(
        y=df[column_name]
    )
    ...
Correlation Analysis

The numerical features are examined using a correlation heatmap:

Categorical Analysis

Categorical variables are identified using:

object_cols = (
    df.select_dtypes(
        include="object"
    ).columns.to_list()
)

Categorical distributions are visualized using count plots.

5. Train/Test Split

The target is separated from the feature matrix:

X = df.drop(
    "Churn",
    axis=1
)


y = df["Churn"]

The dataset is split using an 80/20 stratified split:

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
Why stratification?

The target is imbalanced, so stratify=y ensures that the training and test sets maintain a similar class distribution.

6. Feature Preprocessing

The implementation uses a ColumnTransformer to process numerical and categorical variables separately.

Numerical Pipeline
numeric_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    )
])
Categorical Pipeline
categorical_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),
    (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])

These transformations are combined using:

preprocessor = ColumnTransformer([
    (
        "num",
        numeric_transformer,
        numerical_features
    ),
    (
        "cat",
        categorical_transformer,
        categorical_features
    )
])

This creates a single preprocessing object that can be reused consistently during training and inference.

7. Handling Class Imbalance

The churn target is imbalanced.

Instead of manually oversampling the complete training dataset, SMOTE is integrated into the machine learning pipeline.

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def make_pipeline(model):


    return ImbPipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "smote",
            SMOTE(
                random_state=42
            )
        ),
        (
            "model",
            model
        )
    ])

The resulting architecture is:

Input
  ↓
Preprocessing
  ↓
OneHotEncoder
  ↓
SMOTE
  ↓
Classifier

This is important because preprocessing and resampling remain part of the model-fitting process used during cross-validation.

8. Model Architecture

Three models are evaluated using the same pipeline:

models_pipeline = {


    "Decision Tree": make_pipeline(
        DecisionTreeClassifier(
            random_state=42
        )
    ),


    "Random Forest": make_pipeline(
        RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        )
    ),


    "XGBoost": make_pipeline(
        XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1
        )
    )
}

This ensures that model comparisons are performed using the same preprocessing and class-balancing strategy.

9. Cross-Validation

Five-fold stratified cross-validation is used:

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

The primary scoring metric is F1-score for the churn class:

churn_f1 = make_scorer(
    f1_score,
    pos_label=1
)

Each model is evaluated using:

scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring=churn_f1,
    n_jobs=-1
)

This produces a mean and standard deviation of F1 across the validation folds.

10. Hyperparameter Optimization

The strongest models are further optimized using RandomizedSearchCV.

Random Forest

The Random Forest search explores:

rf_params = {
    "model__n_estimators": [
        200, 300, 400
    ],
    "model__max_depth": [
        None, 10, 15, 20
    ],
    "model__min_samples_split": [
        2, 5, 10
    ],
    "model__min_samples_leaf": [
        1, 2, 4
    ],
    "model__max_features": [
        "sqrt", "log2"
    ],
    "model__class_weight": [
        None, "balanced"
    ]
}

The model is optimized using:

rf_search = RandomizedSearchCV(
    rf_pipeline,
    param_distributions=rf_params,
    n_iter=10,
    scoring=churn_f1,
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)
11. XGBoost Optimization

XGBoost is tuned using a broader parameter search:

xgb_params = {


    "model__n_estimators": [
        200, 300, 500, 700
    ],


    "model__max_depth": [
        3, 4, 5, 6
    ],


    "model__learning_rate": [
        0.01, 0.03, 0.05,
        0.1, 0.2
    ],


    "model__subsample": [
        0.7, 0.8, 0.9, 1.0
    ],


    "model__colsample_bytree": [
        0.7, 0.8, 0.9, 1.0
    ],


    "model__min_child_weight": [
        1, 3, 5
    ]
}

The search is performed with:

xgb_search = RandomizedSearchCV(
    xgb_pipeline,
    param_distributions=xgb_params,
    n_iter=30,
    scoring=churn_f1,
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

The final tuned pipeline is retrieved using:

best_xgb = xgb_search.best_estimator_
12. Final Model Evaluation

The tuned XGBoost pipeline is evaluated against the held-out test set:

y_test_pred = best_xgb.predict(
    X_test
)


y_prob = best_xgb.predict_proba(
    X_test
)[:, 1]

The following metrics are calculated:

accuracy_score(...)
recall_score(...)
f1_score(...)
roc_auc_score(...)
average_precision_score(...)
confusion_matrix(...)
classification_report(...)
Final Test Results
Metric	Score
Accuracy	77.00%
Recall	67.91%
F1 Score	61.06%
ROC-AUC	83.83%
PR-AUC	63.97%

F1 and recall are emphasized because the objective is specifically to identify customers who are likely to churn.

13. Feature Importance

The final XGBoost model is accessed from the fitted pipeline:

preprocessor = (
    best_xgb
    .named_steps["preprocessor"]
)


xgb_model = (
    best_xgb
    .named_steps["model"]
)

Because categorical variables have been one-hot encoded, the transformed feature names are retrieved from the fitted preprocessor:

feature_names = (
    preprocessor
    .get_feature_names_out()
)

Feature importance is then calculated:

importance = pd.Series(
    xgb_model.feature_importances_,
    index=feature_names
).sort_values(
    ascending=False
)

The top 15 features are visualized using a horizontal bar chart.

This provides a global view of which transformed features are most important to the XGBoost model.

14. SHAP Explainability

The project also uses SHAP to explain the final XGBoost model.

The transformed test set is generated using the same fitted preprocessing pipeline:

X_test_transformed = (
    preprocessor.transform(X_test)
)

The transformed data is stored as a DataFrame with the generated feature names:

X_test_transformed_df = pd.DataFrame(
    X_test_transformed,
    columns=feature_names,
    index=X_test.index
)

A TreeExplainer is created for XGBoost:

explainer = shap.TreeExplainer(
    xgb_model
)


shap_values = explainer.shap_values(
    X_test_transformed_df
)

A SHAP summary plot provides a global explanation of the model.

An individual prediction can also be investigated using a waterfall plot:

customer_index = 0


shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[
            customer_index
        ],
        base_values=explainer.expected_value,
        data=X_test_transformed_df.iloc[
            customer_index
        ],
        feature_names=feature_names
    ),
    max_display=15
)

This allows the model to answer:

Why did the model predict this customer as likely to churn?

15. Model Serialization

The complete tuned XGBoost pipeline is serialized using pickle:

model_data = {
    "model": best_xgb
}


with open(
    "customer_churn_model.pkl",
    "wb"
) as f:


    pickle.dump(
        model_data,
        f
    )

The important design decision here is that the entire pipeline is saved, rather than only the XGBoost classifier.

Therefore the saved object contains:

Preprocessor
     │
     ├── Numerical imputation
     └── OneHotEncoder
     
SMOTE
     │
     ▼
XGBoost

This makes the model reusable for inference without manually repeating preprocessing.

16. Loading the Model

The saved pipeline can be restored with:

with open(
    "customer_churn_model.pkl",
    "rb"
) as f:


    model_data = pickle.load(f)


loaded_model = model_data["model"]

The restored object is then ready to receive new customer records.

17. Prediction System

A new customer is represented using the original, human-readable feature values:

input_data = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    ...
}

The dictionary is converted to a DataFrame:

input_data_df = pd.DataFrame(
    [input_data]
)

The saved pipeline handles the preprocessing automatically:

prediction = (
    loaded_model
    .predict(input_data_df)
)


pred_prob = (
    loaded_model
    .predict_proba(input_data_df)[:, 1]
)

The final result is displayed as:

print(
    f"Prediction: "
    f"{'Churn' if prediction[0] == 1 else 'No Churn'}"
)


print(
    f"Churn Probability: "
    f"{pred_prob[0]:.2%}"
)

This means the prediction system does not require separate categorical encoders at inference time.

Implementation Architecture

The core implementation can be summarized as:

                 ┌─────────────────────┐
                 │   Raw Customer Data  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │    Data Cleaning    │
                 │                     │
                 │ TotalCharges        │
                 │ Missing Values      │
                 │ customerID          │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Train/Test Split    │
                 │    stratify=y       │
                 └──────────┬──────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │      ColumnTransformer      │
              │                             │
              │ Numerical → Imputer         │
              │ Categorical → OneHotEncoder│
              └──────────────┬──────────────┘
                             │
                             ▼
                     ┌─────────────┐
                     │    SMOTE    │
                     └──────┬──────┘
                            │
                            ▼
                     ┌─────────────┐
                     │  XGBoost    │
                     └──────┬──────┘
                            │
                ┌───────────┼───────────┐
                ▼           ▼           ▼
           Evaluation   Feature      SHAP
                        Importance
                            │
                            ▼
                     Model Serialization
                            │
                            ▼
                    Prediction System
Why This Implementation?
Pipeline-based preprocessing

Preprocessing is encapsulated in a reusable pipeline instead of manually transforming training and test data separately.

ColumnTransformer

Numerical and categorical features require different preprocessing strategies, so they are handled independently before being combined.

OneHotEncoder

Categorical variables are one-hot encoded while handle_unknown="ignore" allows the pipeline to process previously unseen categories during inference.

SMOTE inside the pipeline

SMOTE is integrated into the imblearn pipeline so that oversampling is performed as part of model fitting rather than manually modifying the complete dataset.

Stratified cross-validation

StratifiedKFold maintains the class distribution across validation folds.

F1-based model selection

F1-score is used as the main optimization metric because identifying the minority churn class is more important than maximizing raw accuracy.

Nested parameter names

Hyperparameters are specified using:

model__parameter

because the classifier is a named step inside the pipeline.

Pipeline serialization

The complete preprocessing and modelling workflow is saved as one object, making inference consistent with training.

Explainability

Feature importance and SHAP provide both global and individual explanations of the final model.

Technologies
Python
Pandas
NumPy
Scikit-learn
imbalanced-learn
XGBoost
SHAP
Matplotlib
Seaborn
Jupyter Notebook
Installation

Clone the repository:

git clone https://github.com/ZORAKEN/customer_churning.git


cd customer_churning

Install dependencies:

pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost shap

Launch Jupyter:

jupyter notebook

Open:

CODE.ipynb

Results

The final tuned XGBoost model achieved:

Accuracy    : 77.00%
Recall      : 67.91%
F1 Score    : 61.06%
ROC-AUC     : 83.83%
PR-AUC      : 63.97%

The model is designed around the practical objective of identifying customers who are at risk of churn rather than optimizing accuracy alone.
