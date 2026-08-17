

# Customer Churn Prediction

> End-to-end machine learning pipeline for predicting customer churn using Scikit-learn, SMOTE, XGBoost, hyperparameter optimization, and SHAP explainability.

## Overview

This project implements a complete machine learning workflow for customer churn prediction, covering:

- Data cleaning and preprocessing
- Exploratory data analysis
- Categorical feature encoding
- Class imbalance handling with SMOTE
- Model comparison
- Cross-validation
- Hyperparameter optimization
- XGBoost classification
- Feature importance
- SHAP explainability
- Model serialization
- Prediction on new customer data

The main design goal is **reproducibility**: preprocessing, resampling, and model training are integrated into a single pipeline rather than being performed as disconnected manual steps.

---

# Architecture

The core machine learning workflow is:

```text
                    Raw Customer Data
                           │
                           ▼
                    Data Cleaning
                           │
              ┌────────────┼────────────┐
              │            │            │
        Remove ID    Convert Charges   Handle
                     to numeric       Missing Data
              │            │            │
              └────────────┴────────────┘
                           │
                           ▼
                  Train / Test Split
                     stratify = y
                           │
                           ▼
                  ColumnTransformer
                    /           \
                   /             \
          Numerical             Categorical
          Imputation            Imputation
                                  │
                           OneHotEncoder
                   \             /
                    \           /
                     └─────┬───┘
                           │
                           ▼
                         SMOTE
                           │
                           ▼
                    Classification
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
       Decision Tree   Random Forest   XGBoost
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                  Cross-Validation
                           │
                           ▼
                 Hyperparameter Tuning
                           │
                           ▼
                    Tuned XGBoost
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
          Evaluation   Importance     SHAP
              │            │            │
              └────────────┼────────────┘
                           ▼
                   Model Serialization
                           │
                           ▼
                  Prediction Pipeline
````

---

# Dataset

The project uses the **Telco Customer Churn** dataset.

The target variable is:

```text
Churn
├── 0 → No Churn
└── 1 → Churn
```

The target is imbalanced, so the implementation focuses on metrics beyond accuracy, particularly **Recall, F1-score, ROC-AUC, and PR-AUC**.

---


---

# 1. Data Preparation

## Removing identifiers

`customerID` is removed because it is an identifier rather than a predictive feature:

## Converting `TotalCharges`

`TotalCharges` is converted from an object/string column into numeric values and missing values are then replaced using the median:


## Target encoding

The target is converted to binary labels:

```python
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})
```

---

# 2. Exploratory Data Analysis

The notebook performs EDA before modelling.

Numerical variables are analysed using:

* Histograms
* Box plots
* Correlation analysis

Categorical variables are examined using count plots.

Reusable plotting functions are created to avoid duplicating visualization code:

```python
def plot_histogram(df, column_name):
    ...
```

```python
def plot_boxplot(df, column_name):
    ...
```

The main numerical features include:

```python
numerical_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]
```

---

# 3. Train/Test Split

The feature matrix and target are separated:

```python
X = df.drop(
    "Churn",
    axis=1
)

y = df["Churn"]
```

An 80/20 stratified split is used:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Using `stratify=y` preserves the churn/non-churn class distribution between the training and test sets.

---

# 4. Preprocessing Pipeline

The preprocessing architecture uses `ColumnTransformer` to apply different transformations to numerical and categorical features.

### Numerical features

```python
numeric_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    )
])
```

### Categorical features

```python
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
```

These are combined into a single preprocessing object:

```python
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
```

This allows the exact same transformations to be reused during training and inference.

---

# 5. SMOTE Pipeline

The churn dataset is class-imbalanced.

Instead of manually applying SMOTE before training, it is integrated directly into an `imblearn` pipeline:

```python
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
```

The resulting pipeline is:

```text
Raw Features
     ↓
Preprocessing
     ↓
OneHotEncoder
     ↓
SMOTE
     ↓
Classifier
```

This keeps preprocessing and resampling inside the model-training workflow and allows the same structure to be used during cross-validation.

---

# 6. Model Comparison

Three classifiers are evaluated using the same pipeline:

```python
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
```

Using the same preprocessing and SMOTE configuration ensures that the models are compared under consistent conditions.

---

# 7. Cross-Validation

Five-fold stratified cross-validation is used:

```python
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

F1-score for the churn class is used as the primary model-selection metric:

```python
churn_f1 = make_scorer(
    f1_score,
    pos_label=1
)
```

Models are evaluated using:

```python
scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring=churn_f1,
    n_jobs=-1
)
```

---

# 8. Hyperparameter Optimization

The strongest models are further optimized using `RandomizedSearchCV`.

## Random Forest

The search includes parameters such as:

```python
rf_params = {
    "model__n_estimators": [...],
    "model__max_depth": [...],
    "model__min_samples_split": [...],
    "model__min_samples_leaf": [...],
    "model__max_features": [...],
    "model__class_weight": [...]
}
```

The important detail is the `model__` prefix.

Because the classifier is a named step inside the pipeline, parameters are accessed using:

```text
model__parameter
```

---

# 9. XGBoost Optimization

XGBoost is tuned across:

```python
xgb_params = {
    "model__n_estimators": [...],
    "model__max_depth": [...],
    "model__learning_rate": [...],
    "model__subsample": [...],
    "model__colsample_bytree": [...],
    "model__min_child_weight": [...]
}
```

The search is performed with:

```python
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
```

The final tuned pipeline is retrieved with:

```python
best_xgb = xgb_search.best_estimator_
```

---

# 10. Final Model

The tuned XGBoost model is used as the final classifier.

Predictions are generated directly from the complete pipeline:

```python
y_test_pred = best_xgb.predict(
    X_test
)

y_prob = best_xgb.predict_proba(
    X_test
)[:, 1]
```

Because preprocessing is contained within the pipeline, `X_test` can remain in its original feature representation.

---

# 11. Model Evaluation

The final model is evaluated using:

```python
accuracy_score
recall_score
f1_score
roc_auc_score
average_precision_score
confusion_matrix
classification_report
```

### Final Test Performance

| Metric   |      Score |
| -------- | ---------: |
| Accuracy | **77.00%** |
| Recall   | **67.91%** |
| F1 Score | **61.06%** |
| ROC-AUC  | **83.83%** |
| PR-AUC   | **63.97%** |

Recall and F1 are particularly important because the objective is to identify customers who are likely to churn.

---

# 12. Feature Importance

The fitted XGBoost model is extracted from the pipeline:

```python
preprocessor = (
    best_xgb
    .named_steps["preprocessor"]
)

xgb_model = (
    best_xgb
    .named_steps["model"]
)
```

The transformed feature names are retrieved from the fitted `ColumnTransformer`:

```python
feature_names = (
    preprocessor
    .get_feature_names_out()
)
```

Feature importance is then calculated:

```python
importance = pd.Series(
    xgb_model.feature_importances_,
    index=feature_names
).sort_values(
    ascending=False
)
```

The top features are visualized as a horizontal bar chart.

This provides a global view of which transformed features contribute most strongly to the XGBoost model.

---

# 13. SHAP Explainability

SHAP is used to provide model explainability at both the global and individual-customer level.

The test data is transformed using the fitted preprocessing stage:

```python
X_test_transformed = (
    preprocessor.transform(X_test)
)

X_test_transformed_df = pd.DataFrame(
    X_test_transformed,
    columns=feature_names,
    index=X_test.index
)
```

A tree-based SHAP explainer is created:

```python
explainer = shap.TreeExplainer(
    xgb_model
)

shap_values = explainer.shap_values(
    X_test_transformed_df
)
```

The SHAP summary plot shows:

* Feature importance
* Direction of influence
* Distribution of feature effects

An individual customer's prediction can also be analysed with a waterfall plot:

```python
shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[customer_index],
        base_values=explainer.expected_value,
        data=X_test_transformed_df.iloc[
            customer_index
        ],
        feature_names=feature_names
    ),
    max_display=15
)
```

This allows the model to answer:

> **Why was this particular customer predicted to churn?**

---

# 14. Model Serialization

The complete tuned XGBoost pipeline is saved:

```python
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
```

The important part is that the **entire pipeline** is serialized.

It contains:

```text
Preprocessor
    │
    ├── Numerical Imputer
    └── OneHotEncoder
    │
    ▼
SMOTE
    │
    ▼
XGBoost
```

This prevents the prediction system from requiring separate preprocessing or categorical encoder files.

---

# 15. Prediction Pipeline

The saved pipeline can be loaded with:

```python
with open(
    "customer_churn_model.pkl",
    "rb"
) as f:

    model_data = pickle.load(f)

loaded_model = model_data["model"]
```

New customer data can be supplied using the original feature values:

```python
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
```

The input is converted into a DataFrame:

```python
input_data_df = pd.DataFrame(
    [input_data]
)
```

The complete pipeline then performs preprocessing and prediction:

```python
prediction = (
    loaded_model
    .predict(input_data_df)
)

churn_probability = (
    loaded_model
    .predict_proba(input_data_df)[:, 1]
)
```

Output:

```text
Prediction: Churn / No Churn
Churn Probability: XX%
```

No separate `LabelEncoder`, `OneHotEncoder`, or preprocessing script is required at inference time.

---

# Implementation Design

The project is built around several design principles.

### 1. Reproducibility

Random seeds are fixed using:

```python
random_state=42
```

for the major stochastic operations.

### 2. Consistent preprocessing

The `ColumnTransformer` ensures numerical and categorical features are transformed consistently.

### 3. Pipeline-based resampling

SMOTE is part of the training pipeline rather than a separate preprocessing step.

### 4. Consistent model comparison

Decision Tree, Random Forest and XGBoost use the same preprocessing and resampling workflow.

### 5. Cross-validation before tuning

Models are initially compared using stratified cross-validation before hyperparameter optimization.

### 6. Pipeline-aware hyperparameter tuning

Parameters are addressed using the pipeline's nested naming convention:

```text
model__parameter
```

### 7. Explainability

The final model is analysed using both feature importance and SHAP.

### 8. Reusable inference

The entire preprocessing and model pipeline is serialized into one object for future predictions.

---

# Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* imbalanced-learn
* XGBoost
* SHAP
* Matplotlib
* Seaborn
* Jupyter Notebook

---

# Installation

Clone the repository:

```bash
git clone https://github.com/ZORAKEN/customer_churning.git

cd customer_churning
```

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost shap
```

Launch Jupyter:

```bash
jupyter notebook
```

Open:

```text
CODE.ipynb
```

---

# Results

```text
Final Model:  Tuned XGBoost

Accuracy:     77.00%
Recall:       67.91%
F1 Score:    61.06%
ROC-AUC:      83.83%
PR-AUC:       63.97%
```

The final model prioritizes the identification of potential churners while maintaining good overall classification performance.

---

# Future Improvements

Potential next steps:

* Build a web-based prediction interface
* Add automated testing
* Add model monitoring
* Containerize the inference system
* Implement automated retraining

---

# Author

**ZORAKEN**

GitHub:
[https://github.com/ZORAKEN](https://github.com/ZORAKEN)



```

