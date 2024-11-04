import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import StackingClassifier
from skopt import BayesSearchCV  # Bayesian optimization

# Load data
def load_data():
    data = pd.read_csv('data/sports_data.csv')
    return data

# Model training with feature selection, stacking, and Bayesian optimization
def train_model():
    data = load_data()
    X = data.drop('outcome', axis=1)  # Drop target column
    y = data['outcome']

    # Feature selection
    feature_selector = SelectKBest(chi2, k=5)  # Adjust `k` to optimize feature count

    # Base models for stacking
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('xgb', XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, random_state=42))
    ]

    # Stacking with Logistic Regression as the final estimator
    stacked_model = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression()
    )

    # Define pipeline with feature selection and stacking model
    pipeline = Pipeline([
        ('selector', feature_selector),
        ('classifier', stacked_model)
    ])

    # Bayesian Optimization for hyperparameters
    param_grid = {
        'classifier__xgb__n_estimators': (50, 200),
        'classifier__xgb__max_depth': (3, 10),
        'classifier__rf__n_estimators': (50, 200),
        'classifier__gb__n_estimators': (50, 200)
    }

    # Bayesian search to optimize the pipeline
    bayes_search = BayesSearchCV(
        estimator=pipeline,
        search_spaces=param_grid,
        n_iter=32,  # Number of iterations for optimization
        cv=5,
        n_jobs=-1,
        scoring='accuracy',
        random_state=42
    )

    # Train-test split for validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    bayes_search.fit(X_train, y_train)

    # Best model and accuracy evaluation
    best_model = bayes_search.best_estimator_
    print("Best Parameters:", bayes_search.best_params_)
    y_pred = best_model.predict(X_test)
    print("Model accuracy:", accuracy_score(y_test, y_pred))

    # Save the optimized model
    joblib.dump(best_model, 'data/prediction_model.joblib')
    return best_model

# Load and predict with the optimized model
model = joblib.load('data/prediction_model.joblib')

# Prediction function
def predict(features):
    prediction = model.predict([features])[0]
    return prediction
