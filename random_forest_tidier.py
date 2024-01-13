# Importing Libraries
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
from data_research import create_df
import matplotlib.pyplot as plt


random_state = 1810
erasmus_db = create_df()

# Move it later to data_research
erasmus_db_for_training = erasmus_db[erasmus_db['status'] != "G"]

factor = pd.factorize(erasmus_db_for_training['status'])
erasmus_db.status = factor[0]
definitions = factor[1]

X = erasmus_db_for_training.drop("status", axis=1)
y = erasmus_db_for_training["status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=random_state)

numeric_features = X.select_dtypes(
    include=['int64', 'float64']).columns.tolist()

categorical_features = X.select_dtypes(
    include=['object', 'string']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[('imputer', SimpleImputer(
            strategy='mean')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), (
            'onehot', OneHotEncoder(drop='first'))]), categorical_features),
    ]
)


param_grid = {
    # Tune max_features from 1 to 200 for the RandomForestClassifier
    'classifier__max_features': list(range(1, 400, 6)),
}

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1': make_scorer(f1_score, average='weighted')
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)

rf_classifier = RandomForestClassifier(
    n_estimators=200, random_state=random_state)

model_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    # Adjust parameters as needed
    ('undersample', RandomUnderSampler(sampling_strategy='auto',
     random_state=random_state)),  # Adjust parameters as needed,
    # This will be replaced with the actual classifier during hyperparameter tuning
    ('classifier', rf_classifier)
])

grid_search = GridSearchCV(estimator=model_pipeline, param_grid=param_grid,
                           scoring=scoring, cv=cv, refit='recall', verbose=2, n_jobs=-1)

grid_search.fit(X_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_)

# Make predictions on the test set using the best model
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)

predictions.value_
# Evaluate the model performance on the test set
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

report = classification_report(y_test, predictions)
print("Classification Report:\n", report)

results = grid_search.cv_results_
max_features_values = param_grid['classifier__max_features']

# Plotting the metrics for different max_features values
plt.figure(figsize=(10, 6))

# Plot accuracy


# Plot precision
plt.plot(max_features_values,
         results['mean_test_precision'], label='Precision', marker='o')

# Plot recall
plt.plot(max_features_values,
         results['mean_test_recall'], label='Recall', marker='o')
plt.plot(max_features_values,
         results['mean_test_accuracy'], label='Accuracy', marker='o')
# Plot F1 score
plt.plot(max_features_values,
         results['mean_test_f1'], label='F1 Score', marker='o')

plt.title('Model Metrics for Different max_features Values')
plt.xlabel('max_features')
plt.ylabel('Score')
plt.legend()
plt.grid(True)

plt.show()
