
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
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
import numpy as np


erasmus_db = create_df()
random_state = 1810
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

boolean_features = X.select_dtypes(
    include=['bool']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[('imputer', SimpleImputer(
            strategy='mean')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), (
            'onehot', OneHotEncoder(drop='first'))]), categorical_features),
        # Placeholder step for boolean features
        ('bool', Pipeline(
            steps=[('placeholder', 'passthrough')]), boolean_features),
    ],
    remainder='passthrough'  # Include the rest of the columns as they are
)

start = 0.001
stop = 0.1
step = 0.005

# Create a sequence using numpy's arange function
sequence = np.arange(start, stop + step, step)


param_grid = {
    'classifier__C':  sequence,
}


scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1': make_scorer(f1_score, average='weighted')
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)

lasso = LogisticRegression(multi_class='ovr', penalty='l1', solver='saga')


model_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    # Adjust parameters as needed
    ('undersample', RandomUnderSampler(sampling_strategy='auto',
     random_state=random_state)),  # Adjust parameters as needed,
    # This will be replaced with the actual classifier during hyperparameter tuning
    ('classifier', lasso)
])

grid_search = GridSearchCV(estimator=model_pipeline, param_grid=param_grid,
                           scoring=scoring, cv=cv, refit='recall', verbose=2, n_jobs=-1)

grid_search.fit(X_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_)

best_model = grid_search.best_estimator_
# joblib.dump(best_model, 'best_model.joblib')
predictions = best_model.predict(X_test)

# Evaluate the model performance on the test set
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

report = classification_report(y_test, predictions)
print("Classification Report:\n", report)

# Predictions including G category

predictions_full = best_model.predict(erasmus_db.drop("status", axis=1))
erasmus_db["predictions"] = predictions_full
# Create the plots

results = grid_search.cv_results_
penalty_values = param_grid['classifier__C']

# Plotting the metrics for different max_features values
plt.figure(figsize=(10, 6))

# Plot accuracy


# Plot precision
plt.plot(penalty_values,
         results['mean_test_precision'], label='Precision', marker='o')

# Plot recall
plt.plot(penalty_values,
         results['mean_test_recall'], label='Recall', marker='o')
# plt.plot(penalty_values,
# results['mean_test_accuracy'], label='Accuracy', marker='o')
# Plot F1 score
plt.plot(penalty_values,
         results['mean_test_f1'], label='F1 Score', marker='o')

plt.title('Model Metrics for Different penalty Values')
plt.xlabel('penalty')
plt.ylabel('Score')
plt.legend()
plt.grid(True)

plt.show()

# Importance of variables
# Fit the best model on the entire training data
best_model.fit(X, y)


# all the things about feature importance i still need to see how to do for lasso
# Get coefficients
coefficients = best_model.named_steps['classifier'].coef_
abs_coef = np.abs(coefficients)
avg_abs_coef = np.mean(abs_coef, axis=0)

# Get feature names from the preprocessor
numeric_features_preprocessed = best_model.named_steps['preprocessor'].transformers_[
    0][2]


boolean_features_preprocessed = best_model.named_steps['preprocessor'].transformers_[
    2][2]

all_features = numeric_features + boolean_features_preprocessed


feature_importance_df = pd.DataFrame(
    {'Feature': all_features, 'Importance': avg_abs_coef})

feature_importance_df = feature_importance_df.sort_values(
    by='Importance', ascending=False)

# Plot the top 10 features
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'][:10],
         feature_importance_df['Importance'][:10])
plt.xlabel('Feature Importance')
plt.title('Top 10 Feature Importances')
plt.show()
