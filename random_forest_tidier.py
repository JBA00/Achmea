# Importing Libraries
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import pandas as pd
from data_research import create_df

random_state = 1810
erasmus_db = create_df()

erasmus_db.select_dtypes(include=['bool'])

factor = pd.factorize(erasmus_db['status'])
erasmus_db.status = factor[0]
definitions = factor[1]

X = erasmus_db.drop("status", axis=1).iloc[:, 0:10]
y = erasmus_db["status"]

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
    'classifier__max_features': list(range(1, 10)),
}

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1': make_scorer(f1_score, average='weighted')
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)

rf_classifier = RandomForestClassifier(
    n_estimators=1000, random_state=random_state)

model_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    # Adjust parameters as needed
    ('smote', SMOTE(sampling_strategy='auto', random_state=random_state)),
    # This will be replaced with the actual classifier during hyperparameter tuning
    ('classifier', rf_classifier)
])

grid_search = GridSearchCV(estimator=model_pipeline, param_grid=param_grid,
                           scoring=scoring, cv=cv, refit='recall', verbose=2, n_jobs=10)

grid_search.fit(X_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_)

# Make predictions on the test set using the best model
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)

# Evaluate the model performance on the test set
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")
