# Importing Libraries
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.pipeline import Pipeline
import pandas as pd
from data_research import create_df

random_state = 1810
erasmus_db = create_df()

factor = pd.factorize(erasmus_db['status'])
erasmus_db.status = factor[0]
definitions = factor[1]

X = erasmus_db.drop("status", axis=1).values
y = erasmus_db.status.values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=random_state)

param_grid = {
    # Tune max_features from 1 to 200 for the RandomForestClassifier
    'clf__max_features': list(range(1, 201)),
}

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1': make_scorer(f1_score, average='weighted')
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

rf_classifier = RandomForestClassifier(
    n_estimators=1000, random_state=random_state)

smote = SMOTE(sampling_strategy='auto', random_state=random_state)

pipeline = Pipeline([
    ('smote', smote),
    ('clf', rf_classifier)
])

grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid,
                           scoring=scoring, cv=cv, refit='f1', verbose=2, n_jobs=-1)

grid_search.fit(X_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_)

# Make predictions on the test set using the best model
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)
