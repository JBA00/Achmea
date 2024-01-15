# Importing Libraries
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
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
import joblib
import numpy as np

erasmus_db = create_df()

# Move it later to data_research
erasmus_db_for_training = erasmus_db[erasmus_db['status'] != "G"]


class InsuranceClassifier:
    def __init__(self, type_of_classifier, full_data, trainable_data, grid,
                 name_of_target_column="status", random_state=1810,
                 number_of_folds=10):
        self.type_of_classifier = type_of_classifier
        self.param_grid = grid
        self.erasmus_db = full_data
        self.erasmus_db_for_training = trainable_data
        self.target_col = name_of_target_column
        self.random_state = random_state
        self.number_of_folds = number_of_folds
        self._initialization()

    def _initialization(self):
        self._prepare_variables()
        self._preprocess_data()
        self._cv_creation()
        self._classify_random_forest()
        self._find_best()

    def _prepare_variables(self):

        factor = pd.factorize(self.erasmus_db_for_training[self.target_col])
        self.erasmus_db_for_training[self.target_col] = factor[0]
        self.definitions = factor[1]

        self.X = self.erasmus_db_for_training.drop(self.target_col, axis=1)
        self.y = self.erasmus_db_for_training[self.target_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.25, random_state=self.random_state)

    def _preprocess_data(self):
        self.numeric_features = self.X.select_dtypes(
            include=['int64', 'float64']).columns.tolist()

        self.categorical_features = self.X.select_dtypes(
            include=['object', 'string']).columns.tolist()

        self.boolean_features = self.X.select_dtypes(
            include=['bool']).columns.tolist()

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[('imputer', SimpleImputer(
                    strategy='mean')), ('scaler', StandardScaler())]), self.numeric_features),
                ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), (
                    'onehot', OneHotEncoder(drop='first'))]), self.categorical_features),
                # Placeholder step for boolean features
                ('bool', Pipeline(
                    steps=[('placeholder', 'passthrough')]), self.boolean_features),
            ],
            remainder='passthrough'  # Include the rest of the columns as they are
        )

    def _cv_creation(self):

        self.cv = StratifiedKFold(
            n_splits=self.number_of_folds, shuffle=True,
            random_state=self.random_state)

    def _classify_random_forest(self):
        self.scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average='weighted'),
            'recall': make_scorer(recall_score, average='weighted'),
            'f1': make_scorer(f1_score, average='weighted')
        }

        if self.type_of_classifier == "Random Forest":
            self.classifier = RandomForestClassifier(
                n_estimators=400, random_state=self.random_state)
        elif self.type_of_classifier == "Lasso":
            self.classifier = LogisticRegression(multi_class='ovr',
                                                 penalty='l1', solver='saga')

        self.model_pipeline = ImbPipeline([
            ('preprocessor', self.preprocessor),
            # Adjust parameters as needed
            ('undersample', RandomUnderSampler(sampling_strategy='auto',
                                               random_state=self.random_state)),  # Adjust parameters as needed,
            # This will be replaced with the actual classifier during hyperparameter tuning
            ('classifier', self.classifier)
        ])

        self.grid_search = GridSearchCV(estimator=self.model_pipeline,
                                        param_grid=self.param_grid,
                                        scoring=self.scoring, cv=self.cv,
                                        refit='recall', verbose=2, n_jobs=-1)

    def _find_best(self):

        self.grid_search.fit(self.X_train, self.y_train)
        self.best_model = self.grid_search.best_estimator_
        self.predictions = self.best_model.predict(self.X_test)
        self.predictions_full = self.best_model.predict(
            self.erasmus_db.drop("status", axis=1))
        self.erasmus_db["predictions"] = self.predictions_full
        defactorized_column = pd.Categorical.from_codes(
            self.erasmus_db['predictions'], self.definitions)
        self.erasmus_db['predictions_defactor'] = defactorized_column

    def get_report(self):
        report = classification_report(self.y_test, self.predictions)
        print("Classification Report:\n", report)

    def get_tuning_graph(self):
        results = self.grid_search.cv_results_
        max_features_values = self.param_grid['classifier__max_features']

        # Plotting the metrics for different max_features values
        plt.figure(figsize=(10, 6))

        # Plot precision
        plt.plot(max_features_values,
                 results['mean_test_precision'], label='Precision', marker='o')

        # Plot recall
        plt.plot(max_features_values,
                 results['mean_test_recall'], label='Recall', marker='o')

        plt.title('Model Metrics for Different max_features Values')
        plt.xlabel('max_features')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)

        plt.show()

    def get_feature_importance_graph(self):
        self.best_model.fit(self.X, self.y)

        if self.type_of_classifier == "Random Forest":
            # Access feature importances
            feature_importances = self.best_model.named_steps['classifier'].feature_importances_
        else:
            coefficients = self.best_model.named_steps['classifier'].coef_
            abs_coef = np.abs(coefficients)
            feature_importances = np.mean(abs_coef, axis=0)

        # Get feature names from the preprocessor
        numeric_features_preprocessed = self.best_model.named_steps['preprocessor'].transformers_[
            0][2]

        boolean_features_preprocessed = self.best_model.named_steps['preprocessor'].transformers_[
            2][2]

        all_features = numeric_features_preprocessed + boolean_features_preprocessed
        # Create a DataFrame for better visualization
        feature_importance_df = pd.DataFrame(
            {'Feature': all_features, 'Importance': feature_importances})

        # Sort by importance in descending order
        feature_importance_df = feature_importance_df.sort_values(
            by='Importance', ascending=False)

        # Plot the top 10 features
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df['Feature'][:10],
                 feature_importance_df['Importance'][:10])
        plt.xlabel('Feature Importance')
        plt.title('Top 10 Feature Importances')
        plt.show()

    def get_misclassified_g_status(self):
        self.g_data = self.erasmus_db[self.erasmus_db[self.target_col] == "G"][[
            self.target_col, "predictions_defactor"]]

        plt.figure(figsize=(10, 6))
        self.g_data["predictions_defactor"].value_counts().plot(
            kind='bar', color='skyblue')
        plt.title('Value Counts of predictions_defactor')
        plt.xlabel('Categories')
        plt.ylabel('Count')
        plt.show()

    def get_missing_amounts(self):
        missed_amount = len(self.g_data["predictions_defactor"][self.g_data["predictions_defactor"] == "A"]) * 15000 + len(self.g_data["predictions_defactor"]
                                                                                                                           [self.g_data["predictions_defactor"] == "P"]) * 1500 - len(self.g_data["predictions_defactor"][self.g_data["predictions_defactor"] == "S"]) * 100
        print(missed_amount)

    def save_the_model(self):
        joblib.dump(self.best_model, 'best_model.joblib')


rf_grid = {'classifier__max_features': list(range(1, 100, 6))}
lasso_grid = {'classifier__C':  np.arange(0.001, 0.105, 0.005)}

my_rf = InsuranceClassifier("Random Forest", erasmus_db, erasmus_db_for_training,
                            rf_grid)


my_rf.get_report()
my_rf.get_tuning_graph()
my_rf.get_feature_importance_graph()
my_rf.get_misclassified_g_status()
my_rf.get_missing_amounts()
