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
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np
import os


class InsuranceClassifier:
    """The class which allows to train the classifier using the best hyperparameters
    using k-fold cross validation. Also, it has internal functions which generate
    plots and reports to assess the quality of the model.
    """

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
        """This is the orchestration function which runs the processes of 
        feature engineering, cross validation and training.
        """
        self._prepare_variables()
        self._preprocess_data()
        self._cv_creation()
        self._classify_model()
        if os.path.exists("best_model.joblib"):
            self.load_the_model()
        else:
            self._find_best()

    def _prepare_variables(self):
        """This function creates training and testing datasets.
        """
        # The id and old predictions columns are saved for a later use, but
        # excluded from training and testing data sets.
        self.ids = self.erasmus_db["id"]
        self.erasmus_db = self.erasmus_db.drop("id", axis=1)
        self.erasmus_db_for_training = self.erasmus_db_for_training.drop(
            "id", axis=1)
        self.erasmus_db = self.erasmus_db.drop("old_predictions", axis=1)
        self.erasmus_db_for_training = self.erasmus_db_for_training.drop(
            "old_predictions", axis=1)

        # Factorization of target features from "S", "A" and "P" to 0, 1, 2.
        factor = pd.factorize(self.erasmus_db_for_training[self.target_col])
        self.erasmus_db_for_training[self.target_col] = factor[0]
        self.definitions = factor[1]

        # Definition of dependent and independent variables and division into
        # train-test datasets.
        self.X = self.erasmus_db_for_training.drop(self.target_col, axis=1)
        self.y = self.erasmus_db_for_training[self.target_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.25, random_state=self.random_state)

    def _preprocess_data(self):
        """Preprocessing numeric, categorical and boolean features.
        """
        self.numeric_features = self.X.select_dtypes(
            include=['int64', 'float64']).columns.tolist()

        self.categorical_features = self.X.select_dtypes(
            include=['object', 'string']).columns.tolist()

        self.boolean_features = self.X.select_dtypes(
            include=['bool']).columns.tolist()

        # Numeric features are normalized. For categorical features, dummy
        # variables are generated. Boolean features are unchanged.
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
        """Definition of the cross validation.
        """
        self.cv = StratifiedKFold(
            n_splits=self.number_of_folds, shuffle=True,
            random_state=self.random_state)

    def _classify_model(self):
        """This function organizes a workflow of the classifier.
        """

        # Scorings used for tuning and assessment of the model.
        self.scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average='weighted'),
            'recall': make_scorer(recall_score, average='weighted'),
            'f1': make_scorer(f1_score, average='weighted'),
            'costs': make_scorer(self.minimize_function, greater_is_better=True)
        }

        # Creation of Random Forest/Lasso models.
        if self.type_of_classifier == "Random Forest":
            self.classifier = RandomForestClassifier(
                n_estimators=400, random_state=self.random_state)
        elif self.type_of_classifier == "Lasso":
            self.classifier = LogisticRegression(multi_class='ovr',
                                                 penalty='l1', solver='saga')

        # Pipeline definition. It was decided to under sample the model, because
        # otherwise tuning process took unreasonable amount of time.
        self.model_pipeline = ImbPipeline([
            ('preprocessor', self.preprocessor),
            ('undersample', RandomUnderSampler(sampling_strategy='auto',
                                               random_state=self.random_state)),
            ('classifier', self.classifier)
        ])

        self.grid_search = GridSearchCV(estimator=self.model_pipeline,
                                        param_grid=self.param_grid,
                                        scoring=self.minimize_function,
                                        cv=self.cv,
                                        refit='costs', verbose=2, n_jobs=-1)

    def _find_best(self):
        """This function finds the best hyperparameters for the model,
        and creates predictions on the full data frame.
        """

        # Choosing the best hyperparameters.
        self.grid_search.fit(self.X_train, self.y_train)
        self.best_model = self.grid_search.best_estimator_

        # Create predictions.
        self.predictions = self.best_model.predict(self.X_test)
        self.predictions_full = self.best_model.predict(
            self.erasmus_db.drop("status", axis=1))
        self.erasmus_db["predictions"] = self.predictions_full

        # Defactorization of predictions column.
        defactorized_column = pd.Categorical.from_codes(
            self.erasmus_db['predictions'], self.definitions)
        self.erasmus_db['predictions_defactor'] = defactorized_column

    def load_the_model(self):
        """If the previously trained saved in the repository, than it is used to 
        generate predictions.
        """
        self.best_model = joblib.load("best_model.joblib")
        self.predictions = self.best_model.predict(self.X_test)
        self.predictions_full = self.best_model.predict(
            self.erasmus_db.drop("status", axis=1))
        self.erasmus_db["predictions"] = self.predictions_full
        defactorized_column = pd.Categorical.from_codes(
            self.erasmus_db['predictions'], self.definitions)
        self.erasmus_db['predictions_defactor'] = defactorized_column

    def get_report(self):
        """This function creates a report with detailed scoring values.
        """
        report = classification_report(self.y_test, self.predictions)
        print(
            f"Classification Report([0, 1, 2] -{pd.Categorical.from_codes([0, 1, 2], self.definitions).tolist()}):\n", report)

    def get_tuning_graph(self):
        """This function generates a graph with the results of the tuning of
        hyperparameters.
        """
        results = self.grid_search.cv_results_
        if self.type_of_classifier == "Random Forest":
            features = self.param_grid['classifier__max_features']
            what_we_are_tuning = "max_features"
        else:
            features = self.param_grid['classifier__C']
            what_we_are_tuning = "penalty"

        # Plotting the metrics for different max_features values.
        plt.figure(figsize=(10, 6))

        plt.plot(features, results['mean_test_score'],
                 label='Custom Loss', marker='o')

        plt.title(f'Model Metrics for Different {what_we_are_tuning} Values')
        plt.xlabel(what_we_are_tuning)
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)

        plt.show()

    def get_feature_importance_graph(self):
        """This function generates the plots with features importance.
        """

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
        """Generate the plot with statuses that were predicted for the actual G clients.
        """
        self.g_data = self.erasmus_db[self.erasmus_db[self.target_col] == "G"][[
            self.target_col, "predictions_defactor"]]
        
        plt.figure(figsize=(10, 6))
  
        # Calculate percentages
        total = len(self.g_data)
        percentages = self.g_data["predictions_defactor"].value_counts(normalize=True) * 100
    
        # Plot percentages
        percentages.plot(kind='bar', color='skyblue')
    
        # Add percentage labels
        for i, percentage in enumerate(percentages):
            plt.text(i, percentage + 1, f'{percentage:.2f}%', ha='center')

        plt.title('Pergentage of assigned categories for G customers')
        plt.xlabel('Categories')
        plt.ylabel('Percentage')
        plt.show()

    def get_missing_amounts(self):
        """This function is used to calculate the potential costs which would 
        occur with the created predictions. Created based on Achmeas suggestion.
        """
        missed_amount = len(self.erasmus_db[(self.erasmus_db["predictions_defactor"] == "A") & (self.erasmus_db["status"] == "S")]) * 100 + len(self.erasmus_db[(self.erasmus_db["predictions_defactor"] == "S") & (self.erasmus_db["status"] == "P")]) * 1500 + len(
            self.erasmus_db[(self.erasmus_db["predictions_defactor"] == "S") & (self.erasmus_db["status"] == "A")]) * 15000 + len(self.erasmus_db[(self.erasmus_db["predictions_defactor"] == "P") & (self.erasmus_db["status"] == "A")]) * 15000
        print(missed_amount)

    def minimize_function(self, estimator, X_test, y_test):
        """This function is used as a loss function to choose the best 
        hyperparameters for the model.

        Args:
            estimator (model): Created model
            X_test (series): Pandas series with independent variable.
            y_test (series): Pandas series with dependent variable.

        Returns:
            integer: the costs created.
        """
        y_true = np.array(y_test)
        y_pred = estimator.predict(X_test)

        # Calculate the different conditions
        condition_1 = np.sum((y_pred == 1) & (y_true == 0)) * 100
        condition_2 = np.sum((y_pred == 0) & (y_true == 2)) * 1500
        condition_3 = np.sum((y_pred == 0) & (y_true == 1)) * 15000
        condition_4 = np.sum((y_pred == 2) & (y_true == 1)) * 15000

        # Total missed amount
        missed_amount = 1 * (condition_1 + condition_2 +
                             condition_3 + condition_4)
        return -missed_amount

    def compare_with_high_low_predictions(self):
        """This function shows the recall for high-low predictions of the model.
        """
        comparison_table = self.erasmus_db[["status", "predictions_defactor"]]
        comparison_table["new_predictions"] = np.where(
            comparison_table["predictions_defactor"] == "S", 0, 1)
        comparison_table["actual_status"] = np.where(
            comparison_table["status"] == "S", 0, 1)

        new_recall = recall_score(
            comparison_table["actual_status"], comparison_table["new_predictions"])

        print(
            f"Overall recall from the model - {new_recall}")

    def save_predictions(self):
        """This function saves predicted values and ids to the new excel file.
        """
        predictions = pd.DataFrame()
        predictions["id"] = self.ids
        predictions["predictions"] = self.erasmus_db["predictions_defactor"]
        predictions.to_excel("predictions.xlsx")

    def save_the_model(self):
        """This function saves the trained model for the future reuse. 
        """
        joblib.dump(self.best_model, 'best_model.joblib')
