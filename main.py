import numpy as np

from data_research import create_df
from insurance_classifier import InsuranceClassifier

# Create the data frame.
erasmus_db = create_df()

# Create the training data frame, which excludes rows with G status.
erasmus_db_for_training = erasmus_db[erasmus_db['status'] != "G"]

# Training grids for random forest and lasso.
model = {"Random Forest": {
    'classifier__max_features': list(range(80, 200, 10))},
    "Lasso":
    {'classifier__C':  np.arange(0.001, 0.105, 0.005)}}

# Initialize the Random Forest classifier.
my_rf = InsuranceClassifier("Random Forest", erasmus_db, erasmus_db_for_training,
                            model["Random Forest"])

# Functions to retrieve the insights about random forest classification.
my_rf.get_report()
my_rf.get_tuning_graph()
my_rf.get_feature_importance_graph()
my_rf.get_misclassified_g_status()
my_rf.get_missing_amounts()
my_rf.compare_with_high_low_predictions()

# Functions which save the predictions and model.
my_rf.save_predictions()
my_rf.save_the_model()

# Initialize the Lasso classifier.
my_lasso = InsuranceClassifier("Lasso", erasmus_db, erasmus_db_for_training,
                               model["Lasso"])

# Functions to retrieve the insights about lasso classification.
my_lasso.get_report()
my_lasso.get_tuning_graph()
my_lasso.get_feature_importance_graph()
my_lasso.get_misclassified_g_status()
my_lasso.get_missing_amounts()
my_lasso.compare_with_high_low_predictions()
