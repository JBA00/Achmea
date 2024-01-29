import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from data_research import create_df
from insurance_classifier import InsuranceClassifier

erasmus_db = create_df()

# Move it later to data_research
erasmus_db_for_training = erasmus_db[erasmus_db['status'] != "G"]
model = {"Random Forest": {
    'classifier__max_features': list(range(80, 200, 10))},
    "Lasso":
    {'classifier__C':  np.arange(0.001, 0.105, 0.005)}}


my_rf = InsuranceClassifier("Random Forest", erasmus_db, erasmus_db_for_training,
                            model["Random Forest"])

my_rf.minimize_function(pd.factorize(my_rf.erasmus_db["status"])[
                        0], my_rf.erasmus_db["predictions"])
my_rf.grid_search.cv_results_["mean_test_score"]
my_rf.grid_search.best_params_
my_rf.grid_search.best_params_
my_rf.get_report()
my_rf.get_tuning_graph()
my_rf.get_feature_importance_graph()
my_rf.get_misclassified_g_status()
my_rf.get_missing_amounts()
my_rf.compare_with_high_low_predictions()

my_rf.save_predictions()

my_rf.save_the_model()

my_rf.save_the_model()

my_lasso = InsuranceClassifier("Lasso", erasmus_db, erasmus_db_for_training,
                               model["Lasso"])


my_lasso.get_report()
my_lasso.get_tuning_graph()
my_lasso.get_feature_importance_graph()
my_lasso.get_misclassified_g_status()
my_lasso.get_missing_amounts()
my_lasso.compare_with_high_low_predictions()
