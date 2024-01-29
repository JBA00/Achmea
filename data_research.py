import pandas as pd


def create_df(what_for="random_forest"):
    """This function creates a dataset for the further analysis. It replaces
    the values, changes the formats of the columns and drops the features
    which can not be further used. For the further understanding of the logic
    for the certain assumptions you can refer to the Technical Footnote.

    Args:
        what_for (str, optional): The parameter which defines the Machine 
        Learning algorithm. Defaults to "random_forest".

    Returns:
        Data Frame: adjusted data frame prepared for the analysis
    """
    # Get data from the excel file.
    erasmus_db = pd.read_csv("data/Erasmus.csv", sep=";")

    # Fix float values, by changing the separator from ,  to . .
    for col in erasmus_db.columns:
        try:
            erasmus_db[col] = erasmus_db[col].str.replace(
                ',', '.').astype(float)
        except Exception:
            continue

    # Create the column, which would account the number of red flag words
    # in the application.
    erasmus_db['SF_woord'] = erasmus_db['SF_woord'].str.replace(
        ' ', '').str.split(',')

    erasmus_db['SF_woord_count'] = [len(x) if isinstance(
        x, list) else None for x in erasmus_db['SF_woord']]

    # Drop the column which can not be used in random forest estimation.
    if what_for == "random_forest":
        erasmus_db = erasmus_db.drop("SF_woord", axis=1)

    erasmus_db = erasmus_db.rename(columns={"prediction": "old_predictions"})

    # It was discovered that the models used do not tolerate NaN values,
    # therefore those were filled in with certain values, taking into account
    # the specificity of the feature. Information regarding the assumptions made
    # can also be found in the technical footnote.
    erasmus_db["SF_woord_count"].fillna(0, inplace=True)

    erasmus_db["woord_74"].fillna(0, inplace=True)
    erasmus_db["woord_121"].fillna(0, inplace=True)

    erasmus_db["D_LG4_1j"].fillna(0, inplace=True)
    erasmus_db["D_cat_1g"].fillna(0, inplace=True)
    erasmus_db["D_cat_4b"].fillna(0, inplace=True)

    erasmus_db["VT_cat_1b"].fillna(999, inplace=True)
    erasmus_db["VT_cat_1d"].fillna(999, inplace=True)
    erasmus_db["VT_cat_1g"].fillna(999, inplace=True)
    erasmus_db["VT_cat_2c"].fillna(999, inplace=True)
    erasmus_db["VT_cat_4b"].fillna(999, inplace=True)
    erasmus_db["VT_cat_5e"].fillna(999, inplace=True)
    erasmus_db["VT_cat_8a"].fillna(999, inplace=True)
    erasmus_db["VT_cat_9a"].fillna(999, inplace=True)
    erasmus_db["VT_cat_10b"].fillna(999, inplace=True)
    erasmus_db["VT_cat_10g.2"].fillna(999, inplace=True)

    # The list of the features taken for the exploration data analytics.
    if what_for == "eda":
        erasmus_db = erasmus_db[["vm", "polis_2", "polis_5",
                                 "age", "status", "LG1", "SF_woord_count"]]

    return erasmus_db
