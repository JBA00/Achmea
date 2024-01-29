import pandas as pd


def create_df(what_for="random_forest"):
    # Create the data frames
    erasmus_db = pd.read_csv("data/Erasmus.csv", sep=";")

    # Fix float values.
    for col in erasmus_db.columns:
        try:
            erasmus_db[col] = erasmus_db[col].str.replace(
                ',', '.').astype(float)
        except Exception:
            continue

    # Translate the column into lists, so that maybe we can calculate the number of red flags.
    erasmus_db['SF_woord'] = erasmus_db['SF_woord'].str.replace(
        ' ', '').str.split(',')

    erasmus_db['SF_woord_count'] = [len(x) if isinstance(
        x, list) else None for x in erasmus_db['SF_woord']]

    if what_for == "random_forest":
        erasmus_db = erasmus_db.drop("SF_woord", axis=1)

    erasmus_db = erasmus_db.rename(columns={"prediction": "old_predictions"})

    erasmus_db["SF_woord_count"].fillna(0, inplace=True)

    # Here are some assumptions about are made, to replace null values- probs better to discuss.
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

    if what_for == "eda":
        erasmus_db = erasmus_db[["vm", "polis_2", "polis_5",
                                 "age", "status", "LG1", "SF_woord_count"]]

    return erasmus_db
