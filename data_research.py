import pandas as pd


def create_df(random_forest=True):
    # Create the data frames
    erasmus_db = pd.read_csv("data/Erasmus.csv", sep=";")
    mutation_log_db = pd.read_csv("data/Mutatielog_id.csv", sep=";")
    toelichting_db = pd.read_excel("data/Toelichting.xlsx")

    # Create new columns for date and time
    mutation_log_db['DatumRegistratie'] = pd.to_datetime(
        mutation_log_db['DatumRegistratie'])

    mutation_log_db['date_column'] = mutation_log_db['DatumRegistratie'].dt.date
    mutation_log_db['time_column'] = mutation_log_db['DatumRegistratie'].dt.time

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

    for col in erasmus_db.filter(regex='^VT_'):
        try:
            erasmus_db[col] = erasmus_db[col].replace(
                999, None).astype(float)
        except Exception:
            continue

    if random_forest:
        erasmus_db.drop("SF_woord", axis=1)
        erasmus_db.drop("id", axis=1)
        erasmus_db.drop("prediction", axis=1)

    return erasmus_db


"""
# Since all the ids are assumed to be finished, we can just get the sum from min and max dates.
id_diff = mutation_log_db.groupby("id")["DatumRegistratie"].agg(
    lambda x: x.max() - x.min()).reset_index()
id_diff["DatumRegistratie"]

erasmus_db_with_time_diff = erasmus_db.merge(id_diff, on=["id"], how="left")

erasmus_db.head(60).to_excel("resulting_tables/changed_erasmus2.xlsx")
"""
