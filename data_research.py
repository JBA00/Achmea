import pandas as pd

# Create the data frames
erasmus_db = pd.read_csv("data/Erasmus.csv", sep=";")
mutation_log_db = pd.read_csv("data/Mutatielog_id.csv", sep=";")
toelichting_db = pd.read_excel("data/Toelichting.xlsx")

# Create new columns for date and time
mutation_log_db['DatumRegistratie'] = pd.to_datetime(
    mutation_log_db['DatumRegistratie'])

mutation_log_db['date_column'] = mutation_log_db['DatumRegistratie'].dt.date
mutation_log_db['time_column'] = mutation_log_db['DatumRegistratie'].dt.time
mutation_log_db.drop('DatumRegistratie', axis=1, inplace=True)

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


"""
erasmus_db.head(60).to_excel("resulting_tables/changed_erasmus2.xlsx")
"""
