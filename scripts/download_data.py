import os
import pandas as pd


URL = (
    "https://archive.ics.uci.edu/"
    "ml/machine-learning-databases/"
    "adult/adult.data"
)

OUTPUT = "data/raw/adult_income.csv"


columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income"
]


os.makedirs(
    "data/raw",
    exist_ok=True
)

df = pd.read_csv(
    URL,
    names=columns,
    skipinitialspace=True
)

df.to_csv(
    OUTPUT,
    index=False
)

print(
    f"Downloaded {len(df)} rows"
)

print(
    f"Saved to {OUTPUT}"
)