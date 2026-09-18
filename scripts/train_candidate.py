import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


DATA = "data/raw/adult_income.csv"
MODEL = "models/candidate_model.pkl"


df = pd.read_csv(DATA)


for col in df.select_dtypes(
    include="object"
).columns:

    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
    )


df["target"] = (
    df["income"]
    .str.replace(".", "", regex=False)
    .map({
        "<=50K": 0,
        ">50K": 1
    })
)

df = df.drop(
    columns=["income"]
)

df = df.dropna(
    subset=["target"]
)


X = df.drop(
    columns=["target"]
)

y = df["target"]


categorical_columns = X.select_dtypes(
    include="object"
).columns.tolist()

numeric_columns = X.select_dtypes(
    exclude="object"
).columns.tolist()


numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])


categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])


preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_columns
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_columns
    )
])


# INTENTIONALLY weaker model
model = RandomForestClassifier(
    n_estimators=30,
    max_depth=5,
    random_state=123,
    n_jobs=-1
)


pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        model
    )
])


# Use EXACT SAME split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


pipeline.fit(
    X_train,
    y_train
)


os.makedirs(
    "models",
    exist_ok=True
)


joblib.dump(
    pipeline,
    MODEL
)


print("Candidate model trained.")
print(
    f"Model saved to {MODEL}"
)