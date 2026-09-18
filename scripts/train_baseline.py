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
MODEL = "models/baseline_model.pkl"
TEST = "data/golden_sets/adult_income_test.csv"


df = pd.read_csv(DATA)

# Clean strings
for col in df.select_dtypes(
    include="object"
).columns:

    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
    )


# Convert target
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

# Remove missing target rows
df = df.dropna(
    subset=["target"]
)

X = df.drop(
    columns=["target"]
)

y = df["target"]


# Identify columns
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


model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    random_state=42,
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

os.makedirs(
    "data/golden_sets",
    exist_ok=True
)


joblib.dump(
    pipeline,
    MODEL
)


# IMPORTANT:
# Save the test set with target.
test_df = X_test.copy()

test_df["target"] = y_test.values

test_df.to_csv(
    TEST,
    index=False
)


print("Baseline model trained.")
print(
    f"Model saved to {MODEL}"
)

print(
    f"Golden test set saved to {TEST}"
)