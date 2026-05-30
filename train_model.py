import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("adult.csv")

# Replace missing values
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

# Drop columns
df.drop(columns=[
    'fnlwgt',
    'race',
    'native.country',
    'education.num'
], inplace=True)

# Store encoders
encoders = {}

# Encode categorical columns
cat_cols = [
    'workclass',
    'education',
    'marital.status',
    'occupation',
    'relationship',
    'sex',
    'income'
]

for col in cat_cols:

    le = LabelEncoder()

    df[col] = le.fit_transform(df[col])

    encoders[col] = le

# Features and target
X = df.drop('income', axis=1)
y = df['income']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Save files
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoders, "encoders.pkl")

print("Everything Saved Successfully")