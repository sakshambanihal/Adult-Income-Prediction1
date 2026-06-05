import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv("adult.csv")

# Replace missing values
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

df.drop(columns=[
    'fnlwgt',
    'race',
    'native.country',
    'education.num'
], inplace=True)

encoders = {}
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

X = df.drop('income', axis=1)
y = df['income']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoders, "encoders.pkl")

print("Everything Saved Successfully")