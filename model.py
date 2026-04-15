# model.py

import pandas as pd
import numpy as np
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
telemetry = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "arnabbiswas1/microsoft-azure-predictive-maintenance",
    "PdM_telemetry.csv"
)

failures = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "arnabbiswas1/microsoft-azure-predictive-maintenance",
    "PdM_failures.csv"
)

# Preprocess
telemetry['datetime'] = pd.to_datetime(telemetry['datetime'])
failures['datetime'] = pd.to_datetime(failures['datetime'])

telemetry = telemetry.sort_values(['machineID', 'datetime'])
telemetry['failure_next_24h'] = 0

for i in range(len(failures)):
    mid = failures.iloc[i]['machineID']
    time = failures.iloc[i]['datetime']

    mask = (
        (telemetry['machineID'] == mid) &
        (telemetry['datetime'] >= time - pd.Timedelta(hours=24)) &
        (telemetry['datetime'] < time)
    )
    telemetry.loc[mask, 'failure_next_24h'] = 1

# Feature Engineering
telemetry['volt_mean'] = telemetry.groupby('machineID')['volt'].transform(lambda x: x.rolling(3).mean())
telemetry['rotate_mean'] = telemetry.groupby('machineID')['rotate'].transform(lambda x: x.rolling(3).mean())
telemetry['pressure_mean'] = telemetry.groupby('machineID')['pressure'].transform(lambda x: x.rolling(3).mean())
telemetry['vibration_mean'] = telemetry.groupby('machineID')['vibration'].transform(lambda x: x.rolling(3).mean())

telemetry = telemetry.dropna()

features = [
    'volt', 'rotate', 'pressure', 'vibration',
    'volt_mean', 'rotate_mean', 'pressure_mean', 'vibration_mean'
]

X = telemetry[features]
y = telemetry['failure_next_24h']

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100, max_depth=10)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")

print("✅ Model trained and saved as model.pkl")