
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error

# Load CSV
df = pd.read_csv(r"crop_yield.csv")

# Clean non-numeric rainfall
def isStr(obj):
    try:
        float(obj)
        return False
    except:
        return True

to_drop = df[df["Annual_Rainfall"].apply(isStr)].index
df = df.drop(to_drop)
df["Annual_Rainfall"] = df["Annual_Rainfall"].astype(float)

# Remove whitespace in strings
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.strip()

# Encode categorical columns
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object' and col not in ['Yield', 'Production']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Define features and targets
features = df.drop(columns=['Yield', 'Production'])
target = df[['Yield', 'Production']]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train_scaled, y_train)

# Save model and preprocessors
with open("dtr.pkl", "wb") as f:
    pickle.dump(model, f)

preprocessor_bundle = {
    "scaler": scaler,
    "label_encoders": label_encoders
}
with open("preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor_bundle, f)

# Evaluate model
y_pred = model.predict(X_test_scaled)
print(f"Model Mean Absolute Error (Yield): {mean_absolute_error(y_test['Yield'], y_pred[:, 0])}")
print(f"Model Mean Absolute Error (Production): {mean_absolute_error(y_test['Production'], y_pred[:, 1])}")

print("✅ Model saved as 'dtr.pkl' and preprocessor as 'preprocessor.pkl'")
