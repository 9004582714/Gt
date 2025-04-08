import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv(r"C:\Users\ANAND BANKAR\crop_yield.csv")
print(df.head())
print("column in dataframe",df.columns)
df.shape
print(df.shape)
df.isnull().sum()
print(df.isnull().sum())
df.info()
print(df.info())
df.duplicated().sum()
print(df.duplicated().sum())
df.describe()
print(df.describe())

# TRANSFORMING AVERAGE RAIN FALL PER YEAR
df["Annual_Rainfall"] 
print(df["Annual_Rainfall"])
def isStr(obj):
    try:
        float(obj)
        return False
    except:
        return True
to_drop = df[df["Annual_Rainfall"].apply(isStr)].index
df = df.drop(to_drop)
df
print(df)
df.info()

# GRAPH FREQUENCY vs state
plt.Figure(figsize=(10,20))
sns.countplot(y=df["State"])

# YEILD PER STATE
df.head(2)
print(df.head(2))
State = (df["State"].unique())
Production=[] 
for State in State:
    Production.append(df[df["State"]==State])
    
# YIELD PER STATE GRAPH
State = (df["State"])
Yield = (df["Yield"])
plt.Figure(figsize=(10,20))
sns.barplot(y=State,x=Yield)

# Graph
plt.figure(figsize=(10,20))
sns.countplot(y=df["Crop"])

# YIELD VS CROP
plt.figure(figsize=(10, 6))
sns.barplot(x=df["Yield"], y=df["Crop"], estimator=np.mean)
plt.title("Average Yield per Crop")
plt.xlabel("Yield")
plt.ylabel("Crop")
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error

# Load and preview dataset
print("Initial Preview:\n", df.head())

# Basic checks
print("Columns:", df.columns)
print("Shape:", df.shape)
print("Missing values:\n", df.isnull().sum())
print("Duplicated rows:", df.duplicated().sum())
print("Data Info:\n")
df.info()

# Drop non-numeric rainfall values
def isStr(obj):
    try:
        float(obj)
        return False
    except:
        return True

to_drop = df[df["Annual_Rainfall"].apply(isStr)].index
df = df.drop(to_drop)
df["Annual_Rainfall"] = df["Annual_Rainfall"].astype(float)

print("After cleaning:\n", df.info())

# Visualizations
plt.figure(figsize=(10, 6))
sns.countplot(y=df["State"])
plt.title("Frequency of Entries per State")
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(y=df["State"], x=df["Yield"])
plt.title("Yield per State")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(y=df["Crop"])
plt.title("Crop Distribution")
plt.show()

# Encode categorical features
label_encoders = {}
class_options = {}
for col in df.columns:
    if df[col].dtype == 'object' and col != 'Yield':
        df[col] = df[col].str.strip()
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        class_options[col] = list(le.classes_)

# Define features and targets (predicting both Yield and Production)
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

# Save model, scaler, and encoders using joblib
joblib.dump(model, 'yield_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'encoders.pkl')

# Evaluation
y_pred = model.predict(X_test_scaled)
print(f"Model Mean Absolute Error (Yield): {mean_absolute_error(y_test['Yield'], y_pred[:, 0])}")
print(f"Model Mean Absolute Error (Production): {mean_absolute_error(y_test['Production'], y_pred[:, 1])}")

# Prediction function
def predict_yield(user_input):
    input_df = pd.DataFrame([user_input])
    for col, le in label_encoders.items():
        if isinstance(user_input[col], str):
            try:
                cleaned_val = user_input[col].strip()
                input_df[col] = le.transform([cleaned_val])
            except ValueError:
                print(f"Invalid input for {col}: '{user_input[col]}' not in training data.")
                print(f"Available options for {col}: {list(le.classes_)}")
                return None
        else:
            input_df[col] = [user_input[col]]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    return {"Yield": prediction[0], "Production": prediction[1]}

# Sample prediction
if __name__ == "__main__":
    print("\n### Example: Predicting Yield and Production from a Sample Row ###")
    sample_row = X_test.iloc[0].to_dict()
    actual_yield = y_test.iloc[0]
    predicted_yield = predict_yield(sample_row)

    print("Input Features:")
    for key, value in sample_row.items():
        print(f"{key}: {value}")
    print(f"Actual Yield: {actual_yield['Yield']}")
    print(f"Actual Production: {actual_yield['Production']}")
    print(f"Predicted Yield: {predicted_yield['Yield']}")
    print(f"Predicted Production: {predicted_yield['Production']}")

    # Custom input
    print("\n### Predict Your Own ###")
    user_input = {}
    for col in features.columns:
        if col in label_encoders:
            print(f"\nAvailable options for {col}: {list(label_encoders[col].classes_)}")
            user_input[col] = input(f"Enter {col} (string): ").strip()
        else:
            user_input[col] = float(input(f"Enter {col} (number): "))
    user_prediction = predict_yield(user_input)
    print(f"Predicted Crop Yield: {user_prediction['Yield']}")
    print(f"Predicted Crop Production: {user_prediction['Production']}")

    # ✅ Save model as dtr.pkl and preprocessing tools as preprocessor.pkl
    import pickle
    with open("dtr.pkl", "wb") as f:
        pickle.dump(model, f)

    preprocessor_bundle = {
        "scaler": scaler,
        "label_encoders": label_encoders
    }
    with open("preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor_bundle, f)

    print("✅ Model saved as 'dtr.pkl' and preprocessor saved as 'preprocessor.pkl'.")
