import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Load dataset
df = pd.read_csv("Data/creditcard.csv")  # ✅ Ensure correct path

# Split features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train logistic regression model
model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
model.fit(X_train_scaled, y_train)

# ✅ Save model and scaler into one file
output_path = "model.pkl"
with open(output_path, "wb") as f:
    pickle.dump({"model": model, "scaler": scaler}, f)

print(f"✅ Successfully saved: {os.path.abspath(output_path)}")
