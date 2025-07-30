import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pickle

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Features & Target
X = df.drop("Class", axis=1)
y = df["Class"]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle imbalance
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_scaled, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Optional: Save scaler if needed during prediction
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ model.pkl and scaler.pkl saved successfully.")
