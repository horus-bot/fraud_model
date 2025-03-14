import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 🚀 Load your dataset (Replace this with actual data loading)
def load_data():
    # If using a CSV file, uncomment the line below:
    # data = pd.read_csv("your_dataset.csv")

    # Simulate a dataset with 8 features and binary classification
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=8, random_state=42)
    
    return X, y

# 📌 Step 1: Load data
X, y = load_data()

# 📌 Step 2: Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Step 3: Initialize and train the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 📌 Step 4: Evaluate model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Trained Successfully! Accuracy: {accuracy:.4f}")

# 📌 Step 5: Save the trained model
model_path = "fraud_detection_model.pkl"
with open(model_path, "wb") as file:
    pickle.dump(model, file)

print(f"✅ Model saved successfully at {model_path}")
