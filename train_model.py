# Step 6.1: Import Machine Learning libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Step 5.3.1: Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split

# Step 5.3.2: Load the dataset
df = pd.read_csv("data/ai4i2020.csv")

# Step 5.3.3: Select input features
feature_columns = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type"
]

X = df[feature_columns].copy()

# Step 5.3.4: Encode product type
X = pd.get_dummies(X, columns=["Type"], dtype=int)

# Step 5.3.5: Select the target variable
y = df["Machine failure"]

# Step 5.3.6: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 5.3.7: Display the dataset split
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())


# Step 6.2.1: Initialize the Random Forest classifier
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

# Step 6.2.2: Train the model
rf_model.fit(X_train, y_train)

print("\nRandom Forest model trained successfully!")

# Step 6.2.3: Predict machine failures
y_pred = rf_model.predict(X_test)

# Step 6.2.4: Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\nRandom Forest Model Performance:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

# Step 6.2.5: Display detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Step 6.2.6: Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Step 7.1: Get predicted failure probabilities
y_prob = rf_model.predict_proba(X_test)[:, 1]

# Step 7.2: Test different classification thresholds
thresholds = [0.50, 0.40, 0.30, 0.20]

print("\nPrediction Threshold Comparison:")
print("-" * 65)

for threshold in thresholds:

    # Step 7.3: Convert probabilities into predictions
    y_pred_threshold = (y_prob >= threshold).astype(int)

    # Step 7.4: Calculate evaluation metrics
    precision = precision_score(
        y_test, y_pred_threshold, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred_threshold, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred_threshold, zero_division=0
    )

    # Step 7.5: Display results
    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1-score: {f1:.4f}"
    )


# Step 7.6.1: Compare original and threshold predictions
y_pred_50 = (y_prob >= 0.50).astype(int)

# Step 7.6.2: Count different predictions
differences = (y_pred != y_pred_50).sum()

print("\nPrediction Comparison:")
print("Number of different predictions:", differences)

# Step 7.6.3: Check exact probability ties
ties = (y_prob == 0.50).sum()

print("Predictions with probability exactly 0.50:", ties)


# Step 8.2.1: Import required libraries
import joblib
import os

# Step 8.2.2: Create the models directory if needed
os.makedirs("models", exist_ok=True)

# Step 8.2.3: Save the trained Random Forest model
joblib.dump(rf_model, "models/random_forest_model.pkl")

# Step 8.2.4: Save the feature column names
joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")

# Step 8.2.5: Confirm successful saving
print("\nModel saved successfully!")
print("Model location: models/random_forest_model.pkl")
print("Feature columns saved: models/feature_columns.pkl")
