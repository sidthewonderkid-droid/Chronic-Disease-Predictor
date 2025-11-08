import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, hamming_loss
from sklearn.preprocessing import StandardScaler
import joblib

# Load and examine data
df = pd.read_csv('asthma.csv')
print("\nData Shape:", df.shape)

# Display class distribution
severity_cols = ["Severity_Mild", "Severity_Moderate", "Severity_None"]
print("\nClass Distribution:")
for col in severity_cols:
    print(f"\n{col}:")
    print(df[col].value_counts(normalize=True))

# Prepare features and targets
X = df.drop(severity_cols, axis=1)
y = df[severity_cols]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y["Severity_Mild"])

# Create and train model with better parameters
model = RandomForestClassifier(
    n_estimators=500,  # More trees
    max_depth=10,      # Prevent overfitting
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("\nModel Evaluation:")
print("Hamming Loss:", hamming_loss(y_test, y_pred))

# Evaluate each severity level separately
for i, severity in enumerate(severity_cols):
    print(f"\nMetrics for {severity}:")
    print(classification_report(y_test[severity], y_pred[:, i]))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Cross-validation score
cv_scores = cross_val_score(model, X, y["Severity_Mild"], cv=5)
print("\nCross-validation accuracy scores:", cv_scores)
print("Average CV accuracy:", cv_scores.mean())

# Save the improved model
joblib.dump(model, "asthma_model_improved.joblib")