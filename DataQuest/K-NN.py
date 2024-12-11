import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Load the dataset
file_path = "shuffled_dataset.xlsx"
df = pd.read_excel(file_path)

# Preview dataset to understand structure
print(df.head())

# Data Preprocessing
# 1. Handle missing values (if any)
df = df.dropna()  # Remove rows with missing values (optional: could also impute)

# 2. If there are categorical variables, apply one-hot encoding
df = pd.get_dummies(df)

# 3. Split into features (X) and target (y)
# Assuming the last column is the target; adjust if necessary
X = df.drop('Vehicle_Type', axis=1)  # Replace 'Vehicle_Type' with the actual column name of your target
y = df['Crash_Severity']

# Manually splitting into training (80%) and testing (20%)
train_size = int(0.8 * len(X))  # 80% training, 20% testing
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Standardize the features (feature scaling)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Set k value to 17
k = 17

# Function to train k-NN classifier
def train_knn(k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model

# Train model with k=17
model = train_knn(k)
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix and Classification Report
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title(f'Confusion Matrix for k={k}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Print Classification Report
print("Classification Report:")
print(class_report)

# Print Time and Space Complexity
print("\nTime Complexity and Space Complexity of k-NN:")
print("Time Complexity: O(n * d + n * log(k)), where n is the number of training samples and d is the number of features.")
print("Space Complexity: O(n * d), where n is the number of training samples and d is the number of features.")