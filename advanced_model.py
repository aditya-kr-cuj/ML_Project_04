# Import Required Libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Set the File Path
data_path = "/Users/adi/Adi_coading/Python3/ML_Project_4/student-mat.csv"  # Replace with your file's exact path

# Check if the File Exists
if not os.path.exists(data_path):
    print(f"Error: File not found at {data_path}. Please check the file path.")
    exit()

# Load Dataset
student_data = pd.read_csv(data_path, sep=";")

# Select Relevant Features for Prediction
selected_features = ['absences', 'failures', 'studytime', 'G1', 'G2']
target_column = 'G3'

# Extract Selected Features and Target
X = student_data[selected_features]
y = student_data[target_column]

# Normalize Selected Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)
print("R-Squared (R²):", r2)

# Visualization: Actual vs Predicted Grades
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color="blue")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "--r", linewidth=2)
plt.title("Actual vs Predicted Grades")
plt.xlabel("Actual Grades")
plt.ylabel("Predicted Grades")
plt.grid()
plt.show()

# Save the Model and Scaler
model_path = "student_grade_model.pkl"
scaler_path = "scaler.pkl"

try:
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved at: {os.path.abspath(model_path)}")
    print(f"Scaler saved at: {os.path.abspath(scaler_path)}")
except Exception as e:
    print(f"Error while saving files: {e}")
