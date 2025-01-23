import tkinter as tk
from tkinter import messagebox
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load("student_grade_model.pkl")
scaler = joblib.load("scaler.pkl")

# Function to Predict
def predict_grade():
    try:
        # Get user inputs
        inputs = [
            float(absences.get()), float(failures.get()), 
            float(studytime.get()), float(G1.get()), float(G2.get())
        ]
        # Normalize input (using relevant scaler)
        inputs = scaler.transform([inputs])
        result = model.predict(inputs)[0]
        messagebox.showinfo("Prediction", f"Predicted Final Grade: {result:.2f}")
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {e}")

# Tkinter GUI
app = tk.Tk()
app.title("Student Grade Predictor")

tk.Label(app, text="Absences (Days):").grid(row=0, column=0)
absences = tk.Entry(app)
absences.grid(row=0, column=1)

tk.Label(app, text="Failures (Count):").grid(row=1, column=0)
failures = tk.Entry(app)
failures.grid(row=1, column=1)

tk.Label(app, text="Study Time (hours):").grid(row=2, column=0)
studytime = tk.Entry(app)
studytime.grid(row=2, column=1)

tk.Label(app, text="First Period Grade (G1):").grid(row=3, column=0)
G1 = tk.Entry(app)
G1.grid(row=3, column=1)

tk.Label(app, text="Second Period Grade (G2):").grid(row=4, column=0)
G2 = tk.Entry(app)
G2.grid(row=4, column=1)

tk.Button(app, text="Predict", command=predict_grade).grid(row=5, column=0, columnspan=2)

app.mainloop()
