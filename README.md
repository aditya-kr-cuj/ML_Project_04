# ML_Project_04

# **Student Performance Prediction Model**

## **Overview**
This project predicts students' final grades based on various features such as attendance, study habits, and previous grades. The model uses a **Random Forest Regressor**, and a **Tkinter GUI** makes it user-friendly for easy predictions.

---

## **Features**
1. **Data Processing**: Preprocessing student data for better model performance.
2. **Machine Learning**: Training and evaluating a Random Forest Regressor for prediction.
3. **Visualization**: Scatter plot of actual vs predicted grades for evaluation.
4. **User Interface**: A GUI built with Tkinter for user-friendly interaction.

---


## **Requirements**
- Python 3.x
- Required Libraries:
  
  pandas
  matplotlib
  scikit-learn
  joblib
  tkinter

## **Dataset**
The dataset used is based on student performance from a publicly available source. Key features include:

- Absences: Number of school days missed.
- Failures: Number of failed subjects.
- Study Time: Weekly study hours.
- G1 and G2: Grades from the first and second periods.
- G3: Final grade (target variable).

## **Model Evaluation**
The model was evaluated using:

Mean Squared Error (MSE)
R² Score
A scatter plot of Actual vs Predicted grades is generated during training for visual analysis.


