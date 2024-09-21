import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Creating the main window
window = tk.Tk()
window.title("Diabetes Prediction System")
window.configure(bg="lightgrey")  # Set the background color of the main window

# Loading the dataset
data = pd.read_csv("C:\\Users\\Arun Kathait\\Documents\\Csv File\\diabetes.csv")

# Separating the features and the target variable
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Splitting the dataset into 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM model
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Train KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

# Dictionary to store models
models = {
    'SVM': svm_model,
    'Random Forest': rf_model,
    'KNN': knn_model
}


def predict_diabetes():
    # Get the input values from the Entry fields
    preg = preg_entry.get()
    glucose = glucose_entry.get()
    bp = bp_entry.get()
    skin_thickness = skin_thickness_entry.get()
    insulin = insulin_entry.get()
    bmi = bmi_entry.get()
    dpf = dpf_entry.get()
    age = age_entry.get()

    # Check if any field is empty
    if not all([preg, glucose, bp, skin_thickness, insulin, bmi, dpf, age]):
        messagebox.showerror("Error", "Please fill all the values.")
        return

    # Convert the input values to the appropriate data types
    try:
        preg = int(preg)
        glucose = int(glucose)
        bp = int(bp)
        skin_thickness = int(skin_thickness)
        insulin = int(insulin)
        bmi = float(bmi)
        dpf = float(dpf)
        age = int(age)
    except ValueError:
        messagebox.showerror("Error", "Please enter valid values.")
        return

    # Create a new instance with the input values
    new_instance = pd.DataFrame(
        [[preg, glucose, bp, skin_thickness, insulin, bmi, dpf, age]],
        columns=X.columns
    )

    # Scale the new instance
    new_instance_scaled = scaler.transform(new_instance)

    # Get the selected model
    selected_model_name = model_var.get()
    selected_model = models[selected_model_name]

    # Make prediction
    prediction = selected_model.predict(new_instance_scaled)

    # Show the prediction in a message box
    messagebox.showinfo("Diabetes Prediction", f"Prediction using {selected_model_name}: {'Diabetic' if prediction[0] == 1 else 'Not Diabetic'}")


def clear_input_fields():
    # Clear all the input fields
    preg_entry.delete(0, tk.END)
    glucose_entry.delete(0, tk.END)
    bp_entry.delete(0, tk.END)
    skin_thickness_entry.delete(0, tk.END)
    insulin_entry.delete(0, tk.END)
    bmi_entry.delete(0, tk.END)
    dpf_entry.delete(0, tk.END)
    age_entry.delete(0, tk.END)


# Create labels and entry fields for the input variables
preg_label = tk.Label(window, text="Pregnancies:", bg="lightgray")
preg_label.pack()
preg_entry = tk.Entry(window, bg="white")
preg_entry.pack()

glucose_label = tk.Label(window, text="Glucose:", bg="lightgray")
glucose_label.pack()
glucose_entry = tk.Entry(window, bg="white")
glucose_entry.pack()

bp_label = tk.Label(window, text="Blood Pressure:", bg="lightgray")
bp_label.pack()
bp_entry = tk.Entry(window, bg="white")
bp_entry.pack()

skin_thickness_label = tk.Label(window, text="Skin Thickness:", bg="lightgray")
skin_thickness_label.pack()
skin_thickness_entry = tk.Entry(window, bg="white")
skin_thickness_entry.pack()

insulin_label = tk.Label(window, text="Insulin:", bg="lightgray")
insulin_label.pack()
insulin_entry = tk.Entry(window, bg="white")
insulin_entry.pack()

bmi_label = tk.Label(window, text="BMI:", bg="lightgray")
bmi_label.pack()
bmi_entry = tk.Entry(window, bg="white")
bmi_entry.pack()

dpf_label = tk.Label(window, text="Diabetes Pedigree Function:", bg="lightgray")
dpf_label.pack()
dpf_entry = tk.Entry(window, bg="white")
dpf_entry.pack()

age_label = tk.Label(window, text="Age:", bg="lightgray")
age_label.pack()
age_entry = tk.Entry(window, bg="white")
age_entry.pack()

# Create radio buttons for selecting the model
model_var = tk.StringVar(value='SVM')
svm_radio = tk.Radiobutton(window, text="SVM", variable=model_var, value='SVM', bg="lightgray")
svm_radio.pack()
rf_radio = tk.Radiobutton(window, text="Random Forest", variable=model_var, value='Random Forest', bg="lightgray")
rf_radio.pack()
knn_radio = tk.Radiobutton(window, text="KNN", variable=model_var, value='KNN', bg="lightgray")
knn_radio.pack()

# Create a button to predict diabetes
predict_button = tk.Button(window, text="Predict", command=predict_diabetes)
predict_button.pack()

# Create a button to clear the input fields
clear_button = tk.Button(window, text="Clear", command=clear_input_fields)
clear_button.pack()

# Start the main event loop
window.mainloop()
