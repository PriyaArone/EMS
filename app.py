pip install streamlit


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os
import matplotlib.pyplot as plt
from google.colab import files

# Upload dataset
uploaded = files.upload()

# Load the dataset
df = pd.read_csv('paddy_energy_dataset.csv')

# Select input features and target
X = df.iloc[:, 1:13]  # Input columns (features)
y = df['Total Output (MJ/ha)']  # Target column (output)

# Scale the input features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Create and train the model
model = MLPRegressor(hidden_layer_sizes=(12, 8, 1), max_iter=1000, random_state=1)
model.fit(X_train, y_train)

# Predict the output
y_pred = model.predict(X_test)

# Print the R² and RMSE scores
print("R²:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Create a scatter plot for actual vs predicted values
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Total Output (MJ/ha)')
plt.grid(True)
plt.show()

# Ensure the model directory exists
os.makedirs('model', exist_ok=True)

# Save the trained model and scaler as .pkl files
joblib.dump(model, 'model/trained_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

# Optionally, download the model files for use in your app
files.download('model/trained_model.pkl')
files.download('model/scaler.pkl')


import os

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Now save your model and scaler
import joblib
joblib.dump(model, 'model/trained_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')


import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('model/trained_model.pkl')
scaler = joblib.load('model/scaler.pkl')

# Streamlit app to collect user inputs
st.title("Farm Energy Efficiency Prediction")

# Collect user inputs
human_labour = st.number_input("Human Labour (h)", min_value=0.0)
machinery_use = st.number_input("Machinery Use (h)", min_value=0.0)
diesel = st.number_input("Diesel Fuel (L)", min_value=0.0)
nitrogen = st.number_input("Nitrogen Fertilizer (kg)", min_value=0.0)
phosphate = st.number_input("Phosphate Fertilizer (kg)", min_value=0.0)
potassium = st.number_input("Potassium Fertilizer (kg)", min_value=0.0)
zinc = st.number_input("Zinc Fertilizer (kg)", min_value=0.0)
fym = st.number_input("Farmyard Manure (kg)", min_value=0.0)
chemicals = st.number_input("Chemicals (Pesticides/Herbicides) (kg)", min_value=0.0)
water = st.number_input("Water for Irrigation (m³)", min_value=0.0)
electricity = st.number_input("Electricity (kWh)", min_value=0.0)
seeds = st.number_input("Seeds (kg)", min_value=0.0)

# Prepare input array
input_data = np.array([[human_labour, machinery_use, diesel, nitrogen, phosphate,
                        potassium, zinc, fym, chemicals, water, electricity, seeds]])

# Scale the input data
input_scaled = scaler.transform(input_data)

# Predict using the model
prediction = model.predict(input_scaled)

# Show the prediction
st.success(f"Predicted Total Energy Output (MJ/ha): {prediction[0]:.2f}")
