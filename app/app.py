import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Loading model

model = joblib.load("models/house_price_model.pkl")

st.title("House Price Prediction App")



# Defining Inputs
GrLivArea = st.number_input("Living Area (sq ft)", value=1000)
OverallQual = st.slider("Overall Quality", 1, 10, 5)
GarageCars = st.number_input("Garage Capacity", value=1)
GarageArea = st.number_input("Garage Area", value=200)
TotalBsmtSF = st.number_input("Basement Area", value=500)
FullBath = st.number_input("Full Bathrooms", value=1)
YearBuilt = st.number_input("Year Built", value=2000)

Neighborhood = st.selectbox("Neighborhood", [
    'NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst', 'NridgHt'
])

KitchenQual = st.selectbox("Kitchen Quality", ['Ex', 'Gd', 'TA', 'Fa'])
ExterQual = st.selectbox("Exterior Quality", ['Ex', 'Gd', 'TA', 'Fa'])
BsmtQual = st.selectbox("Basement Quality", ['Ex', 'Gd', 'TA', 'Fa', 'Po'])




# FEATURE ENGINEERING
HouseAge = 2024 - YearBuilt




# PREDICTION 

if st.button("Predict Price"):
    try:
        input_data = pd.DataFrame([{
            'OverallQual': OverallQual,
            'GrLivArea': GrLivArea,
            'Neighborhood': Neighborhood,
            'TotalBsmtSF': TotalBsmtSF,
            'GarageCars': GarageCars,
            'GarageArea': GarageArea,
            'KitchenQual': KitchenQual,
            'ExterQual': ExterQual,
            'BsmtQual': BsmtQual,
            'FullBath': FullBath,
            'HouseAge': HouseAge
        }])

        prediction = model.predict(input_data)

       
        
        # converting log into actual price
        
        prediction = np.exp(prediction)

        st.success(f"Predicted House Price: ${prediction[0]:,.2f}")

    except Exception as e:
        st.error(f"Error: {e}")