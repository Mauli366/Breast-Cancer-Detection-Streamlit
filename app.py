
import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open('breast_cancer_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Breast Cancer Detection App")

st.write("Enter the values below:")
# ===== INPUT FIELDS (31 FEATURES) =====

patient_id = st.number_input("id")
radius_mean = st.number_input("radius_mean")
texture_mean = st.number_input("texture_mean")
perimeter_mean = st.number_input("perimeter_mean")
area_mean = st.number_input("area_mean")
smoothness_mean = st.number_input("smoothness_mean")
compactness_mean = st.number_input("compactness_mean")
concavity_mean = st.number_input("concavity_mean")
concave_points_mean = st.number_input("concave points_mean")
symmetry_mean = st.number_input("symmetry_mean")
fractal_dimension_mean = st.number_input("fractal_dimension_mean")

radius_se = st.number_input("radius_se")
texture_se = st.number_input("texture_se")
perimeter_se = st.number_input("perimeter_se")
area_se = st.number_input("area_se")
smoothness_se = st.number_input("smoothness_se")
compactness_se = st.number_input("compactness_se")
concavity_se = st.number_input("concavity_se")
concave_points_se = st.number_input("concave points_se")
symmetry_se = st.number_input("symmetry_se")
fractal_dimension_se = st.number_input("fractal_dimension_se")

radius_worst = st.number_input("radius_worst")
texture_worst = st.number_input("texture_worst")
perimeter_worst = st.number_input("perimeter_worst")
area_worst = st.number_input("area_worst")
smoothness_worst = st.number_input("smoothness_worst")
compactness_worst = st.number_input("compactness_worst")
concavity_worst = st.number_input("concavity_worst")
concave_points_worst = st.number_input("concave points_worst")
symmetry_worst = st.number_input("symmetry_worst")
fractal_dimension_worst = st.number_input("fractal_dimension_worst")

# ===== PREDICTION =====

if st.button("Predict"):

    input_data = np.array([[
        patient_id,
        radius_mean,
        texture_mean,
        perimeter_mean,
        area_mean,
        smoothness_mean,
        compactness_mean,
        concavity_mean,
        concave_points_mean,
        symmetry_mean,
        fractal_dimension_mean,
        radius_se,
        texture_se,
        perimeter_se,
        area_se,
        smoothness_se,
        compactness_se,
        concavity_se,
        concave_points_se,
        symmetry_se,
        fractal_dimension_se,
        radius_worst,
        texture_worst,
        perimeter_worst,
        area_worst,
        smoothness_worst,
        compactness_worst,
        concavity_worst,
        concave_points_worst,
        symmetry_worst,
        fractal_dimension_worst
    ]])
    #st.write("Input shape:", input_data.shape)
   # st.write("Scaler expects:", scaler.n_features_in_)

    # Scale input
    input_data_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_data_scaled)

    # Output
    if prediction[0] == 1:
        st.success("Benign Tumor")
    else:
        st.error("Malignant Tumor")