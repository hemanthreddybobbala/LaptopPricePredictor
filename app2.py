import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and preprocessing tools
model_path = "src/laptop_price_model.pkl"
ohe_path = "src/ohe_encoder.pkl"
scaler_path = "src/scaler.pkl"

try:
    model = joblib.load(model_path)
    ohe = joblib.load(ohe_path)
    scaler = joblib.load(scaler_path)
except FileNotFoundError as e:
    st.error(f"Error loading files: {e}. Please ensure model and encoder files exist in the correct path.")
    st.stop()

# Get expected feature names from the trained model (this must match the training DataFrame)
expected_feature_names = model.feature_names_in_

# Streamlit UI
st.title("💻 Laptop Price Prediction")
st.sidebar.header("Enter Laptop Specifications")

# User Inputs
brand_options = ["HP", "Lenovo", "Dell", "Asus", "Acer", "MSI", "Apple", "Samsung", "Other"]
processor_brand_options = ["Intel", "AMD", "Apple", "MediaTek", "Qualcomm", "Microsoft", "Other"]
processor_name_options = ["Intel Core i3 (11th Gen)", "Intel Core i5 (11th Gen)", "Intel Core i7 (11th Gen)", 
                          "AMD Ryzen 5", "AMD Ryzen 7", "Apple M1", "Apple M2", "MediaTek Octa-core", "Other"]
gpu_options = ["GeForce GTX 1650 GPU, 4 GB", "Radeon Graphics", "Intel Iris Xe", "GeForce RTX 3050 GPU, 4 GB", 
               "Unknown", "Other"]
gpu_brand_options = ["NVIDIA", "AMD", "Intel", "Apple", "Unknown", "Other"]

# Additional inputs (not encoded by ohe, just for display or future use)
ram_expandable_options = ["Not Expandable", "12 GB Expandable", "16 GB Expandable", "32 GB Expandable"]
ram_type_options = ["DDR4 RAM", "DDR5 RAM", "LPDDR5", "Other"]
display_type_options = ["LED", "LCD", "OLED", "IPS", "Other"]
display_options = ["11.6", "13.3", "14", "15.6", "16", "17.3", "Other"]
adapter_options = ["45", "65", "90", "120", "Other"]

brand = st.sidebar.selectbox("Brand", brand_options)
processor_brand = st.sidebar.selectbox("Processor Brand", processor_brand_options)
processor_name = st.sidebar.selectbox("Processor Name", processor_name_options)
gpu = st.sidebar.selectbox("GPU", gpu_options)
gpu_brand = st.sidebar.selectbox("GPU Brand", gpu_brand_options)
ram = st.sidebar.slider("RAM (GB)", 4, 64, 8, step=4)
ghz = st.sidebar.number_input("Processor Speed (GHz)", min_value=1.0, max_value=5.5, value=2.4, step=0.1)
ssd = st.sidebar.slider("SSD Storage (GB)", 0, 4096, 512, step=64)
hdd = st.sidebar.slider("HDD Storage (GB)", 0, 2048, 0, step=64)
battery_life = st.sidebar.slider("Battery Life (hrs)", 2, 100, 10, step=1)

# Additional inputs (not used in transformation)
ram_expandable = st.sidebar.selectbox("RAM Expandable", ram_expandable_options)
ram_type = st.sidebar.selectbox("RAM Type", ram_type_options)
display_type = st.sidebar.selectbox("Display Type", display_type_options)
display = st.sidebar.selectbox("Display Size (inches)", display_options)
adapter = st.sidebar.selectbox("Adapter (Watts)", adapter_options)

# Define columns used during model training
categorical_cols = ["Brand", "Processor_Name", "Processor_Brand", "GPU", "GPU_Brand"]
numerical_cols = ["RAM", "Ghz", "SSD", "HDD", "Battery_Life"]

# Create input DataFrame
input_data = pd.DataFrame(
    [[brand, processor_name, processor_brand, gpu, gpu_brand, ram, ghz, ssd, hdd, battery_life]],
    columns=categorical_cols + numerical_cols
)

# Calculate additional features that were present during training
input_data["Total_Storage"] = input_data["SSD"] + input_data["HDD"]
input_data["Price_per_RAM_GB"] = 1000  # Placeholder value

# Extended numerical columns (order must match training)
extended_numerical_cols = numerical_cols + ["Total_Storage", "Price_per_RAM_GB"]

# Preprocessing
# Separate numerical and categorical data
numerical_data = input_data[extended_numerical_cols].copy()
categorical_data = input_data[categorical_cols].copy()

# One-hot encode categorical features using the pre-fitted encoder
categorical_encoded = ohe.transform(categorical_data)
input_data_encoded = pd.DataFrame(categorical_encoded, columns=ohe.get_feature_names_out())

# Combine numerical and encoded categorical data
input_data_final = pd.concat([input_data_encoded.reset_index(drop=True), numerical_data.reset_index(drop=True)], axis=1)

# Reindex the final DataFrame to match exactly the features used during model training
input_data_final = input_data_final.reindex(columns=expected_feature_names, fill_value=0)

# Scale numerical features (using the pre-fitted scaler)
input_data_final[extended_numerical_cols] = scaler.transform(input_data_final[extended_numerical_cols])

# Predict Laptop Price
if st.sidebar.button("Predict Price"):
    try:
        price_prediction = model.predict(input_data_final)[0]
        st.success(f"💰 Estimated Laptop Price: ₹{price_prediction:,.2f}")
        
        # Calculate derived metrics for display
        total_storage = ssd + hdd
        price_per_ram = price_prediction / ram if ram > 0 else 0
        
        # Display input for verification
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Hardware Specs")
            st.write(f"**Brand:** {brand}")
            st.write(f"**Processor:** {processor_brand} {processor_name}")
            st.write(f"**Speed:** {ghz} GHz")
            st.write(f"**RAM:** {ram} GB ({ram_type})")
            st.write(f"**Storage:** {ssd} GB SSD + {hdd} GB HDD")
            st.write(f"**GPU:** {gpu_brand} {gpu}")
        with col2:
            st.subheader("Display & Power")
            st.write(f"**Display:** {display} inches {display_type}")
            st.write(f"**Battery:** {battery_life} hours")
            st.write(f"**Adapter:** {adapter} Watts")
            st.write(f"**RAM Expandable:** {ram_expandable}")
            st.write(f"**Total Storage:** {total_storage} GB")
            st.write(f"**Price/RAM GB:** ₹{price_per_ram:,.2f}")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.error(f"Input data columns: {list(input_data_final.columns)}")
        with st.expander("Debug Information"):
            st.write("### Model Expected Features:")
            st.write(expected_feature_names)
            st.write("### Your Input Data (Encoded):")
            st.write(input_data_final)

# st.write(\"\"\"  
# ### How It Works  
# This app uses a machine learning model trained on laptop specifications to predict retail prices.  
# Enter the details of your desired laptop configuration to get an estimated market price.  

#     #### Key Price Factors  
#     - RAM and processor specifications typically have the largest impact on price  
#     - Premium brands like Apple command higher prices regardless of specs  
#     - SSD storage is more expensive than HDD storage  
#     - Battery life and display quality also significantly affect pricing  \"\"\")
