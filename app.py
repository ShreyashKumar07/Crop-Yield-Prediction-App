import streamlit as st
import pandas as pd
import joblib

# Set page title and layout
st.set_page_config(page_title="Crop Yield Predictor", layout="wide")
st.title("🌾 Crop Yield Predictor")
st.markdown("Enter the details below to predict crop yield (hg/ha) based on historical data.")

# Define the path to files
BASE_PATH = r"C:\Users\shrey\OneDrive\Desktop\crop\Predictor"

# Load the original dataset and preprocessing objects
@st.cache_data
def load_data_and_models():
    df = pd.read_csv(f"{BASE_PATH}\\yield_df.csv")
    mlp = joblib.load(f"{BASE_PATH}\\mlp_regressor.pkl")
    scaler = joblib.load(f"{BASE_PATH}\\scaler.pkl")
    le = joblib.load(f"{BASE_PATH}\\label_encoder_area.pkl")
    ohe = joblib.load(f"{BASE_PATH}\\one_hot_encoder_item.pkl")
    return df, mlp, scaler, le, ohe

# Load data and models
try:
    df, mlp, scaler, le, ohe = load_data_and_models()
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# Get unique values for dropdowns
areas = sorted(df['Area'].unique())
items = sorted(df['Item'].unique())

# Define numerical column ranges for user input (based on yield_df.csv)
year_min, year_max = int(df['Year'].min()), int(df['Year'].max())
rain_min, rain_max = 51.0, 3240.0  # From the provided range
pesticides_min, pesticides_max = 0.04, 367778.0  # From the provided range
temp_min, temp_max = 1.3, 30.65  # From the provided range

# Create two columns for layout
col1, col2 = st.columns(2)

# User inputs in the first column
with col1:
    st.subheader("Input Parameters")
    area = st.selectbox("Select Area", areas)
    item = st.selectbox("Select Crop", items)
    year = st.number_input("Year", min_value=year_min, max_value=year_max, value=year_min, step=1)
    rainfall = st.slider("Average Rainfall (mm/year)", min_value=rain_min, max_value=rain_max, value=rain_min, step=1.0)
    pesticides = st.slider("Pesticides (tonnes)", min_value=pesticides_min, max_value=pesticides_max, value=pesticides_min, step=0.01)
    temp = st.slider("Average Temperature (°C)", min_value=temp_min, max_value=temp_max, value=temp_min, step=0.01)

# Predict button
if st.button("Predict Yield"):
    # Create a DataFrame from user inputs
    input_data = pd.DataFrame({
        'Area': [area],
        'Item': [item],
        'Year': [year],
        'average_rain_fall_mm_per_year': [rainfall],
        'pesticides_tonnes': [pesticides],
        'avg_temp': [temp]
    })

    # Preprocess the input data
    try:
        # Encode 'Area'
        input_data['Area'] = le.transform(input_data['Area'])

        # Encode 'Item'
        item_encoded = ohe.transform(input_data[['Item']])
        item_encoded_df = pd.DataFrame(item_encoded, columns=ohe.get_feature_names_out(['Item']))
        input_data = pd.concat([input_data, item_encoded_df], axis=1)
        input_data = input_data.drop('Item', axis=1)

        # Scale numerical columns
        numerical_columns = ['Area', 'Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
        input_data[numerical_columns] = scaler.transform(input_data[numerical_columns])

        # Ensure the feature order matches the training data
        expected_columns = ['Area', 'Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp',
                           'Item_Cassava', 'Item_Maize', 'Item_Plantains and others', 'Item_Potatoes',
                           'Item_Rice, paddy', 'Item_Sorghum', 'Item_Soybeans', 'Item_Sweet potatoes',
                           'Item_Wheat', 'Item_Yams']
        input_data = input_data[expected_columns]

        # Make prediction
        predicted_yield = mlp.predict(input_data)[0]

        # Display the prediction in the second column
        with col2:
            st.subheader("Prediction Result")
            st.success(f"Predicted Yield: {predicted_yield:,.0f} hg/ha")
            st.markdown(f"This means approximately **{predicted_yield/10:,.0f} kg/ha**.")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Add some footer information
st.markdown("---")
st.markdown("**Note:** This app predicts crop yield based on historical data using a machine learning model. Results are estimates and may vary due to real-world factors.")