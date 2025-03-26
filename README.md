# CropCast: Crop Yield Prediction App 🌾

## Overview
CropCast is a simple web application built with Streamlit that predicts crop yields (in hg/ha) based on historical data. Users can input parameters like area, crop type, year, rainfall, pesticides, and temperature to get a predicted yield using a pre-trained MLP (Multi-Layer Perceptron) model. This project is a minimal viable product (MVP) designed for farmers, researchers, or enthusiasts interested in agricultural yield forecasting.

## Dataset
The dataset used in this project is sourced from the [Crop Yield Prediction Dataset](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset) on Kaggle, created by Patelris. It contains historical crop yield data across various countries, crops, and environmental factors.

## Files in This Repository
- **`yield_df.csv`**: The dataset containing historical crop yield data.
- **`mlp_regressor.pkl`**: The pre-trained MLP model for yield prediction.
- **`scaler.pkl`**: The scaler object used for preprocessing numerical features.
- **`label_encoder_area.pkl`**: The label encoder for the `Area` column.
- **`one_hot_encoder_item.pkl`**: The one-hot encoder for the `Item` (crop type) column.
- **`app.py`**: The Streamlit app script to run the prediction interface.
- **`Crop Prediction MLP.ipynb`**: The Jupyter Notebook containing the data preprocessing, model training, and evaluation steps.

## How to Run the App Locally
Follow these steps to run CropCast on your local machine:

1. Clone the Repository:
   ```bash
   git clone https://github.com/<your-username>/CropCast.git
   cd CropCast

2. Set Up a Virtual Environment (optional but recommended):
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Dependencies:
   pip install streamlit pandas scikit-learn joblib

4. Run the Streamlit App:
   streamlit run app.py

This will launch the app in your default web browser (e.g., at http://localhost:8501).

5. Interact with the App:
Select an Area and Item (crop type) from the dropdowns.
Enter a Year and use the sliders to set Average Rainfall, Pesticides, and Average Temperature.
Click "Predict Yield" to see the predicted crop yield in hg/ha and kg/ha.

Credits
The dataset is sourced from the Crop Yield Prediction Dataset on Kaggle, created by Patelris.
This project was developed as part of an effort to create a user-friendly tool for crop yield prediction.
License
This project is licensed under the MIT License.

Happy forecasting with CropCast! 🌱
