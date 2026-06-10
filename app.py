import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .price-tag {
        font-size: 3rem;
        font-weight: 800;
        color: #2e7d32;
        text-align: center;
        padding: 2rem;
        background-color: #e8f5e9;
        border-radius: 10px;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_and_train_model():
    file_path = "kc_house_data.csv"
    
    if not os.path.exists(file_path):
        return None, None, "Dataset not found. Ensure 'kc_house_data.csv' is in the directory."
        
    try:
        df = pd.read_csv(file_path)
        
        features = [
            'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 
            'floors', 'waterfront', 'view', 'condition', 'grade', 'yr_built'
        ]
        target = 'price'
        
        X = df[features]
        y = df[target]
        
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        return model, features, "Success"
        
    except Exception as e:
        return None, None, str(e)

st.title("🏡 House Price Prediction AI")
st.markdown("Enter the specifications of your house below, and our Machine Learning model will estimate its market value instantly.")

with st.spinner("Initializing AI Model..."):
    model, features, status = load_and_train_model()

if status != "Success":
    st.error(f"Failed to load model: {status}")
else:
    st.sidebar.header("🔧 Property Details")
    st.sidebar.markdown("Adjust the sliders below to see the predicted price change in real-time.")
    
    col_input1, col_input2 = st.columns([1, 2])
    
    with st.sidebar:
        bedrooms = st.slider("Bedrooms", min_value=1, max_value=10, value=3)
        bathrooms = st.slider("Bathrooms", min_value=1.0, max_value=8.0, value=2.0, step=0.25)
        sqft_living = st.number_input("Living Area (sqft)", min_value=500, max_value=15000, value=2000, step=100)
        sqft_lot = st.number_input("Lot Size (sqft)", min_value=500, max_value=100000, value=5000, step=500)
        floors = st.selectbox("Floors", options=[1.0, 1.5, 2.0, 2.5, 3.0], index=0)
        waterfront = st.selectbox("Waterfront", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        view = st.slider("View Quality (0-4)", min_value=0, max_value=4, value=0)
        condition = st.slider("Condition (1-5)", min_value=1, max_value=5, value=3)
        grade = st.slider("Construction Grade (1-13)", min_value=1, max_value=13, value=7)
        yr_built = st.slider("Year Built", min_value=1900, max_value=2024, value=1990)

    input_data = pd.DataFrame([[
        bedrooms, bathrooms, sqft_living, sqft_lot, 
        floors, waterfront, view, condition, grade, yr_built
    ]], columns=features)
    
    predicted_price = model.predict(input_data)[0]
    
    st.markdown("### 🏷️ Estimated Market Value")
    st.markdown(f"<div class='price-tag'>${predicted_price:,.2f}</div>", unsafe_allow_html=True)
    
    st.info("💡 **Tip:** Try changing the 'Waterfront' or 'Grade' options in the sidebar to see how drastically they impact the house price!")
    
    with st.expander("ℹ️ About the Model"):
        st.write("This application uses a **Gradient Boosting Regressor** trained on historical house sales data. It evaluates 10 key structural features to estimate the final sale price.")