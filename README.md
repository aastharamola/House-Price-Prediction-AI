<img width="1920" height="1080" alt="Screenshot (126)" src="https://github.com/user-attachments/assets/f0e934a8-fb0f-437b-be4e-53e00abcdfcb" />

# 🏡 House Price Prediction using Machine Learning

A beginner-friendly project to predict house prices based on various factors such as square footage, number of bedrooms, location, and more. This project walks through data exploration, visualization, linear regression, and gradient boosting to achieve high model accuracy.
![Uploading Screenshot (126).png…]()

---

## 📂 Project Structure

```

data/                         # Dataset used for training and testing
house_price_prediction.ipynb  # Main Jupyter Notebook with all code
app.py                        # Streamlit web application
requirements.txt              # Python dependencies
README.md                     # Project documentation

````

---

## 📊 Dataset

This dataset contains details of house listings including:

- Number of bedrooms and bathrooms  
- Square footage (with and without basement)  
- Waterfront presence  
- Location via latitude and longitude  
- Zipcode  
- Year built and renovated  
- Price of the house  

**Source:** _[Add dataset source or link here]_

---

## 🧪 Models Used

### 🔹 Linear Regression
- First model used to understand relationships in data  
- Achieved ~73% accuracy  

### 🔹 Gradient Boosting Regressor
- Powerful ensemble model using decision trees  
- Achieved **~91.94% accuracy**

---

## 📈 Key Visualizations

- Most common house types by bedroom count  
- Price vs. Living Area  
- Price vs. Location (Latitude and Longitude)  
- Influence of features like:
  - Basement area  
  - Floors  
  - Condition  
  - Waterfront  

---

## 🔧 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
````

### Main Libraries Used

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* streamlit

---

## 🚀 How to Run

1. **Clone the repository**

```bash
git clone https://github.com/aastharamola/House-Price-Prediction-AI.git
cd house-price-prediction
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the Jupyter Notebook (Optional)**

```bash
jupyter notebook house_price_prediction.ipynb
```

4. **Run the Web Application**

```bash
streamlit run app.py
```

---

## 🎯 Goal

To achieve over **85% prediction accuracy** in estimating house prices using regression techniques.

✅ Final model using Gradient Boosting achieved **91.94% accuracy**.

---

## 📚 Learning Points

* Data cleaning and preprocessing
* Exploratory data analysis
* Feature engineering
* Linear Regression vs Gradient Boosting
* Model evaluation using `r2_score`
