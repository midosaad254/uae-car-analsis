# uae-car-analsis
 # UAE Used Cars Analysis 🚗

![Header Image](https://img.shields.io/badge/License-MIT-blue.svg)  
**A Dash-based web application for analyzing and predicting used car prices in the UAE using XGBoost.**

---

## 📋 Project Overview

This project provides an interactive web application built with [Dash](https://dash.plotly.com/) to analyze the used car market in the United Arab Emirates. It leverages a dataset of 10,000 used cars (`uae_used_cars_10k.csv`) and employs an **XGBoost** model to predict car prices based on features like year, mileage, cylinders, and more. The app includes:

- **Dashboard**: Visualize market trends, price distributions, and car locations.
- **Price Prediction**: Input car details to get an estimated price with explanations (SHAP) and market positioning.

---

## 📊 Features

- **Interactive Dashboard**: Filter cars by make, location, price range, and year. View 3D scatter plots, heatmaps, and price trends.
- **Price Prediction**: Predict car prices with sentiment analysis of descriptions and radar charts for car evaluation.
- **Data Insights**: Export filtered data as CSV and explore similar cars using Nearest Neighbors.
- **Explainability**: SHAP visualizations to understand feature impacts on predictions.

---

## 🛠️ Requirements

To run this project locally, you'll need the following Python libraries (listed in `requirements.txt`):
dash
pandas
numpy
xgboost
scikit-learn
plotly
dash-leaflet
textblob
shap
joblib


---

## ⚙️ Installation

Follow these steps to set up and run the project on your machine:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/midosaad254/uae-car-analsis.git
   cd uae-car-analsis

   pip install -r requirements.txt
   python app.py
 
## 📂 Project Structure

uae-car-analsis/
├── data/
│   └── uae_used_cars_10k.csv       # Dataset of 10,000 used cars
├── models/
│   ├── stacking_model.pkl          # Trained XGBoost model
│   ├── scaler.pkl                  # StandardScaler for preprocessing
│   └── models.py                   # Prediction and analysis functions
├── app.py                          # Main Dash application
├── callbacks.py                    # Dash callbacks for interactivity
├── layouts.py                      # UI layout definitions
├── train_model.py                  # Model training script
├── utils.py                        # Utility functions
├── requirements.txt                # Required Python libraries
└── README.md                       # Project documentation

## 🎮 Usage

Dashboard Tab:
Apply filters (make, location, price, year) to explore the dataset.
View visualizations like price distribution, 3D scatter plots, and market trends.
Export filtered data using the "Export Data" button.
Price Prediction Tab:
Enter car details (make, model, year, mileage, etc.).
Click "Predict Price" to get an estimated price with confidence intervals.
Analyze feature impacts with SHAP plots and car evaluation via radar charts.

## Dataset

The dataset (data/uae_used_cars_10k.csv) contains 10,000 records of used cars in the UAE with the following columns:

Make, Model, Year, Mileage, Cylinders, Price, Transmission, Fuel Type, Color, Description, [Location] (if available).
Explore the dataset on Kaggle: UAE Used Cars 10k (Update this link once uploaded)

Built with ❤️ by Mohammed Saadat in 2025




   

