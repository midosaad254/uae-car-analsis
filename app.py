# app.py
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import os
import pandas as pd
import joblib
from layouts import get_layout
from callbacks import register_callbacks

# مسارات الملفات
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'uae_used_cars_10k.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'stacking_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')

# تحميل البيانات والنموذج
try:
    df = pd.read_csv(DATA_PATH)
    # تحويل الأعمدة الرقمية لأرقام
    df['Cylinders'] = pd.to_numeric(df['Cylinders'], errors='coerce')
    df['Mileage'] = pd.to_numeric(df['Mileage'], errors='coerce')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    
    # حساب الأعمدة المحسوبة
    df['Mileage_Km'] = df['Mileage'] * 1.60934
    df['Age'] = 2025 - df['Year']
    df['Km_per_Year'] = df['Mileage_Km'] / df['Age'].replace(0, 1)
    df['Car_Value_Index'] = (0.4 * (1 - df['Age']/df['Age'].max()) + 
                            0.3 * (1 - df['Mileage_Km']/df['Mileage_Km'].max()) + 
                            0.1 * df['Cylinders']/df['Cylinders'].max())
    df['Rarity_Score'] = df.groupby('Model')['Model'].transform('count').apply(lambda x: 1 / (x / len(df)))
    df['Price_per_Mile'] = df['Price'] / df['Mileage'].replace(0, 1)
    
    # ملء القيم المفقودة
    df = df.fillna(df.mean(numeric_only=True))
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError as e:
    raise FileNotFoundError(f"خطأ في تحميل الملفات: {str(e)}")

# إعداد التطبيق
app = dash.Dash(__name__, external_stylesheets=[
    "https://cdnjs.cloudflare.com/ajax/libs/bootstrap-rtl/3.4.0/css/bootstrap-rtl.min.css",
    "https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap"
])
app.title = "تحليل سوق السيارات المستعملة في الإمارات"
app.layout = get_layout(df)

# تسجيل الدوال التفاعلية
register_callbacks(app, df, model, scaler)

# تشغيل السيرفر
if __name__ == '__main__':
    app.run_server(debug=True)