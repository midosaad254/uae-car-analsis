# utils.py
import pandas as pd
import joblib
import os

def load_data(file_path: str) -> pd.DataFrame:
    """تحميل ومعالجة البيانات من ملف CSV."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ملف البيانات غير موجود: {file_path}")
    
    df = pd.read_csv(file_path)
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(exclude=['number']).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    df['Cylinders'] = pd.to_numeric(df['Cylinders'], errors='coerce')
    df['Cylinders'] = df['Cylinders'].fillna(df['Cylinders'].median())
    
    df['Mileage_Km'] = df['Mileage'] * 1.60934
    df['Age'] = 2025 - df['Year']
    df['Km_per_Year'] = df['Mileage_Km'] / df['Age'].replace(0, 1)
    df['Price_per_Km'] = df['Price'] / df['Mileage_Km'].replace(0, 1)
    df['Car_Value_Index'] = (0.4 * (1 - df['Age']/df['Age'].max()) + 
                            0.3 * (1 - df['Mileage_Km']/df['Mileage_Km'].max()) + 
                            0.1 * df['Cylinders']/df['Cylinders'].max())
    
    # إضافة إحداثيات افتراضية للمواقع
    location_coords = {
        'Dubai': (25.2048, 55.2708),
        'Abu Dhabi': (24.4539, 54.3773),
        'Sharjah': (25.3463, 55.4209),
        'Ajman': (25.4111, 55.4354),
        'Ras Al Khaimah': (25.7874, 55.9422),
        'Fujairah': (25.1288, 56.3265),
        'Umm Al Quwain': (25.5639, 55.5522)
    }
    df['Lat'] = df['Location'].map(lambda x: location_coords.get(x, (25.2048, 55.2708))[0])
    df['Long'] = df['Location'].map(lambda x: location_coords.get(x, (25.2048, 55.2708))[1])
    return df

def load_model_and_scaler(model_path: str, scaler_path: str) -> tuple:
    """تحميل النموذج والمقياس."""
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"ملف النموذج أو المقياس غير موجود: {model_path}, {scaler_path}")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def validate_input(df, make, model, year, mileage, cylinders, transmission, fuel_type, color) -> list:
    """التحقق من صحة المدخلات قبل معالجتها."""
    errors = []
    if make not in df['Make'].unique():
        errors.append(f"الماركة '{make}' غير موجودة في قاعدة البيانات")
    if model not in df['Model'].unique():
        errors.append(f"الموديل '{model}' غير موجود في قاعدة البيانات")
    if year < 1900 or year > 2025:
        errors.append(f"السنة '{year}' خارج النطاق المقبول (1900-2025)")
    if mileage < 0 or mileage > 1000000:
        errors.append(f"المسافة '{mileage}' خارج النطاق المقبول")
    if cylinders <= 0 or cylinders > 16:
        errors.append(f"عدد الأسطوانات '{cylinders}' خارج النطاق المقبول")
    if transmission not in df['Transmission'].unique():
        errors.append(f"ناقل الحركة '{transmission}' غير موجود في قاعدة البيانات")
    if fuel_type not in df['Fuel Type'].unique():
        errors.append(f"نوع الوقود '{fuel_type}' غير موجود في قاعدة البيانات")
    if color not in df['Color'].unique():
        errors.append(f"اللون '{color}' غير موجود في قاعدة البيانات")
    return errors