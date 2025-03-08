# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# تحميل ومعالجة البيانات
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # معالجة القيم المفقودة
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(exclude=['number']).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # تحويل المسافة من ميل إلى كيلومترات وإضافة ميزات جديدة
    df['Mileage_Km'] = df['Mileage'] * 1.60934
    df['Age'] = 2025 - df['Year']
    df['Km_per_Year'] = df['Mileage_Km'] / df['Age'].replace(0, 1)
    df['Price_per_Km'] = df['Price'] / df['Mileage_Km'].replace(0, 1)
    df['Car_Value_Index'] = (0.4 * (1 - df['Age']/df['Age'].max()) + 
                             0.3 * (1 - df['Mileage_Km']/df['Mileage_Km'].max()) + 
                             0.1 * df['Cylinders']/df['Cylinders'].max())
    
    return df

# تحضير البيانات للتدريب
def prepare_features(df):
    # تحويل الأعمدة الفئوية إلى أرقام
    categorical_cols = ['Make', 'Model', 'Transmission', 'Fuel Type', 'Color', 'Body Type', 'Location']
    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes
    
    # اختيار الميزات
    features = ['Make', 'Model', 'Year', 'Mileage_Km', 'Cylinders', 'Transmission', 
                'Fuel Type', 'Color', 'Age', 'Km_per_Year', 'Price_per_Km', 'Car_Value_Index']
    X = df[features]
    y = df['Price']
    
    return X, y

# تدريب النموذج
def train_model(X, y):
    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # تهيئة المقياس
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # تعريف النماذج الأساسية
    estimators = [
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ('lr', LinearRegression())
    ]
    
    # تعريف نموذج الـ Stacking
    stacking_model = StackingRegressor(
        estimators=estimators,
        final_estimator=LinearRegression(),
        cv=5
    )
    
    # تدريب النموذج
    stacking_model.fit(X_train_scaled, y_train)
    
    # تقييم النموذج
    y_pred = stacking_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    return stacking_model, scaler

# التنفيذ الرئيسي
if __name__ == "__main__":
    # تحميل البيانات
    file_path = r'D:\trans\ai\machine learning 3\projects\1\2\uae car analsis\uae_used_cars_10k.csv'
    df = load_and_preprocess_data(file_path)
    
    # تحضير الميزات
    X, y = prepare_features(df)
    
    # تدريب النموذج
    model, scaler = train_model(X, y)
    
    # حفظ النموذج والمقياس
    joblib.dump(model, 'stacking_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("تم حفظ النموذج والمقياس بنجاح!")