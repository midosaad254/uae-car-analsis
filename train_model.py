# train_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor  # نرجع لـ XGBoost بس
from sklearn.preprocessing import StandardScaler
import os

# مسار البيانات
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'uae_used_cars_10k.csv')

# تحميل البيانات
df = pd.read_csv(DATA_PATH)

# تحويل الأعمدة لأرقام
df['Mileage'] = pd.to_numeric(df['Mileage'], errors='coerce')
df['Cylinders'] = pd.to_numeric(df['Cylinders'], errors='coerce')

# تحضير الأعمدة المحسوبة
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

# تحضير البيانات
features = ['Year', 'Mileage', 'Cylinders', 'Age', 'Km_per_Year', 'Car_Value_Index', 'Rarity_Score', 'Price_per_Mile']
X = df[features]
y = df['Price']

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تطبيع البيانات
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# تدريب نموذج XGBoost
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X_train_scaled, y_train)

# حفظ النموذج والمقياس
joblib.dump(model, os.path.join(BASE_DIR, 'stacking_model.pkl'))
joblib.dump(scaler, os.path.join(BASE_DIR, 'scaler.pkl'))

# اختبار الدقة
score = model.score(X_test_scaled, y_test)
print(f"دقة النموذج: {score:.2f}")