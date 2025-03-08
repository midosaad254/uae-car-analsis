# models.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import shap

def prepare_input(df, make, model, year, mileage, cylinders, transmission, fuel_type, color, description):
    """تحضير مدخلات المستخدم للتنبؤ."""
    input_data = pd.DataFrame({
        'Make': [make], 'Model': [model], 'Year': [year], 'Mileage': [mileage], 
        'Cylinders': [cylinders], 'Transmission': [transmission], 'Fuel Type': [fuel_type], 
        'Color': [color], 'Description': [description]
    })
    input_data['Mileage_Km'] = input_data['Mileage'] * 1.60934
    input_data['Age'] = 2025 - input_data['Year']
    input_data['Km_per_Year'] = input_data['Mileage_Km'] / input_data['Age'].replace(0, 1)
    input_data['Car_Value_Index'] = (0.4 * (1 - input_data['Age']/df['Age'].max()) + 
                                    0.3 * (1 - input_data['Mileage_Km']/df['Mileage_Km'].max()) + 
                                    0.1 * input_data['Cylinders']/df['Cylinders'].max())
    rarity = df.groupby('Model')['Model'].count() / len(df)
    input_data['Rarity_Score'] = 1 / rarity.get(model, 0.01)
    input_data['Price_per_Mile'] = df['Price_per_Mile'].mean()  # نستخدم متوسط القيمة لأن السعر لسه متوقع
    return input_data

def features():
    """قائمة المميزات المستخدمة في النموذج."""
    return ['Year', 'Mileage', 'Cylinders', 'Age', 'Km_per_Year', 'Car_Value_Index', 'Rarity_Score', 'Price_per_Mile']

def explain_prediction(model, input_scaled):
    """شرح التنبؤ باستخدام SHAP."""
    explainer = shap.TreeExplainer(model)  # رجعنا لـ TreeExplainer
    shap_values = explainer.shap_values(input_scaled)
    fig = shap.summary_plot(shap_values, input_scaled, feature_names=features(), show=False, plot_type="bar")
    return fig

def market_position(pred, model_name, df):
    """تحديد مكانة السعر في السوق."""
    model_prices = df[df['Model'] == model_name]['Price']
    percentile = np.mean(model_prices < pred) * 100 if not model_prices.empty else 50
    avg_price = model_prices.mean() if not model_prices.empty else pred
    return f"السعر المتوقع في النسبة المئوية {percentile:.1f}% مقارنة بـ {model_name}. المتوسط: {avg_price:,.0f} درهم"

def analyze_description(description):
    """تحليل الوصف وتعديل السعر."""
    from textblob import TextBlob
    blob = TextBlob(description)
    sentiment = blob.sentiment.polarity
    desc_factor = 1 + (sentiment * 0.1)
    return desc_factor, sentiment

def analyze_price_trends(df):
    """تحليل اتجاهات الأسعار."""
    yearly_prices = df.groupby('Year')['Price'].mean().reset_index()
    if len(yearly_prices) < 2:
        return None
    slope = np.polyfit(yearly_prices['Year'], yearly_prices['Price'], 1)[0]
    next_year = yearly_prices['Year'].max() + 1
    next_year_prediction = yearly_prices['Price'].iloc[-1] + slope
    trend_message = "مرتفع" if slope > 0 else "منخفض"
    trend_color = '#009739' if slope > 0 else '#EF323D'
    return {
        'yearly_prices': yearly_prices.to_dict('records'),
        'next_year': next_year,
        'next_year_prediction': next_year_prediction,
        'trend_message': trend_message,
        'trend_color': trend_color
    }

def prepare_similarity_model(df):
    """تحضير نموذج الجيران الأقرب للبحث عن السيارات المشابهة."""
    df_encoded = pd.get_dummies(df[['Make', 'Model', 'Year', 'Mileage_Km', 'Cylinders']], 
                               columns=['Make', 'Model'])
    X = df_encoded.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    nn_model = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(X_scaled)
    return nn_model, df_encoded, scaler

def get_city_coordinates(city_name):
    """تحويل اسم المدينة إلى إحداثيات."""
    city_coords = {
        'Dubai': [25.2048, 55.2708],
        'Abu Dhabi': [24.4539, 54.3773],
        'Sharjah': [25.3463, 55.4209],
        'Ajman': [25.4111, 55.4354],
        'Ras Al Khaimah': [25.7895, 55.9432],
        'Fujairah': [25.1281, 56.3265],
        'Umm Al Quwain': [25.5556, 55.5556]
    }
    return city_coords.get(city_name, [25.2048, 55.2708])  # دبي كافتراضي