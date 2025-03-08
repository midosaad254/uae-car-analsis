# dashboard.py
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import joblib
import shap
from functools import lru_cache

# إعدادات عامة
DATA_PATH = r'D:\trans\ai\machine learning 3\projects\1\2\uae car analsis\uae_used_cars_10k.csv'
MODEL_PATH = 'stacking_model.pkl'
SCALER_PATH = 'scaler.pkl'
COLORS = {
    'red': '#EF323D', 'green': '#009739', 'white': '#FFFFFF', 'black': '#000000', 
    'gold': '#CDAA63', 'light_gray': '#f8f9fa', 'dark_gray': '#343a40'
}

# تحميل البيانات
def load_data(file_path):
    df = pd.read_csv(file_path)
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(exclude=['number']).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # تحويل عمود Cylinders لأرقام
    df['Cylinders'] = pd.to_numeric(df['Cylinders'], errors='coerce')
    df['Cylinders'] = df['Cylinders'].fillna(df['Cylinders'].median())
    
    # إضافة الميزات الجديدة
    df['Mileage_Km'] = df['Mileage'] * 1.60934
    df['Age'] = 2025 - df['Year']
    df['Km_per_Year'] = df['Mileage_Km'] / df['Age'].replace(0, 1)
    df['Price_per_Km'] = df['Price'] / df['Mileage_Km'].replace(0, 1)
    df['Car_Value_Index'] = (0.4 * (1 - df['Age']/df['Age'].max()) + 
                             0.3 * (1 - df['Mileage_Km']/df['Mileage_Km'].max()) + 
                             0.1 * df['Cylinders']/df['Cylinders'].max())
    return df

# تحميل النموذج والمقياس
def load_model_and_scaler(model_path, scaler_path):
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except FileNotFoundError as e:
        print(f"خطأ: {e}. تأكد من وجود الملفات في المسار الصحيح.")
        raise

# تهيئة التطبيق
app = dash.Dash(__name__, 
                external_stylesheets=[
                    'https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.1.3/css/bootstrap.min.css',
                    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
                ],
                suppress_callback_exceptions=True)
server = app.server

# تحميل البيانات والنموذج
df = load_data(DATA_PATH)
model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)

# إعداد نموذج التشابه
@lru_cache(maxsize=1)
def prepare_similarity_model():
    features = ['Make', 'Model', 'Year', 'Mileage_Km', 'Cylinders']
    df_encoded = pd.get_dummies(df[features], columns=['Make', 'Model'])
    scaler_nn = StandardScaler()
    df_scaled = scaler_nn.fit_transform(df_encoded)
    nn_model = NearestNeighbors(n_neighbors=5)
    nn_model.fit(df_scaled)
    return nn_model, df_encoded, scaler_nn

nn_model, df_encoded, scaler_nn = prepare_similarity_model()

# دالة تحضير المدخلات
@lru_cache(maxsize=128)
def prepare_input(make, model_input, year, mileage, cylinders, transmission, fuel_type, color, desc):
    mileage_km = mileage * 1.60934
    age = 2025 - year
    km_per_year = mileage_km / max(age, 1)
    car_value_index = (0.4 * (1 - age/df['Age'].max()) + 
                       0.3 * (1 - mileage_km/df['Mileage_Km'].max()) + 
                       0.1 * cylinders/df['Cylinders'].max())
    input_df = pd.DataFrame({
        'Make': [df['Make'].unique().tolist().index(make) if make in df['Make'].unique() else 0],
        'Model': [df['Model'].unique().tolist().index(model_input) if model_input in df['Model'].unique() else 0],
        'Year': [year], 'Mileage_Km': [mileage_km], 'Cylinders': [cylinders],
        'Transmission': [df['Transmission'].unique().tolist().index(transmission)],
        'Fuel Type': [df['Fuel Type'].unique().tolist().index(fuel_type)],
        'Color': [df['Color'].unique().tolist().index(color)],
        'Age': [age], 'Km_per_Year': [km_per_year],
        'Price_per_Km': [df['Price_per_Km'].mean()], 'Car_Value_Index': [car_value_index]
    })
    return input_df

# دالة تحليل SHAP
def explain_prediction(input_data):
    explainer = shap.Explainer(model)
    shap_values = explainer(input_data)
    feature_importance = pd.DataFrame({
        'Feature': input_data.columns,
        'Importance': np.abs(shap_values.values[0]),
        'Effect': shap_values.values[0]
    }).sort_values('Importance', ascending=False)
    fig = px.bar(feature_importance.head(10), x='Importance', y='Feature', color='Effect',
                 color_continuous_scale=['#EF323D', '#CDAA63', '#009739'],
                 title='العوامل المؤثرة في سعر السيارة')
    return fig

# الواجهة
app.layout = html.Div([
    html.Div([
        html.Img(src='https://via.placeholder.com/50', style={'float': 'right', 'margin': '10px'}),
        html.H1("سوق السيارات المستعملة في الإمارات", className='mb-3'),
        html.H4("تحليل وتوقع الأسعار باستخدام الذكاء الاصطناعي", className='mb-3'),
    ], className='header mb-4', style={'background': f'linear-gradient(135deg, {COLORS["red"]}, {COLORS["green"]})', 
                                      'color': COLORS['white'], 'border-bottom': f'5px solid {COLORS["gold"]}'}),

    dcc.Tabs([
        dcc.Tab(label="لوحة المعلومات", children=[
            html.Div([
                html.Div([
                    html.Div([html.H3(f"{df['Price'].mean():,.0f} درهم", style={'color': COLORS['green']}), html.P("متوسط السعر")], className='kpi-card text-center'),
                    html.Div([html.H3(f"{df['Price'].max():,.0f} درهم", style={'color': COLORS['red']}), html.P("أعلى سعر")], className='kpi-card text-center'),
                    html.Div([html.H3(f"{df['Price'].min():,.0f} درهم", style={'color': COLORS['gold']}), html.P("أقل سعر")], className='kpi-card text-center'),
                    html.Div([html.H3(f"{len(df):,}", style={'color': COLORS['black']}), html.P("عدد السيارات")], className='kpi-card text-center'),
                ], className='d-flex justify-content-around mb-4'),

                html.Div([
                    dcc.Graph(id='price-distribution'),
                    dcc.Graph(id='price-vs-age-mileage'),
                    dcc.Graph(id='price-heatmap'),
                    dcc.Graph(id='price-trend'),
                ], className='row mb-4'),

                html.Div([
                    html.H4("فلاتر البحث", className='mb-3 ar'),
                    dcc.Dropdown(id='filter-make', options=[{'label': m, 'value': m} for m in df['Make'].unique()], multi=True, placeholder="اختر الماركة"),
                    dcc.Dropdown(id='filter-location', options=[{'label': l, 'value': l} for l in df['Location'].unique()], multi=True, placeholder="اختر الموقع"),
                    dcc.RangeSlider(id='filter-price', min=int(df['Price'].min()), max=int(df['Price'].max()), step=10000,
                                    marks={int(df['Price'].min()): f"{int(df['Price'].min()):,}", int(df['Price'].max()): f"{int(df['Price'].max()):,}"},
                                    value=[int(df['Price'].min()), int(df['Price'].max())]),
                    html.Button('تطبيق الفلاتر', id='apply-filters', n_clicks=0, className='btn btn-success mt-2'),
                ], className='p-3 bg-light rounded mt-4'),
            ], className='container-fluid p-4')
        ]),

        dcc.Tab(label="توقع السعر", children=[
            html.Div([
                html.H2("توقع سعر سيارتك", className='text-center mb-4 ar'),
                html.Div([
                    html.Div([
                        dcc.Dropdown(id='make', options=[{'label': m, 'value': m} for m in df['Make'].unique()], value=df['Make'].iloc[0], placeholder="الماركة"),
                        dcc.Dropdown(id='model', placeholder="الموديل"),
                        dcc.Slider(id='year', min=2000, max=2025, value=2016, marks={i: str(i) for i in range(2000, 2026, 5)}, tooltip={"placement": "bottom", "always_visible": True}),
                        dcc.Input(id='mileage', type='number', value=100000, placeholder="المسافة (ميل)", className='form-control mt-2'),
                        dcc.Input(id='cylinders', type='number', value=df['Cylinders'].mean(), placeholder="عدد الأسطوانات", className='form-control mt-2'),
                        dcc.Dropdown(id='transmission', options=[{'label': t, 'value': t} for t in df['Transmission'].unique()], value=df['Transmission'].mode()[0], placeholder="ناقل الحركة"),
                        dcc.Dropdown(id='fuel-type', options=[{'label': f, 'value': f} for f in df['Fuel Type'].unique()], value=df['Fuel Type'].mode()[0], placeholder="نوع الوقود"),
                        dcc.Dropdown(id='color', options=[{'label': c, 'value': c} for c in df['Color'].unique()], value=df['Color'].mode()[0], placeholder="اللون"),
                        dcc.Textarea(id='description', value="Good condition", placeholder="الوصف (اختياري)", className='form-control mt-2', style={'height': '100px'}),
                        html.Button('توقع السعر', id='predict-btn', n_clicks=0, className='btn btn-primary mt-3'),
                        dcc.Loading(id="loading", type="circle", children=[]),
                    ], className='col-md-6'),

                    html.Div([
                        html.Div(id='prediction-output', className='p-3 bg-light rounded mb-4'),
                        html.Div(id='similar-cars', className='mb-4'),
                        dcc.Graph(id='shap-plot'),
                    ], className='col-md-6'),
                ], className='row'),
            ], className='container p-4')
        ])
    ])
], style={'min-height': '100vh', 'display': 'flex', 'flex-direction': 'column', 'background-color': COLORS['light_gray']})

# ردود الفعل (Callbacks)
@app.callback(
    [Output('model', 'options'), Output('model', 'value')],
    [Input('make', 'value')]
)
def update_models(make):
    models = [{'label': m, 'value': m} for m in df[df['Make'] == make]['Model'].unique()]
    return models, models[0]['value'] if models else None

@app.callback(
    [Output('price-distribution', 'figure'),
     Output('price-vs-age-mileage', 'figure'),
     Output('price-heatmap', 'figure'),
     Output('price-trend', 'figure')],
    [Input('apply-filters', 'n_clicks')],
    [State('filter-make', 'value'), State('filter-location', 'value'), State('filter-price', 'value')]
)
def update_charts(n_clicks, makes, locations, price_range):
    filtered_df = df.copy()
    if n_clicks > 0:
        if makes:
            filtered_df = filtered_df[filtered_df['Make'].isin(makes)]
        if locations:
            filtered_df = filtered_df[filtered_df['Location'].isin(locations)]
        filtered_df = filtered_df[(filtered_df['Price'] >= price_range[0]) & (filtered_df['Price'] <= price_range[1])]
    
    price_dist_fig = px.histogram(filtered_df, x='Price', nbins=50, title='توزيع الأسعار', color_discrete_sequence=[COLORS['green']])
    price_age_fig = px.scatter(filtered_df, x='Age', y='Price', size='Mileage_Km', color='Cylinders', title='السعر مقابل العمر والمسافة')
    heatmap_data = filtered_df.groupby(['Make', 'Model'])['Price'].mean().unstack().iloc[:10, :10]
    heatmap_fig = go.Figure(data=go.Heatmap(z=heatmap_data.values, x=heatmap_data.columns, y=heatmap_data.index, colorscale='Viridis'))
    heatmap_fig.update_layout(title='خريطة حرارية للأسعار')
    trend_fig = px.line(filtered_df.groupby('Year')['Price'].mean().reset_index(), x='Year', y='Price', title='اتجاهات الأسعار', color_discrete_sequence=[COLORS['red']])
    
    return price_dist_fig, price_age_fig, heatmap_fig, trend_fig

@app.callback(
    [Output('prediction-output', 'children'),
     Output('similar-cars', 'children'),
     Output('shap-plot', 'figure'),
     Output('loading', 'children')],
    [Input('predict-btn', 'n_clicks')],
    [State('make', 'value'), State('model', 'value'), State('year', 'value'), 
     State('mileage', 'value'), State('cylinders', 'value'), State('transmission', 'value'), 
     State('fuel-type', 'value'), State('color', 'value'), State('description', 'value')]
)
def update_prediction(n_clicks, make, model_input, year, mileage, cylinders, transmission, fuel_type, color, description):
    if n_clicks == 0:
        return "", "", go.Figure(), []
    
    loading = [html.Div("جاري التوقع...", className='spinner')]
    input_data = prepare_input(make, model_input, year, mileage, cylinders, transmission, fuel_type, color, description)
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)[0]
    std = df[df['Model'] == model_input]['Price'].std() if not df[df['Model'] == model_input].empty else 10000
    
    pred_output = html.Div([
        html.H4("السعر المتوقع", className='ar'),
        html.P(f"{pred:,.2f} درهم (± {std * 1.96:,.2f})", style={'color': COLORS['green'], 'font-size': '24px'}),
    ])
    
    input_encoded = pd.get_dummies(input_data[['Make', 'Model', 'Year', 'Mileage_Km', 'Cylinders']], columns=['Make', 'Model'])
    input_encoded = input_encoded.reindex(columns=df_encoded.columns, fill_value=0)
    input_scaled_nn = scaler_nn.transform(input_encoded)
    distances, indices = nn_model.kneighbors(input_scaled_nn)
    similar_cars = df.iloc[indices[0]][['Make', 'Model', 'Year', 'Mileage', 'Price']]
    similar_cars_output = html.Div([
        html.H4("سيارات مشابهة", className='ar'),
        dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in similar_cars.columns],
            data=similar_cars.to_dict('records'),
            style_table={'overflowX': 'auto'}
        )
    ])
    
    shap_fig = explain_prediction(input_scaled)
    
    return pred_output, similar_cars_output, shap_fig, []

# تشغيل التطبيق
if __name__ == "__main__":
    app.run_server(debug=True)