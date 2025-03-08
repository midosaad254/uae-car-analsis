# layouts.py
from dash import dcc, html, dash_table
import dash_leaflet as dl
import plotly.graph_objects as go

COLORS = {
    'red': '#EF323D', 'green': '#009739', 'white': '#FFFFFF', 'black': '#000000', 
    'gold': '#CDAA63', 'light_gray': '#f8f9fa', 'dark_gray': '#343a40', 'blue': '#007bff'
}

def get_layout(df):
    """إنشاء تخطيط واجهة المستخدم."""
    return html.Div([
        html.Meta(name="viewport", content="width=device-width, initial-scale=1.0"),
        dcc.Store(id='filtered-data'),
        html.Div([
            html.Link(rel="stylesheet", href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap-rtl/3.4.0/css/bootstrap-rtl.min.css"),
            html.Link(rel="stylesheet", href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css"),  # أيقونات FontAwesome
            html.Link(rel="stylesheet", href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap"),
        ], id='rtl-support', style={
            'font-family': "'Tajawal', sans-serif",
            'direction': 'rtl',
            'text-align': 'right'
        }),
        
        # مؤشر التحميل
        dcc.Loading(id="loading-overlay", type="cube", color=COLORS['blue'], children=[
            html.Div("جاري المعالجة...", style={'color': COLORS['white'], 'text-align': 'center', 'margin-top': '20%'})
        ], fullscreen=True, style={'display': 'none'}),
        
        # الهيدر
        html.Div([
            html.Img(src='/assets/logo.png', style={'float': 'right', 'margin': '10px', 'height': '50px'}),
            html.H1([html.I(className="fas fa-car mr-2"), " سوق السيارات المستعملة في الإمارات"], className='mb-2 ar'),
            html.H4("تحليل وتوقع الأسعار باستخدام الذكاء الاصطناعي", className='mb-3 ar', style={'color': COLORS['gold']}),
        ], className='header mb-4', style={
            'background': f'linear-gradient(135deg, {COLORS["red"]}, {COLORS["green"]})', 
            'color': COLORS['white'], 
            'padding': '20px',
            'border-radius': '10px',
            'box-shadow': '0 4px 8px rgba(0,0,0,0.1)'
        }),

        dcc.Tabs([
            # تبويب لوحة المعلومات
            dcc.Tab(label="لوحة المعلومات", children=[
                html.Div([
                    # KPI Cards
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-dollar-sign fa-2x", style={'color': COLORS['green']}),
                            html.H3(f"{df['Price'].mean():,.0f} درهم", style={'color': COLORS['green']}),
                            html.P("متوسط السعر", className='ar')
                        ], className='kpi-card text-center', style={'background': COLORS['white'], 'border-radius': '10px', 'padding': '15px', 'box-shadow': '0 2px 4px rgba(0,0,0,0.1)'}),
                        html.Div([
                            html.I(className="fas fa-arrow-up fa-2x", style={'color': COLORS['red']}),
                            html.H3(f"{df['Price'].max():,.0f} درهم", style={'color': COLORS['red']}),
                            html.P("أعلى سعر", className='ar')
                        ], className='kpi-card text-center', style={'background': COLORS['white'], 'border-radius': '10px', 'padding': '15px', 'box-shadow': '0 2px 4px rgba(0,0,0,0.1)'}),
                        html.Div([
                            html.I(className="fas fa-arrow-down fa-2x", style={'color': COLORS['gold']}),
                            html.H3(f"{df['Price'].min():,.0f} درهم", style={'color': COLORS['gold']}),
                            html.P("أقل سعر", className='ar')
                        ], className='kpi-card text-center', style={'background': COLORS['white'], 'border-radius': '10px', 'padding': '15px', 'box-shadow': '0 2px 4px rgba(0,0,0,0.1)'}),
                        html.Div([
                            html.I(className="fas fa-car fa-2x", style={'color': COLORS['black']}),
                            html.H3(f"{len(df):,}", style={'color': COLORS['black']}),
                            html.P("عدد السيارات", className='ar')
                        ], className='kpi-card text-center', style={'background': COLORS['white'], 'border-radius': '10px', 'padding': '15px', 'box-shadow': '0 2px 4px rgba(0,0,0,0.1)'}),
                    ], className='d-flex justify-content-around mb-4', style={'gap': '20px'}),

                    # الرسومات
                    html.Div([
                        html.Div([dcc.Graph(id='price-distribution')], className='col-md-6'),
                        html.Div([dcc.Graph(id='price-vs-age-mileage-3d')], className='col-md-6'),
                        html.Div([dcc.Graph(id='price-heatmap')], className='col-md-6'),
                        html.Div([dcc.Graph(id='price-trend')], className='col-md-6'),
                        html.Div([dcc.Graph(id='make-distribution')], className='col-md-12'),  # رسم جديد لتوزيع الماركات
                        html.Div(id='location-map', children=[
                            dl.Map([
                                dl.TileLayer(),
                                dl.LayerGroup(id="map-markers")
                            ], center=[25.2048, 55.2708], zoom=7, style={'height': '400px', 'border-radius': '10px'})
                        ], className='col-md-12 mt-4'),
                        html.Button('تصدير البيانات', id='export-btn', n_clicks=0, className='btn btn-primary mt-3', 
                                   style={'background': COLORS['blue'], 'border': 'none', 'padding': '10px 20px', 'border-radius': '5px'}),
                        dcc.Download(id="download-dataframe-csv")
                    ], className='row mb-4'),

                    # الفلاتر
                    html.Div([
                        html.H4([html.I(className="fas fa-filter mr-2"), " فلاتر البحث"], className='mb-3 ar'),
                        dcc.Dropdown(id='filter-make', options=[{'label': m, 'value': m} for m in df['Make'].unique()], multi=True, placeholder="اختر الماركة", style={'margin-bottom': '15px'}),
                        dcc.Dropdown(id='filter-location', options=[{'label': l, 'value': l} for l in df['Location'].unique()], multi=True, placeholder="اختر الموقع", style={'margin-bottom': '15px'}),
                        dcc.RangeSlider(id='filter-price', min=int(df['Price'].min()), max=int(df['Price'].max()), step=10000,
                                        marks={int(df['Price'].min()): f"{int(df['Price'].min()):,}", int(df['Price'].max()): f"{int(df['Price'].max()):,}"},
                                        value=[int(df['Price'].min()), int(df['Price'].max())], tooltip={"placement": "bottom", "always_visible": True}),
                        html.Div(  # نضيف Div حوالين RangeSlider
                            dcc.RangeSlider(id='filter-year', min=int(df['Year'].min()), max=int(df['Year'].max()), step=1,
                                            marks={int(df['Year'].min()): str(int(df['Year'].min())), int(df['Year'].max()): str(int(df['Year'].max()))},
                                            value=[int(df['Year'].min()), int(df['Year'].max())], tooltip={"placement": "bottom", "always_visible": True}),
                            style={'margin-top': '20px'}
                        ),
                        html.Button('تطبيق الفلاتر', id='apply-filters', n_clicks=0, className='btn btn-success mt-3', 
                                   style={'background': COLORS['green'], 'border': 'none', 'padding': '10px 20px', 'border-radius': '5px'})
                    ], className='p-4 bg-light rounded mt-4', style={'box-shadow': '0 4px 8px rgba(0,0,0,0.1)'})
                ], className='container-fluid p-4')
            ]),

            # تبويب توقع السعر
            dcc.Tab(label="توقع السعر", children=[
                html.Div([
                    html.H2([html.I(className="fas fa-calculator mr-2"), " توقع سعر سيارتك"], className='text-center mb-4 ar'),
                    html.Div([
                        # حقول الإدخال
                        html.Div([
                            dcc.Dropdown(id='make', options=[{'label': m, 'value': m} for m in df['Make'].unique()], value=df['Make'].iloc[0], placeholder="الماركة", style={'margin-bottom': '15px'}),
                            dcc.Dropdown(id='model', placeholder="الموديل", style={'margin-bottom': '15px'}),
                            dcc.Slider(id='year', min=2000, max=2025, value=2016, marks={i: str(i) for i in range(2000, 2026, 5)}, tooltip={"placement": "bottom", "always_visible": True}),
                            dcc.Input(id='mileage', type='number', min=0, step=1000, value=100000, placeholder="المسافة (ميل)", className='form-control mt-2', debounce=True),
                            dcc.Input(id='cylinders', type='number', min=1, step=1, value=df['Cylinders'].mean(), placeholder="عدد الأسطوانات", className='form-control mt-2', debounce=True),
                            dcc.Dropdown(id='transmission', options=[{'label': t, 'value': t} for t in df['Transmission'].unique()], value=df['Transmission'].mode()[0], placeholder="ناقل الحركة", style={'margin-bottom': '15px'}),
                            dcc.Dropdown(id='fuel-type', options=[{'label': f, 'value': f} for f in df['Fuel Type'].unique()], value=df['Fuel Type'].mode()[0], placeholder="نوع الوقود", style={'margin-bottom': '15px'}),
                            dcc.Dropdown(id='color', options=[{'label': c, 'value': c} for c in df['Color'].unique()], value=df['Color'].mode()[0], placeholder="اللون", style={'margin-bottom': '15px'}),
                            dcc.Textarea(id='description', value="Good condition", placeholder="الوصف (اختياري)", className='form-control mt-2', style={'height': '100px', 'margin-bottom': '15px'}),
                            html.Div([
                                html.Button('توقع السعر', id='predict-btn', n_clicks=0, className='btn btn-primary mr-2', 
                                           style={'background': COLORS['blue'], 'border': 'none', 'padding': '10px 20px', 'border-radius': '5px'}),
                                html.Button('مسح الحقول', id='reset-btn', n_clicks=0, className='btn btn-secondary', 
                                           style={'background': COLORS['dark_gray'], 'border': 'none', 'padding': '10px 20px', 'border-radius': '5px'})
                            ], className='d-flex justify-content-between'),
                        ], className='col-md-6 p-3 bg-light rounded', style={'box-shadow': '0 4px 8px rgba(0,0,0,0.1)'}),
                        
                        # النتائج
                        html.Div([
                            html.Div(id='prediction-output', className='p-3 bg-light rounded mb-4', style={'border': f'1px solid {COLORS["green"]}'}), 
                            html.Div(id='market-position', className='p-3 bg-light rounded mb-4', style={'border': f'1px solid {COLORS["gold"]}'}), 
                            html.Div(id='similar-cars', className='p-3 bg-light rounded mb-4', style={'border': f'1px solid {COLORS["blue"]}'}), 
                            dcc.Graph(id='radar-chart', className='mb-4'),
                            dcc.Graph(id='shap-plot'),
                        ], className='col-md-6'),
                    ], className='row'),
                ], className='container p-4')
            ])
        ])
    ], style={'min-height': '100vh', 'background-color': COLORS['light_gray'], 'padding-bottom': '20px'})