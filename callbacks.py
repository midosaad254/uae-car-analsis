# callbacks.py
from dash.dependencies import Input, Output, State
from dash import html, dcc, dash
import plotly.express as px
import plotly.graph_objects as go
from models import prepare_input, explain_prediction, market_position, analyze_description, analyze_price_trends, prepare_similarity_model, features, get_city_coordinates
from utils import validate_input
import pandas as pd
import dash_leaflet as dl
import dash_table

def register_callbacks(app, df, model, scaler):
    """تسجيل الدوال التفاعلية لتحديث الواجهة."""
    nn_model, df_encoded, scaler_nn = prepare_similarity_model(df)

    @app.callback(
        [Output('model', 'options'), Output('model', 'value')],
        [Input('make', 'value')]
    )
    def update_models(make):
        models = [{'label': m, 'value': m} for m in df[df['Make'] == make]['Model'].unique()]
        return models, models[0]['value'] if models else None

    @app.callback(
        [Output('price-distribution', 'figure'),
         Output('price-vs-age-mileage-3d', 'figure'),
         Output('price-heatmap', 'figure'),
         Output('price-trend', 'figure'),
         Output('make-distribution', 'figure'),
         Output('map-markers', 'children'),
         Output('filtered-data', 'data')],
        [Input('apply-filters', 'n_clicks')],
        [State('filter-make', 'value'), State('filter-location', 'value'), State('filter-price', 'value'), State('filter-year', 'value')]
    )
    def update_charts(n_clicks, makes, locations, price_range, year_range):
        filtered_df = df.copy()
        if n_clicks > 0:
            if makes:
                filtered_df = filtered_df[filtered_df['Make'].isin(makes)]
            if locations:
                filtered_df = filtered_df[filtered_df['Location'].isin(locations)]
            filtered_df = filtered_df[(filtered_df['Price'] >= price_range[0]) & (filtered_df['Price'] <= price_range[1])]
            filtered_df = filtered_df[(filtered_df['Year'] >= year_range[0]) & (filtered_df['Year'] <= year_range[1])]
        
        price_dist_fig = px.histogram(filtered_df, x='Price', nbins=50, title='توزيع الأسعار', color_discrete_sequence=['#009739'])
        price_age_fig = go.Figure(data=[go.Scatter3d(
            x=filtered_df['Age'], y=filtered_df['Mileage_Km'], z=filtered_df['Price'],
            mode='markers', marker=dict(size=5, color=filtered_df['Cylinders'], colorscale='Viridis', showscale=True)
        )])
        price_age_fig.update_layout(
            title='السعر مقابل العمر والمسافة (3D)',
            scene=dict(xaxis_title='العمر', yaxis_title='المسافة (كم)', zaxis_title='السعر')
        )
        heatmap_data = filtered_df.groupby(['Make', 'Model'])['Price'].mean().unstack().iloc[:10, :10]
        heatmap_fig = go.Figure(data=go.Heatmap(z=heatmap_data.values, x=heatmap_data.columns, y=heatmap_data.index, colorscale='Viridis'))
        heatmap_fig.update_layout(title='خريطة حرارية للأسعار')
        
        trends = analyze_price_trends(filtered_df)
        if trends:
            trend_fig = px.line(pd.DataFrame(trends['yearly_prices']), x='Year', y='Price', 
                               title=f"اتجاهات الأسعار: {trends['trend_message']}", 
                               color_discrete_sequence=[trends['trend_color']])
            trend_fig.add_scatter(x=[trends['next_year']], y=[trends['next_year_prediction']], 
                                mode='markers+text', text=[f"توقع {trends['next_year']}"], 
                                textposition="top center", marker=dict(size=12, color=trends['trend_color']))
        else:
            trend_fig = px.line(filtered_df.groupby('Year')['Price'].mean().reset_index(), x='Year', y='Price', 
                               title='اتجاهات الأسعار', color_discrete_sequence=['#EF323D'])
        
        make_dist_fig = px.pie(filtered_df, names='Make', title='توزيع الماركات', hole=0.3, color_discrete_sequence=px.colors.qualitative.Bold)
        
        markers = [
            dl.Marker(
                position=get_city_coordinates(row['Location']),
                children=dl.Tooltip(f"{row['Make']} {row['Model']} - {row['Price']:,.0f} درهم")
            ) for _, row in filtered_df.iterrows() if 'Location' in row and pd.notna(row['Location'])
        ]
        
        return price_dist_fig, price_age_fig, heatmap_fig, trend_fig, make_dist_fig, markers, filtered_df.to_json()

    @app.callback(
        [Output('prediction-output', 'children'),
         Output('market-position', 'children'),
         Output('similar-cars', 'children'),
         Output('shap-plot', 'figure'),
         Output('radar-chart', 'figure'),
         Output('loading-overlay', 'style')],
        [Input('predict-btn', 'n_clicks')],
        [State('make', 'value'), State('model', 'value'), State('year', 'value'), 
         State('mileage', 'value'), State('cylinders', 'value'), State('transmission', 'value'), 
         State('fuel-type', 'value'), State('color', 'value'), State('description', 'value')]
    )
    def update_prediction(n_clicks, make, model_input, year, mileage, cylinders, transmission, fuel_type, color, description):
        if n_clicks == 0:
            return "", "", "", go.Figure(), go.Figure(), {'display': 'none'}
        
        loading_style = {'display': 'block'}
        inputs = [make, model_input, year, mileage, cylinders, transmission, fuel_type, color]
        if None in inputs or any(x == "" for x in inputs if isinstance(x, str)):
            return html.Div("يرجى ملء جميع الحقول المطلوبة", style={'color': '#EF323D'}), "", "", go.Figure(), go.Figure(), {'display': 'none'}
        
        try:
            year = int(year) if year is not None else 2020
            mileage = float(mileage) if mileage is not None else 0
            cylinders = float(cylinders) if cylinders is not None else df['Cylinders'].mean()
            description = description or "No description"

            errors = validate_input(df, make, model_input, year, mileage, cylinders, transmission, fuel_type, color)
            if errors:
                return html.Div([html.P(err, style={'color': '#EF323D'}) for err in errors]), "", "", go.Figure(), go.Figure(), {'display': 'none'}
            
            input_data = prepare_input(df, make, model_input, year, mileage, cylinders, transmission, fuel_type, color, description)
            input_scaled = scaler.transform(input_data[features()].values)
            pred = model.predict(input_scaled)[0]
            
            desc_factor, sentiment = analyze_description(description)
            adjusted_pred = pred * desc_factor
            input_data['Price_per_Mile'] = adjusted_pred / input_data['Mileage'].replace(0, 1)
            
            std = df[df['Model'] == model_input]['Price'].std() if not df[df['Model'] == model_input].empty else 10000
            
            pred_output = html.Div([
                html.H4("السعر المتوقع", className='ar'),
                html.P(f"{adjusted_pred:,.2f} درهم (± {std * 1.96:,.2f})", style={'color': '#009739', 'font-size': '24px'}),
                html.P(f"معامل التعديل (الوصف): {desc_factor:.2f}", style={'color': '#CDAA63', 'font-size': '14px'}),
                html.P(f"المشاعر: {'إيجابي' if sentiment > 0 else 'سلبي' if sentiment < 0 else 'محايد'} ({sentiment:.2f})", 
                       style={'color': '#000000', 'font-size': '14px'})
            ])
            
            market_pos = market_position(adjusted_pred, model_input, df)
            
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
            
            shap_fig = explain_prediction(model, input_scaled)
            
            radar_fig = go.Figure(data=go.Scatterpolar(
                r=[input_data['Car_Value_Index'].iloc[0]*10, (1 - input_data['Age'].iloc[0]/df['Age'].max())*10, 
                   (1 - input_data['Mileage_Km'].iloc[0]/df['Mileage_Km'].max())*10, 
                   input_data['Cylinders'].iloc[0]/df['Cylinders'].max()*10, input_data['Rarity_Score'].iloc[0]/df['Rarity_Score'].max()*10],
                theta=['القيمة الإجمالية', 'العمر', 'المسافة', 'المحرك', 'الندرة'],
                fill='toself',
                name='تقييم السيارة'
            ))
            radar_fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                title="تقييم جوانب السيارة",
                showlegend=True
            )
            
            return pred_output, market_pos, similar_cars_output, shap_fig, radar_fig, {'display': 'none'}
        
        except Exception as e:
            error_msg = f"خطأ: {str(e)}"
            print(error_msg)
            return html.Div(error_msg, style={'color': '#EF323D'}), "", "", go.Figure(), go.Figure(), {'display': 'none'}

    @app.callback(
        [Output('make', 'value', allow_duplicate=True),
         Output('model', 'value', allow_duplicate=True),
         Output('year', 'value', allow_duplicate=True),
         Output('mileage', 'value', allow_duplicate=True),
         Output('cylinders', 'value', allow_duplicate=True),
         Output('transmission', 'value', allow_duplicate=True),
         Output('fuel-type', 'value', allow_duplicate=True),
         Output('color', 'value', allow_duplicate=True),
         Output('description', 'value', allow_duplicate=True)],
        [Input('reset-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def reset_inputs(n_clicks):
        if n_clicks > 0:
            return (df['Make'].iloc[0], None, 2016, 100000, df['Cylinders'].mean(), 
                    df['Transmission'].mode()[0], df['Fuel Type'].mode()[0], df['Color'].mode()[0], "Good condition")
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    @app.callback(
        Output("download-dataframe-csv", "data"),
        Input("export-btn", "n_clicks"),
        State("filtered-data", "data"),
        prevent_initial_call=True,
    )
    def export_data(n_clicks, filtered_data):
        if n_clicks > 0 and filtered_data:
            filtered_df = pd.read_json(filtered_data)
            return dcc.send_data_frame(filtered_df.to_csv, "filtered_cars_data.csv", index=False)