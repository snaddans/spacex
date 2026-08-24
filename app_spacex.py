import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
import joblib

import os

print("Loading SpaceX model package...")

# Use absolute path resolved dynamically for deployment compatibility
load_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spacex_model.joblib")

try:
    artifacts = joblib.load(load_path)
    rf_model = artifacts['model']
    label_encoders = artifacts['encoders']
    features = artifacts['features']
    df = artifacts['df']
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print(f"❌ Error: Could not find the model file at {load_path}")
    exit()

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("🚀 SpaceX Falcon 9 Landing Predictor", style={'textAlign': 'center'}),
    
    dcc.Tabs([
        dcc.Tab(label='Launch Analytics', children=[
            html.Div([
                # Existing Launch Site Chart
                dcc.Graph(
                    figure=px.histogram(df, x='Launch site', color='Landing_Success', barmode='group', title='Landings by Launch Site')
                ),
                
                # NEW Orbit Chart
                dcc.Graph(
                    figure=px.histogram(df, x='Orbit', color='Landing_Success', barmode='group', title='Landings by Orbit')
                ),
                
                # Existing Payload Chart
                dcc.Graph(
                    figure=px.box(df, x='Landing_Success', y='Payload mass (kg)', title='Payload Mass vs Landing Success')
                )
            ])
        ]),
        
        dcc.Tab(label='Custom Prediction Tool', children=[
            html.Div(style={'padding': '20px'}, children=[
                html.H3("Enter Mission Parameters:"),
                
                html.Label("Payload Mass (kg) (0-25000):"),
                dcc.Input(id='in-payload', type='number', value=5000, min=0, max=25000, style={'margin': '10px'}),
                
                html.Label("Orbit:"),
                dcc.Dropdown(
                    id='in-orbit', 
                    options=[{'label': o, 'value': o} for o in label_encoders.get('Orbit', {}).classes_] if 'Orbit' in label_encoders else [], 
                    style={'margin': '10px'}
                ),
                
                html.Label("Launch Site:"),
                dcc.Dropdown(
                    id='in-site', 
                    options=[{'label': s, 'value': s} for s in label_encoders.get('Launch site', {}).classes_] if 'Launch site' in label_encoders else [], 
                    style={'margin': '10px'}
                ),
                
                html.Button('Predict Landing', id='predict-btn', n_clicks=0, style={'padding': '10px', 'backgroundColor': 'black', 'color': 'white'}),
                html.H2(id='prediction-output', style={'marginTop': '20px'})
            ])
        ])
    ])
])

@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-btn', 'n_clicks')],
    [State('in-payload', 'value'), State('in-orbit', 'value'), State('in-site', 'value')]
)
def predict_landing(n_clicks, payload, orbit, site):
    if n_clicks == 0 or not orbit or not site:
        return ""
    
    if payload is None or payload < 0 or payload > 25000:
        return "⚠️ Prediction can't be determined (Payload must be between 0 and 25,000 kg)"
    
    try:
        orbit_encoded = label_encoders['Orbit'].transform([orbit])[0]
        site_encoded = label_encoders['Launch site'].transform([site])[0]
        
        input_data = pd.DataFrame([[payload, orbit_encoded, site_encoded]], columns=features)
        prediction = rf_model.predict(input_data)[0]
        
        if prediction == 1:
            return "🟢 Prediction: Successful Landing!"
        else:
            return "🔴 Prediction: Landing Failure / No Attempt"
    except Exception as e:
        return f"Error processing input: {e}"

# Expose the Flask server for gunicorn
server = app.server

if __name__ == '__main__':
    app.run(debug=True)