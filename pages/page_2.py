import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import dash_table
import numpy as np
import datetime
from sklearn import preprocessing
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

from assets.metrics import load_models
from assets.training import RandomGenerator
from assets.kalman_filter_vasicek import VaR

# 1. Import the EONIA data
data = pd.read_csv("data/EONIA.csv", sep=";")
dates_EONIA = np.ravel(data.Date[:775][::-1].values).astype(str)
dates_EONIA = [datetime.datetime.strptime(d,"%d-%m-%Y").date()
               for d in dates_EONIA]
data = np.array(data.iloc[:775,2])[::-1] # Filter for only EONIA

# Import the data and apply the transformation
# We import the data until 6-10-2017
df = pd.read_csv("data/Master_EONIA.csv", sep=";")
dates_t_var = df.iloc[3798:, 0]
dates_t_var = [datetime.datetime.strptime(d,"%d-%m-%Y").date()
               for d in dates_t_var]
df = df.iloc[:, 1:] # Remove the Date variable from the dataset
df.EONIA[1:] = np.diff(df.EONIA)
df = df.iloc[1:, :]
scaler = preprocessing.MinMaxScaler().fit(df)

# Create the VaR estimate for Vasicek
upperVasicek = []
lowerVasicek = []

for j in [1, 10, 20]:
    upperVasicek.append(VaR(data[500], data[300:500], j, percentile=0.99, upward=True))
    lowerVasicek.append(VaR(data[500], data[300:500], j, percentile=0.99, upward=False))

upperVasicek = np.round(np.ravel(upperVasicek) - data[500], 4)
lowerVasicek = np.round(np.ravel(lowerVasicek) - data[500], 4)

# Define the settings
hparams = []
hidden_dim = 4

# Import the pre-trained models
embedder_model, recovery_model, supervisor_model, generator_model, discriminator_model = load_models(8250, hparams, hidden_dim)

# Calculate the upper-VaR and lower-VaR for 1-day, 10-day and 20-day
upperVaR = []
lowerVaR = []
for i in [1, 10, 20]:
    Z_mb = RandomGenerator(1000, [i, hidden_dim])
    samples = recovery_model(generator_model(Z_mb)).numpy()
    reshaped_data = samples.reshape((samples.shape[0]*samples.shape[1], 
                                    samples.shape[2]))

    scaled_reshaped_data = scaler.inverse_transform(reshaped_data)
    simulations = scaled_reshaped_data.reshape(((samples.shape[0],
                                                samples.shape[1], 
                                                samples.shape[2])))

    results = np.sum(simulations[:,:,8], axis=1)
    results.sort()

    upperVaR.append(results[990])
    lowerVaR.append(results[10])

upperVaR = np.round(upperVaR, 4)
lowerVaR = np.round(lowerVaR, 4)

# Simulate 50 EONIA short rates as an example
Z_mb = RandomGenerator(50, [20, hidden_dim])
samples = recovery_model(generator_model(Z_mb)).numpy()
reshaped_data = samples.reshape((samples.shape[0]*samples.shape[1], 
                                samples.shape[2]))

scaled_reshaped_data = scaler.inverse_transform(reshaped_data)
simulations = scaled_reshaped_data.reshape(((samples.shape[0],
                                             samples.shape[1], 
                                             samples.shape[2])))

# Only get the EONIA simulations
simulations = simulations[:, :, 8]

dates2 = pd.date_range(start = '03-12-2018',
                     end = '31-12-2018',
                     periods = 20)

# Create plot of 20-day simulations
#fig2 = px.line(go.Scatter(x=dates2, y=simulations))
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=dates2, y=simulations,
                    mode='lines',
                    name='lines'))

fig2.update_layout(yaxis_title = 'Simulation of short rate', xaxis_title = 'Date')
fig2.update_layout({
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'paper_bgcolor': 'rgba(0,0,0,0)',
})
fig2.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)

# Create a second plot of the ESTER data
df = pd.read_csv('data/df_full_pre_ester.csv', sep=';')
df2 = pd.read_csv('data/EONIA_rate.csv', sep=';')

dates = pd.date_range(start = '01-01-2017',
                     end = '15-05-2020',
                     periods = 649)
fig = px.line(x=dates, y=df.WT)
fig.update_layout(yaxis_title = 'Short rate', xaxis_title = 'Date')
              
fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)

fig.update_layout({
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'paper_bgcolor': 'rgba(0,0,0,0)',
})


# Create Table with the VaR estimates
VaRs = pd.DataFrame([np.append("TimeGAN - upper", upperVaR), 
                    np.append("Vasicek - upper", upperVasicek),
                    np.append("TimeGAN with PLS+FM - lower", lowerVaR), 
                    np.append("Vasicek - lower", lowerVasicek)])

VaRs.columns=['Model', '1 day', '10 days', '20 days']
VaRs.index=['TimeGAN with PLS+FM', 'Vasicek' ,'TimeGAN with PLS+FM', 'Vasicek']

page_2_layout = html.Div([    
    
    dcc.Markdown('''###### This figure shows ESTER simulations based on TimeGAN'''),
    html.Div(id='page-1-content'),
    
    dcc.Graph(figure = fig2),

    #dbc.Row(children=[dbc.Col(children=[latent_slider_1], className="col-md-8"),
    #                      dbc.Col(children=["Slider for latent variable"], className="col-md-4")]),

    dcc.Graph(figure = fig), 

    dcc.Markdown('''
                ###### 2.) Value-at-Risk for ESTER based on TimeGAN and 1-factor Vasicek.''', style = {"padding":"3px"}),
    
    dash_table.DataTable(
        id = 'VaR-table',
        data=VaRs.to_dict('records'),
        columns=[{'id': c, 'name': c} for c in VaRs.columns],

        style_header={'backgroundColor': 'rgb(38, 43, 61)'},
        style_cell={
            'backgroundColor': 'rgb(38, 43, 61)',
            'color': 'white'
        },
    ),

], style={"margin-right":"2%", "margin-left":"2%"})