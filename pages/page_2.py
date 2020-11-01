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
from dash_table.Format import Format, Scheme

#from app import app

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

def simulate_yeah(simus):

    # Define the settings
    hparams = []
    hidden_dim = 4

    # Import the pre-trained models
    embedder_model, recovery_model, supervisor_model, generator_model, discriminator_model = load_models(8250, hparams, hidden_dim)

    # Simulate 50 EONIA short rates as an example
    Z_mb = RandomGenerator(simus, [20, hidden_dim])
    samples = recovery_model(generator_model(Z_mb)).numpy()
    reshaped_data = samples.reshape((samples.shape[0]*samples.shape[1], 
                                    samples.shape[2]))

    scaled_reshaped_data = scaler.inverse_transform(reshaped_data)
    simulations = scaled_reshaped_data.reshape(((samples.shape[0],
                                                samples.shape[1], 
                                                samples.shape[2])))

    # Only get the EONIA simulations
    simulations = simulations[:, :, 8]

    dates2 = pd.date_range(start = '12-03-2018',
                        end = '12-31-2018',
                        periods = 20)

    df5 = pd.DataFrame(simulations.T)
    df5['Date'] = dates2

    fig2 = px.line(df5, x='Date', y=[x for x in range(simus)])
    fig2.update_layout(yaxis_title = 'Daily difference EONIA', xaxis_title = 'Date', 
                    title={'text':'Simulations of EONIA using TimeGAN', 'x':0.5, 'font':dict(color='white')})

    fig2.update_layout(
        legend={'title':{'text':'Simulation', 'font':dict(color='white')}, 'font':dict(color='white')},
        yaxis_color = 'white',
        xaxis_color = 'white',
        yaxis=dict(range=[-0.006, 0.006])
    )

    fig2.update_layout({
        'plot_bgcolor': 'rgba(255,255,255,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
    })

    fig2.update_xaxes(tickfont=dict(color='white'), visible=True)
    fig2.update_yaxes(tickfont=dict(color='white'))

    return fig2

# Create Table with the VaR estimates
VaRs = pd.DataFrame([np.append("TimeGAN with PLS+FM - upper", upperVaR), 
                    np.append("Vasicek - upper", upperVasicek),
                    np.append("TimeGAN with PLS+FM - lower", lowerVaR), 
                    np.append("Vasicek - lower", lowerVasicek)])

# Rename the columns and rows
VaRs.columns=['Model \ T', '1 day', '10 days', '20 days']
VaRs.index=['TimeGAN with PLS+FM', 'Vasicek' ,'TimeGAN with PLS+FM', 'Vasicek']

pd.options.display.float_format = '${:.2f}'.format

page_2_layout = html.Div([    
    
    dcc.Markdown('''###### The figure shows €STER simulations based on TimeGAN. \
                Double click on one of simulations in the legend to isolate a €STER simulation. '''),

    html.Div([html.Button('Simulate 1 EONIA path', id='simulate_again_1', n_clicks=0, style={'margin-right':'1rem'}),
              html.Button('Simulate 10 EONIA paths', id='simulate_again_10', n_clicks=0, style={'margin-right':'1rem'}),
              html.Button('Simulate 100 EONIA paths', id='simulate_again_100', n_clicks=0),
              html.Div(id='clicked-button', children='btn1:0 btn2:0 btn3:0 last:nan', style={'display':'none'})], style={'display':'inline-block'}),

    dcc.Graph(id='page-2-graph'),

    dcc.Markdown('''###### The table below shows the Value-at-Risk for EONIA based on \
                 TimeGAN and 1-factor Vasicek for multiple T based on 1,000 EONIA simulations.''', style = {"padding":"3px"}),
    
    dbc.Row(dbc.Col(
        dash_table.DataTable(
        id = 'VaR-table',
        data=VaRs.to_dict('records'),
        columns=[{'id': c, 'format': Format( 
                precision=5,
                scheme=Scheme.fixed,
                decimal_delimiter='.'), 'name': c} for c in VaRs.columns],
        style_header={'backgroundColor': 'rgb(21, 46, 64)',
                      'fontWeight': 'bold',
                      'border': '2px solid white'},
        style_cell={
            'backgroundColor': 'rgb(38, 43, 61)',
            'color': 'white',
            'padding': '5px',
        },
    ), width={'offset':2, 'size':8})),

    html.Div(id='page-1-content')

], style={"margin-right":"2%", "margin-left":"2%"})