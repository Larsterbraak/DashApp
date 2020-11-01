import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import tensorflow as tf
from sklearn import preprocessing
from assets.training import RandomGenerator
import numpy as np
from assets.metrics import load_models
from dash.dependencies import Input, Output

tf.get_logger().setLevel('ERROR')

def latent_plot(a, b, c, d):
    # Import the data and apply the transformation
    # We import the data until 6-10-2017
    df = pd.read_csv("data/Master_EONIA.csv", sep=";")
    df = df.iloc[:, 1:] # Remove the Date variable from the dataset
    df.EONIA[1:] = np.diff(df.EONIA)
    df = df.iloc[1:, :]
    scaler = preprocessing.MinMaxScaler().fit(df)

    # Define the settings
    hparams = []
    hidden_dim = 4

    # Import the pre-trained models
    em, recovery_model, sup, generator_model, dis = load_models(8250, hparams, hidden_dim)

    # Fix the random generator at a specific point
    Z_mb = np.zeros(shape=(10, 20, hidden_dim))

    # Generate the latent space
    latent_space = generator_model(Z_mb).numpy()

    # Adjust the first dimension of the latent space - latent variable 1
    latent_space[:, :, 0] = latent_space[:, :, 0] + a
    latent_space[:, :, 1] = latent_space[:, :, 1] + b
    latent_space[:, :, 2] = latent_space[:, :, 2] + c
    latent_space[:, :, 3] = latent_space[:, :, 3] + d

    # Produce the samples
    samples = recovery_model(latent_space).numpy()
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

    fig2 = px.line(df5, x='Date', y=[x for x in range(10)])
    fig2.update_layout(yaxis_title = 'daily difference EONIA', xaxis_title='', 
                    title={'text':'Simulations of EONIA using TimeGAN based on latent space', 'x':0.5, 'font':dict(color='white')})

    fig2.update_layout({
        'plot_bgcolor': 'rgba(255,255,255,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
    })

    fig2.update_layout(
        legend={'title':{'text':'Simulation', 'font':dict(color='white')}, 'font':dict(color='white')},
        yaxis_color = 'white',
        xaxis_color = 'white',
        yaxis=dict(range=[-0.25, 0.25])
    )

    fig2.update_xaxes(tickfont=dict(color='white'))
    fig2.update_yaxes(tickfont=dict(color='white'))

    return fig2

latent_slider_1 = dcc.Slider(id='latent-input-1', min=0, max=1, step=0.01, value=0.1, marks={x: f"{1*x:.2f}" for x in np.linspace(0, 1, 11)})
latent_slider_2 = dcc.Slider(id="latent-input-2", min=0, max=1, step=0.01, value=0.1, marks={x: f"{1*x:.2f}" for x in np.linspace(0, 1, 11)})
latent_slider_3 = dcc.Slider(id="latent-input-3", min=0, max=1, step=0.01, value=0.1, marks={x: f"{1*x:.2f}" for x in np.linspace(0, 1, 11)})
latent_slider_4 = dcc.Slider(id="latent-input-4", min=0, max=1, step=0.01, value=0.1, marks={x: f"{1*x:.2f}" for x in np.linspace(0, 1, 11)})

page_3_layout = html.Div([    
    dcc.Markdown('''###### This figure shows the ESTER simulation conditioned on latent space.'''),

    dcc.Graph(id='test-figure', style={'padding':'0', 'margin-bottom':'0%'}),

    dbc.Row(children=[dbc.Col(children=[latent_slider_1], className="col-md-11"),
                      dbc.Col(html.Img(src='assets/latent_var_1.png', style={'width':'80%'}))]),
    
    dbc.Row(children=[dbc.Col(children=[latent_slider_2], className="col-md-11"),
                      dbc.Col(html.Img(src='assets/latent_var_2.png', style={'width':'80%'}))]),

    dbc.Row(children=[dbc.Col(children=[latent_slider_3], className="col-md-11"),
                      dbc.Col(html.Img(src='assets/latent_var_3.png', style={'width':'80%'}))]),
    
    dbc.Row(children=[dbc.Col(children=[latent_slider_4], className="col-md-11"),
                     dbc.Col(html.Img(src='assets/latent_var_4.png', style={'width':'80%'}))]),

    html.Div(id='page-3-content')
], style={"margin-right":"2%", "margin-left":"2%"})