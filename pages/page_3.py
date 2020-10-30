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

#from app import app

tf.get_logger().setLevel('ERROR')

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
Z_mb = np.zeros(shape=(1000, 20, hidden_dim))

# Create the possible changes
changes = np.linspace(0, 1, 10)

# Generate the latent space
latent_space = generator_model(Z_mb).numpy()

# Adjust the first dimension of the latent space - latent variable 1
latent_space[:, :, 0] = latent_space[:, :, 0] + changes[7]

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

dates2 = pd.date_range(start = '03-12-2018',
                      end = '31-12-2018',
                      periods = 20)

df5 = pd.DataFrame(simulations.T)
df5['Date'] = dates2

fig2 = px.line(df5, x='Date', y=[x for x in range(20)])
fig2.update_layout(yaxis_title = 'daily difference EONIA', xaxis_title = 'Date', title='Simulations of EONIA using TimeGAN based on latent space', title_x=0.5)
fig2.update_layout({
    'plot_bgcolor': 'rgba(255,255,255,0)',
    'paper_bgcolor': 'rgba(0,0,0,0)',
})

latent_slider_1 = dcc.Slider(id='latent-input-1', min=0, max=1, step=0.01, value=0.5, marks={x: f"{1*x:.2f}" for x in np.linspace(0, 1, 11)})
latent_slider_2 = dcc.Slider(id="latent-input-2", min=0, max=1, step=0.01, value=0.5, marks={x: f"{1*x:.2f}" for x in np.linspace(0, 1, 11)})
latent_slider_3 = dcc.Slider(id="latent-input-3", min=0, max=1, step=0.01, value=0.5, marks={x: f"{1*x:.2f}" for x in np.linspace(0, 1, 11)})
latent_slider_4 = dcc.Slider(id="latent-input-4", min=0, max=1, step=0.01, value=0.5, marks={x: f"{1*x:.2f}" for x in np.linspace(0, 1, 11)})

page_3_layout = html.Div([    
    dcc.Markdown('''###### This figure shows the ESTER simulation conditioned on a specific latent space. \
                 Use the sliders below to adjust the latent spaces and see the effect.'''),
    
    dcc.Graph(figure = fig2, style={'padding':'0', 'margin-bottom':'0%'}),

    #dcc.Graph(id='test-figure-123'),

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

# @page_3_layout.callback(Output(component_id='test-figure-123', component_property='figure'),
#                         [Input('latent-input-1', 'value'),
#                          Input('latent-input-2', 'value'),
#                          Input('latent-input-3', 'value'),
#                          Input('latent-input-4', 'value')])
# def update_figure(value):
#     # Fix the random generator at a specific point
#     Z_mb = np.zeros(shape=(1000, 20, hidden_dim))

#     # Generate the latent space
#     latent_space = generator_model(Z_mb).numpy()

#     # Adjust the dimensions of the latent space
#     latent_space[:, :, 0] = latent_space[:, :, 0] + value[0].astype(np.float64)
#     latent_space[:, :, 1] = latent_space[:, :, 0] + value[1].astype(np.float64)
#     latent_space[:, :, 2] = latent_space[:, :, 0] + value[2].astype(np.float64)
#     latent_space[:, :, 3] = latent_space[:, :, 0] + value[3].astype(np.float64)

#     # Produce the samples
#     samples = recovery_model(latent_space).numpy()
#     reshaped_data = samples.reshape((samples.shape[0]*samples.shape[1], 
#                                     samples.shape[2]))
                
#     scaled_reshaped_data = scaler.inverse_transform(reshaped_data)
#     simulations = scaled_reshaped_data.reshape(((samples.shape[0],
#                                                 samples.shape[1], 
#                                                 samples.shape[2])))

#     # Only get the EONIA simulations
#     simulations = simulations[:, :, 8]

#     dates2 = pd.date_range(start = '03-12-2018',
#                         end = '31-12-2018',
#                         periods = 20)

#     df5 = pd.DataFrame(simulations.T)
#     df5['Date'] = dates2

#     fig = px.line(df5, x='Date', y=[x for x in range(20)])
#     fig.update_layout(yaxis_title = 'daily difference EONIA', xaxis_title = 'Date', title='Simulations of EONIA using TimeGAN based on latent space', title_x=0.5)
#     fig.update_layout({
#         'plot_bgcolor': 'rgba(255,255,255,0)',
#         'paper_bgcolor': 'rgba(0,0,0,0)',
#     })
    
#     fig.update_layout(transition_duration=500)

#     return fig