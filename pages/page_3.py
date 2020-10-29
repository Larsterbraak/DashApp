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
import numpy as np

from assets.metrics import load_models

tf.keras.backend.set_floatx('float64')
tf.get_logger().setLevel('INFO')

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
embedder_model, recovery_model, supervisor_model, generator_model, discriminator_model = load_models(8250, hparams, hidden_dim)

# Fix the random generator at a specific point
#Z_mb = tf.zeros(shape=(1000, 20, 4))
#changes = np.linspace(0, 1, 5)

#latent_space = generator_model(Z_mb)

# Adjust the first dimension of the latent space
#latent_space[:, :, 0] = latent_space[:, :, 0] #+ changes[2]

#latent_space = tf.cast(latent_space, tf.float64)

#samples = recovery_model(Z_mb).numpy()
#reshaped_data = samples.reshape((samples.shape[0]*samples.shape[1], 
#                                samples.shape[2]))
            
#scaled_reshaped_data = scaler.inverse_transform(reshaped_data)
#simulations = scaled_reshaped_data.reshape(((samples.shape[0],
#                                             samples.shape[1], 
#                                             samples.shape[2])))

# Only get the EONIA simulations
#simulations = simulations[:, :, 8]

#dates2 = pd.date_range(start = '03-12-2018',
#                     end = '31-12-2018',
#                     periods = 20)

#df5 = pd.DataFrame(simulations.T)
#df5['Date'] = dates2

#fig2 = px.line(df5, x='Date', y=[x for x in range(20)])
#fig2.update_layout(yaxis_title = 'EONIA [%]', xaxis_title = 'Date', title='Simulations of EONIA using TimeGAN based on latent space', title_x=0.5)
#fig2.update_layout({
#    'plot_bgcolor': 'rgba(255,255,255,0)',
#    'paper_bgcolor': 'rgba(0,0,0,0)',
#})

latent_slider_1 = dcc.Slider(id="latent_input_1", min=0, max=1, step=0.01, value=0.25, marks={x: f"{1*x:.2f}" for x in np.linspace(0, 1, 11)})
latent_slider_2 = dcc.Slider(id="latent_input_2", min=0, max=1, step=0.01, value=0.25, marks={x: f"{1*x:.2f}" for x in np.linspace(0, 1, 11)})
latent_slider_3 = dcc.Slider(id="latent_input_3", min=0, max=1, step=0.01, value=0.25, marks={x: f"{1*x:.2f}" for x in np.linspace(0, 1, 11)})
latent_slider_4 = dcc.Slider(id="latent_input_4", min=0, max=1, step=0.01, value=0.25, marks={x: f"{1*x:.2f}" for x in np.linspace(0, 1, 11)})

page_3_layout = html.Div([    
    dcc.Markdown('''###### This figure shows the ESTER simulation conditioned on a specific latent space. \
                 Use the sliders below to adjust the latent spaces and see the effect.'''),
    
    #dcc.Graph(figure = fig2),

    dbc.Row(children=[dbc.Col(children=[latent_slider_1], className="col-md-10"),
                      dbc.Col(children=["Latent var 1"], className="col-md-2")]),
    
    dbc.Row(children=[dbc.Col(children=[latent_slider_2], className="col-md-10"),
                     dbc.Col(children=["Latent var 2"], className="col-md-2")]),
    
    dbc.Row(children=[dbc.Col(children=[latent_slider_3], className="col-md-10"),
                     dbc.Col(children=["Latent var 3"], className="col-md-2")]),

    dbc.Row(children=[dbc.Col(children=[latent_slider_4], className="col-md-10"),
                     dbc.Col(children=["Latent var 4"], className="col-md-2")]),

    html.Div(id='page-3-content')
], style={"margin-right":"2%", "margin-left":"2%"})