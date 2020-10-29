import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np

page_1_layout = html.Div(children=[
    dcc.Markdown('''###### Equation (1) shows the min-max objective for the Generator and Discriminator.'''),
    html.Div(html.Img(src='assets/normal_min_max.png', style={'width':'100%'})),

    dcc.Markdown('''###### TimeGAN performs a GAN on a latent space. Equation (2) shows the \
                recovery loss due to mapping to the latent space and recovering the original data. \
                Equation (3) shows the supervised loss on the latent space.'''),
    dbc.Row([dbc.Col(html.Div(html.Img(src='assets/reconstruction_loss.png', style={'width':'100%'})), width={"offset":1, 'size':5}),
            dbc.Col(html.Div(html.Img(src='assets/supervised_loss.png', style={'width':'100%'})), width={"offset":0, 'size':5})]),

    dcc.Markdown('''###### Equation (4) shows objective for the Autoencoder and Supervisor and Equation (5) show the objective \
                 for the GAN and Supervisor.'''),

    dbc.Row([dbc.Col(html.Div(html.Img(src='assets/first_objective.png', style={'width':'85%'})), width={"offset":1, 'size':5}),
            dbc.Col(html.Div(html.Img(src='assets/second_objective.png', style={'width':'100%'})), width={"offset":0, 'size':5})]),

    dcc.Markdown('''###### Equation (6) shows the normal unsupervised loss, i.e. negative cross-entropy.'''),

    dbc.Row(dbc.Col(html.Div(html.Img(src='assets/negative_entropy_loss.png', style={'width':'80%'})), width={'offset':1})),    

    dcc.Markdown('''###### In this study, we also evaluate a different unsupervised loss. \
                 We implement the Wasserstein-1 loss with Gradient Penalty shown in Equation (7). '''),
    dbc.Row(dbc.Col(html.Div(html.Img(src='assets/wasserstein_loss.png', style={'width':'75%'})), width={'offset':1})),

    dcc.Markdown(''' ###### We also evaluate Feature Matching as shown in Eq. (9) as penalization for mode collapse.'''),
    dbc.Row(dbc.Col(html.Div(html.Img(src='assets/feature_matching_loss.png', style={'width':'80%'})), width={'offset':1})),

    dcc.Markdown(''' ###### The Feature Matching loss is added to the second objective of TimeGAN in Eq. (10).'''),
    dbc.Row(dbc.Col(html.Div(html.Img(src='assets/feature_matching_gan.png', style={'width':'70%'})), width={'offset':2})),

    dcc.Markdown(''' ###### The left figure shows the block diagram of \
                the TimeGAN model and the right figure shows the training schema.'''),

    html.Div([html.Img(src='assets/block_diagram.png', style={'width':'45%'}),
              html.Img(src='assets/training_schema.png', style={'width':'51%'})]),
], style={"margin-right":"2%", "margin-left":"2%", "margin-bottom":"2%"})