import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np

page_1_layout = html.Div(children=[
    dcc.Markdown('''###### Equation (1) shows the min-max objective for the Generator and Discriminator in a regular GAN setup.'''),
    html.Div(html.Img(src='assets/normal_min_max.png', style={'width':'100%'})),

    dcc.Markdown('''###### The TimeGAN models is in essence a regular GAN on a latent space. Equation (2) shows the \
                recovery loss due to mapping to the latent space and recovering the original data. Next to that, the TimeGAN \
                model supervises the latent space using Equation (3).'''),
    dbc.Row([dbc.Col(html.Div(html.Img(src='assets/reconstruction_loss.png', style={'width':'100%'})), width={"offset":1, 'size':5}),
            dbc.Col(html.Div(html.Img(src='assets/supervised_loss.png', style={'width':'100%'})), width={"offset":0, 'size':5})]),

    dcc.Markdown('''###### The TimeGAN model formulates two objectives. Equation (4) shows objective for the Autoencoder and Supervisor and Equation (5) show the objective \
                 for the GAN and the Supervisor. Both the objectives are simulataneously trained.'''),

    dbc.Row([dbc.Col(html.Div(html.Img(src='assets/first_objective.png', style={'width':'85%'})), width={"offset":1, 'size':5}),
            dbc.Col(html.Div(html.Img(src='assets/second_objective.png', style={'width':'100%'})), width={"offset":0, 'size':5})]),
    
    dcc.Markdown(''' ###### This results in the the block diagram (left) and the the training schema (right) of the TimeGAN model.'''),
    html.Div([html.Img(src='assets/block_diagram.png', style={'width':'45%', 'padding':'10pt'}),
              html.Img(src='assets/training_schema.png', style={'width':'51%', 'padding':'10pt'})]),
    
    dcc.Markdown('''###### **Our contribution:** In a regular GAN model, the unsupervised loss is the negative cross-entropy as shown in Equation (6). \
                 In this study, we also evaluate a different unsupervised loss. \
                 We implement the Wasserstein-1 loss with Gradient Penalty as shown in Equation (7). '''),

    dbc.Row(dbc.Col(html.Div(html.Img(src='assets/negative_entropy_loss.png', style={'width':'80%'})), width={'offset':1})),    
    dbc.Row(dbc.Col(html.Div(html.Img(src='assets/wasserstein_loss.png', style={'width':'80%'})), width={'offset':1})),

    dcc.Markdown(''' ###### We also implement Feature Matching as shown in Equation (9) as a penalization for mode collapse. \
                 We add the Feature Matching loss to the second objective of the TimeGAN model which results in a new \
                 objective in Equation (10). '''),
    dbc.Row(dbc.Col(html.Div(html.Img(src='assets/feature_matching_loss.png', style={'width':'80%'})), width={'offset':1}), style={'margin-bottom':'2%'}),
    dbc.Row(dbc.Col(html.Div(html.Img(src='assets/feature_matching_gan.png', style={'width':'80%'})), width={'offset':1})),

    dcc.Markdown(''' ###### Lastly, we also implement Positive Label Smoothing to the unsupervised loss to prevent mode collapse. \
                 We define the new optimal Discriminator in Equation (11). '''),
    dbc.Row(dbc.Col(html.Div(html.Img(src='assets/positive_label_smoothing.png', style={'width':'80%'})), width={'offset':1}))

], style={"margin-right":"2%", "margin-left":"2%", "margin-bottom":"2%"})