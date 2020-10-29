import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np

deltas1 = (np.random.rand(16)-0.5) * 10000
deltas2 = (np.random.rand(16)*2-1) * 10000
deltas3 = (np.random.rand(16)*3-1.5) * 10000

tenors = ["01M", "02M", "03M", "04M", "05M", "06M", "09M", "01Y",
          "02Y", "03Y", "04Y", "05Y", "10Y", "20Y", "30Y", "50Y"]

traceIRS1 = go.Bar(x = tenors, y = deltas1, name = "Market maker")
traceIRS2 = go.Bar(x = tenors, y = deltas2, name = "Pension fund")
traceIRS3 = go.Bar(x = tenors, y = deltas3, name = "Hedge fund")

vasicek = (np.random.rand(1000)*2)
hull_white = (np.random.rand(1000)*1.92)

page_1_layout = html.Div(children=[
    dcc.Markdown('''###### Equation (1) shows the min-max objective for the Generator and Discriminator.'''),
    html.Div(html.Img(src='assets/normal_min_max.png', style={'width':'100%'})),

    dcc.Markdown('''###### TimeGAN performs a GAN on a latent space. Equation (2) shows the \
                recovery loss due to mapping to the latent space and recovering the original data. \
                Equation (3) shows the supervised loss on the latent space.'''),
    dbc.Row([dbc.Col(html.Div(html.Img(src='assets/reconstruction_loss.png', style={'width':'100%'})), width={"offset":1, 'size':5}),
            dbc.Col(html.Div(html.Img(src='assets/supervised_loss.png', style={'width':'100%'})), width={"offset":0, 'size':5})]),

    dcc.Markdown('''###### Equation (4) shows objective 1 and Equation (5) the GAN objective.'''),

    dbc.Row([dbc.Col(html.Div(html.Img(src='assets/first_objective.png', style={'width':'90%'})), width={"offset":1, 'size':5}),
            dbc.Col(html.Div(html.Img(src='assets/second_objective.png', style={'width':'100%'})), width={"offset":0, 'size':5})]),

    dcc.Markdown('''###### Equation (6) shows the unsupervised loss. In the study, we evaluate a different \
                 unsupervised loss metric. We implement the Wasserstein loss shown in Equation (7). This \
                 results in the new objective in Eq. (8).'''),

    html.Div(html.Img(src='assets/negative_entropy_loss.png', style={'width':'90%'})),
    html.Div(html.Img(src='assets/wasserstein_loss.png', style={'width':'90%'})),
    html.Div(html.Img(src='assets/wasserstein_gan.png', style={'width':'90%'})),

    dcc.Markdown(''' ###### We also add Feature Matching shown in Eq. (9) as a penalization for mode collapse. \
                 The new objective now becomes Eq. (10).'''),
    html.Div(html.Img(src='assets/feature_matching_loss.png', style={'width':'90%'})),
    html.Div(html.Img(src='assets/feature_matching_gan.png', style={'width':'80%'})),

    dcc.Markdown(''' ###### The figure on the left shows the block diagram of \
                the TimeGAN model. The figure on the right shows the training schema, i.e. backpropagation schema.'''),

    html.Div([html.Img(src='assets/block_diagram.png', style={'width':'45%'}),
              html.Img(src='assets/training_schema.png', style={'width':'51%'})]),

    #dcc.Markdown(''' ###### 1.) Delta ladder for a fictitious fixed income portfolio''', style = {"padding":"3px"}),
    
    #dcc.Graph(
    #   id = 'delta-ladder',
    #   figure={
    #       'data': [traceIRS1, traceIRS2, traceIRS3],
    #       'layout': go.Layout(title='Delta ladder of IRS portfolio',  
    #                           barmode='stack', 
    #                           plot_bgcolor='rgb(38,43,61)',  paper_bgcolor='rgb(38,43,61)',
    #                           )
    #   }),
], style={"margin-right":"2%", "margin-left":"2%", "margin-bottom":"2%"})