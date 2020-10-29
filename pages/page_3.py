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
import numpy as np

tf.keras.backend.set_floatx('float64')
tf.get_logger().setLevel('INFO')

VaRs = pd.DataFrame([["TimeGAN", 0.19, 0.21, 0.18], ["Vasicek", 0.53, 0.75, 0.85], ["Hull-White", 1.21, 1.59, 1.52]])
VaRs.columns=['Model', '10-days', '20-days', '50-days']
VaRs.index=['TimeGAN', 'Vasicek', 'Hull-White']

df = pd.read_csv('data/df_full_pre_ester.csv', sep=';')

dates = pd.date_range(start = '01-01-2017',
                     end = '15-05-2020',
                     periods = 649)


fig = px.line(x=dates, y=df.WT)
fig.update_layout(yaxis_title = 'Short rate', xaxis_title = 'Date $\tau$')
fig.add_trace(go.Scatter(x=dates, y=[-0.4531 for x in range(649)], name='VaR(99%)',
                         line=dict(color='firebrick', width=2, dash = 'dash')))
fig.update_layout({
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'paper_bgcolor': 'rgba(0,0,0,0)',
})

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

# The t-SNE projections
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

X, Target = load_digits(return_X_y=True)

tsne = TSNE()
tsne_results = tsne.fit_transform(X)

traceTSNE = go.Scatter(x = tsne_results[:,0],
                       y = tsne_results[:,1],
                    #   name = Target,
                    #  hoveron = Target,
                       mode = 'markers',
                    #  text = Target.unique(),
                       showlegend = True,
                       marker = dict(size = 8,
                                    color = Target,
                                    colorscale ='Jet',
                                    showscale = False,
                                    
                                    line = dict(
                                    width = 2,
                                    color = 'rgb(255, 255, 255)'
                                    ),
                                opacity = 0.8
                            )
                    )

data = [traceTSNE]

layout = dict(title = 'TSNE (T-Distributed Stochastic Neighbour Embedding)',
              hovermode= 'closest',
              yaxis = dict(zeroline = False),
              #yaxis_title = 'Dimension 2',
              xaxis = dict(zeroline = False),
              #xaxis_title = 'Dimension 1 $$\tau$$',
              showlegend= True,
              plot_bgcolor = 'rgba(0,0,0,0)',
              paper_bgcolor = 'rgba(0,0,0,0)',

             )

fig2 = dict(data=data, layout=layout)

latent_slider_1 = dcc.Slider(id="latent_input_1", min=0, max=1, step=0.01, value=0.25, marks={x: f"{1*x:.0f}" for x in np.linspace(0, 1, 11)})
latent_slider_2 = dcc.Slider(id="latent_input_2", min=0, max=1, step=0.01, value=0.25, marks={x: f"{1*x:.0f}" for x in np.linspace(0, 1, 11)})
latent_slider_3 = dcc.Slider(id="latent_input_3", min=0, max=1, step=0.01, value=0.25, marks={x: f"{1*x:.0f}" for x in np.linspace(0, 1, 11)})
latent_slider_4 = dcc.Slider(id="latent_input_4", min=0, max=1, step=0.01, value=0.25, marks={x: f"{1*x:.0f}" for x in np.linspace(0, 1, 11)})

page_3_layout = html.Div([    
    dcc.Markdown('''###### Use the sliders to adjust the latent spaces.'''),
    
    #dcc.Graph(figure = fig),

    dbc.Row(children=[dbc.Col(children=[latent_slider_1], className="col-md-8"),
                      dbc.Col(children=["Latent variable 1"], className="col-md-4")]),
    
    dbc.Row(children=[dbc.Col(children=[latent_slider_2], className="col-md-8"),
                     dbc.Col(children=["Latent variable 2"], className="col-md-4")]),
    
    dbc.Row(children=[dbc.Col(children=[latent_slider_3], className="col-md-8"),
                     dbc.Col(children=[html.Img(src='first_objective.png')], className="col-md-4")]),

    dbc.Row(children=[dbc.Col(children=[latent_slider_4], className="col-md-8"),
                     dbc.Col(children=["Latent variable 4"], className="col-md-4")]),

    #dcc.Markdown('''
    #            ###### 2.) Qualtitative assessment of difference between simulated and real short rates based on t-SNE''', style = {"padding":"3px"}),
    #dcc.Graph(figure = fig2),
   
    #dcc.Markdown('''
                ###### 3.) Discriminative score for an ad-hoc discriminator training''', style = {"padding":"3px"}),
    
    #dash_table.DataTable(
    #    id = 'VaR-table',
    #    data=VaRs.to_dict('records'),
    #    columns=[{'id': c, 'name': c} for c in VaRs.columns],
        
    #    style_cell={
    #        'backgroundColor': 'rgb(38, 43, 61)',
    #        'color': 'white'
    #    },
    #),

    html.Div(id='page-3-content')
], style={"margin-right":"2%", "margin-left":"2%"})