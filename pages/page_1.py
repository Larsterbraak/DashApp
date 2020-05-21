import dash
import dash_core_components as dcc
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
    html.H4('Interest rate risk for fictious IRS portfolio', style = {"font-size":"24pt", "font-weight":"200", "letter-spacing":"1px"}),
    
    html.H4('1.) Delta ladder for a fictitious fixed income portfolio', style = {"font-size":"16pt", "font-weight":"200", "letter-spacing":"1px"}),
    html.Div(id='home-content'),
    dcc.Graph(
        id = 'delta-ladder',
        figure={
            'data': [traceIRS1, traceIRS2, traceIRS3],
            'layout': go.Layout(title='Delta ladder of IRS portfolio',  
                                barmode='stack', 
                                plot_bgcolor='rgb(38,43,61)',  paper_bgcolor='rgb(38,43,61)',
                                )
        }),
    
], style={"margin-right":"5%", "margin-left":"5%"})