import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import dash_table

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

VaRs = pd.DataFrame([["TimeGAN", 0.19, 0.21, 0.18], ["Vasicek", 0.53, 0.75, 0.85], ["Hull-White", 1.21, 1.59, 1.52]])
VaRs.columns=['Model', '10-days', '20-days', '50-days']
VaRs.index=['TimeGAN', 'Vasicek', 'Hull-White']

page_2_layout = html.Div([    
    html.H4('Interest rate risk via Monte-Carlo simulation', style = {"font-size":"24pt", "font-weight":"200", "letter-spacing":"1px"}),
    
    dcc.Markdown('''
                ###### 1.) ESTER simulations based on TimeGAN''', style = {"padding":"3px"}),

    dcc.Dropdown(
        id='page-1-dropdown',
        options=[{'label': i, 'value': i} for i in ['ESTER', 'PRE-ESTER', 'EONIA']],
        value='ESTER', 
        className = 'select-control'
    ),

    html.Div(id='page-1-content'),
    
    dcc.Graph(figure = fig), 
    
    dcc.Markdown('''
                ###### 2.) Value-at-Risk for ESTER based on TimeGAN and reference models''', style = {"padding":"3px"}),
    
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


], style={"margin-right":"5%", "margin-left":"5%"})