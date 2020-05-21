import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

VaRs = pd.DataFrame([[0.19, 0.21, 0.18], [0.53, 0.75, 0.85], [1.21, 1.59, 1.52]])
VaRs.columns=['10-days', '20-days', '50-days']
VaRs.index=['TimeGAN', 'Vasicek', 'Hull-White']

df = pd.read_csv('data/df_full_pre_ester.csv', sep=';')

dates = pd.date_range(start = '01-01-2017',
                     end = '15-05-2020',
                     periods = 649)


fig = px.line(x=dates, y=df.WT)
fig.update_layout(yaxis_title = 'Short rate', xaxis_title = 'Date')
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

page_3_layout = html.Div([
    html.H4('Training results for the TimeGAN', style = {"font-size":"24pt", "font-weight":"200", "letter-spacing":"1px"}),
    html.P('Check out the predicted VaR(99%) based on the TimeGAN in this plot!'),
    dcc.Graph(figure = fig),
    html.P('Check out the VaR(99%) for different time intervals!'),
    dash_table.DataTable(
        id = 'VaR-table',
        data=VaRs.to_dict('records'),
        columns=[{'id': c, 'name': c} for c in VaRs.columns],
        
        style_cell={
            'backgroundColor': 'rgb(38, 43, 61)',
            'color': 'white'
        },
    ),

    html.Div(id='page-3-content')
], style={"margin-right":"10%", "margin-left":"10%"})