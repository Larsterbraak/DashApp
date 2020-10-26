import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import dash_table

from assets.metrics import load_models

# 1. Import the EONIA data
data = pd.read_csv("data/EONIA.csv", sep=";")
dates_EONIA = np.ravel(data.Date[:525][::-1].values).astype(str)
dates_EONIA = [datetime.datetime.strptime(d,"%d-%m-%Y").date()
               for d in dates_EONIA]
data = np.array(data.iloc[:525,2])[::-1] # Filter for only EONIA

# Import the data and apply the transformation
# We import the data until 6-10-2017
df = pd.read_csv("data/Master_EONIA.csv", sep=";")
dates_t_var = df.iloc[3798:, 0]
dates_t_var = [datetime.datetime.strptime(d,"%d-%m-%Y").date()
               for d in dates_t_var]
df = df.iloc[:, 1:] # Remove the Date variable from the dataset
df.EONIA[1:] = np.diff(df.EONIA)
df = df.iloc[1:, :]
scaler = preprocessing.MinMaxScaler().fit(df)

# Define the settings
hparams = []
hidden_dim = 4

# Import the pre-trained models
embedder_model, recovery_model, supervisor_model, generator_model, discriminator_model = load_models(2700, hparams, hidden_dim)

# Simulate 250 EONIA short rates
Z_mb = RandomGenerator(10000, [20, hidden_dim])
samples = recovery_model(generator_model(Z_mb)).numpy()
reshaped_data = samples.reshape((samples.shape[0]*samples.shape[1], 
                                    samples.shape[2]))

scaled_reshaped_data = scaler.inverse_transform(reshaped_data)
simulations = scaled_reshaped_data.reshape(((samples.shape[0],
                                                samples.shape[1], 
                                                samples.shape[2])))

upperVaR = []
lowerVaR = []
for i in [1, 10, 20]:
    # Simulate 250 EONIA short rates
    Z_mb = RandomGenerator(10000, [i, hidden_dim])
    samples = recovery_model(generator_model(Z_mb)).numpy()
    reshaped_data = samples.reshape((samples.shape[0]*samples.shape[1], 
                                    samples.shape[2]))

    scaled_reshaped_data = scaler.inverse_transform(reshaped_data)
    simulations = scaled_reshaped_data.reshape(((samples.shape[0],
                                                samples.shape[1], 
                                                samples.shape[2])))

    results = np.sum(simulations[:,:,8], axis=1)
    results.sort()

    upperVaR.append(results[9900])
    lowerVaR.append(results[100])

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

VaRs = pd.DataFrame([["TimeGAN - upper", upperVaR], 
                    ["TimeGAN with PLS+FM - lower", lowerVaR], 
                    ["Vasicek", 1.21, 1.59, 1.52]])

VaRs.columns=['Model', '1 day', '10 days', '20 days']
VaRs.index=['TimeGAN', 'TimeGAN with PLS+FM', 'Vasicek']

page_2_layout = html.Div([    
    html.H4('Interest rate simulation', style = {"font-size":"20pt", "font-weight":"200", "letter-spacing":"1px"}),
    
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
                ###### 2.) Value-at-Risk for ESTER based on TimeGAN and 1-factor Vasicek.''', style = {"padding":"3px"}),
    
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