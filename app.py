import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from sklearn import preprocessing
import pandas as pd
import numpy as np
import plotly.express as px

from pages import page_1
from pages import page_2
from pages import page_3

from assets.training import RandomGenerator
from assets.metrics import load_models

# external CSS stylesheets
external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    {
        'href': 'https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css',
        'rel': 'stylesheet',
        'integrity': 'sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO',
        'crossorigin': 'anonymous'
    },
    #'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML'
]

app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets)

#mathjax = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML'
#app.scripts.append_script({ 'external_url' : mathjax })

server = app.server
app.config.suppress_callback_exceptions = True
app.scripts.config.serve_locally = True

SIDEBAR_STYLE = {
    "top": "4rem",
    "left": "4rem",
    "bottom": 0,
    "width": "30%",
    "margin-left": "2%",
    "margin-right": "0.5%",
    "padding": "2rem 1rem",
    "background-color": "rgb(38,43,61)",
    "color": "rgb(226,239,250)",
    'border': '1px solid rgb(38,43,61)',
    'borderRadius': '15px',
    'overflow': 'hidden',
    'font':{"family":'Helvetica', "size":"10"}
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "top": "10rem",
    "left": 0,
    "right": 0,
    "margin-left": "1%",
    "margin-right": "2%",
    "padding": "2rem 1rem",
    "width": "64%",
    "background-color": "rgb(38,43,61)",
    "color": "rgb(226,239,250)",
    'border': '1px solid rgb(38,43,61)',
    'borderRadius': '15px',
    'overflow': 'hidden',
    'font':'Helvetica'
}

NAVBAR_STYLE = {
    "margin-left":"1rem",
    "margin-right": "1rem",
    "margin-bottom": "1rem",
    "margin-top": "0rem",
    "width":"98%",
    "padding":"0.1rem 0.1rem",
    "background-color": "rgb(38,43,61)",
    "color": "rgb(226,239,250)",
    'border': '1px solid rgb(38,43,61)',
    'borderRadius': '15px',
    'overflow': 'hidden',
    'font':'Helvetica'
}

TEST_STYLE = {
    "background-color": "rgb(38,43,61)",
    "color": "rgb(226,239,250)",
    'border': '0.5px solid white',
    'font':'Helvetica',
    "heigth":"1%",
}

TEST_STYLE_2 = {
    "background-color": "rgb(38,43,61)",
    "color": "#7CD3F7",
    'borderRight': '0.5px solid white',
    'borderLeft': '0.5px solid white',
    "float":"center",
    'font':'Helvetica',
    "margin-left":"5%",
    "borderTop":"3px solid #7CD3F7",
    'borderBottom':'0px solid #d6d6d6',
    "heigth":"250px",
}

TEST_STYLE_3 = {
    "background-color": "rgb(38,43,61)",
    "color": "rgb(226,239,250)",
    'borderRight': '0.5px solid white', 
    "float":"center",
    'font':'Helvetica',
    "borderTop":"0.5px solid #d6d6d6",
    'borderBottom':'0.5px solid #d6d6d6',
    "heigth":"250px",
}

TEST_STYLE_4 = {
    "background-color": "rgb(38,43,61)",
    "color": "rgb(226,239,250)",
    "float":"center",
    "heigth":"250px",
    "borderTop":"0.5px solid #d6d6d6",
    'font':'Helvetica',
    'borderBottom':'0.5px solid #d6d6d6',
    'borderRight': '0.5px solid white', 
}

tab_selected_style = {
    "background-color": "rgb(38,43,61)",
    "color": "#7CD3F7",
    "borderTop":"3px solid #7CD3F7",
    'borderBottom':'0px solid #d6d6d6',
    "heigth":"5%",
}

DROPDOWN_STYLE = {
    "background-color": "rgb(38,43,61)",
    "color": "blue",
    'font':'Helvetica'
}

NAV_STYLE = {
    "background-color": "rgb(38,43,61)",
    "color": "rgb(226,239,250)",
    'border': '0.5px solid white',
    "margin-left":"1rem",
    "width":"30%"
}

tabs = html.Div(dcc.Tabs(id = 'circos-control-tabs', value = 'what-is', children = [
    
    dcc.Tab(
        label = 'About',
        value = 'what-is',
        style=TEST_STYLE, selected_style=tab_selected_style,
        children = html.Div(id = 'control-tab', children = [
            dcc.Markdown(''' 
                        ###### This webapp is a demonstration of the MSc Thesis [TimeGAN for short rates](https://github.com/Larsterbraak/TimeGAN-short-rates/blob/master/Thesis.pdf) \
                        commisioned by the Erasmus University Quantitative Finance department and \
                        performed by [Lars ter Braak](https://www.linkedin.com/in/lars-ter-braak/). Feel free to have a look around and use \
                        the TimeGAN to simulate short rates or see the effect of the latent space. \
                        You can check a visualization of the Tensorflow training for the best TimeGAN model on [Tensorboard](https://tensorboard.dev/experiment/kqNuBA7aR96gB07zuM7z5g).''', style = {"font-size":"4pt", "font-weight":"100", "letter-spacing":"1px"}),
            ])
    ),

    dcc.Tab(
            label = 'Contents',
            value = 'data',      
            style=TEST_STYLE, selected_style=tab_selected_style,
            
            children = html.Div(className = 'circos-tab', children = [

                dcc.Markdown('''
                             ###### On page "1. Inner workings" we describe the Generative Adversarial Network (GAN) \
                             and discuss the TimeGAN model. On page "2. Short rate simulations" we shows a tool to simulate EONIA or €STER using \
                             the TimeGAN model and compare the performance to baseline models. On page "3. Inspect latent space" we show a tool to check the \
                             effect of the latent variables on the short rate simulations. Play around and see what type of short rate TimeGAN can produce. \
                             ''', style = {"font-size":"8pt", "font-weight":"10", "letter-spacing":"0.5px"}),
        
                ])
        )
]))

options = html.Div(tabs, className = 'item-a', style={"margin-right":"3%", "margin-left":"3%", "margin-top":"1%"})

sidebar = html.Div([
        options,
    ], style=SIDEBAR_STYLE)

content = html.Div([dbc.Nav(children = [

            dbc.NavItem(dbc.NavLink("1. Inner workings (math heavy)", active=False, href="/page-1", id="page-1-link", style={'font-weight':'bold'})),         
            dbc.NavItem(dbc.NavLink("2. Short rate simulation", active=False, href="/page-2", id="page-2-link", style={'font-weight':'bold'})),
            dbc.NavItem(dbc.NavLink("3. Inspect latent space", active=False, href="/page-3", id="page-3-link", style={'font-weight':'bold'}))
    
            ], pills=True, justified=True),

            html.Div(id="page-content")
        ], style=CONTENT_STYLE)

navbar = html.Nav(className = "navbar navbar-default navbar-static-top", children=[          
            html.Div([
                html.Button(html.A('Check out my LinkedIn', href="https://www.linkedin.com/in/lars-ter-braak/"),
                                    id='linked_in', n_clicks=0, className="button-primary"),
                html.Img(src='assets/linkedin_icon.png', style={"height":"40px", "margin-left":"2rem", "float":"left"}),
            ], className = 'row',  style = {"float":"left", "margin-left": "2rem", "margin-top":"1rem", "margin-bottom":"1rem"}),
            
            html.H2('TimeGAN for EONIA-€STER transition', style = {"float": "center", 'font':'Helvetica', "margin-top":"1rem", "margin-left": "4rem", "margin-right":"4rem", "margin-bottom":"1rem"}), 

            html.Div([
                html.Img(src='assets/github.png', style={"height":"40px", "margin-right":"2rem", "float":"right"}),
                html.Button(html.A('View on Github', href="https://github.com/Larsterbraak/TimeGAN-short-rates"), 
                                    id='github', n_clicks=0, className="button-primary")                
            ], className = 'row',  style = {"margin-right": "2rem", "float": "right", "margin-top":"1rem", "margin-bottom":"1rem"})
            
            ], style = NAVBAR_STYLE)

app.layout = html.Div([dcc.Location(id="url"), navbar, html.Div([sidebar, content], className='row')], style={"background-color":"rgb(66, 75, 107)"})

app.title = 'TimeGAN for short rates | Lars ter Braak'

# this callback uses the current pathname to set the active state of the
# corresponding nav link to true, allowing users to tell see page they are on
@app.callback(
    [Output(f"page-{i}-link", "active") for i in range(1, 4)],
    [Input("url", "pathname")],
)
def toggle_active_links(pathname):
    if pathname == "/":
        # Treat page 1 as the homepage / index
        return True, False, False
    return [pathname == f"/page-{i}" for i in range(1, 4)]

# Index Page callback
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def render_page_content(pathname):
    if pathname == '/':
        return dbc.Jumbotron(
        [
           dcc.Markdown('''
                       ###### Feel free to read up on this app on the left side of the page. \
                       If you are ready to dive into the TimeGAN, select one of the pages above. ''', style = {"padding":"5px"}),
        ], style = {"background-color":"rgb(38,43,61)"}
    )
    elif pathname == '/page-1':
        return page_1.page_1_layout
    elif pathname == '/page-2':
        return page_2.page_2_layout
    elif pathname == '/page-3':
        return page_3.page_3_layout

    # If the user tries to reach a different page, return a 404 message
    else:
        return dbc.Jumbotron([
            html.Hr(),
            dcc.Markdown('''
                        ###### Feel free to read up on this app on the left side of the page. \
                        If you are ready to dive into the TimeGAN, select one of the pages above.''', style = {"padding":"5px"}),
        ], style = {"background-color":"rgb(38,43,61)"})

if __name__=='__main__':
    app.run_server(debug=True)