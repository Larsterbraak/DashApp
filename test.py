import os
import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash(__name__)
server = app.server

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    #"position": "fixed",
    "top": "10rem",
    "left": "4rem",
    "bottom": 0,
    "width": "30%",
    "margin-left": "4rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "background-color": "rgb(38,43,61)",
    "color": "rgb(226,239,250)",
    'border': '1px solid rgb(38,43,61)',
    'borderRadius': '15px',
    'overflow': 'hidden',
    'font':{"family":'Helvetica', "size":"12"}
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "top": "10rem",
    "left": 0,
    "right": 0,
    "margin-left": "1rem",
    "margin-right": "1rem",
    "padding": "2rem 1rem",
    "width": "60%",
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
    "margin-top": "1rem",
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
}

TEST_STYLE_2 = {
    "background-color": "rgb(38,43,61)",
    "color": "rgb(226,239,250)",
    'borderRight': '0.5px solid white',
    "float":"center",
    "margin-left":"10%",
    "heigth":"5%",
}

TEST_STYLE_3 = {
    "background-color": "rgb(38,43,61)",
    "color": "rgb(226,239,250)",
    'borderRight': '0.5px solid white', 
    "float":"center",
    "heigth":"5%",
}

TEST_STYLE_4 = {
    "background-color": "rgb(38,43,61)",
    "color": "rgb(226,239,250)",
    "float":"center",
    "heigth":"5%",
}

tab_selected_style = {
    "background-color": "rgb(38,43,61)",
    "color": "#7CD3F7",
    "borderTop":"3px solid #7CD3F7",
    'borderBottom':'0px solid #d6d6d6',
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
            
            html.H4("What is TimeGAN for short rates?", style = {"font-size":"24pt", "font-weight":"200", "letter-spacing":"1px"}),

            dcc.Markdown(''' 
                        ###### This app is a demonstration of the MSc Thesis [TimeGAN for short rates](https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks.pdf) \
                        commisioned by the Erasmus University Quantitative Finance department and \
                        performed by [Lars ter Braak](https://www.linkedin.com/in/lars-ter-braak/). Feel free to have a look around and play with \
                        the TimeGAN to simulate short rates and see the implications on fixed-income portfolios. \
                        You can also check out my training scheme on [Tensorboard](https://tensorboard.dev/experiment/qi0Do7FMQpS5vC%20jVkIwBQg/).''', style = {"padding":"5px"}),

            dcc.Markdown('''
                        ###### On the first page you will find the current €STER short rate and the Value-at-Risk for different \
                        time periods according using the TimeGAN for short rates.''', style = {"padding":"3px"}),
            
            dcc.Markdown('''
                        ###### The second page will display the results section for the training of the TimeGAN.''', style = {"padding":"3px"}),

            dcc.Markdown('''
                        ###### The third page show a practical application of the simulation of the €STER short rate. \
                        This page shows you a delta ladder for a fictious Interest Rate Swap portfolio. As a comparison, \
                        the Vasicek and Hull-White short rate models are also displayed.''', style = {"padding":"3px"})
        ])
    ),

    dcc.Tab(
            label = 'Use own data',
            value = 'data',      
            style=TEST_STYLE, selected_style=tab_selected_style,
            
            children = html.Div(className = 'circos-tab', children = [
                
                html.H4("Use your own short rate data?", style = {"font-size":"24pt", "font-weight":"200", "letter-spacing":"1px"}),
                
                dcc.Markdown('''
                            ###### For research purposes it would be helpful if you specify for \
                            which region and for what term you intend to use the TimeGAN for short rates.''', style = {"padding":"3px"}),

                html.Div(className = 'app-controls-block', children= [
                    
                    html.H4("Region", style = {"font-size":"18pt", "font-weight":"200", "letter-spacing":"1px"}),
                    
                    html.Div(dcc.Dropdown(
                        id = "cmap",
                        options=[{'label': i, 'value': i} for i in ['US', 'EU', 'Asia', 'Other']])
                    ),

                    html.H4("Term of rate", style = {"font-size":"18pt", "font-weight":"200", "letter-spacing":"1px"}),
                    
                    html.Div(dcc.Dropdown(
                        id = "background",
                        options=[{'label': i, 'value': i} for i in ['< 1M', '1M < x < 6M', '> 6M']],
                        className = 'select-control'),
                    ),

                    dcc.Markdown('''
                                ###### Your file must be in .xlsx or .csv file format to be working. Please upload the file below.''', style = {"padding":"5px"}),

                    dcc.Upload(id='upload-data',
                                children=html.Div([
                                    'Drag and Drop or ',
                                    html.A('Select Files', style={"borderBottom":"1px solid white"})
                                ]),
                                style={
                                    'width': '100%',
                                    'height': '60px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin': '10px'
                                },
                                # Allow multiple files to be uploaded
                                multiple=True
                            ),
                            html.Div(id='output-data-upload'),
                ])
            ])
        )
]))

options = html.Div(tabs, className = 'item-a', style={"margin-right":"10%", "margin-left":"10%", "margin-top":"3%"})

sidebar = html.Div([
        options,
    ], style=SIDEBAR_STYLE)

content = html.Div([dbc.Nav(children = [

            dbc.NavLink("Interest rate risk in different fixed income portfolios", href="/page-1", id="page-1-link", style=TEST_STYLE_2),
            dbc.NavLink("Interest rate risk", href="/page-2", id="page-2-link", style=TEST_STYLE_3),
            dbc.NavLink("TimeGAN training results", href="/page-3", id="page-3-link", style=TEST_STYLE_4)
    
            ], fill=True, style={"padding":"15px"}),

            html.Div(id="page-content")
        ], style=CONTENT_STYLE)

navbar = html.Nav(className = "navbar navbar-default navbar-static-top", children=[          
            html.Div([
                html.Button(html.A('Check out my LinkedIn', href="https://www.linkedin.com/in/lars-ter-braak/"),
                                    id='linked_in', n_clicks=0, className="button-primary"),
            ], className = 'row',  style = {"float":"left", "margin-left": "2rem", "margin-top":"1rem", "margin-bottom":"1rem"}),
            
            html.H2('TimeGAN for EONIA-€STER transition', style = {"float": "center", "margin-top":"1rem", "margin-left": "4rem", "margin-right":"4rem", "margin-bottom":"1rem"}), 

            html.Div([
                html.Img(src='assets/github.png', style={"height":"40px", "margin-right":"2rem", "float":"right"}),
                html.Button(html.A('View on Github', href="https://github.com/Larsterbraak/TimeGAN"), 
                                    id='github', n_clicks=0, className="button-primary")                
            ], className = 'row',  style = {"margin-right": "2rem", "float": "right", "margin-top":"1rem", "margin-bottom":"1rem"})
            
            ], style = NAVBAR_STYLE)

app.layout = html.Div([dcc.Location(id="url"), navbar, html.Div([sidebar, content], className='row')], style={"background-color":"rgb(66, 75, 107)"})

# app.layout = html.Div([
#     html.H2('Hello World'),
#     dcc.Dropdown(
#         id='dropdown',
#         options=[{'label': i, 'value': i} for i in ['LA', 'NYC', 'MTL']],
#         value='LA'
#     ),
#     html.Div(id='display-value'),
#     sidebar
# ])

app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})


@app.callback(dash.dependencies.Output('display-value', 'children'), [dash.dependencies.Input('dropdown', 'value')])
def display_value(value):
    return 'You have selected "{}"'.format(value)

if __name__ == '__main__':
    app.run_server(debug=True)