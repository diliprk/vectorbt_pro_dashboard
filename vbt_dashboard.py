## Load Libraries
import time
import numpy as np
import pandas as pd
import dash_daq as daq
import vectorbtpro as vbt
import plotly.express as px
import plotly.graph_objects as go
from dateutil import parser
from datetime import datetime, date
from dash_iconify import DashIconify
from dash_extensions import WebSocket
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, dash_table
import dash.dash_table.FormatTemplate as FormatTemplate
from dash.dependencies import Input, Output, ClientsideFunction

# region - LOAD VBT PICKLE FILE OBJECTS
## Load pickle files of saved results from VBT
resample_time_periods = ['15m',  '4h']
# mtf_data = vbt.Config.load('data/mtf_data.pickle') ## Multi Time-Frame data with entries and exits
pf = vbt.Portfolio.load('data/pf_sim.pickle') ## Portfolio Simulation Results
symbols = list(pf.trade_history['Column'].unique())

price_data = vbt.Config.load('data/price_data.pickle')
vbt_indicators_data = vbt.Config.load('data/vbt_indicators_data.pickle')
pandas_indicators_data = vbt.Config.load('data/pandas_indicator_data.pickle')
entries_exits_data = vbt.Config.load('data/entries_exits_data.pickle')
print(type(vbt_indicators_data), vbt_indicators_data["m15_rsi_bbands"]["GBPUSD"].lowerband)


## Load data from pickle files 
## m15_data
m15_data = price_data["m15_data"]
m15_open  = m15_data.get('Open')
m15_close = m15_data.get('Close')
m15_high  = m15_data.get('High')
m15_low   = m15_data.get('Low')

## h4 data
h4_data = price_data["h4_data"]
h4_open  = h4_data.get('Open')
h4_close = h4_data.get('Close')
h4_high  = h4_data.get('High')
h4_low   = h4_data.get('Low')

m15_bbands_price = vbt_indicators_data["m15_price_bbands"]
h4_bbands_price  = vbt_indicators_data["h4_price_bbands"]

m15_rsi = pandas_indicators_data["m15_rsi"]
h4_rsi = pandas_indicators_data["h4_rsi"]
m15_bbands_rsi = vbt_indicators_data["m15_rsi_bbands"]
h4_bbands_rsi  = vbt_indicators_data["h4_rsi_bbands"]

clean_entries = entries_exits_data['clean_entries']
clean_exits = entries_exits_data['clean_exits']
clean_entries_h4 = clean_entries.vbt.resample_apply("4h", "any", wrap_kwargs=dict(dtype=bool))
clean_exits_h4 = clean_exits.vbt.resample_apply("4h", "any", wrap_kwargs=dict(dtype=bool))
# endregion

## GLOBAL VARIABLES
app = Dash(__name__, meta_tags=[{"name": "viewport", "contentZ": "width=device-width, initial-scale=1"}])
app.title = "VectorBT Dashboard"
server = app.server
app.config["suppress_callback_exceptions"] = True

theme = {
    'dark': True,
    'detail': '#2d3038',  # Background-card
    'primary': '#007439',  # Green
    'secondary': '#FFD15F',  # Accent
}

## Global VBT Plot Settings
vbt.settings.set_theme("dark")
vbt.settings['plotting']['layout']['width'] = 1280

def build_banner():
    return html.Div(
        id="banner",
        className="banner",
        children=[
            html.Div(
                id="banner-text",
                children=[
                    html.H5("VectorBT Dashboard"),
                    html.H6("Portfolio Simulation (Backtest) Results and Strategy Visualizer"),
                ],
            ),
            html.Div(
                id="banner-logo",
                children=[html.A(html.Img(id="logo",src=app.get_asset_url("vbt_logo.png")),href="https://vectorbt.pro"),
                ],
            ),
        ],
    )

def build_tabs():
    return html.Div(id="tabs", className="tabs", 
    children=[dcc.Tabs(id="app-tabs",value="tab2",className="custom-tabs",
             children=[dcc.Tab(id="sim-res-tab", label="Portfolio Simulation", value="tab1",className="custom-tab",
                        selected_className="custom-tab--selected" ),
                       dcc.Tab(id="strategy-viz-tab", label="Strategy Visualizer", value="tab2", className="custom-tab",
                        selected_className="custom-tab--selected" )
                ],
            )
        ],
    )

sel_symbol = symbols[0]
sel_period = resample_time_periods[1]

symbols_dropdown = html.Div([
                        html.P('Select Symbol:',style={"font-weight":"bold"}),
                        dcc.Dropdown(id = 'select-symbol-dropdown',
                        options = list({'label': symbol, 'value': symbol} for symbol in symbols),
                        style = {'width':'40%','text-align': 'left'},
                        value = sel_symbol, optionHeight = 25)
                        ])

time_periods = html.Div([
                        html.P('(Resample) Time period:',style={"font-weight":"bold"}),
                        dcc.Dropdown(id = 'select-resample-dropdown',
                        options = list({'label': period, 'value': period} for period in resample_time_periods),
                        style = {'width':'40%','text-align': 'left'},
                        value = sel_period, optionHeight = 25)
                        ])       


def build_tab_1(symbols_dropdown, time_periods, sel_symbol, sel_period):
    return [
        dbc.Row([dbc.Col([symbols_dropdown]), dbc.Col([time_periods])],
                    # style = {'display' : 'inline','align': 'right'} 
                    ),        
        html.Div(children = [
            dcc.Graph(id = 'pf-orders', figure = pf[sel_symbol].resample(sel_period).plot()),
            dcc.Graph(id = 'drawdown-plot', figure =  pf[sel_symbol].drawdowns.plot(**{"title_text" : f"Drawdowns Plot for {sel_symbol}"})),
            dcc.Graph(id = 'underwater-plot', figure =  pf[sel_symbol].plot_underwater(**{"title_text" : f"Underwater Plot for {sel_symbol}"}))        
        ])
        ]

    sel_symbol = symbols[0]
    sel_period = time_periods[3]
    symbols_dropdown = html.Div([
                            html.P('Select Symbol:',style={"font-weight":"bold"}),
                            dcc.Dropdown(id = 'select-symbol-dropdown',
                            options = list({'label': symbol, 'value': symbol} for symbol in symbols),
                            style = {'width':'40%','text-align': 'left'},
                            value = sel_symbol, optionHeight = 25)
                            ])
    resampler_options = html.Div([
                            html.P('(Resample) Time period:',style={"font-weight":"bold"}),
                            dcc.Dropdown(id = 'select-resample-dropdown',
                            options = list({'label': period, 'value': period} for period in time_periods),
                            style = {'width':'40%','text-align': 'left'},
                            value = sel_period, optionHeight = 25)
                            ])
    return [
        # Manually select symbols
        dbc.Row([dbc.Col([symbols_dropdown]), dbc.Col([resampler_options])],
                    # style = {'display' : 'inline','align': 'right'} 
                    ),
        html.Div(children = [dcc.Graph(id = 'pf-orders', figure = pf[sel_symbol].resample(sel_period).plot()),
        dcc.Graph(id = 'drawdown-plot', figure =  pf[sel_symbol].drawdowns.plot(**{"title_text" : f"Drawdowns Plot for {sel_symbol}"}))
        ])
    ]
# callback for Tab 1
# ------------------------------------------------------------
@app.callback(
    [Output('pf-orders', 'figure'),
     Output('drawdown-plot', 'figure'),
     Output('underwater-plot', 'figure')],
    Input('select-symbol-dropdown', 'value'),
    Input('select-resample-dropdown', 'value')
    )
def render_symbol_charts(symbol, period):
    order_plt_kwargs = {"title_text" : f"{symbol} - {period}"}
    drawdown_plt_kwargs = {"title_text" : f"Drawdowns Plot for {symbol}",'title_x': 0.5}
    underwater_plt_kwargs = {"title_text" : f"Underwater Plot for {symbol}",'title_x': 0.5}
    order_plot = pf[symbol].resample(period).plot(**order_plt_kwargs)
    drawdown_plot = pf[symbol].drawdowns.plot(**drawdown_plt_kwargs)
    underwater_plt_kwargs = pf[symbol].plot_underwater(**underwater_plt_kwargs)
    return [order_plot, drawdown_plot, underwater_plt_kwargs]

date_picker_range = html.Div([
    dcc.DatePickerRange(
        id='date-picker',
        clearable=True,
        reopen_calendar_on_clear=True,
        persistence=False,
        min_date_allowed = m15_open.index[0].date(),
        max_date_allowed = m15_close.index[-1].date(),
        initial_visible_month = m15_open.index[0].date(),
        start_date = h4_open.index[0].date(),
        end_date= h4_close.index[200].date())
])


# callback for Tab 2
# ------------------------------------------------------------
@app.callback(
    Output('ohlcv-plot', 'figure'),
    [
     Input(component_id = 'date-picker', component_property = 'start_date'),
     Input(component_id = 'date-picker', component_property = 'end_date'),
     Input('select-symbol-dropdown', 'value'),
     Input('select-resample-dropdown', 'value')]
    )
def main_chart(start_date, end_date, symbol, time_period):
    start_date_txt = datetime.strptime(start_date, '%Y-%m-%d').strftime("%b %d, %Y")
    end_date_txt = datetime.strptime(end_date, '%Y-%m-%d').strftime("%b %d, %Y")
    start_date = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y.%m.%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y.%m.%d')

    kwargs1 = {"title_text" : f"H4 OHLCV with BBands on Price for {symbol} from {start_date_txt} to {end_date_txt}", 
               "title_font_size" : 18,
               "height" : 720,
               "legend" : dict(yanchor="top",y=0.99, xanchor="left",x= 0.1)}
    

    # print("START DATE:", start_date, '||', "END DATE:", end_date)

    df_ohlc = pd.concat([h4_open[symbol], h4_high[symbol], h4_low[symbol], h4_close[symbol] ], 
                axis = 1, keys= ['Open', 'High', 'Low', 'Close'])
    ## Filter Data according to date slice
    df_slice = df_ohlc[["Open", "High", "Low", "Close"]][start_date : end_date]
    ## Retrieve datetime index of rows where price data is NULL
    # retrieve the dates that are in the original datset
    dt_obs = df_slice.index.to_list()
    # Drop rows with missing values
    dt_obs_dropped = df_slice['Close'].dropna().index.to_list()
    # Store dates with missing values
    dt_breaks = [d for d in dt_obs if d not in dt_obs_dropped]

    fig =  df_slice.vbt.ohlcv.plot(**kwargs1) 
     ## Plots Long Entries / Exits and Short Entries / Exits
    pf[symbol][start_date:end_date].plot_trade_signals(fig=fig, plot_close=False, plot_positions="lines")
    bb_line_style = dict(color="white",width=1, dash="dot")
    h4_bbands_price[symbol].plot(fig=fig, **kwargs1 ,
                lowerband_trace_kwargs=dict(fill=None, name = 'BB_Price_Lower', connectgaps=True, line = bb_line_style), 
                upperband_trace_kwargs=dict(fill=None, name = 'BB_Price_Upper', connectgaps=True, line = bb_line_style),
                middleband_trace_kwargs=dict(fill=None, name = 'BB_Price_Middle', connectgaps=True))    


    
    ## Plot Trade Profit or Loss Boxes
    pf[symbol].trades.direction_long[start_date : end_date].plot(fig=fig,plot_close = False, plot_markers = False)
    pf[symbol].trades.direction_short[start_date : end_date].plot(fig=fig,plot_close = False, plot_markers = False)
    fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])
    return fig


def rsi_indicator(start_date, end_date, rsi, bb_rsi, entries, exits):
    start_date = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y.%m.%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y.%m.%d')
    rsi = rsi[start_date : end_date]
    bb_rsi = bb_rsi[start_date : end_date]
    fig = rsi.rename("RSI").vbt.plot(trace_kwargs = dict(connectgaps=True))
    bb_line_style = dict(color="white",width=1, dash="dot")
    bb_rsi.plot(fig=fig,
                lowerband_trace_kwargs=dict(fill=None, name = 'BB_RSI_Lower', connectgaps=True,line = bb_line_style), 
                upperband_trace_kwargs=dict(fill=None, name = 'BB_RSI_Upper', connectgaps=True,line = bb_line_style),
                middleband_trace_kwargs=dict(fill=None, name = 'BB_RSI_Middle', connectgaps=True, visible = False))  
    
    if (entries is not None) & (exits is not None):
        ## Slice Entries and Exits
        entries = entries[start_date : end_date]
        exits = exits[start_date : end_date]
        ## Add Entries and Long Exits on RSI in lower subplot
        entries.vbt.signals.plot_as_entries(rsi, fig = fig,
                                            trace_kwargs=dict(name = "Long Entry", marker=dict(color="limegreen") ))  
        exits.vbt.signals.plot_as_exits(rsi, fig = fig, 
                                        trace_kwargs=dict(name = "Short Entry",  marker=dict(color="red"),
                                                        # showlegend = False ## To hide this from the legend
                                                        ))     
    return fig

@app.callback(
    [Output('indicator1', 'figure'),Output('rsi_label', 'children')],
    [Input(component_id = 'date-picker', component_property = 'start_date'),
     Input(component_id = 'date-picker', component_property = 'end_date'),
     Input('select-symbol-dropdown', 'value'),
     Input('indicator-resampler', 'value')])

def contruct_rsi(start_date, end_date, symbol, time_period):
    if time_period == "15m":
        rsi = m15_rsi[symbol]
        bb_rsi = m15_bbands_rsi[symbol]
        entries = clean_entries[symbol]
        exits = clean_exits[symbol]
    elif time_period == "4h":
        rsi = h4_rsi[symbol]
        bb_rsi = h4_bbands_rsi[symbol]
        entries = clean_entries_h4[symbol]
        exits = clean_exits_h4[symbol]
    fig = rsi_indicator(start_date, end_date, rsi, bb_rsi, entries, exits)
    rsi_title = f"RSI plot for {symbol} on {time_period} time period"
    return fig, rsi_title

def build_tab_2(symbols_dropdown, time_periods):
    return [
        dbc.Row([dbc.Col([symbols_dropdown]), dbc.Col([time_periods])],
            # style = {'display' : 'inline','align': 'right'} 
            ),
        html.Br(),
        date_picker_range,
        html.Br(),
        html.Div(children = [ dcc.Graph(id = 'ohlcv-plot') ] ),
        html.Div([dcc.Dropdown(id = 'indicator-resampler',
                options = list({'label': period, 'value': period} for period in resample_time_periods),
                style = {'width':'40%','text-align': 'left'},
                value = "15m", optionHeight = 25)
                ]),
        html.Div(children = [html.P(id = 'rsi_label',style={"font-weight":"bold"}),
                             dcc.Graph(id = 'indicator1') ])
    ]

@app.callback(
    Output("app-content", "children"),
    [Input("app-tabs", "value")]
)
def render_tab_content(tab):
    if tab == "tab1":
        return build_tab_1(symbols_dropdown, time_periods,
                            sel_symbol, sel_period)
    elif tab == "tab2":
        # return html.Div([html.P("Welcome to Tab2")])
        return build_tab_2(symbols_dropdown, time_periods)


app.layout = html.Div(
    children=[
        build_banner(),
        build_tabs(),
        # Main app
        html.Div(id='app-content', className='container scalable')
    ]
)

# Run the App
if __name__ == "__main__":
    app.run_server(port=8001,debug=True)



