import ta
import pandas as pd
import numpy as np
import plotly.graph_objs as go

from plotly.offline import iplot


def timeSeriesPlot(df, title=None, mode='lines'):
    data = []
    for col in df.columns:
        data.append(
            go.Scatter(x=df.index,  y=df[col], name=col, mode=mode),
        )
        
    lay = go.Layout(title=title)

    fig = go.Figure(data=data, layout=lay)
    return iplot(fig)


def daily_feature(df):
    df = df.rolling(2).apply(lambda x: x.iloc[-1] - x.iloc[0]) \
                      .add_suffix(f'_daily_change') \
                      .fillna(0)
    return df


def change_feature(df, window_size = 3):
    df = df.rolling(window_size).apply(lambda x: (x.iloc[-1] - x.iloc[0])*100/x.iloc[0]) \
                                .add_suffix(f'_change_{window_size}(%)') \
                                .fillna(0).replace(np.inf, 0)
    return df


def ema_feature(df, window_size = 10):
    df = ta.trend.ema(df, window_size).fillna(0).add_suffix(f'_EMA_{window_size}')
    return df


def lags_feature(df, steps=5):
    res_df, columns = [], []
    
    for step in range(1, steps + 1):
        res_df.append(df.shift(step))
        for col in df.columns:
            columns.append(f"{col}(t-{step})")

    res_df = pd.concat(res_df, axis=1)
    res_df.columns = columns

    res_df.fillna(method='bfill', inplace=True)
    res_df.fillna(method='ffill', inplace=True)
    
    return res_df
