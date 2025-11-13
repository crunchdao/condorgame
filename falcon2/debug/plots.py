import pandas as pd

from falcon2.debug.densitytoscale import densities_to_scales
from falcon2.prices import PriceStore


def plot_quarantine(asset, quarantine_entry, prices: PriceStore, title):

    ts, predictions, step = quarantine_entry

    if ts > prices.get_last_price(asset)[0]:
        return

    scales_df = densities_to_scales(predictions)
    scales_df["ts"] = [ts - step * i for i in range(len(scales_df) - 1, -1, -1)]
    scales_df["time"] = pd.to_datetime(scales_df["ts"], unit="s", utc=True)
    scales_df["price"] = [prices.get_closest_price(asset, ts)[1] for ts in scales_df["ts"]]


    import plotly.graph_objects as go

    # Create a filled band between q05 and q95
    fig = go.Figure()

    # Lower bound (q05)
    fig.add_trace(go.Scatter(
        x=scales_df["time"],
        y=scales_df["q05"],
        mode='lines',
        line=dict(width=0),
        name='5th percentile',
        showlegend=False
    ))

    # Upper bound (q95) with fill to previous trace
    fig.add_trace(go.Scatter(
        x=scales_df["time"],
        y=scales_df["q95"],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',  # <-- fills area between q05 and q95
        fillcolor='rgba(255,165,0,0.5)',  # translucent blue band
        name='90% interval'
    ))

    fig.add_trace(go.Scatter(
        x=scales_df["time"],
        y=scales_df["mean"],
        mode='lines',
        line=dict(color='red', width=2),
        name='Price mean predicted'
    ))



    # Main line for price
    fig.add_trace(go.Scatter(
        x=scales_df["time"],
        y=scales_df["price"],
        mode='lines',
        line=dict(color='blue', width=2),
        name='Price'
    ))

    # Layout
    fig.update_layout(
        title=title,
        hovermode='x unified',
        xaxis_title='Time',
        yaxis_title='Price',
    )

    fig.show()


def plot_prices(data, title):
    import plotly.express as px

    df = pd.DataFrame(data, columns=["ts", "price"])

    df["time"] = pd.to_datetime(df["ts"], unit="s", utc=True)

    # Create a line graph using Plotly
    fig = px.line(
        df,
        x="time",
        y="price",
        title=title,
    )

    # Show the graph
    fig.show()
