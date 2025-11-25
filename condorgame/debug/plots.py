import pandas as pd
import numpy as np

from condorgame.debug.densitytosimulations import simulate_paths
from condorgame.prices import PriceStore


def plot_quarantine(asset, quarantine_entry, prices: PriceStore, mode="direct", title=""):
    """
    Plot the predicted price/return distribution (quarantine) for a given asset.

    Parameters
    ----------
    asset : str
        Asset symbol, e.g., "BTC".
    quarantine_entry : tuple
        Tuple of (timestamp, predictions, step) representing a single quarantine entry.
        - timestamp (int): the reference time of the quarantine
        - predictions (list): predicted distributions for each future step
        - step (int): time interval in seconds between prediction steps
    prices : PriceStore
        Object that holds historical price data for the asset.
    mode : str, default "direct"
        - "direct": predictions are returns
        - otherwise: predictions are in price space
    title : str
        Plot title.
    """

    ts, predictions, step = quarantine_entry

    # Skip if the reference timestamp is after the last known price
    if ts > prices.get_last_price(asset)[0]:
        return

    # Determine starting price for simulation
    # If returns ("direct") mode, use 0.0 as the starting point
    start_price = prices.get_closest_price(asset, ts - step * len(predictions))[1]

    # Simulate multiple paths based on the predictions
    # This creates a Monte Carlo distribution of potential price trajectories   
    simulations = simulate_paths(
        predictions,
        start_point=0.0 if mode=="direct" else start_price,
        num_paths=10000,
        step_minutes=None,
        start_time=None,
        mode=mode
    )

    # Create a DataFrame to store simulated mean and confidence intervals
    scales_df = pd.DataFrame({
        "mean": simulations["mean"] if mode=="direct" else simulations["mean"],
        "q_low_paths": simulations["q_low_paths"] if mode=="direct" else simulations["q_low_paths"],
        "q_high_paths": simulations["q_high_paths"] if mode=="direct" else simulations["q_high_paths"],
    })

    # Map timestamps for each prediction step
    scales_df["ts"] = [ts - step * i for i in range(len(scales_df) - 1, -1, -1)]
    scales_df["time"] = pd.to_datetime(scales_df["ts"], unit="s", utc=True)

    # Attach the historical price for each timestamp
    scales_df["price"] = [prices.get_closest_price(asset, ts)[1] for ts in scales_df["ts"]]
    scales_df["return"] = scales_df["price"].diff().fillna(0.0)
    # print(scales_df)

    import plotly.graph_objects as go

    title=f"Predicted {asset} {'return price' if mode=="direct" else "price"} distribution at {scales_df["time"].iloc[0]}"

    # Create a filled band between q05 and q95
    fig = go.Figure()

    # Lower bound (q05)
    fig.add_trace(go.Scatter(
        x=scales_df["time"],
        y=scales_df["q_low_paths"],
        mode='lines',
        line=dict(width=0),
        name='5th percentile',
        showlegend=False
    ))

    # Upper bound (q95) with fill to previous trace
    fig.add_trace(go.Scatter(
        x=scales_df["time"],
        y=scales_df["q_high_paths"],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',  # <-- fills area between q05 and q95
        fillcolor='rgba(255,165,0,0.5)',  # translucent blue band
        name='95% interval'
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
        y=scales_df["return"] if mode=="direct" else scales_df["price"],
        mode='lines',
        line=dict(color='blue', width=2),
        name='Return price' if mode=="direct" else "Price"
    ))

    # Layout
    fig.update_layout(
        title=title,
        hovermode='x unified',
        xaxis_title='Time',
        yaxis_title='Return price' if mode=="direct" else "Price",
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


def plot_return_prices(data, title):
    import plotly.express as px

    df = pd.DataFrame(data, columns=["ts", "price"])

    df["return"] = df["price"].diff().fillna(0.0)

    # print(df.describe())

    df["time"] = pd.to_datetime(df["ts"], unit="s", utc=True)

    # Create a line graph using Plotly
    fig = px.line(
        df,
        x="time",
        y="return",
        title=title,
    )

    # Show the graph
    fig.show()

def plot_scores(data):
    import plotly.express as px

    df = pd.DataFrame([
        {"asset": asset, "ts": ts, "score": score}
        for asset, records in data.items()
        for ts, score in records
    ])

    df["time"] = pd.to_datetime(df["ts"], unit="s", utc=True)

    # average score accross asset
    df_avg = df.groupby("time", as_index=False)["score"].mean()

    start_scores = df_avg['time'].iloc[0]
    end_scores = df_avg['time'].iloc[-1]
    assets = df["asset"].unique().tolist()
    title = f"{assets} CRPS scores from {start_scores} to {end_scores}"
    
    # Create a line graph using Plotly
    fig = px.line(
        df_avg,
        x="time",
        y="score",
        title=title,
    )

    # Show the graph
    fig.show()