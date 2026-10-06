"""
charts.py - Plotly figures for the apps. Each function takes a DataFrame and
returns a figure; Streamlit shows it with st.plotly_chart(fig).

Design rules followed:
  * Gains vs losses use BLUE vs RED, not green vs red, because about 1 in 12
    men can't tell green from red apart. Blue/red works for everyone.
  * One measure per chart, one y-axis, no legend on single-series charts
    (the title says what it is).
  * Hover tooltips on everything; thin 2px lines; quiet grid lines.
  * No background colour set, so Streamlit's light/dark theme applies.
"""

import pandas as pd
import plotly.graph_objects as go

GAIN = "#2a78d6"   # blue
LOSS = "#e34948"   # red
LINE = "#2a78d6"   # single-series line colour
NEUTRAL = "#8a8984"  # reference lines (e.g. starting cash)


def _layout(fig: go.Figure, title: str, prefix: str = "₹", y_format: str = ",.0f") -> go.Figure:
    """Shared look. The unit goes in front of each tick ("₹1,000,000") rather
    than in a rotated axis title, which is harder to read."""
    fig.update_layout(
        title=title, height=340, margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False, hovermode="x unified",
        yaxis=dict(title=None, tickprefix=prefix, tickformat=y_format),
        xaxis=dict(title=None),
    )
    return fig


def equity_chart(curve: pd.DataFrame, start_value: float) -> go.Figure:
    """Account value over time, with a dashed line at the starting cash."""
    fig = go.Figure(go.Scatter(
        x=curve["date"], y=curve["equity"], mode="lines", line=dict(color=LINE, width=2),
        hovertemplate="₹%{y:,.2f}<extra></extra>",
    ))
    fig.add_hline(y=start_value, line=dict(color=NEUTRAL, width=1, dash="dash"),
                  annotation_text="starting cash", annotation_position="bottom right")
    return _layout(fig, "Equity (account value)")


def drawdown_chart(curve: pd.DataFrame) -> go.Figure:
    """How far below its best level the account is, as a filled area."""
    fig = go.Figure(go.Scatter(
        x=curve["date"], y=curve["drawdown"], mode="lines", fill="tozeroy",
        line=dict(color=LOSS, width=2), hovertemplate="%{y:.2%}<extra></extra>",
    ))
    return _layout(fig, "Drawdown (distance below the peak)", prefix="", y_format=".1%")


def daily_pnl_chart(curve: pd.DataFrame) -> go.Figure:
    """Profit or loss of each day: blue bars up, red bars down."""
    colours = [GAIN if v >= 0 else LOSS for v in curve["day_pnl"]]
    fig = go.Figure(go.Bar(
        x=curve["date"], y=curve["day_pnl"], marker=dict(color=colours, cornerradius=4),
        hovertemplate="₹%{y:,.2f}<extra></extra>",
    ))
    return _layout(fig, "Daily P&L")


def pnl_by_symbol_chart(by_symbol: pd.DataFrame) -> go.Figure:
    """Total P&L per stock as horizontal bars (long names stay readable)."""
    df = by_symbol.iloc[::-1]  # biggest at the top
    colours = [GAIN if v >= 0 else LOSS for v in df["total_pnl"]]
    fig = go.Figure(go.Bar(
        x=df["total_pnl"], y=df["symbol"], orientation="h",
        marker=dict(color=colours, cornerradius=4),
        customdata=df[["realised_pnl", "unrealised_pnl"]],
        hovertemplate=("<b>%{y}</b><br>Total ₹%{x:,.2f}<br>Realised ₹%{customdata[0]:,.2f}"
                       "<br>Unrealised ₹%{customdata[1]:,.2f}<extra></extra>"),
    ))
    fig.update_layout(title="P&L by stock", height=max(220, 60 + 34 * len(df)),
                      margin=dict(l=10, r=10, t=50, b=10), showlegend=False,
                      xaxis=dict(tickprefix="₹", tickformat=",.0f"), hovermode="closest")
    return fig


def candle_chart(bars: pd.DataFrame, symbol: str) -> go.Figure:
    """Daily candlesticks: blue = closed higher than it opened, red = lower."""
    fig = go.Figure(go.Candlestick(
        x=bars["timestamp"], open=bars["open"], high=bars["high"], low=bars["low"], close=bars["close"],
        increasing=dict(line=dict(color=GAIN, width=1), fillcolor=GAIN),
        decreasing=dict(line=dict(color=LOSS, width=1), fillcolor=LOSS),
    ))
    fig.update_layout(xaxis_rangeslider_visible=False)
    # Markets are shut at weekends: skip Sat-Sun so candles sit side by side
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return _layout(fig, f"{symbol} daily price", y_format=",.2f")
