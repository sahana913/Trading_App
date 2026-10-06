"""
charts.py - Plotly figures for the apps. Each function takes a DataFrame and
returns a figure; Streamlit shows it with st.plotly_chart(fig).

Design rules followed:
  * Gains are green and losses red (the trading convention). Because some
    people can't tell green from red, colour is never the only signal:
    tables add ▲/▼ and +/- signs, and bar direction (up/down) shows the sign.
  * One measure per y-axis (price and volume get separate stacked panels,
    never two scales on one axis); legends only where there are 2+ lines.
  * Hover tooltips on everything; thin 2px lines; quiet grid lines.
  * No background colour set, so Streamlit's light/dark theme applies.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

GAIN = "#22a55a"     # green
LOSS = "#e34948"     # red
LINE = "#3987e5"     # single-series line colour (blue, readable on dark)
NEUTRAL = "#8a8984"  # reference lines (e.g. starting cash)
MA_COLOURS = ["#f0a020", "#b48cf2", "#3fc1c9"]  # moving-average lines, in order


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


def add_moving_averages(bars: pd.DataFrame, windows: tuple[int, ...]) -> pd.DataFrame:
    """Add one column per window, e.g. "MA20" = average close of the last 20 bars.
    Compute this on the FULL history before cutting to the visible bars, so
    the first visible bars already have a value."""
    bars = bars.copy()
    for w in windows:
        bars[f"MA{w}"] = bars["close"].rolling(w).mean()
    return bars


def price_volume_chart(bars: pd.DataFrame, symbol: str, ma_windows: tuple[int, ...] = (20, 50),
                       last: int = 120) -> go.Figure:
    """Candlesticks + moving averages on top, volume bars underneath.

    bars: the whole price history up to "now" (timestamp, open, high, low,
    close, volume). Only the `last` bars are drawn.
    """
    bars = add_moving_averages(bars, ma_windows).tail(last)
    up = bars["close"] >= bars["open"]

    # Two stacked panels sharing the date axis: price (75%) and volume (25%)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25],
                        vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=bars["timestamp"], open=bars["open"], high=bars["high"], low=bars["low"],
        close=bars["close"], name=symbol, showlegend=False,
        increasing=dict(line=dict(color=GAIN, width=1), fillcolor=GAIN),
        decreasing=dict(line=dict(color=LOSS, width=1), fillcolor=LOSS),
    ), row=1, col=1)
    for w, colour in zip(ma_windows, MA_COLOURS):
        fig.add_trace(go.Scatter(
            x=bars["timestamp"], y=bars[f"MA{w}"], mode="lines", name=f"{w}-day average",
            line=dict(color=colour, width=1.5), hovertemplate="₹%{y:,.2f}",
        ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=bars["timestamp"], y=bars["volume"], name="Volume", showlegend=False,
        marker=dict(color=[GAIN if u else LOSS for u in up], opacity=0.6),
        hovertemplate="%{y:,.0f}",
    ), row=2, col=1)

    fig.update_layout(
        title=f"{symbol} · daily", height=520, margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified", xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
    )
    fig.update_yaxes(tickprefix="₹", tickformat=",.0f", row=1, col=1)
    fig.update_yaxes(tickformat="~s", row=2, col=1)  # 12M instead of 12,000,000
    # Markets are shut at weekends: skip Sat-Sun so candles sit side by side
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig


def pnl_calendar_chart(grid: pd.DataFrame) -> go.Figure:
    """Daily P&L calendar: one square per trading day, weeks left to right.

    grid comes from metrics.calendar_frame (rows Mon-Fri, columns = week's Monday).
    Colour scale is diverging: red = loss, grey = about zero, green = gain,
    centred on 0 so a small gain never looks like a loss.
    """
    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    # The real date of every cell, for the hover text
    dates = [[(week + pd.Timedelta(days=d)).strftime("%a %d %b %Y") for week in grid.columns]
             for d in grid.index]
    limit = float(abs(grid).max().max()) if grid.notna().any().any() else 1.0
    fig = go.Figure(go.Heatmap(
        z=grid.to_numpy(), x=list(grid.columns), y=weekdays, customdata=dates,
        colorscale=[[0.0, LOSS], [0.5, "#3a3a38"], [1.0, GAIN]],
        zmid=0, zmin=-limit, zmax=limit,  # symmetric, so equal gains/losses look equally strong
        xgap=3, ygap=3,                   # thin gaps so each day is its own square
        hovertemplate="%{customdata}<br>₹%{z:,.2f}<extra></extra>",
        colorbar=dict(title="₹", tickformat=",.0f", thickness=12),
    ))
    fig.update_layout(title="Daily P&L calendar", height=260, margin=dict(l=10, r=10, t=50, b=10))
    fig.update_yaxes(autorange="reversed")  # Monday on top, like a calendar
    fig.update_xaxes(tickformat="%d %b", title=None)
    return fig


# ---------------------------------------------------------------------------
# ML charts (AI Insights page)
# ---------------------------------------------------------------------------
STRATEGY = "#3987e5"   # the model's strategy (blue)
BENCHMARK = "#f0a020"  # buy-and-hold (orange): two clearly different hues


def feature_importance_chart(importance: pd.Series, top: int = 12) -> go.Figure:
    """Horizontal bars, most important feature at the top."""
    s = importance.head(top).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=s.to_numpy(), y=s.index, orientation="h", marker=dict(color=STRATEGY, cornerradius=4),
        hovertemplate="<b>%{y}</b><br>%{x:.1%} of total importance<extra></extra>",
    ))
    fig.update_layout(title="What the model relies on (feature importance)",
                      height=max(260, 60 + 28 * len(s)), margin=dict(l=10, r=10, t=50, b=10),
                      showlegend=False, xaxis=dict(tickformat=".0%"), hovermode="closest")
    return fig


def model_auc_chart(test_scores: dict, chosen: str) -> go.Figure:
    """Test ROC-AUC of every model vs the 0.5 'coin flip' line. The chosen
    model is drawn solid, the others faded."""
    names = list(test_scores)
    aucs = [test_scores[n]["roc_auc"] for n in names]
    fig = go.Figure(go.Bar(
        x=names, y=aucs, marker=dict(color=[STRATEGY if n == chosen else NEUTRAL for n in names],
                                     cornerradius=4),
        text=[f"{a:.3f}" for a in aucs], textposition="outside",
        hovertemplate="<b>%{x}</b><br>ROC-AUC %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0.5, line=dict(color=LOSS, width=1, dash="dash"),
                  annotation_text="coin flip", annotation_position="right")  # in the margin, clear of bar labels
    lo = min(0.4, min(aucs) - 0.02)
    hi = max(0.6, max(aucs) + 0.03)
    fig.update_layout(title="Test ROC-AUC by model (higher is better)", height=320,
                      margin=dict(l=10, r=70, t=50, b=10), showlegend=False,
                      yaxis=dict(range=[lo, hi], tickformat=".2f"), hovermode="closest")
    return fig


def backtest_chart(curves: pd.DataFrame, label: str) -> go.Figure:
    """Growth of ₹1 over the test period: model strategy vs buy-and-hold."""
    fig = go.Figure()
    for col, name, colour in (("strategy", f"Model signal ({label})", STRATEGY),
                              ("buy_and_hold", "Buy and hold", BENCHMARK)):
        fig.add_trace(go.Scatter(
            x=curves["timestamp"], y=curves[col], mode="lines", name=name,
            line=dict(color=colour, width=2), hovertemplate="%{y:.3f}",
        ))
        # Direct label at the end of each line, so colour isn't the only key
        fig.add_annotation(x=curves["timestamp"].iloc[-1], y=curves[col].iloc[-1],
                           text=f"{curves[col].iloc[-1] - 1:+.1%}", showarrow=False,
                           xanchor="left", xshift=6, font=dict(color=colour))
    fig.add_hline(y=1.0, line=dict(color=NEUTRAL, width=1, dash="dash"))
    fig.update_layout(title="Backtest on the test period: value of ₹1, after costs", height=380,
                      margin=dict(l=10, r=50, t=50, b=10), hovermode="x unified",
                      yaxis=dict(tickformat=".2f"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1))
    return fig
