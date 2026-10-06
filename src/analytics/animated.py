"""
animated.py - The "alive" visuals: Plotly figures with animation frames.

How Plotly animation works (all from Python, no JavaScript to write):
  * fig.data          what is drawn when the page loads. We always put the
                      FINAL picture here, so the chart is complete at rest.
  * fig.frames        a list of snapshots; each one replaces some of fig.data.
  * a Play button     (updatemenus) steps through the frames in the browser.
So a chart opens finished, and Play replays how it got there.

Builders here:
  market_heatmap        sector treemap: size = money traded, colour = % change
  equity_replay         the equity curve drawing itself day by day
  leaderboard_race      traders' equity bars re-ranking over the days
  forecast_fan          simulated future price paths + 50% / 90% bands
  signal_waterfall      why the model says UP or DOWN (per-feature contributions)
  correlation_network   stocks pulled together by correlation, settling into clusters
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.analytics.charts import BENCHMARK, GAIN, LINE, LOSS, NEUTRAL, STRATEGY, _layout

SECTOR_COLOURS = {"Banks": "#3987e5", "IT": "#9085e9", "FMCG": "#199e70", "Energy": "#d95926",
                  "Telecom": "#d55181", "Infrastructure": "#c98500", "Other": "#8a8984"}


def _play_controls(frame_ms: int, transition_ms: int) -> dict:
    """A ▶ Play / ⏸ Pause button pair under the chart."""
    return dict(
        type="buttons", direction="left", showactive=False, x=0, y=-0.12, xanchor="left", yanchor="top",
        pad=dict(t=0, r=6),
        buttons=[
            dict(label="▶ Play", method="animate",
                 args=[None, dict(frame=dict(duration=frame_ms, redraw=True), fromcurrent=False,
                                  transition=dict(duration=transition_ms, easing="cubic-in-out"))]),
            dict(label="⏸ Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
        ],
    )


# ---------------------------------------------------------------------------
# Market heatmap
# ---------------------------------------------------------------------------
def market_heatmap(snapshot: pd.DataFrame) -> go.Figure:
    """snapshot: symbol, sector, ltp, change_pct, turnover (₹ crore).
    Tiles grouped by sector, sized by turnover, green/red by today's move."""
    df = snapshot.copy()
    df["label"] = [f"{'▲' if c >= 0 else '▼'} {c:+.2f}%" for c in df["change_pct"]]
    limit = max(1.0, float(df["change_pct"].abs().max()))
    fig = px.treemap(
        df, path=[px.Constant("Market"), "sector", "symbol"], values="turnover", color="change_pct",
        color_continuous_scale=[[0, LOSS], [0.5, "#3a3a38"], [1, GAIN]],
        range_color=[-limit, limit], custom_data=["ltp", "change_pct", "turnover", "label"],
    )
    fig.update_traces(
        texttemplate="<b>%{label}</b><br>%{customdata[3]}", textposition="middle center",
        hovertemplate="<b>%{label}</b><br>₹%{customdata[0]:,.2f} · %{customdata[1]:+.2f}%"
                      "<br>Traded ₹%{customdata[2]:,.0f} cr<extra></extra>",
        marker=dict(line=dict(width=2, color="#0e1117")), root_color="rgba(0,0,0,0)",
    )
    fig.update_layout(height=430, margin=dict(l=4, r=4, t=36, b=4), title="Market heatmap · size = money traded",
                      coloraxis_colorbar=dict(title="%", thickness=10, ticksuffix="%"))
    return fig


# ---------------------------------------------------------------------------
# Equity replay
# ---------------------------------------------------------------------------
def equity_replay(curve: pd.DataFrame, start_value: float, max_frames: int = 60) -> go.Figure:
    """The equity line drawn day by day, with a dot at the newest point."""
    dates, equity = list(curve["date"]), list(curve["equity"])
    n = len(dates)
    stops = sorted(set(np.linspace(1, n, min(n, max_frames)).round().astype(int)))  # at most max_frames frames
    lo, hi = min(min(equity), start_value), max(max(equity), start_value)
    pad = (hi - lo) * 0.08 or start_value * 0.01

    def traces(k: int):
        return [go.Scatter(x=dates[:k], y=equity[:k], mode="lines", line=dict(color=LINE, width=2),
                           fill="tozeroy", fillcolor="rgba(57,135,229,0.10)", hovertemplate="₹%{y:,.2f}<extra></extra>"),
                go.Scatter(x=[dates[k - 1]], y=[equity[k - 1]], mode="markers",
                           marker=dict(size=10, color=GAIN if equity[k - 1] >= start_value else LOSS,
                                       line=dict(width=2, color="#0e1117")), hoverinfo="skip")]

    fig = go.Figure(data=traces(n), frames=[go.Frame(data=traces(k), name=str(k)) for k in stops])
    fig.add_hline(y=start_value, line=dict(color=NEUTRAL, width=1, dash="dash"),
                  annotation_text="starting cash", annotation_position="bottom right")
    _layout(fig, "Equity replay (press Play)")
    fig.update_layout(updatemenus=[_play_controls(110, 80)], margin=dict(l=10, r=10, t=50, b=60),
                      xaxis=dict(range=[dates[0], dates[-1]]), yaxis=dict(range=[lo - pad, hi + pad]))
    return fig


# ---------------------------------------------------------------------------
# Leaderboard race
# ---------------------------------------------------------------------------
def leaderboard_race(daily: pd.DataFrame, top: int = 8) -> go.Figure:
    """daily: username, date, equity (one row per trader per day).
    Horizontal bars per day; the order changes as traders overtake each other."""
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    last = df["date"].max()
    leaders = df[df["date"] == last].nlargest(top, "equity")["username"].tolist()
    df = df[df["username"].isin(leaders)]
    palette = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9", "#008300", "#e66767"]
    colour = {u: palette[i % len(palette)] for i, u in enumerate(sorted(leaders))}  # colour follows the trader
    x_max = df["equity"].max() * 1.02
    x_min = min(df["equity"].min() * 0.995, x_max * 0.97)

    def bars(day) -> go.Bar:
        d = df[df["date"] == day].sort_values("equity")
        return go.Bar(x=d["equity"], y=d["username"], orientation="h",
                      marker=dict(color=[colour[u] for u in d["username"]], cornerradius=4),
                      text=[f"₹{v / 1e5:,.2f} L" for v in d["equity"]], textposition="outside",
                      hovertemplate="<b>%{y}</b><br>₹%{x:,.0f}<extra></extra>")

    days = sorted(df["date"].unique())
    fig = go.Figure(data=[bars(last)], frames=[go.Frame(data=[bars(d)], name=str(pd.Timestamp(d).date()),
                                                        layout=dict(title=f"Leaderboard race · {pd.Timestamp(d):%d %b %Y}"))
                                               for d in days])
    fig.update_layout(title=f"Leaderboard race · {pd.Timestamp(last):%d %b %Y}", height=110 + 46 * len(leaders),
                      margin=dict(l=10, r=10, t=50, b=60), showlegend=False,
                      xaxis=dict(range=[x_min, x_max], tickprefix="₹", tickformat=",.0f"),
                      updatemenus=[_play_controls(350, 300)])
    return fig


# ---------------------------------------------------------------------------
# Forecast fan
# ---------------------------------------------------------------------------
def simulate_paths(last_close: float, sigma: float, days: int, sims: int, seed: int = 42) -> np.ndarray:
    """Monte Carlo: geometric Brownian motion with zero drift.
    price_(t+1) = price_t x exp(sigma x Z - sigma^2 / 2),  Z ~ N(0, 1)
    (the -sigma^2/2 keeps the AVERAGE future price equal to today's).
    Returns an array of shape (sims, days + 1); column 0 is today."""
    rng = np.random.default_rng(seed)
    steps = np.exp(rng.standard_normal((sims, days)) * sigma - sigma ** 2 / 2)
    return last_close * np.concatenate([np.ones((sims, 1)), np.cumprod(steps, axis=1)], axis=1)


def forecast_fan(history: pd.DataFrame, symbol: str, days: int = 20, sims: int = 400,
                 shown_paths: int = 12) -> tuple[go.Figure, dict]:
    """history: timestamp, close (up to now). Returns the figure and a summary
    {sigma, p5, p50, p95} for the last simulated day."""
    closes = history["close"].astype(float).to_numpy()
    sigma = float(np.diff(np.log(closes[-61:])).std())  # volatility of the last 60 daily returns
    paths = simulate_paths(closes[-1], sigma, days, sims)
    q = {p: np.percentile(paths, p, axis=0) for p in (5, 25, 50, 75, 95)}

    past_x = list(pd.to_datetime(history["timestamp"]).iloc[-60:])
    future_x = list(pd.bdate_range(past_x[-1], periods=days + 1))  # business days ahead
    band = lambda lo, hi, op, name: go.Scatter(  # noqa: E731
        x=future_x + future_x[::-1], y=list(q[hi]) + list(q[lo][::-1]), fill="toself", mode="lines",
        line=dict(width=0), fillcolor=f"rgba(57,135,229,{op})", name=name, hoverinfo="skip")

    def path_traces(k: int):  # sample paths revealed up to day k
        return [go.Scatter(x=future_x[: k + 1], y=paths[i, : k + 1], mode="lines", showlegend=False,
                           line=dict(color=BENCHMARK, width=1), opacity=0.55, hoverinfo="skip")
                for i in range(shown_paths)]

    static = [
        band(5, 95, 0.14, "90% of paths"), band(25, 75, 0.28, "50% of paths"),
        go.Scatter(x=past_x, y=closes[-60:], mode="lines", line=dict(color="#e7e9ef", width=2), name="Close",
                   hovertemplate="₹%{y:,.2f}<extra></extra>"),
        go.Scatter(x=future_x, y=q[50], mode="lines", line=dict(color="#e7e9ef", width=1.5, dash="dash"),
                   name="Middle path", hovertemplate="₹%{y:,.2f}<extra></extra>"),
    ]
    fig = go.Figure(data=static + path_traces(days),
                    frames=[go.Frame(data=path_traces(k), traces=list(range(len(static), len(static) + shown_paths)),
                                     name=str(k)) for k in range(1, days + 1)])
    _layout(fig, f"{symbol}: the next {days} trading days", y_format=",.0f")
    # legend inside the plot's top-left corner, so it never collides with the title
    fig.update_layout(showlegend=True, legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top",
                                                   bgcolor="rgba(14,17,23,0.6)", font=dict(size=11)),
                      updatemenus=[_play_controls(90, 60)], margin=dict(l=10, r=10, t=60, b=60))
    return fig, {"sigma": sigma, "p5": float(q[5][-1]), "p50": float(q[50][-1]), "p95": float(q[95][-1]),
                 "last": float(closes[-1])}


# ---------------------------------------------------------------------------
# Signal waterfall
# ---------------------------------------------------------------------------
def signal_waterfall(contrib: pd.DataFrame, base: float, symbol: str, top: int = 8) -> go.Figure:
    """contrib: feature, label, value_text, contribution (log-odds), biggest first.
    Starts at the model's base rate and adds each feature's push, ending at P(up).
    The axis is in log-odds (that is how the contributions add up); the hover
    shows probabilities."""
    shown = contrib.head(top)
    rest = contrib["contribution"].iloc[top:].sum()
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    names = ["Base rate"] + [f"{r.label} ({r.value_text})" for r in shown.itertuples()] + \
            ([f"{len(contrib) - top} other features"] if len(contrib) > top else []) + [f"P(up) for {symbol}"]
    values = [base] + list(shown["contribution"]) + ([rest] if len(contrib) > top else []) + [0]
    measures = ["absolute"] + ["relative"] * (len(values) - 2) + ["total"]
    final = base + contrib["contribution"].sum()
    fig = go.Figure(go.Waterfall(
        orientation="h", y=names, x=values, measure=measures,
        increasing=dict(marker=dict(color=GAIN)), decreasing=dict(marker=dict(color=LOSS)),
        totals=dict(marker=dict(color=STRATEGY)), connector=dict(line=dict(color=NEUTRAL, dash="dot", width=1)),
        text=[f"{sig(base):.1%}"] + [f"{v:+.3f}" for v in values[1:-1]] + [f"{sig(final):.1%}"],
        textposition="outside", hovertemplate="%{y}<br>%{x:+.3f} log-odds<extra></extra>",
    ))
    # x range from the running totals, with room on the right for the value labels
    running = np.cumsum([base] + values[1:-1])
    lo, hi = min(0.0, running.min()), max(0.0, running.max(), final)
    pad = (hi - lo) * 0.28 or 0.1
    fig.update_layout(title=f"Why the model says {'UP' if final > 0 else 'DOWN'} for {symbol}",
                      height=120 + 34 * len(names), margin=dict(l=10, r=20, t=50, b=10), showlegend=False,
                      yaxis=dict(autorange="reversed"),
                      xaxis=dict(title="log-odds (0 = 50%)", zeroline=True, range=[lo - pad * 0.4, hi + pad]))
    return fig


# ---------------------------------------------------------------------------
# Correlation network
# ---------------------------------------------------------------------------
def force_layout(corr: pd.DataFrame, threshold: float, iterations: int = 90, seed: int = 7) -> list[np.ndarray]:
    """A small spring simulation (no extra library):
      * every pair of stocks pushes apart (like charges)
      * linked pairs (correlation >= threshold) pull together, harder when more correlated
      * a weak pull to the centre keeps everything on screen
    Returns the positions after each iteration (for the animation)."""
    n = len(corr)
    rng = np.random.default_rng(seed)
    pos = rng.normal(0, 1, (n, 2))
    vel = np.zeros_like(pos)
    c = corr.to_numpy()
    link = (c >= threshold) & ~np.eye(n, dtype=bool)
    history = [pos.copy()]
    for _ in range(iterations):
        diff = pos[:, None, :] - pos[None, :, :]                  # vector from j to i
        dist = np.linalg.norm(diff, axis=2) + np.eye(n)           # avoid dividing by 0 on the diagonal
        repel = (diff / dist[..., None] ** 3).sum(axis=1) * 0.15
        rest = 1.6 - 1.4 * c                                      # more correlated -> shorter spring
        spring = -(diff / dist[..., None] * ((dist - rest) * link * c)[..., None]).sum(axis=1) * 0.12
        vel = (vel + repel + spring - pos * 0.02) * 0.82
        pos = pos + vel
        history.append(pos.copy())
    return history


def correlation_network(returns: pd.DataFrame, sectors: dict, threshold: float = 0.34,
                        frames_every: int = 3) -> go.Figure:
    """returns: daily returns, one column per stock. Opens settled; Play shows
    the stocks starting scattered and pulling into their clusters."""
    corr = returns.corr()
    names = list(corr.columns)
    steps = force_layout(corr, threshold)
    pairs = [(i, j, corr.iloc[i, j]) for i in range(len(names)) for j in range(i + 1, len(names))
             if corr.iloc[i, j] >= threshold]

    def traces(pos: np.ndarray):
        out = []
        for i, j, c in pairs:  # one line per link so each can have its own thickness
            out.append(go.Scatter(x=[pos[i, 0], pos[j, 0]], y=[pos[i, 1], pos[j, 1]], mode="lines",
                                  # thicker and brighter for stronger correlation (clamped: never negative)
                                  line=dict(width=1 + max(c - 0.25, 0) * 14,
                                            color=f"rgba(231,233,239,{min(max(0.15 + c * 0.6, 0.1), 0.9):.2f})"),
                                  hoverinfo="text", text=f"{names[i]} – {names[j]}: {c:.2f}", showlegend=False))
        out.append(go.Scatter(
            x=pos[:, 0], y=pos[:, 1], mode="markers+text", text=names, textposition="top center",
            textfont=dict(color="#e7e9ef", size=12),
            marker=dict(size=18, color=[SECTOR_COLOURS.get(sectors.get(s, "Other"), "#8a8984") for s in names],
                        line=dict(width=2, color="#0e1117")),
            customdata=[sectors.get(s, "Other") for s in names],
            hovertemplate="<b>%{text}</b><br>%{customdata}<extra></extra>", showlegend=False))
        return out

    final = steps[-1]
    span = np.abs(np.concatenate(steps)).max() * 1.15
    fig = go.Figure(data=traces(final),
                    frames=[go.Frame(data=traces(p), name=str(k)) for k, p in enumerate(steps[::frames_every])])
    for sector, colour in SECTOR_COLOURS.items():  # legend entries for the sectors present
        if sector in {sectors.get(s, "Other") for s in names}:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=10, color=colour),
                                     name=sector, showlegend=True))
    fig.update_layout(title=f"Who moves together (links: correlation ≥ {threshold:.2f})", height=470,
                      margin=dict(l=10, r=10, t=50, b=60),
                      xaxis=dict(visible=False, range=[-span, span]), yaxis=dict(visible=False, range=[-span, span]),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
                      updatemenus=[_play_controls(70, 50)])
    return fig
