import streamlit as st
import pandas as pd
import numpy as np
from xata.client import XataClient
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

is_local = os.getenv("IS_LOCAL")

st.set_page_config(
    layout="wide", page_title="Automated Trading System Dashboard", page_icon="./favicon.png")

xata = XataClient(
    db_url=os.getenv("XATA_DB_URL"),
    api_key=os.getenv("XATA_API_KEY")
)


@st.cache_data
def load_data(date):
    trades = []
    df = None

    try:
        records = xata.data().query("trades_v1", {
            "page": {
                "size": 1000
            }
        })
        trades.extend(records["records"])
        while records.has_more_results():
            records = xata.data().query("trades_v1", {
                "page": {
                    "after": records.get_cursor(),
                    "size": 1000
                }
            })
            trades.extend(records["records"])

        df = pd.DataFrame(trades)
        df = df.drop(["xata", "id"], axis=1)
        df.to_csv('./data/trades_v1.csv', index=False)
    except Exception as e:
        df = pd.read_csv('./data/trades_v1.csv')
        print(e)
    return df


# trades = load_data()

# df = pd.DataFrame(trades)
# df = df.drop(["xata", "id"], axis=1)
# df

strategies = ['MEAN_REVERSION',
              'INTRADAY_GAP_UP', 'SWING_STOCHASTIC_RSI']

if 'strategies' not in st.session_state:
    st.session_state['strategies'] = strategies

if is_local == "1":
    original_df = pd.read_csv('./data/trades_v1.csv')
else:
    original_df = load_data(datetime.now().strftime("%Y-%m-%d"))


df = original_df.copy()
df["date"] = pd.to_datetime(df["date"]).dt.date

df = df[df["strategy"].isin(st.session_state["strategies"] or strategies)]

daily_df = df.groupby("date")["net_pnl"].sum().reset_index().set_index("date")
daily_df.index = pd.to_datetime(daily_df.index)
daily_df["cum_pnl"] = daily_df["net_pnl"].cumsum()

monthly_df = daily_df.resample("M").sum()
# monthly_df["cum_pnl"] = monthly_df["net_pnl"].cumsum()

# monthly_df = df.groupby(["strategy", "date"])[
#     "net_pnl"].sum().reset_index().set_index(["date"])
# monthly_df.index = pd.to_datetime(monthly_df.index)
# monthly_df = monthly_df.groupby("strategy").resample("M").sum().drop(
#     "strategy", axis=1).reset_index().set_index("date")

st.title("Automated Trading System Dashboard")
st.markdown('#')


st.header("Overivew", divider='rainbow')
st.markdown('#')

st.multiselect(
    'Select Strategy',
    strategies,
    key="strategies",
)
st.markdown('##')

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total PNL", f"₹{round(daily_df['net_pnl'].sum(), 2)}")

with col2:
    st.metric("Last 1 day PNL",
              f"₹{ round(daily_df['net_pnl'].tail(1).sum(), 2) }",
              delta=round(daily_df['net_pnl'].tail(1).sum() -
                          daily_df['net_pnl'].tail(2).sum(), 2))

with col3:
    st.metric("Last 7 days PNL",
              f"₹{ round(daily_df['net_pnl'].tail(7).sum(), 2) }", delta=round(daily_df['net_pnl'].tail(7).sum() - daily_df['net_pnl'].tail(14).sum(), 2))

with col4:
    st.metric("Last 30 days PNL",
              f"₹{round(daily_df['net_pnl'].tail(30).sum(), 2)}", delta=round(daily_df['net_pnl'].tail(30).sum() - daily_df['net_pnl'].tail(60).sum(), 2))

with col5:
    st.metric("Total trades", len(df))

st.markdown("### Cumulative PNL")
st.plotly_chart(px.line(daily_df, x=daily_df.index,
                y="cum_pnl"), use_container_width=True)

st.markdown("### Monthly PNL")
st.plotly_chart(px.bar(monthly_df, x=monthly_df.index,
                y="net_pnl"), use_container_width=True)

st.markdown("#")
st.header("Individual Strategies", divider='rainbow')
st.markdown('#')

analysis_df = original_df.copy()


monthly_df = analysis_df.groupby(["strategy", "date"])[
    "net_pnl"].sum().reset_index().set_index(["date"])
monthly_df.index = pd.to_datetime(monthly_df.index)
monthly_df = monthly_df.groupby("strategy").resample("M").sum().drop(
    "strategy", axis=1).reset_index().set_index("date")

monthly_pnl_fig = px.bar(monthly_df, x=monthly_df.index, y="net_pnl",
                         color="strategy", barmode='group')

monthly_pnl_fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Monthly PnL",
    legend_orientation="h",
    legend_yanchor="top",
    legend_xanchor="auto",
    legend_y=1.1,
    legend_title_text=' ',
)


analysis_df["result"] = np.where(analysis_df["net_pnl"] > 0, 1, 0)
analysis_df["winning_amount_pct"] = (
    analysis_df["net_pnl"]*100)/analysis_df["buy_value"]

strategy_df = analysis_df.groupby(["strategy"]).agg(
    {"result": ["sum", "count"], "winning_amount_pct": ["mean", "max", "min"]})
strategy_df.columns = ["Wins", "Total_Trades", "Mean_Return_Pct",
                       "Max_Return_Pct", "Min_Return_Pct"]
strategy_df["Win_Rate"] = (strategy_df["Wins"]*100)/strategy_df["Total_Trades"]


for strategy in strategies:
    st.subheader(strategy.replace("_", " "))

    indi_df = strategy_df.loc[strategy]

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("Total Trades", int(indi_df["Total_Trades"]))
    with col2:
        st.metric("Wins", int(indi_df["Wins"]))
    with col3:
        st.metric("Win Rate", round(indi_df["Win_Rate"], 2))
    with col4:
        st.metric("Mean Return Pct", round(indi_df["Mean_Return_Pct"], 3))
    with col5:
        st.metric("Max Return Pct", round(indi_df["Max_Return_Pct"], 2))
    with col6:
        st.metric("Min Return Pct", round(indi_df["Min_Return_Pct"], 2))

    st.divider()


st.markdown("#")
st.markdown("### Monthly PNL")
st.plotly_chart(monthly_pnl_fig, use_container_width=True)
