from typing import List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


def fetch_quotes(tickers: List[str]) -> pd.DataFrame:
    """
    Fetch latest quote data (price, % change, volume) for a list of tickers.

    Uses yfinance (Yahoo! Finance). Data is typically delayed (up to ~15 minutes)
    but good enough to prove out the dashboard plumbing. You can later swap this
    out for a Webull-based data source.
    """
    data = []
    for t in tickers:
        ticker = yf.Ticker(t)
        info = ticker.fast_info
        last_price = info.last_price
        prev_close = info.previous_close
        volume = info.last_volume
        pct_change = (
            (last_price - prev_close) / prev_close * 100 if prev_close else 0.0
        )
        data.append(
            {
                "ticker": t.upper(),
                "price": last_price,
                "prev_close": prev_close,
                "pct_change": pct_change,
                "volume": volume,
                "currency": info.currency,
            }
        )
    return pd.DataFrame(data)


def fetch_intraday_series(ticker: str) -> pd.DataFrame:
    """
    Fetch intraday price series for plotting.
    """
    df = yf.download(
        tickers=ticker,
        period="1d",
        interval="1m",
        progress=False,
        auto_adjust=False,
    )
    if not df.empty:
        df = df.reset_index()

        # yfinance sometimes returns MultiIndex columns (field, ticker).
        # Flatten that so we can work with simple column names.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        # Standardize the time column name.
        if "Datetime" in df.columns:
            time_col = "Datetime"
        elif "Date" in df.columns:
            time_col = "Date"
        else:
            time_col = df.columns[0]

        df = df.rename(columns={time_col: "time"})
    return df


def main() -> None:
    st.set_page_config(page_title="Stock Dashboard", layout="wide")

    st.title("Stock Price / Volume Dashboard")

    st.write(
        "Backend is using free Yahoo! Finance data via `yfinance` "
        "(typically delayed). This is just to prove the dashboard. "
        "Later you can plug in Webull or any other real-time feed."
    )

    default_tickers = "AAPL, MSFT, TSLA"
    tickers_input = st.text_input("Tickers (comma separated)", value=default_tickers)
    if st.button("Refresh quotes and chart"):
        # This just causes the script to re-run with the same state.
        st.experimental_rerun()

    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    if not tickers:
        st.warning("Enter at least one ticker symbol.")
        return

    # Main layout: left = table, right = chart
    col_table, col_chart = st.columns([1, 2], gap="large")

    with col_table:
        with st.spinner("Fetching quotes..."):
            quotes = fetch_quotes(tickers)
        if quotes.empty:
            st.error("No quote data returned.")
        else:
            # Color for % change
            def format_pct(val: float) -> str:
                return f"{val:+.2f}%"

            quotes_display = quotes.copy()
            quotes_display["pct_change"] = quotes_display["pct_change"].map(
                format_pct
            )
            st.subheader("Latest Quotes")
            st.dataframe(
                quotes_display.set_index("ticker"),
                use_container_width=True,
            )

    with col_chart:
        selected = st.selectbox("Chart ticker", options=tickers)
        with st.spinner(f"Fetching intraday series for {selected}..."):
            series = fetch_intraday_series(selected)

        if series.empty:
            st.error("No intraday data available.")
        else:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=series["time"],
                    y=series["Close"],
                    mode="lines",
                    name="Price",
                )
            )
            fig.update_layout(
                title=f"{selected} intraday price",
                xaxis_title="Time",
                yaxis_title="Price",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Note about Webull latency
    st.markdown(
        """
**About Webull latency**  
- Webull gives real-time quotes for U.S. stocks and ETFs to app users; in practice you're usually looking at a 1–2 second network/processing delay vs the exchange.  
- Options are typically **15 minutes delayed** unless you subscribe to OPRA real-time data.  
- This dashboard is wired so you can swap `yfinance` out for a Webull client later (e.g., an unofficial API wrapper) and keep the UI the same.
"""
    )


if __name__ == "__main__":
    main()

