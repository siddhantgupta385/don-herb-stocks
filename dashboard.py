import time
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from yfinance.exceptions import YFRateLimitError


@st.cache_data(ttl=60)  # Cache for 60 seconds to reduce API calls
def fetch_quotes(tickers: List[str]) -> pd.DataFrame:
    """
    Fetch latest quote data (price, % change, volume) for a list of tickers.

    Uses yfinance (Yahoo! Finance). Data is typically delayed (up to ~15 minutes)
    but good enough to prove out the dashboard plumbing. You can later swap this
    out for a Webull-based data source.
    
    Includes retry logic with exponential backoff to handle rate limiting.
    """
    data = []
    max_retries = 3
    base_delay = 2  # seconds
    
    for t in tickers:
        ticker = yf.Ticker(t)
        
        # Retry logic for rate limiting
        for attempt in range(max_retries):
            try:
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
                break  # Success, exit retry loop
            except YFRateLimitError:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    time.sleep(delay)
                else:
                    # Last attempt failed, add placeholder
                    data.append(
                        {
                            "ticker": t.upper(),
                            "price": None,
                            "prev_close": None,
                            "pct_change": None,
                            "volume": None,
                            "currency": "N/A",
                        }
                    )
            except Exception as e:
                # Other errors - add placeholder
                data.append(
                    {
                        "ticker": t.upper(),
                        "price": None,
                        "prev_close": None,
                        "pct_change": None,
                        "volume": None,
                        "currency": "N/A",
                    }
                )
                break
        
        # Add delay between tickers to avoid rate limiting
        if t != tickers[-1]:  # Don't delay after last ticker
            time.sleep(0.5)
    
    return pd.DataFrame(data)


@st.cache_data(ttl=300)  # Cache for 5 minutes (intraday data changes less frequently)
def fetch_intraday_series(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Fetch intraday price series for multiple tickers.
    Returns a dict mapping ticker -> DataFrame with 'time' and 'Close' columns.
    
    Includes retry logic with exponential backoff to handle rate limiting.
    Falls back to fetching tickers individually if batch fetch fails.
    """
    result = {}
    max_retries = 2
    base_delay = 2  # seconds
    
    # Try to download all tickers at once first (more efficient)
    df = None
    batch_success = False
    
    for attempt in range(max_retries):
        try:
            df = yf.download(
                tickers=tickers,
                period="1d",
                interval="1m",
                progress=False,
                auto_adjust=False,
            )
            if not df.empty:
                batch_success = True
                break  # Success, exit retry loop
        except YFRateLimitError:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
        except Exception:
            pass  # Will fall back to individual fetches
    
    # Process batch result if successful
    if batch_success and df is not None and not df.empty:
        df = df.reset_index()
        
        # Handle MultiIndex columns (multiple tickers)
        if isinstance(df.columns, pd.MultiIndex):
            # Get unique tickers from second level
            unique_tickers = df.columns.get_level_values(1).unique()
            
            for ticker in unique_tickers:
                try:
                    # Get all columns for this ticker
                    ticker_cols = [col for col in df.columns if col[1] == ticker]
                    
                    # Find time column
                    time_col = None
                    for col in df.columns:
                        if isinstance(col, tuple) and col[0] in ["Datetime", "Date"]:
                            time_col = col
                            break
                    
                    if time_col is None:
                        time_col = df.columns[0]
                    
                    # Extract ticker data
                    cols_to_use = [time_col] + ticker_cols
                    ticker_df = df[cols_to_use].copy()
                    
                    # Flatten column names
                    new_cols = []
                    for col in ticker_df.columns:
                        if isinstance(col, tuple):
                            new_cols.append(col[0])
                        else:
                            new_cols.append(col)
                    ticker_df.columns = new_cols
                    
                    # Rename time column
                    if "Datetime" in ticker_df.columns:
                        ticker_df = ticker_df.rename(columns={"Datetime": "time"})
                    elif "Date" in ticker_df.columns:
                        ticker_df = ticker_df.rename(columns={"Date": "time"})
                    elif len(ticker_df.columns) > 0:
                        ticker_df = ticker_df.rename(columns={ticker_df.columns[0]: "time"})
                    
                    # Ensure Close column exists and add to result
                    if "Close" in ticker_df.columns and not ticker_df.empty:
                        result[ticker.upper()] = ticker_df[["time", "Close"]].copy()
                except Exception:
                    continue  # Skip this ticker if processing fails
        else:
            # Single ticker case - simple columns
            try:
                if "Datetime" in df.columns:
                    time_col = "Datetime"
                elif "Date" in df.columns:
                    time_col = "Date"
                else:
                    time_col = df.columns[0]
                
                df = df.rename(columns={time_col: "time"})
                if "Close" in df.columns and not df.empty:
                    result[tickers[0].upper()] = df[["time", "Close"]].copy()
            except Exception:
                pass
    
    # Fallback: fetch tickers individually if batch failed or returned empty
    if not result:
        for ticker in tickers:
            for attempt in range(max_retries):
                try:
                    ticker_df = yf.download(
                        tickers=ticker,
                        period="1d",
                        interval="1m",
                        progress=False,
                        auto_adjust=False,
                    )
                    
                    if not ticker_df.empty:
                        ticker_df = ticker_df.reset_index()
                        
                        # Rename time column
                        if "Datetime" in ticker_df.columns:
                            ticker_df = ticker_df.rename(columns={"Datetime": "time"})
                        elif "Date" in ticker_df.columns:
                            ticker_df = ticker_df.rename(columns={"Date": "time"})
                        else:
                            ticker_df = ticker_df.rename(columns={ticker_df.columns[0]: "time"})
                        
                        # Extract Close column
                        if "Close" in ticker_df.columns and not ticker_df.empty:
                            result[ticker.upper()] = ticker_df[["time", "Close"]].copy()
                            break  # Success for this ticker
                except YFRateLimitError:
                    if attempt < max_retries - 1:
                        time.sleep(base_delay * (2 ** attempt))
                    else:
                        break  # Give up on this ticker
                except Exception:
                    break  # Give up on this ticker
            
            # Small delay between individual fetches
            if ticker != tickers[-1]:
                time.sleep(0.5)
    
    return result


@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_historical_series(tickers: List[str], period: str = "1mo", interval: str = "1d") -> Dict[str, pd.DataFrame]:
    """
    Fetch historical price series for multiple tickers.
    Returns a dict mapping ticker -> DataFrame with 'time' and 'Close' columns.
    
    Args:
        tickers: List of ticker symbols
        period: Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        interval: Valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    """
    result = {}
    max_retries = 3  # More retries for historical data
    base_delay = 3  # Longer base delay
    
    # Try to download all tickers at once first using yf.download
    df = None
    batch_success = False
    last_error = None
    
    for attempt in range(max_retries):
        try:
            df = yf.download(
                tickers=tickers,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=False,
            )
            if df is not None and not df.empty:
                batch_success = True
                break
        except YFRateLimitError as e:
            last_error = f"Rate limit error: {str(e)}"
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
        except Exception as e:
            last_error = f"Error: {str(e)}"
            # For non-rate-limit errors, try once more with delay
            if attempt < max_retries - 1:
                time.sleep(base_delay)
    
    # If batch download failed, try individual Ticker().history() calls (more reliable)
    if not batch_success and len(tickers) == 1:
        try:
            ticker_obj = yf.Ticker(tickers[0])
            df = ticker_obj.history(
                period=period,
                interval=interval,
                auto_adjust=False,
            )
            if df is not None and not df.empty:
                batch_success = True
        except Exception:
            pass
    
    # Process batch result if successful
    if batch_success and df is not None and not df.empty:
        df = df.reset_index()
        
        # Handle MultiIndex columns (multiple tickers)
        if isinstance(df.columns, pd.MultiIndex):
            unique_tickers = df.columns.get_level_values(1).unique()
            
            for ticker in unique_tickers:
                try:
                    ticker_cols = [col for col in df.columns if col[1] == ticker]
                    
                    # Find time column
                    time_col = None
                    for col in df.columns:
                        if isinstance(col, tuple) and col[0] in ["Datetime", "Date"]:
                            time_col = col
                            break
                    
                    if time_col is None:
                        time_col = df.columns[0]
                    
                    cols_to_use = [time_col] + ticker_cols
                    ticker_df = df[cols_to_use].copy()
                    
                    # Flatten column names
                    new_cols = []
                    for col in ticker_df.columns:
                        if isinstance(col, tuple):
                            new_cols.append(col[0])
                        else:
                            new_cols.append(col)
                    ticker_df.columns = new_cols
                    
                    # Rename time column
                    if "Datetime" in ticker_df.columns:
                        ticker_df = ticker_df.rename(columns={"Datetime": "time"})
                    elif "Date" in ticker_df.columns:
                        ticker_df = ticker_df.rename(columns={"Date": "time"})
                    elif len(ticker_df.columns) > 0:
                        ticker_df = ticker_df.rename(columns={ticker_df.columns[0]: "time"})
                    
                    if "Close" in ticker_df.columns and not ticker_df.empty:
                        result[ticker.upper()] = ticker_df[["time", "Close"]].copy()
                except Exception:
                    continue
        else:
            # Single ticker case
            try:
                if "Datetime" in df.columns:
                    time_col = "Datetime"
                elif "Date" in df.columns:
                    time_col = "Date"
                else:
                    time_col = df.columns[0]
                
                df = df.rename(columns={time_col: "time"})
                if "Close" in df.columns and not df.empty:
                    result[tickers[0].upper()] = df[["time", "Close"]].copy()
            except Exception:
                pass
    
    # Fallback: fetch tickers individually using Ticker().history() (more reliable)
    if not result:
        for ticker in tickers:
            ticker_success = False
            for attempt in range(max_retries):
                try:
                    # Try using Ticker().history() first (often more reliable)
                    ticker_obj = yf.Ticker(ticker)
                    ticker_df = ticker_obj.history(
                        period=period,
                        interval=interval,
                        auto_adjust=False,
                    )
                    
                    if ticker_df is not None and not ticker_df.empty:
                        ticker_df = ticker_df.reset_index()
                        
                        # Handle MultiIndex if present
                        if isinstance(ticker_df.columns, pd.MultiIndex):
                            ticker_df.columns = [col[0] if isinstance(col, tuple) else col for col in ticker_df.columns]
                        
                        # Rename time column
                        if "Datetime" in ticker_df.columns:
                            ticker_df = ticker_df.rename(columns={"Datetime": "time"})
                        elif "Date" in ticker_df.columns:
                            ticker_df = ticker_df.rename(columns={"Date": "time"})
                        elif len(ticker_df.columns) > 0:
                            ticker_df = ticker_df.rename(columns={ticker_df.columns[0]: "time"})
                        
                        if "Close" in ticker_df.columns and not ticker_df.empty:
                            result[ticker.upper()] = ticker_df[["time", "Close"]].copy()
                            ticker_success = True
                            break
                except YFRateLimitError:
                    if attempt < max_retries - 1:
                        time.sleep(base_delay * (2 ** attempt))
                    else:
                        # Last attempt: try yf.download as final fallback
                        try:
                            ticker_df = yf.download(
                                tickers=ticker,
                                period=period,
                                interval=interval,
                                progress=False,
                                auto_adjust=False,
                            )
                            if ticker_df is not None and not ticker_df.empty:
                                ticker_df = ticker_df.reset_index()
                                if isinstance(ticker_df.columns, pd.MultiIndex):
                                    ticker_df.columns = [col[0] if isinstance(col, tuple) else col for col in ticker_df.columns]
                                if "Datetime" in ticker_df.columns:
                                    ticker_df = ticker_df.rename(columns={"Datetime": "time"})
                                elif "Date" in ticker_df.columns:
                                    ticker_df = ticker_df.rename(columns={"Date": "time"})
                                elif len(ticker_df.columns) > 0:
                                    ticker_df = ticker_df.rename(columns={ticker_df.columns[0]: "time"})
                                if "Close" in ticker_df.columns and not ticker_df.empty:
                                    result[ticker.upper()] = ticker_df[["time", "Close"]].copy()
                                    ticker_success = True
                        except Exception:
                            pass
                        break
                except Exception as e:
                    # Try once more for other errors
                    if attempt < max_retries - 1:
                        time.sleep(base_delay)
                    else:
                        # Last attempt: try yf.download as final fallback
                        try:
                            ticker_df = yf.download(
                                tickers=ticker,
                                period=period,
                                interval=interval,
                                progress=False,
                                auto_adjust=False,
                            )
                            if ticker_df is not None and not ticker_df.empty:
                                ticker_df = ticker_df.reset_index()
                                if isinstance(ticker_df.columns, pd.MultiIndex):
                                    ticker_df.columns = [col[0] if isinstance(col, tuple) else col for col in ticker_df.columns]
                                if "Datetime" in ticker_df.columns:
                                    ticker_df = ticker_df.rename(columns={"Datetime": "time"})
                                elif "Date" in ticker_df.columns:
                                    ticker_df = ticker_df.rename(columns={"Date": "time"})
                                elif len(ticker_df.columns) > 0:
                                    ticker_df = ticker_df.rename(columns={ticker_df.columns[0]: "time"})
                                if "Close" in ticker_df.columns and not ticker_df.empty:
                                    result[ticker.upper()] = ticker_df[["time", "Close"]].copy()
                                    ticker_success = True
                        except Exception:
                            pass
                        break
            
            # Delay between tickers to avoid rate limiting
            if ticker != tickers[-1]:
                time.sleep(1.0)  # Longer delay between individual fetches
    
    return result


def main() -> None:
    st.set_page_config(page_title="Stock Dashboard", layout="wide")

    st.title("Stock Price / Volume Dashboard")

    st.write(
        "Backend is using free Yahoo! Finance data via `yfinance` "
        "(typically delayed). This is just to prove the dashboard. "
        "Later you can plug in Webull or any other real-time feed."
    )
    st.info("ℹ️ **Note:** Data is cached for 60 seconds to reduce API calls. If you see rate limit errors, wait a moment before refreshing.")

    default_tickers = "AAPL, MSFT, TSLA"
    tickers_input = st.text_input("Tickers (comma separated)", value=default_tickers)
    if st.button("Refresh quotes and chart"):
        # This just causes the script to re-run with the same state.
        st.rerun()

    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    if not tickers:
        st.warning("Enter at least one ticker symbol.")
        return

    # Main layout: left = table, right = chart
    col_table, col_chart = st.columns([1, 2], gap="large")

    with col_table:
        with st.spinner("Fetching quotes..."):
            try:
                quotes = fetch_quotes(tickers)
            except Exception as e:
                st.error(f"Error fetching quotes: {str(e)}")
                quotes = pd.DataFrame()
        
        if quotes.empty:
            st.error("No quote data returned.")
        elif quotes["price"].isna().any():
            st.warning("⚠️ Rate limited by Yahoo Finance. Some data may be missing. Please wait a moment and refresh.")
            # Show what we have
            quotes_display = quotes[~quotes["price"].isna()].copy()
            if not quotes_display.empty:
                def format_pct(val: float) -> str:
                    if pd.isna(val):
                        return "N/A"
                    return f"{val:+.2f}%"
                
                quotes_display["pct_change"] = quotes_display["pct_change"].map(format_pct)
                st.subheader("Latest Quotes (partial)")
                st.dataframe(
                    quotes_display.set_index("ticker"),
                    use_container_width=True,
                )
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
        selected_tickers = st.multiselect(
            "Select tickers for chart (multiple allowed)",
            options=tickers,
            default=tickers[:min(3, len(tickers))] if tickers else []
        )
        
        if not selected_tickers:
            st.info("Select at least one ticker to display the chart.")
        else:
            # Data type selection
            data_type = st.radio(
                "Chart type",
                options=["Intraday", "Historical"],
                horizontal=True,
                index=0
            )
            
            # Period and interval selection for historical data
            period = "1d"
            interval = "1m"
            chart_title_prefix = "Intraday"
            
            if data_type == "Historical":
                col_period, col_interval = st.columns(2)
                
                with col_period:
                    period = st.selectbox(
                        "Period",
                        options=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
                        index=2,  # Default to 1mo
                        help="Time range for historical data"
                    )
                
                with col_interval:
                    # Interval options depend on period
                    if period in ["1d", "5d"]:
                        interval_options = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d"]
                        default_idx = 2  # 5m
                    elif period in ["1mo", "3mo"]:
                        interval_options = ["1d", "5d", "1wk"]
                        default_idx = 0  # 1d (more reliable than 1h)
                    else:
                        interval_options = ["1d", "5d", "1wk", "1mo", "3mo"]
                        default_idx = 0  # 1d
                    
                    interval = st.selectbox(
                        "Interval",
                        options=interval_options,
                        index=default_idx,
                        help="Data point frequency"
                    )
                
                chart_title_prefix = "Historical"
            
            # Fetch data based on selection
            if data_type == "Intraday":
                with st.spinner(f"Fetching intraday data for {', '.join(selected_tickers)}..."):
                    try:
                        series_dict = fetch_intraday_series(selected_tickers)
                    except Exception as e:
                        st.error(f"Error fetching intraday data: {str(e)}")
                        series_dict = {}
                
                if not series_dict:
                    st.warning(
                        "⚠️ No intraday data available. This may be due to:\n"
                        "- Rate limiting by Yahoo Finance (wait 30-60 seconds and try again)\n"
                        "- Market is closed (intraday data only available during trading hours)\n"
                        "- Invalid ticker symbols\n\n"
                        "**Tip:** Try refreshing in a moment, or check if the market is open."
                    )
                else:
                    _plot_chart(series_dict, selected_tickers, f"{chart_title_prefix} Price Comparison")
            else:
                # Historical data
                with st.spinner(f"Fetching historical data for {', '.join(selected_tickers)} (period: {period}, interval: {interval})..."):
                    try:
                        series_dict = fetch_historical_series(selected_tickers, period=period, interval=interval)
                    except Exception as e:
                        st.error(f"Error fetching historical data: {str(e)}")
                        series_dict = {}
                
                if not series_dict:
                    st.warning(
                        "⚠️ No historical data available. This may be due to:\n"
                        "- Rate limiting by Yahoo Finance (wait 30-60 seconds and try again)\n"
                        "- Invalid ticker symbols or period/interval combination\n"
                        "- Data not available for the selected period\n\n"
                        "**Tips:**\n"
                        "- Try a simpler combination: Period='1mo' with Interval='1d' (most reliable)\n"
                        "- For longer periods (1y+), use Interval='1d' or '1wk'\n"
                        "- Wait 30-60 seconds if you see rate limit errors, then refresh"
                    )
                else:
                    _plot_chart(series_dict, selected_tickers, f"{chart_title_prefix} Price Comparison ({period})")

    # Note about Webull latency
    st.markdown(
        """
**About Webull latency**  
- Webull gives real-time quotes for U.S. stocks and ETFs to app users; in practice you're usually looking at a 1–2 second network/processing delay vs the exchange.  
- Options are typically **15 minutes delayed** unless you subscribe to OPRA real-time data.  
- This dashboard is wired so you can swap `yfinance` out for a Webull client later (e.g., an unofficial API wrapper) and keep the UI the same.
"""
    )


def _plot_chart(series_dict: Dict[str, pd.DataFrame], selected_tickers: List[str], title: str) -> None:
    """Helper function to plot chart from series dictionary."""
    fig = go.Figure()
    
    # Color palette for different tickers
    colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # olive
        '#17becf',  # cyan
    ]
    
    for idx, ticker in enumerate(selected_tickers):
        if ticker in series_dict:
            series = series_dict[ticker]
            if not series.empty and "Close" in series.columns:
                color = colors[idx % len(colors)]
                fig.add_trace(
                    go.Scatter(
                        x=series["time"],
                        y=series["Close"],
                        mode="lines",
                        name=ticker,
                        line=dict(color=color, width=2),
                    )
                )
    
    if len(fig.data) == 0:
        st.error("No valid data to plot.")
    else:
        fig.update_layout(
            title=f"{title}: {', '.join(selected_tickers)}",
            xaxis_title="Time",
            yaxis_title="Price ($)",
            height=600,
            width=None,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()

