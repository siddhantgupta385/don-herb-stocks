import logging
import time
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

# Enable logging to see yfinance API calls
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable yfinance debug logging to see HTTP requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


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
        logger.info(f"🔍 Fetching quote data for {t} from Yahoo Finance...")
        
        # Retry logic for rate limiting
        for attempt in range(max_retries):
            try:
                info = ticker.fast_info
                last_price = info.last_price
                prev_close = info.previous_close
                volume = info.last_volume
                
                # Validate we actually got data
                if last_price is None or pd.isna(last_price):
                    logger.warning(f"⚠️ Got empty data for {t}, retrying...")
                    if attempt < max_retries - 1:
                        time.sleep(base_delay)
                        continue
                    else:
                        raise ValueError(f"No price data for {t}")
                
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
                logger.info(f"✅ Successfully fetched quote data for {t}: ${last_price:.2f}")
                break  # Success, exit retry loop
            except YFRateLimitError:
                logger.warning(f"⚠️ Rate limited for {t}, attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"⏳ Waiting {delay}s before retry...")
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


@st.cache_data(ttl=60)  # Reduced cache time to 60 seconds
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
    max_retries = 2  # Reduced retries to speed up
    base_delay = 1  # Shorter delay - we'll retry faster
    
    # Fetch tickers individually using Ticker().history() - most reliable method
    # This avoids batch download rate limiting issues
    for ticker in tickers:
        ticker_success = False
        for attempt in range(max_retries):
                try:
                    # PRIMARY: Use Ticker().history() - most reliable method
                    logger.info(f"🔍 Fetching historical data for {ticker} (period={period}, interval={interval}) from Yahoo Finance...")
                    ticker_obj = yf.Ticker(ticker)
                    ticker_df = ticker_obj.history(
                        period=period,
                        interval=interval,
                        auto_adjust=False,
                    )
                    
                    if ticker_df is not None and not ticker_df.empty:
                        logger.info(f"✅ Successfully fetched {len(ticker_df)} data points for {ticker}")
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
                            logger.info(f"✅ Added {ticker.upper()} to result with {len(ticker_df)} rows")
                            ticker_success = True
                            break
                        else:
                            logger.warning(f"⚠️ No 'Close' column found for {ticker} or DataFrame is empty")
                    else:
                        logger.warning(f"⚠️ Empty DataFrame returned for {ticker}")
                except YFRateLimitError:
                    if attempt < max_retries - 1:
                        time.sleep(base_delay * (2 ** attempt))
                    else:
                        # Last attempt: try simpler period/interval as fallback
                        try:
                            # Try with simpler interval if current one failed
                            simple_interval = "1d" if interval != "1d" else "1wk"
                            ticker_obj = yf.Ticker(ticker)
                            ticker_df = ticker_obj.history(
                                period=period,
                                interval=simple_interval,
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
                        # Last attempt: try simpler period/interval
                        try:
                            simple_interval = "1d" if interval != "1d" else "1wk"
                            ticker_obj = yf.Ticker(ticker)
                            ticker_df = ticker_obj.history(
                                period=period,
                                interval=simple_interval,
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
            time.sleep(0.5)  # Shorter delay for faster loading
    
    logger.info(f"📦 Returning historical data: {len(result)} tickers - {list(result.keys())}")
    for ticker_key, df in result.items():
        logger.info(f"  {ticker_key}: {len(df)} rows, columns: {list(df.columns)}")
    
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
    
    # Info about supported tickers
    with st.expander("ℹ️ What tickers are supported?", expanded=False):
        st.markdown("""
        **This dashboard supports any ticker symbol available on Yahoo Finance, including:**
        
        - **U.S. Stocks**: AAPL, MSFT, TSLA, GOOGL, AMZN, etc.
        - **International Stocks**: ASML, TSM, NVO, SAP, etc. (use Yahoo Finance format)
        - **ETFs**: SPY, QQQ, VTI, VOO, etc.
        - **Indices**: ^GSPC (S&P 500), ^DJI (Dow), ^IXIC (Nasdaq)
        - **Crypto (via ETFs)**: GBTC, ETHE, BITO, COIN
        - **Currencies**: EURUSD=X, GBPUSD=X, etc.
        - **Commodities**: GC=F (Gold), CL=F (Crude Oil), etc.
        
        **To find ticker symbols:**
        - Search on [Yahoo Finance](https://finance.yahoo.com)
        - Use the Quick Add buttons below for popular options
        - Type any valid Yahoo Finance ticker symbol
        
        **Note:** Some tickers may have different formats (e.g., indices use ^ prefix, currencies use =X suffix)
        """)

    # Popular tickers organized by category
    POPULAR_TICKERS = {
        "Tech Giants": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX"],
        "Finance": ["JPM", "BAC", "WFC", "GS", "MS", "C", "V", "MA"],
        "Healthcare": ["JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT", "LLY"],
        "Consumer": ["WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "COST", "LOW"],
        "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "VLO", "PSX"],
        "Industrials": ["BA", "CAT", "GE", "HON", "LMT", "RTX", "DE", "EMR"],
        "Indices & ETFs": ["SPY", "QQQ", "DIA", "IWM", "VTI", "VOO", "VEA", "VWO"],
        "Crypto (via ETFs)": ["GBTC", "ETHE", "BITO", "COIN"],
        "International": ["ASML", "TSM", "NVO", "SAP", "UL", "BP", "SHEL", "TM"],
    }
    
    # Flatten all popular tickers for easy access
    ALL_POPULAR_TICKERS = [ticker for category in POPULAR_TICKERS.values() for ticker in category]
    
    st.subheader("📊 Ticker Selection")
    
    # Two-column layout for ticker selection
    col_ticker_input, col_quick_add = st.columns([2, 1])
    
    with col_ticker_input:
        default_tickers = "AAPL, MSFT, TSLA"
        tickers_input = st.text_input(
            "Enter ticker symbols (comma separated)", 
            value=default_tickers,
            help="Type any stock ticker symbol available on Yahoo Finance. Examples: AAPL, MSFT, TSLA, GOOGL, AMZN, SPY, QQQ. You can also search for international stocks, ETFs, and more!"
        )
        
        # Show autocomplete suggestions
        if tickers_input:
            current_input = tickers_input.split(",")[-1].strip().upper()
            if current_input and len(current_input) >= 1:
                # Find matching popular tickers
                matches = [t for t in ALL_POPULAR_TICKERS if t.startswith(current_input)][:5]
                if matches:
                    st.caption(f"💡 Suggestions: {', '.join(matches)}")
    
    with col_quick_add:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    # Quick add popular tickers with expandable sections
    with st.expander("🚀 Quick Add Popular Tickers (Click to expand)", expanded=False):
        cols = st.columns(3)
        
        category_idx = 0
        for category, ticker_list in POPULAR_TICKERS.items():
            col = cols[category_idx % 3]
            with col:
                # Show category with ticker count
                st.markdown(f"**{category}** ({len(ticker_list)} tickers)")
                
                # Create buttons for each ticker in this category
                for ticker in ticker_list[:4]:  # Show first 4, rest in tooltip
                    current_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
                    if ticker not in current_tickers:
                        if st.button(f"➕ {ticker}", key=f"add_{ticker}", use_container_width=True):
                            current_tickers.append(ticker)
                            st.session_state['tickers_input'] = ", ".join(current_tickers)
                            st.rerun()
                    else:
                        st.button(f"✓ {ticker}", key=f"added_{ticker}", use_container_width=True, disabled=True)
                
                # Show "more" if there are more tickers
                if len(ticker_list) > 4:
                    remaining = ", ".join(ticker_list[4:])
                    st.caption(f"Also: {remaining}")
            
            category_idx += 1
    
    # Use session state to preserve ticker input if quick-add was used
    if 'tickers_input' in st.session_state:
        tickers_input = st.session_state['tickers_input']
        del st.session_state['tickers_input']  # Clear after use
    
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    
    # Remove duplicates while preserving order
    seen = set()
    tickers = [t for t in tickers if not (t in seen or seen.add(t))]
    
    if not tickers:
        st.warning("Enter at least one ticker symbol or use the Quick Add buttons above.")
        return
    
    # Show selected tickers
    if len(tickers) > 0:
        st.info(f"📈 **Selected Tickers ({len(tickers)}):** {', '.join(tickers)}")

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
            logger.warning("⚠️ Quotes DataFrame is empty")
            st.error("No quote data returned.")
        elif quotes["price"].isna().any():
            logger.warning(f"⚠️ Some quotes have NaN prices: {quotes[quotes['price'].isna()]['ticker'].tolist()}")
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
        st.subheader("📈 Chart Configuration")
        
        # Enhanced ticker selection with search and popular options
        st.markdown("**Select tickers to display on chart:**")
        
        # Create a combined list of current tickers + popular tickers for selection
        # Prioritize current tickers first, then popular ones
        all_available_tickers = tickers + [t for t in ALL_POPULAR_TICKERS if t not in tickers]
        
        selected_tickers = st.multiselect(
            "Choose tickers for chart (search or select multiple)",
            options=all_available_tickers,
            default=tickers[:min(3, len(tickers))] if tickers else [],
            help=f"Search for any ticker symbol. Currently loaded: {len(tickers)} tickers. Popular tickers ({len(ALL_POPULAR_TICKERS)} total) are also available. Select multiple to compare on the same chart."
        )
        
        # Allow adding custom tickers directly in chart selection
        if selected_tickers:
            st.caption(f"📊 Selected for chart: {', '.join(selected_tickers)}")
        
        # Show quick add buttons for popular categories
        if not selected_tickers:
            st.info("💡 **Tip:** Select tickers above or use Quick Add buttons to add popular stocks, then select them here for charting.")
        
        if not selected_tickers:
            st.info("Select at least one ticker to display the chart.")
        else:
            # Data type selection - Default to Historical (more reliable)
            data_type = st.radio(
                "Chart type",
                options=["Intraday", "Historical"],
                horizontal=True,
                index=1  # Default to Historical
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
                    logger.warning(f"⚠️ No historical data returned for {selected_tickers}")
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
                    logger.info(f"✅ Plotting chart with {len(series_dict)} tickers: {list(series_dict.keys())}")
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
    logger.info(f"📊 Plotting chart: {title} with {len(series_dict)} series")
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
    
    traces_added = 0
    for idx, ticker in enumerate(selected_tickers):
        if ticker in series_dict:
            series = series_dict[ticker]
            logger.info(f"  Checking {ticker}: empty={series.empty}, has Close={'Close' in series.columns if not series.empty else False}")
            if not series.empty and "Close" in series.columns:
                color = colors[idx % len(colors)]
                logger.info(f"  ✅ Adding trace for {ticker} with {len(series)} points")
                fig.add_trace(
                    go.Scatter(
                        x=series["time"],
                        y=series["Close"],
                        mode="lines",
                        name=ticker,
                        line=dict(color=color, width=2),
                    )
                )
                traces_added += 1
            else:
                logger.warning(f"  ⚠️ Skipping {ticker}: empty={series.empty}, has Close={'Close' in series.columns if not series.empty else False}")
        else:
            logger.warning(f"  ⚠️ {ticker} not found in series_dict. Available: {list(series_dict.keys())}")
    
    logger.info(f"📊 Added {traces_added} traces to chart")
    if len(fig.data) == 0:
        logger.error("❌ No valid data to plot - all series were empty or missing Close column")
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

