import logging
import time
from typing import Dict, List
from datetime import datetime, timedelta

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

# Global rate limiting state
if 'last_rate_limit_time' not in st.session_state:
    st.session_state.last_rate_limit_time = None
if 'rate_limit_cooldown' not in st.session_state:
    st.session_state.rate_limit_cooldown = False


def check_rate_limit_cooldown() -> bool:
    """Check if we're in a cooldown period after rate limiting."""
    if st.session_state.rate_limit_cooldown:
        if st.session_state.last_rate_limit_time:
            elapsed = (datetime.now() - st.session_state.last_rate_limit_time).total_seconds()
            if elapsed < 60:  # 60 second cooldown
                remaining = int(60 - elapsed)
                logger.warning(f"⏸️ Rate limit cooldown active. Please wait {remaining} more seconds.")
                return True
            else:
                # Cooldown expired
                st.session_state.rate_limit_cooldown = False
                st.session_state.last_rate_limit_time = None
    return False


def mark_rate_limited():
    """Mark that we've been rate limited."""
    st.session_state.last_rate_limit_time = datetime.now()
    st.session_state.rate_limit_cooldown = True


@st.cache_data(ttl=300)  # Cache for 5 minutes to reduce API calls
def fetch_quotes_with_period(tickers: List[str], period: str = "1d") -> pd.DataFrame:
    """
    Fetch quote data with percentage change calculated over a specified time period.
    
    Args:
        tickers: List of ticker symbols
        period: Time period for percentage change calculation
                Options: "1d" (daily), "5d", "1wk", "1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"
    
    Returns:
        DataFrame with ticker, price, pct_change (over selected period), volume, currency
    """
    # Check if we're in cooldown
    if check_rate_limit_cooldown():
        # Return empty data with placeholders
        return pd.DataFrame([{
            "ticker": t.upper(),
            "price": None,
            "prev_close": None,
            "pct_change": None,
            "volume": None,
            "currency": "N/A",
        } for t in tickers])
    
    data = []
    max_retries = 2
    base_delay = 10
    
    # Initial delay before starting requests
    time.sleep(2)
    
    # Map period to yfinance period format
    period_map = {
        "1d": "1d",
        "5d": "5d", 
        "1wk": "1wk",
        "1mo": "1mo",
        "3mo": "3mo",
        "6mo": "6mo",
        "1y": "1y",
        "2y": "2y",
        "5y": "5y",
        "ytd": "ytd",
        "max": "max"
    }
    yf_period = period_map.get(period, "1d")
    
    for t in tickers:
        ticker = yf.Ticker(t)
        logger.info(f"🔍 Fetching quote data for {t} with {period} period change...")
        
        for attempt in range(max_retries):
            try:
                # Get current price
                info = ticker.fast_info
                last_price = info.last_price
                volume = info.last_volume
                
                if last_price is None or pd.isna(last_price):
                    logger.warning(f"⚠️ Got empty data for {t}, retrying...")
                    if attempt < max_retries - 1:
                        time.sleep(base_delay)
                        continue
                    else:
                        raise ValueError(f"No price data for {t}")
                
                # Calculate percentage change over the selected period
                if yf_period == "1d":
                    # For daily change, use previous close
                    prev_close = info.previous_close
                    pct_change = ((last_price - prev_close) / prev_close * 100) if prev_close else 0.0
                else:
                    # For longer periods, fetch historical data
                    try:
                        # Determine appropriate interval based on period
                        if yf_period in ["5d", "1wk"]:
                            interval = "1d"
                        elif yf_period in ["1mo", "3mo"]:
                            interval = "1d"
                        else:
                            interval = "1d"
                        
                        hist = ticker.history(period=yf_period, interval=interval)
                        if hist is not None and not hist.empty and len(hist) > 0:
                            # Get the first close price in the period (oldest)
                            first_close = hist['Close'].iloc[0]
                            # Calculate percentage change
                            pct_change = ((last_price - first_close) / first_close * 100) if first_close else 0.0
                            prev_close = first_close
                        else:
                            # Fallback to previous close if historical data unavailable
                            prev_close = info.previous_close
                            pct_change = ((last_price - prev_close) / prev_close * 100) if prev_close else 0.0
                    except Exception as e:
                        logger.warning(f"⚠️ Could not fetch historical data for {t} ({yf_period}), using previous close: {e}")
                        prev_close = info.previous_close
                        pct_change = ((last_price - prev_close) / prev_close * 100) if prev_close else 0.0
                
                data.append({
                    "ticker": t.upper(),
                    "price": last_price,
                    "prev_close": prev_close,
                    "pct_change": pct_change,
                    "volume": volume,
                    "currency": info.currency,
                })
                logger.info(f"✅ Successfully fetched quote data for {t}: ${last_price:.2f} ({pct_change:+.2f}% over {period})")
                break
            except YFRateLimitError:
                logger.warning(f"⚠️ Rate limited for {t}, attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.info(f"⏳ Waiting {delay}s before retry...")
                    time.sleep(delay)
                else:
                    mark_rate_limited()
                    logger.error(f"❌ Rate limited for {t} after {max_retries} attempts. Entering 60-second cooldown period.")
                    data.append({
                        "ticker": t.upper(),
                        "price": None,
                        "prev_close": None,
                        "pct_change": None,
                        "volume": None,
                        "currency": "N/A",
                    })
                    remaining_tickers = tickers[tickers.index(t) + 1:]
                    for remaining_t in remaining_tickers:
                        data.append({
                            "ticker": remaining_t.upper(),
                            "price": None,
                            "prev_close": None,
                            "pct_change": None,
                            "volume": None,
                            "currency": "N/A",
                        })
                    break
            except Exception as e:
                logger.error(f"❌ Error fetching data for {t}: {e}")
                data.append({
                    "ticker": t.upper(),
                    "price": None,
                    "prev_close": None,
                    "pct_change": None,
                    "volume": None,
                    "currency": "N/A",
                })
                break
        
        if t != tickers[-1]:
            time.sleep(5)
    
    return pd.DataFrame(data)


@st.cache_data(ttl=300)  # Cache for 5 minutes to reduce API calls
def fetch_quotes(tickers: List[str]) -> pd.DataFrame:
    """
    Fetch latest quote data (price, % change, volume) for a list of tickers.

    Uses yfinance (Yahoo! Finance). Data is typically delayed (up to ~15 minutes)
    but good enough to prove out the dashboard plumbing. You can later swap this
    out for a Webull-based data source.
    
    Includes retry logic with exponential backoff to handle rate limiting.
    """
    # Check if we're in cooldown
    if check_rate_limit_cooldown():
        # Return empty data with placeholders
        return pd.DataFrame([{
            "ticker": t.upper(),
            "price": None,
            "prev_close": None,
            "pct_change": None,
            "volume": None,
            "currency": "N/A",
        } for t in tickers])
    
    data = []
    max_retries = 2  # Reduced retries to fail faster and enter cooldown
    base_delay = 10  # Increased base delay significantly
    
    # Initial delay before starting requests
    time.sleep(2)
    
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
                    delay = base_delay * (2 ** attempt)  # Exponential backoff: 10s, 20s
                    logger.info(f"⏳ Waiting {delay}s before retry...")
                    time.sleep(delay)
                else:
                    # Last attempt failed - mark as rate limited and enter cooldown
                    mark_rate_limited()
                    logger.error(f"❌ Rate limited for {t} after {max_retries} attempts. Entering 60-second cooldown period.")
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
                    # Add placeholders for remaining tickers and stop
                    remaining_tickers = tickers[tickers.index(t) + 1:]
                    for remaining_t in remaining_tickers:
                        data.append(
                            {
                                "ticker": remaining_t.upper(),
                                "price": None,
                                "prev_close": None,
                                "pct_change": None,
                                "volume": None,
                                "currency": "N/A",
                            }
                        )
                    break  # Exit ticker loop
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
            time.sleep(5)  # Increased delay significantly to reduce rate limiting
    
    return pd.DataFrame(data)


@st.cache_data(ttl=300)  # Cache for 5 minutes (intraday data changes less frequently)
def fetch_intraday_series(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Fetch intraday price series for multiple tickers.
    Returns a dict mapping ticker -> DataFrame with 'time' and 'Close' columns.
    
    Includes retry logic with exponential backoff to handle rate limiting.
    Falls back to fetching tickers individually if batch fetch fails.
    """
    # Check if we're in cooldown
    if check_rate_limit_cooldown():
        return {}
    
    result = {}
    max_retries = 2
    base_delay = 10  # seconds - increased significantly
    
    # Initial delay
    time.sleep(2)
    
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
                time.sleep(5)  # Increased delay significantly to reduce rate limiting
    
    return result


@st.cache_data(ttl=300)  # Cache for 5 minutes to reduce API calls
def fetch_historical_series(tickers: List[str], period: str = "1mo", interval: str = "1d") -> Dict[str, pd.DataFrame]:
    """
    Fetch historical price series for multiple tickers.
    Returns a dict mapping ticker -> DataFrame with 'time' and 'Close' columns.
    
    Args:
        tickers: List of ticker symbols
        period: Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        interval: Valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    """
    # Check if we're in cooldown
    if check_rate_limit_cooldown():
        return {}
    
    result = {}
    max_retries = 2  # Reduced to fail faster
    base_delay = 10  # Increased delay significantly
    
    # Initial delay
    time.sleep(2)
    
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
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"⚠️ Rate limited for {ticker}, waiting {delay}s before retry {attempt + 1}/{max_retries}")
                        time.sleep(delay)
                    else:
                        # Last attempt failed - mark as rate limited
                        mark_rate_limited()
                        logger.error(f"❌ Rate limited for {ticker} after {max_retries} attempts. Entering 60-second cooldown period.")
                        break  # Stop trying this ticker
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
            time.sleep(5)  # Increased delay significantly to reduce rate limiting
    
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
    # Show rate limit cooldown status if active
    if st.session_state.rate_limit_cooldown and st.session_state.last_rate_limit_time:
        elapsed = (datetime.now() - st.session_state.last_rate_limit_time).total_seconds()
        if elapsed < 60:
            remaining = int(60 - elapsed)
            st.error(f"⏸️ **Rate Limit Cooldown Active:** Please wait {remaining} more seconds before refreshing. Yahoo Finance has temporarily rate-limited requests.")
        else:
            st.session_state.rate_limit_cooldown = False
            st.session_state.last_rate_limit_time = None
    
    st.info("ℹ️ **Note:** Data is cached for 5 minutes to reduce API calls. If you see rate limit errors, the app will automatically enter a 60-second cooldown period.")
    
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
        
        if not selected_tickers:
            st.info("💡 **Tip:** Select tickers above or use Quick Add buttons to add popular stocks, then select them here for charting.")
        else:
            # Data type selection - Default to Historical (more reliable)
            data_type = st.radio(
                "Chart type",
                options=["Intraday", "Historical", "Performance Bars"],
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
            if data_type == "Performance Bars":
                # Time period selection for percentage change
                col_period_label, col_period_select = st.columns([1, 2])
                with col_period_label:
                    st.markdown("**Time Period:**")
                with col_period_select:
                    pct_change_period = st.selectbox(
                        "Calculate percentage change over:",
                        options=["1d", "5d", "1wk", "1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"],
                        index=0,  # Default to 1d (daily)
                        help="Select the time period over which to calculate percentage change. '1d' = daily change, '1wk' = weekly change, etc.",
                        label_visibility="collapsed"
                    )
                
                # Vertical bar chart showing percentage changes
                period_label = {
                    "1d": "Daily",
                    "5d": "5-Day",
                    "1wk": "Weekly",
                    "1mo": "Monthly",
                    "3mo": "3-Month",
                    "6mo": "6-Month",
                    "1y": "1-Year",
                    "2y": "2-Year",
                    "5y": "5-Year",
                    "ytd": "Year-to-Date",
                    "max": "All-Time"
                }.get(pct_change_period, pct_change_period)
                
                with st.spinner(f"Fetching {period_label.lower()} percentage change data for {', '.join(selected_tickers)}..."):
                    try:
                        quotes_data = fetch_quotes_with_period(selected_tickers, period=pct_change_period)
                    except Exception as e:
                        st.error(f"Error fetching quote data: {str(e)}")
                        quotes_data = pd.DataFrame()
                
                if quotes_data.empty or quotes_data["pct_change"].isna().all():
                    st.warning(
                        "⚠️ No quote data available for performance bars. This may be due to:\n"
                        "- Rate limiting by Yahoo Finance (wait 30-60 seconds and try again)\n"
                        "- Invalid ticker symbols\n\n"
                        "**Tip:** Try refreshing in a moment."
                    )
                else:
                    _plot_performance_bars(quotes_data, selected_tickers, period_label)
            elif data_type == "Intraday":
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


def _plot_performance_bars(quotes_df: pd.DataFrame, selected_tickers: List[str], period_label: str = "Daily") -> None:
    """
    Plot a vertical rectangular bar chart showing percentage changes grouped by ranges.
    Matches the reference design with segments, ticker labels on sides, and percentage labels.
    
    Args:
        quotes_df: DataFrame with ticker, price, pct_change data
        selected_tickers: List of selected ticker symbols
        period_label: Label for the time period (e.g., "Daily", "Weekly", "Monthly")
    """
    logger.info(f"📊 Plotting performance bars for {len(quotes_df)} tickers ({period_label} change)")
    
    # Filter to only selected tickers and valid data
    quotes_df = quotes_df[quotes_df["ticker"].isin([t.upper() for t in selected_tickers])].copy()
    quotes_df = quotes_df[~quotes_df["pct_change"].isna()].copy()
    
    if quotes_df.empty:
        st.error("No valid percentage change data to plot.")
        return
    
    # Sort by percentage change (ascending for bottom to top display)
    quotes_df = quotes_df.sort_values("pct_change", ascending=True).copy()
    
    # Define percentage ranges (bins) with realistic stock movement ranges
    # Negative ranges (losses) - Red tones
    # Small positive ranges (typical daily moves) - Yellow/Orange tones
    # Medium positive ranges - Green/Blue tones
    # Large positive ranges - Bright Green tones
    ranges = [
        (float('-inf'), -10, "-10%+", '#8B0000'),      # Dark Red (large losses)
        (-10, -5, "-5% to -10%", '#DC143C'),           # Crimson (moderate losses)
        (-5, -2, "-2% to -5%", '#FF6347'),              # Tomato (small losses)
        (-2, 0, "0% to -2%", '#FFA07A'),                # Light Salmon (minimal losses)
        (0, 1, "+0% to +1%", '#FFD700'),                # Gold (flat to small gain)
        (1, 2, "+1% to +2%", '#FFA500'),                # Orange (small gain)
        (2, 3, "+2% to +3%", '#FF8C00'),                # Dark Orange (moderate gain)
        (3, 5, "+3% to +5%", '#32CD32'),                # Lime Green (good gain)
        (5, 10, "+5% to +10%", '#00CED1'),              # Dark Turquoise (strong gain)
        (10, 20, "+10% to +20%", '#1E90FF'),            # Dodger Blue (very strong gain)
        (20, float('inf'), "+20%+", '#00FF00'),         # Bright Green (exceptional gain)
    ]
    
    # Group tickers by percentage range
    range_groups = {range_label: [] for _, _, range_label, _ in ranges}
    
    for _, row in quotes_df.iterrows():
        pct = row["pct_change"]
        ticker = row["ticker"]
        
        # Find which range this ticker belongs to
        for min_pct, max_pct, range_label, _ in ranges:
            # Handle different range types
            if min_pct == float('-inf'):
                # Negative infinity: match if pct < max_pct
                if pct < max_pct:
                    range_groups[range_label].append({
                        "ticker": ticker,
                        "pct_change": pct,
                        "price": row["price"]
                    })
                    break
            elif max_pct == float('inf'):
                # Positive infinity: match if pct >= min_pct
                if pct >= min_pct:
                    range_groups[range_label].append({
                        "ticker": ticker,
                        "pct_change": pct,
                        "price": row["price"]
                    })
                    break
            else:
                # Normal range: match if min_pct <= pct < max_pct
                if min_pct <= pct < max_pct:
                    range_groups[range_label].append({
                        "ticker": ticker,
                        "pct_change": pct,
                        "price": row["price"]
                    })
                    break
    
    # Create the vertical bar chart
    fig = go.Figure()
    
    # Build stacked bars from bottom to top with equal segment heights
    segment_height = 10  # Fixed height for each segment for visual consistency
    y_bottom = 0
    all_annotations = []
    
    # Track which segments have data
    segments_with_data = []
    for min_pct, max_pct, range_label, color in ranges:
        if range_groups[range_label]:
            segments_with_data.append((min_pct, max_pct, range_label, color))
    
    # Build the chart from bottom to top
    for idx, (min_pct, max_pct, range_label, color) in enumerate(segments_with_data):
        tickers_in_range = range_groups[range_label]
        y_top = y_bottom + segment_height
        y_mid = (y_bottom + y_top) / 2
        
        # Create the bar segment with black border
        fig.add_trace(go.Bar(
            x=[0.5],  # Center the bar
            y=[segment_height],
            base=[y_bottom],
            name=range_label,
            marker_color=color,
            marker_line=dict(color='black', width=2),
            hovertemplate=f"<b>{range_label}</b><br>" +
                         "<br>".join([f"{t['ticker']}: {t['pct_change']:.2f}% (${t['price']:.2f})" 
                                     for t in tickers_in_range]) +
                         "<extra></extra>",
            orientation='v',
            showlegend=False
        ))
        
        # Get ticker labels
        ticker_labels = ", ".join([t['ticker'] for t in tickers_in_range])
        
        # Alternate ticker labels between left and right sides
        # Even indices (0, 2, 4...) go on right, odd indices (1, 3, 5...) go on left
        if idx % 2 == 0:
            # Right side for even indices
            all_annotations.append(dict(
                x=0.95,
                y=y_mid,
                text=ticker_labels,
                showarrow=False,
                xref="paper",
                yref="y",
                font=dict(size=12, color="black", family="Arial Black"),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="black",
                borderwidth=1.5,
                align="right"
            ))
            # Percentage label on right
            all_annotations.append(dict(
                x=0.98,
                y=y_mid,
                text=range_label,
                showarrow=False,
                xref="paper",
                yref="y",
                font=dict(size=11, color="black", family="Arial Black", weight="bold"),
                align="right"
            ))
        else:
            # Left side for odd indices
            all_annotations.append(dict(
                x=0.05,
                y=y_mid,
                text=ticker_labels,
                showarrow=False,
                xref="paper",
                yref="y",
                font=dict(size=12, color="black", family="Arial Black"),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="black",
                borderwidth=1.5,
                align="left"
            ))
            # Percentage label on left
            all_annotations.append(dict(
                x=0.02,
                y=y_mid,
                text=range_label,
                showarrow=False,
                xref="paper",
                yref="y",
                font=dict(size=11, color="black", family="Arial Black", weight="bold"),
                align="left"
            ))
        
        # Add horizontal dashed lines at segment boundaries
        if idx < len(segments_with_data) - 1:  # Don't add line at the top
            fig.add_shape(
                type="line",
                x0=0.3, y0=y_top, x1=0.7, y1=y_top,
                line=dict(color="black", width=1, dash="dash"),
                xref="paper",
                yref="y"
            )
        
        y_bottom = y_top
    
    # Calculate max y for arrow positioning
    max_y = len(segments_with_data) * segment_height if segments_with_data else 100
    
    # Add large upward arrow on the left side
    all_annotations.append(dict(
        x=0.1,
        y=max_y + 5,
        text="↑",
        showarrow=False,
        xref="paper",
        yref="y",
        font=dict(size=40, color="black"),
        align="center"
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"📊 Vertical Chart Display: {period_label} Percentage Change by Range",
            font=dict(size=18, family="Arial Black")
        ),
        xaxis=dict(
            title="",
            showticklabels=False,
            range=[0, 1],
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            title="Percentage Change (%)",
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            gridwidth=1,
            side="left",
            showline=True,
            linecolor="black",
            linewidth=2
        ),
        height=1000,
        width=None,
        barmode='stack',
        showlegend=False,
        hovermode='closest',
        plot_bgcolor='rgba(240,240,240,0.5)',
        paper_bgcolor='white',
        annotations=all_annotations,
        margin=dict(l=100, r=100, t=80, b=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Also show a summary table
    with st.expander("📊 Detailed Performance Data", expanded=False):
        display_df = quotes_df[["ticker", "price", "pct_change", "volume"]].copy().sort_values("pct_change", ascending=False)
        display_df["pct_change"] = display_df["pct_change"].apply(lambda x: f"{x:+.2f}%")
        st.dataframe(display_df.set_index("ticker"), use_container_width=True)


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

