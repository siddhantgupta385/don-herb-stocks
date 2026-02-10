import logging
import time
from typing import Dict, List, Optional
from datetime import datetime
import os
import signal
from contextlib import contextmanager

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import requests

# Alpha Vantage API Configuration
ALPHA_VANTAGE_API_KEY = "INN8K9M8D1426XX4"
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

# Rate Limit Configuration
# Free tier: 5 requests/minute, 25 requests/day
# Premium tiers: 75, 150, 300, 600, or 1200 requests/minute (no daily limit)
# Set this based on your subscription tier
ALPHA_VANTAGE_REQUESTS_PER_MINUTE = 5  # Change this when you upgrade:
# - 75 requests/min: $49.99/month
# - 150 requests/min: $99.99/month  
# - 300 requests/min: $149.99/month
# - 600 requests/min: $199.99/month
# - 1200 requests/min: $249.99/month

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@contextmanager
def timeout_handler(seconds=10):
    """Context manager to handle timeouts for API calls."""
    def timeout_signal_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set up signal handler for timeout (Unix only)
    if hasattr(signal, 'SIGALRM'):
        old_handler = signal.signal(signal.SIGALRM, timeout_signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # Windows doesn't support SIGALRM, just yield without timeout
        yield

# Try to import streamlit-autorefresh, fallback if not available
try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False
    logger.warning("streamlit-autorefresh not installed. Install with: pip install streamlit-autorefresh")

# Global rate limiting state
if 'last_rate_limit_time' not in st.session_state:
    st.session_state.last_rate_limit_time = None
if 'rate_limit_cooldown' not in st.session_state:
    st.session_state.rate_limit_cooldown = False
if 'graphs' not in st.session_state:
    st.session_state.graphs = [{"stocks": []} for _ in range(12)]


def check_rate_limit_cooldown() -> bool:
    """Check if we're in a cooldown period after rate limiting."""
    if st.session_state.rate_limit_cooldown:
        if st.session_state.last_rate_limit_time:
            elapsed = (datetime.now() - st.session_state.last_rate_limit_time).total_seconds()
            if elapsed < 60:
                return True
            else:
                st.session_state.rate_limit_cooldown = False
                st.session_state.last_rate_limit_time = None
    return False


def mark_rate_limited():
    """Mark that we've been rate limited."""
    st.session_state.last_rate_limit_time = datetime.now()
    st.session_state.rate_limit_cooldown = True


def check_market_status() -> tuple:
    """Check if US market is open. Returns (is_open, status_message)."""
    now = datetime.now()
    current_time = now.time()
    current_day = now.weekday()  # 0=Monday, 6=Sunday
    
    # Market hours: 9:30 AM - 4:00 PM ET (convert to local if needed, simplified check)
    market_open = datetime.strptime("09:30", "%H:%M").time()
    market_close = datetime.strptime("16:00", "%H:%M").time()
    
    # Weekend check
    if current_day >= 5:  # Saturday or Sunday
        return False, "Market is closed (Weekend)"
    
    # Market hours check (simplified - assumes ET timezone)
    if current_time < market_open or current_time > market_close:
        return False, "Market is closed (Outside trading hours: 9:30 AM - 4:00 PM ET)"
    
    return True, "Market is open"


@st.cache_data(ttl=10, show_spinner=False)  # Reduced cache to 10 seconds for near real-time updates
def fetch_current_quotes(tickers: List[str]) -> pd.DataFrame:
    """Fetch current quote data (price, % change, volume) for a list of tickers using Alpha Vantage API.
    Works even when market is closed by using previous close data.
    Alpha Vantage free tier: 5 calls/minute, 500 calls/day"""
    if check_rate_limit_cooldown():
        return pd.DataFrame([{
            "ticker": t.upper(),
            "price": None,
            "prev_close": None,
            "pct_change": None,
            "volume": None,
            "currency": "N/A",
        } for t in tickers])
    
    data = []
    
    # Track API call timing for rate limiting (5 calls per minute for free tier)
    if 'alpha_vantage_calls' not in st.session_state:
        st.session_state.alpha_vantage_calls = []
    
    # Clean old calls (older than 1 minute)
    current_time = time.time()
    st.session_state.alpha_vantage_calls = [
        call_time for call_time in st.session_state.alpha_vantage_calls
        if current_time - call_time < 60
    ]
    
    for idx, t in enumerate(tickers):
        ticker = t.upper()
        last_price = None
        prev_close = None
        volume = None
        currency = "USD"
        pct_change = None
        
        logger.info(f"Fetching data for {ticker} from Alpha Vantage... ({idx + 1}/{len(tickers)})")
        
        try:
            # Check rate limit before making call
            current_time = time.time()
            recent_calls = [
                call_time for call_time in st.session_state.alpha_vantage_calls
                if current_time - call_time < 60
            ]
            
            if len(recent_calls) >= ALPHA_VANTAGE_REQUESTS_PER_MINUTE:
                # Wait until the oldest call is more than 1 minute old
                oldest_call = min(recent_calls)
                wait_time = 60 - (current_time - oldest_call) + 1  # Add 1 second buffer
                if wait_time > 0:
                    logger.warning(f"Rate limit reached ({ALPHA_VANTAGE_REQUESTS_PER_MINUTE} calls/min). Waiting {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    # Clean calls again after waiting
                    current_time = time.time()
                    st.session_state.alpha_vantage_calls = [
                        call_time for call_time in st.session_state.alpha_vantage_calls
                        if current_time - call_time < 60
                    ]
            
            # Make API call to Alpha Vantage
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": ticker,
                "apikey": ALPHA_VANTAGE_API_KEY
            }
            
            response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            # Track this API call
            st.session_state.alpha_vantage_calls.append(time.time())
            
            # Check for API errors
            if "Error Message" in result:
                error_msg = result["Error Message"]
                logger.error(f"Alpha Vantage API error for {ticker}: {error_msg}")
                if "API call frequency" in error_msg or "rate limit" in error_msg.lower():
                    mark_rate_limited()
                data.append({
                    "ticker": ticker,
                    "price": None,
                    "prev_close": None,
                    "pct_change": None,
                    "volume": None,
                    "currency": "N/A",
                })
                continue
            
            if "Note" in result:
                note = result["Note"]
                logger.warning(f"Alpha Vantage note for {ticker}: {note}")
                if "API call frequency" in note or "rate limit" in note.lower():
                    mark_rate_limited()
            
            # Parse the Global Quote response
            if "Global Quote" in result and result["Global Quote"]:
                quote = result["Global Quote"]
                
                # Extract data from Alpha Vantage response
                # Alpha Vantage uses keys like "05. price", "08. previous close", etc.
                try:
                    price_str = quote.get("05. price", "0")
                    prev_close_str = quote.get("08. previous close", "0")
                    change_str = quote.get("09. change", "0")
                    change_percent_str = quote.get("10. change percent", "0%")
                    volume_str = quote.get("06. volume", "0")
                    
                    # Convert to appropriate types
                    last_price = float(price_str) if price_str and price_str != "None" else None
                    prev_close = float(prev_close_str) if prev_close_str and prev_close_str != "None" else None
                    volume = int(float(volume_str)) if volume_str and volume_str != "None" else None
                    
                    # Parse percentage change (format: "X.XX%")
                    if change_percent_str and change_percent_str != "None":
                        change_percent_str = change_percent_str.replace("%", "").strip()
                        pct_change = float(change_percent_str) if change_percent_str else None
                    else:
                        # Calculate from price and prev_close if available
                        if last_price is not None and prev_close is not None and prev_close != 0:
                            pct_change = ((last_price - prev_close) / prev_close) * 100
                        else:
                            pct_change = None
                    
                    # If no current price but we have previous close, use it
                    if (last_price is None or last_price == 0) and prev_close is not None:
                        last_price = prev_close
                        pct_change = 0.0
                    
                    logger.info(f"Successfully fetched data for {ticker}: price={last_price}, prev_close={prev_close}, pct_change={pct_change}")
                    
                except (ValueError, TypeError) as e:
                    logger.error(f"Error parsing Alpha Vantage response for {ticker}: {e}")
                    last_price = None
                    prev_close = None
                    pct_change = None
                    volume = None
            else:
                logger.warning(f"No Global Quote data in response for {ticker}")
                last_price = None
                prev_close = None
                pct_change = None
                volume = None
            
            data.append({
                "ticker": ticker,
                "price": last_price,
                "prev_close": prev_close,
                "pct_change": pct_change if pct_change is not None else 0.0,
                "volume": volume,
                "currency": currency,
            })
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching {ticker} from Alpha Vantage: {e}")
            data.append({
                "ticker": ticker,
                "price": None,
                "prev_close": None,
                "pct_change": None,
                "volume": None,
                "currency": "N/A",
            })
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            data.append({
                "ticker": ticker,
                "price": None,
                "prev_close": None,
                "pct_change": None,
                "volume": None,
                "currency": "N/A",
            })
        
        # Rate limiting: Small delay between requests to avoid hitting rate limits too quickly
        # The main rate limiting is handled by checking call history above
        if len(tickers) > 1 and t != tickers[-1]:
            # Adjust delay based on rate limit - faster for premium tiers
            if ALPHA_VANTAGE_REQUESTS_PER_MINUTE >= 75:
                time.sleep(0.2)  # Premium tier - minimal delay
            else:
                time.sleep(0.5)  # Free tier - slightly longer delay
    
    return pd.DataFrame(data)


def generate_demo_data(tickers: List[str]) -> pd.DataFrame:
    """Generate demo/mock data with meaningful % moves (many ≥10%) for slab charts."""
    import random

    if 'demo_base_prices' not in st.session_state:
        st.session_state.demo_base_prices = {
            "AAPL": 259.0, "MSFT": 479.0, "GOOGL": 328.0, "AMZN": 247.0,
            "META": 653.0, "NVDA": 875.0, "TSLA": 445.0, "NFLX": 89.0,
            "JPM": 329.0, "BAC": 58.0, "WFC": 96.0, "GS": 485.0,
            "JNJ": 168.0, "PFE": 28.0, "UNH": 545.0, "ABBV": 178.0,
            "WMT": 114.0, "HD": 385.0, "MCD": 295.0, "NKE": 95.0,
            "XOM": 118.0, "CVX": 152.0, "SPY": 580.0, "QQQ": 485.0,
        }
    if 'demo_prev_prices' not in st.session_state:
        st.session_state.demo_prev_prices = {}

    data = []
    for t in tickers:
        ticker = t.upper()
        if ticker not in st.session_state.demo_base_prices:
            st.session_state.demo_base_prices[ticker] = random.uniform(50, 500)
        base_price = st.session_state.demo_base_prices[ticker]
        if ticker in st.session_state.demo_prev_prices:
            prev_price = st.session_state.demo_prev_prices[ticker]
        else:
            prev_price = base_price
            st.session_state.demo_prev_prices[ticker] = prev_price

        # Larger fluctuations so many stocks have ≥10% change (for slab visibility)
        fluctuation = random.uniform(-0.18, 0.18)
        current_price = prev_price * (1 + fluctuation)
        st.session_state.demo_prev_prices[ticker] = current_price
        pct_change = ((current_price - base_price) / base_price) * 100
        if abs(pct_change) > 25:
            current_price = base_price * (1 + random.uniform(-0.22, 0.22))
            st.session_state.demo_prev_prices[ticker] = current_price
            pct_change = ((current_price - base_price) / base_price) * 100

        data.append({
            "ticker": ticker,
            "price": round(current_price, 2),
            "prev_close": round(base_price, 2),
            "pct_change": round(pct_change, 2),
            "volume": random.randint(5000000, 100000000),
            "currency": "USD",
        })
    return pd.DataFrame(data)


# Slab order (bottom to top in the single stacked bar)
SLAB_ORDER = [
    "Down >2%",
    "Down 0–2%",
    "Flat",
    "Up 0–2%",
    "Up 2–5%",
    "Up 5–10%",
    "Up 10–20%",
    "Up 20%+",
]

# Distinct color per slab (like reference: red down, orange/yellow/green up)
SLAB_COLORS = {
    "Down >2%": "#dc2626",
    "Down 0–2%": "#ef4444",
    "Flat": "#94a3b8",
    "Up 0–2%": "#22c55e",
    "Up 2–5%": "#4ade80",
    "Up 5–10%": "#facc15",
    "Up 10–20%": "#fb923c",
    "Up 20%+": "#f97316",
}


def get_slab(pct_change: float) -> str:
    """Assign slab label from percentage change."""
    if pd.isna(pct_change):
        return "Flat"
    if pct_change < -2:
        return "Down >2%"
    if pct_change < 0:
        return "Down 0–2%"
    if pct_change == 0:
        return "Flat"
    if pct_change <= 2:
        return "Up 0–2%"
    if pct_change <= 5:
        return "Up 2–5%"
    if pct_change <= 10:
        return "Up 5–10%"
    if pct_change <= 20:
        return "Up 10–20%"
    return "Up 20%+"


def plot_stacked_by_slab(
    quotes_df: pd.DataFrame,
    height: Optional[int] = None,
    compact: bool = False,
    title: Optional[str] = None,
) -> None:
    """One vertical bar: all stocks in one bar, stacked by slab (percentage range), each slab a different color."""
    if quotes_df.empty:
        st.warning("No data to plot")
        return

    if "pct_change" not in quotes_df.columns or quotes_df["pct_change"].isna().all():
        st.warning("No percentage change data available for selected stocks")
        return

    valid_data = quotes_df[~quotes_df["pct_change"].isna()].copy()
    if valid_data.empty:
        st.warning("No valid percentage change data after filtering")
        return

    # Only show stocks with at least 10% change (abs)
    valid_data = valid_data[valid_data["pct_change"].abs() >= 10].copy()
    if valid_data.empty:
        st.caption("No stocks with ≥10% change")
        return

    valid_data["slab"] = valid_data["pct_change"].apply(get_slab)
    valid_data["price"] = valid_data["price"].fillna(pd.NA)

    bar_label = "Stocks"
    fig = go.Figure()

    for slab in SLAB_ORDER:
        slab_df = valid_data[valid_data["slab"] == slab]
        if slab_df.empty:
            continue
        count = len(slab_df)
        tickers_in_slab = ", ".join(slab_df["ticker"].tolist())
        hover_lines = [f"<b>{slab}</b>", f"Stocks: {tickers_in_slab}", f"Count: {count}"]
        for _, row in slab_df.iterrows():
            price_str = f"${row['price']:.2f}" if pd.notna(row["price"]) else "N/A"
            hover_lines.append(f"{row['ticker']}: {row['pct_change']:+.2f}% · {price_str}")
        hover_text = "<br>".join(hover_lines)
        text_size = 8 if compact else 10
        fig.add_trace(
            go.Bar(
                x=[bar_label],
                y=[count],
                name=slab,
                marker=dict(
                    color=SLAB_COLORS.get(slab, "#94a3b8"),
                    line=dict(color="rgba(0,0,0,0.25)", width=1),
                ),
                text=f"{slab}<br>{tickers_in_slab}" if count <= 4 else f"{slab}<br>{tickers_in_slab}",
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(size=text_size),
                hovertemplate=hover_text + "<extra></extra>",
            )
        )

    if height is None:
        height = 180 if compact else 420
    chart_title = title if title else "Stock Performance by Slab - Daily Change (%)"
    # Use native int/float for layout to avoid Plotly serialization errors
    margin_l = 32 if compact else 60
    margin_r = 12 if compact else 40
    margin_t = 28 if compact else 50
    margin_b = 24 if compact else 60
    fig.update_layout(
        barmode="stack",
        title=dict(
            text=str(chart_title),
            font=dict(size=int(9 if compact else 12), family="Arial"),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            title="",
            showgrid=False,
            zeroline=False,
            tickvals=[bar_label],
            ticktext=[bar_label],
            tickfont=dict(size=int(8 if compact else 11)),
        ),
        yaxis=dict(
            title="Count" if compact else "Stocks (count)",
            showgrid=True,
            gridcolor="rgba(128,128,128,0.3)",
            zeroline=False,
            titlefont=dict(size=int(8 if compact else 11)),
            tickfont=dict(size=int(7 if compact else 10)),
        ),
        height=int(height),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=int(margin_l), r=int(margin_r), t=int(margin_t), b=int(margin_b)),
        showlegend=not compact,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=7)),
        bargap=0.3,
    )
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Stock Dashboard", layout="wide", initial_sidebar_state="collapsed")
    
    st.title("Stock Dashboard")
    
    # Initialize auto-refresh state
    if 'auto_refresh_enabled' not in st.session_state:
        st.session_state.auto_refresh_enabled = False
    if 'refresh_interval' not in st.session_state:
        st.session_state.refresh_interval = 1
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = None
    if 'update_count' not in st.session_state:
        st.session_state.update_count = 0
    # Auto-refresh will be handled by streamlit-autorefresh component below
    
    # Sidebar controls
    st.sidebar.header("⚙️ Settings")
    
    # Demo mode toggle
    use_demo = st.sidebar.checkbox("🧪 Use Demo Data (for testing)", value=True, help="Enable to use mock data instead of real API calls.")
    
    # Auto-refresh controls
    st.sidebar.divider()
    st.sidebar.subheader("🔄 Real-time Monitoring")
    auto_refresh = st.sidebar.checkbox(
        "Enable Auto-Refresh", 
        value=st.session_state.auto_refresh_enabled,
        key="auto_refresh_checkbox",
        help="Automatically refresh stock data at specified intervals"
    )
    st.session_state.auto_refresh_enabled = auto_refresh
    
    if auto_refresh:
        refresh_interval = st.sidebar.slider(
            "Refresh Interval (seconds)",
            min_value=1,
            max_value=300,
            value=st.session_state.refresh_interval,
            step=1,
            key="refresh_interval_slider",
            help="How often to refresh stock data (minimum 1 second)"
        )
        st.session_state.refresh_interval = refresh_interval
    
    # Popular stocks list
    POPULAR_STOCKS = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX",
        "JPM", "BAC", "WFC", "GS", "MS", "C", "V", "MA",
        "JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT", "LLY",
        "WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "COST", "LOW",
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "VLO", "PSX",
        "BA", "CAT", "GE", "HON", "LMT", "RTX", "DE", "EMR",
        "SPY", "QQQ", "DIA", "IWM", "VTI", "VOO", "VEA", "VWO",
        "GBTC", "ETHE", "BITO", "COIN",
        "ASML", "TSM", "NVO", "SAP", "UL", "BP", "SHEL", "TM",
        "DIS", "NFLX", "CMCSA", "VZ", "T", "TMUS",
        "PG", "KO", "PEP", "CL", "UL", "MDT",
        "ORCL", "CRM", "ADBE", "INTC", "AMD", "QCOM",
    ]
    
    if 'all_used_stocks_set' not in st.session_state:
        st.session_state.all_used_stocks_set = set()

    current_used_stocks = set()
    for graph_idx, graph in enumerate(st.session_state.graphs):
        key = f"graph_stocks_{graph_idx}"
        current_used_stocks.update(st.session_state.get(key, graph.get("stocks", [])))
    new_stocks = current_used_stocks - st.session_state.all_used_stocks_set
    if new_stocks:
        st.session_state.all_used_stocks_set.update(new_stocks)
        st.session_state.all_available_stocks_cached = sorted(list(set(POPULAR_STOCKS + list(st.session_state.all_used_stocks_set))))
    if 'all_available_stocks_cached' not in st.session_state:
        st.session_state.all_available_stocks_cached = sorted(list(set(POPULAR_STOCKS + list(st.session_state.all_used_stocks_set))))
    all_available_stocks = st.session_state.all_available_stocks_cached
    
    # Status bar - only update when data actually changes to prevent flickering
    status_col1, status_col2, status_col3 = st.columns([2, 2, 1])
    with status_col1:
        if st.session_state.last_update_time:
            update_time_str = st.session_state.last_update_time.strftime("%H:%M:%S")
            st.caption(f"🟢 Last update: {update_time_str}")
        else:
            st.caption("⚪ Waiting for first update...")
    
    with status_col2:
        if st.session_state.auto_refresh_enabled:
            st.caption(f"🔄 Auto-refresh: Every {st.session_state.refresh_interval}s")
        else:
            st.caption("⏸️ Auto-refresh: Disabled")
    
    with status_col3:
        st.caption(f"📊 Updates: {st.session_state.update_count}")
    
    # Controls section - always visible
    col_add, col_refresh, col_clear = st.columns([1, 1, 1])
    with col_add:
        if st.button("+ Add Graph", use_container_width=True):
            st.session_state.graphs.append({"stocks": []})
            st.rerun()
    with col_refresh:
        if st.button("🔄 Refresh Data", use_container_width=True):
            # Clear cache for background refresh (non-blocking)
            fetch_current_quotes.clear()
            # Clear cached quotes to force fresh fetch
            st.session_state.cached_quotes = pd.DataFrame()
            st.session_state.last_update_time = datetime.now()
            st.session_state.update_count += 1
            st.rerun()
    
    with col_clear:
        if st.button("🗑️ Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache cleared!")
            st.rerun()
    
    all_stocks_in_graphs = set()
    for graph_idx, graph in enumerate(st.session_state.graphs):
        key = f"graph_stocks_{graph_idx}"
        all_stocks_in_graphs.update(st.session_state.get(key, graph.get("stocks", [])))
    
    # Initialize valid_quotes as empty DataFrame
    valid_quotes = pd.DataFrame()
    
    # Track which stocks are being loaded
    if 'loading_stocks' not in st.session_state:
        st.session_state.loading_stocks = set()
    if 'cached_quotes' not in st.session_state:
        st.session_state.cached_quotes = pd.DataFrame()
    
    # Track last fetched stocks to prevent unnecessary refetches
    if 'last_fetched_stocks' not in st.session_state:
        st.session_state.last_fetched_stocks = set()
    
    # Fetch data only for stocks that are in graphs
    if all_stocks_in_graphs:
        stocks_to_fetch = sorted(list(all_stocks_in_graphs))
        
        # Check which stocks are new (not in cached data)
        cached_tickers = set(st.session_state.cached_quotes['ticker'].tolist()) if not st.session_state.cached_quotes.empty else set()
        new_stocks = set(stocks_to_fetch) - cached_tickers
        existing_stocks = set(stocks_to_fetch) & cached_tickers
        
        # Show existing cached data immediately (non-blocking)
        if not st.session_state.cached_quotes.empty and existing_stocks:
            existing_quotes = st.session_state.cached_quotes[
                st.session_state.cached_quotes['ticker'].isin(existing_stocks)
            ].copy()
            valid_quotes = existing_quotes.copy()
        else:
            valid_quotes = pd.DataFrame()
        
        # Fetch data in background (non-blocking) - only if stocks changed or cache expired
        stocks_set = set(stocks_to_fetch)
        
        # Determine if we need to fetch data
        needs_fetch = False
        if st.session_state.cached_quotes.empty:
            # Always fetch if no cached data
            needs_fetch = True
        elif stocks_set != st.session_state.last_fetched_stocks:
            # Fetch if stocks changed
            needs_fetch = True
        elif (st.session_state.auto_refresh_enabled and 
              st.session_state.last_update_time and
              (datetime.now() - st.session_state.last_update_time).total_seconds() >= st.session_state.refresh_interval):
            # Fetch if auto-refresh interval passed
            needs_fetch = True
        
        if stocks_to_fetch and needs_fetch:
            try:
                logger.info(f"Starting data fetch for stocks: {stocks_to_fetch}")
                # Fetch current data (this happens in background, UI remains responsive)
                if use_demo:
                    quotes = generate_demo_data(stocks_to_fetch)
                    logger.info("Using demo data")
                else:
                    quotes = fetch_current_quotes(stocks_to_fetch)
                    logger.info(f"Fetched {len(quotes)} quotes, {len(quotes[~quotes['price'].isna()])} with valid prices")
                
                # Update cached quotes
                st.session_state.cached_quotes = quotes.copy()
                st.session_state.last_fetched_stocks = stocks_set
                
                # Update timestamp and count
                st.session_state.last_update_time = datetime.now()
                st.session_state.update_count += 1
                
                # Get valid quotes
                valid_quotes = quotes[~quotes["price"].isna()].copy()
                
                st.session_state.loading_stocks = set()
            except Exception as e:
                error_msg = f"Error fetching data: {str(e)}"
                logger.error(error_msg)
                # Show error to user
                st.error(f"⚠️ {error_msg}")
                # Keep existing cached data if available
                if valid_quotes.empty and not st.session_state.cached_quotes.empty:
                    valid_quotes = st.session_state.cached_quotes[
                        st.session_state.cached_quotes['ticker'].isin(stocks_to_fetch)
                    ].copy()
        elif not st.session_state.cached_quotes.empty:
            # Use cached data if available and no fetch needed
            valid_quotes = st.session_state.cached_quotes[
                st.session_state.cached_quotes['ticker'].isin(stocks_to_fetch)
            ].copy()
        else:
            # No data available
            valid_quotes = pd.DataFrame()
    else:
        valid_quotes = pd.DataFrame()
    
    # Store for use inside fragments (fragment reruns don't re-execute main)
    st.session_state._valid_quotes = valid_quotes
    st.session_state._all_available_stocks = all_available_stocks
    st.session_state._all_stocks_in_graphs = all_stocks_in_graphs
    st.session_state._needs_fetch = needs_fetch if 'needs_fetch' in dir() else False

    # Compact grid: 4 per row, small charts + expander for options so 3+ rows fit in viewport
    st.markdown(
        "<style>div[data-testid='stExpander'] > div { padding-top: 0.25rem; padding-bottom: 0.25rem; } "
        ".stButton > button { padding: 0.2rem 0.5rem; font-size: 0.8rem; }</style>",
        unsafe_allow_html=True,
    )
    st.subheader("📈 Stock performance by slab")
    GRAPHS_PER_ROW = 4
    COMPACT_CHART_HEIGHT = 180
    num_graphs = len(st.session_state.graphs)
    num_rows = (num_graphs + GRAPHS_PER_ROW - 1) // GRAPHS_PER_ROW

    for row in range(num_rows):
        cols = st.columns(GRAPHS_PER_ROW)
        for col_idx in range(GRAPHS_PER_ROW):
            graph_idx = row * GRAPHS_PER_ROW + col_idx
            if graph_idx >= num_graphs:
                break
            with cols[col_idx]:
                multiselect_key = f"graph_stocks_{graph_idx}"
                graph = st.session_state.graphs[graph_idx]
                applied = st.session_state.get(multiselect_key, graph.get("stocks", []))
                n_stocks = len(applied)
                stocks_in_other = set()
                for oi, og in enumerate(st.session_state.graphs):
                    if oi != graph_idx:
                        stocks_in_other.update(st.session_state.get(f"graph_stocks_{oi}", og.get("stocks", [])))
                available = [s for s in all_available_stocks if s not in stocks_in_other or s in applied]
                for s in applied:
                    if s not in available:
                        available.append(s)
                available = sorted(available)

                # One short row: expander "G1 (n)" + tiny Delete
                head_col, del_col = st.columns([1, 0.06])
                with head_col:
                    with st.expander(f"**G{graph_idx + 1}** ({n_stocks})", expanded=False):
                        with st.form(key=f"graph_form_{graph_idx}", clear_on_submit=False):
                            sel = st.multiselect(
                                "Stocks",
                                options=available,
                                default=applied,
                                key=f"graph_multiselect_{graph_idx}",
                                label_visibility="collapsed",
                            )
                            if st.form_submit_button("Update"):
                                st.session_state[multiselect_key] = list(sel)
                                st.session_state.graphs[graph_idx]["stocks"] = list(sel)
                                st.rerun()
                with del_col:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("✕", key=f"delete_{graph_idx}", help="Delete"):
                        if multiselect_key in st.session_state:
                            del st.session_state[multiselect_key]
                        st.session_state.graphs.pop(graph_idx)
                        st.rerun()

                graph_stocks = st.session_state.get(multiselect_key, [])
                if graph_stocks:
                    if valid_quotes.empty:
                        if st.session_state.get("_needs_fetch", False) and all_stocks_in_graphs:
                            st.caption("⏳ Fetching...")
                        else:
                            st.caption("⚠️ No data. Refresh or Update.")
                    else:
                        gq = valid_quotes[valid_quotes["ticker"].isin(graph_stocks)].copy()
                        if gq.empty:
                            st.caption("⚠️ No data for selected")
                        else:
                            plot_stacked_by_slab(
                                gq,
                                height=COMPACT_CHART_HEIGHT,
                                compact=True,
                                title=f"G{graph_idx + 1}",
                            )
                else:
                    st.caption("Select stocks → Update")

    if all_stocks_in_graphs and not valid_quotes.empty:
        with st.expander("📊 View Raw Data", expanded=False):
            display_quotes = valid_quotes.copy()
            display_quotes["price"] = display_quotes["price"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
            display_quotes["pct_change"] = display_quotes["pct_change"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
            display_quotes["volume"] = display_quotes["volume"].apply(lambda x: f"{x/1e6:.2f}M" if pd.notna(x) and x >= 1e6 else f"{x/1e3:.2f}K" if pd.notna(x) and x >= 1e3 else "N/A" if pd.isna(x) else f"{x:.0f}")
            st.dataframe(display_quotes.set_index("ticker"), use_container_width=True)
    
    # Auto-refresh mechanism using streamlit-autorefresh (reliable method)
    if st.session_state.auto_refresh_enabled and all_stocks_in_graphs:
        if AUTOREFRESH_AVAILABLE:
            # Use streamlit-autorefresh component (most reliable)
            refresh_count = st_autorefresh(
                interval=st.session_state.refresh_interval * 1000,  # Convert to milliseconds
                limit=None,  # Unlimited refreshes
                key="stock_dashboard_autorefresh"
            )
            
            # Background refresh - clear cache and update in background (non-blocking)
            # Only clear cache if we haven't just refreshed (prevent unnecessary reruns)
            if 'last_autorefresh_time' not in st.session_state:
                st.session_state.last_autorefresh_time = None
            
            current_time = datetime.now()
            if (st.session_state.last_autorefresh_time is None or 
                (current_time - st.session_state.last_autorefresh_time).total_seconds() >= st.session_state.refresh_interval):
                fetch_current_quotes.clear()
                st.session_state.last_autorefresh_time = current_time
        else:
            # Fallback: manual refresh reminder
            st.sidebar.warning("⚠️ Auto-refresh requires: `pip install streamlit-autorefresh`")
            if st.sidebar.button("🔄 Refresh Now"):
                fetch_current_quotes.clear()
                st.session_state.last_update_time = datetime.now()
                st.session_state.update_count += 1
                st.rerun()


if __name__ == "__main__":
    main()
