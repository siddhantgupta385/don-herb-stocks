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
    st.session_state.graphs = [{"stocks": []}]  # Start with one graph by default


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
    """Generate demo/mock data for testing when API is rate limited.
    Simulates real-time fluctuations with random variations."""
    import random
    
    # Initialize base prices in session state if not exists (for consistency across refreshes)
    if 'demo_base_prices' not in st.session_state:
        st.session_state.demo_base_prices = {
            "AAPL": 259.0, "MSFT": 479.0, "GOOGL": 328.0, "AMZN": 247.0,
            "META": 653.0, "NVDA": 875.0, "TSLA": 445.0, "NFLX": 89.0,
            "JPM": 329.0, "BAC": 58.0, "WFC": 96.0, "GS": 485.0,
            "JNJ": 168.0, "PFE": 28.0, "UNH": 545.0, "ABBV": 178.0,
            "WMT": 114.0, "HD": 385.0, "MCD": 295.0, "NKE": 95.0,
            "XOM": 118.0, "CVX": 152.0, "SPY": 580.0, "QQQ": 485.0,
        }
    
    # Initialize previous prices tracking for realistic fluctuations
    if 'demo_prev_prices' not in st.session_state:
        st.session_state.demo_prev_prices = {}
    
    data = []
    
    for t in tickers:
        ticker = t.upper()
        
        # Get or initialize base price
        if ticker not in st.session_state.demo_base_prices:
            st.session_state.demo_base_prices[ticker] = random.uniform(50, 500)
        
        base_price = st.session_state.demo_base_prices[ticker]
        
        # Get previous price (for realistic fluctuations)
        if ticker in st.session_state.demo_prev_prices:
            prev_price = st.session_state.demo_prev_prices[ticker]
        else:
            prev_price = base_price
            st.session_state.demo_prev_prices[ticker] = prev_price
        
        # Simulate real-time fluctuation: small random change from previous price
        # This creates more realistic "live" updates
        fluctuation = random.uniform(-0.015, 0.015)  # -1.5% to +1.5% per update
        current_price = prev_price * (1 + fluctuation)
        
        # Update previous price for next refresh
        st.session_state.demo_prev_prices[ticker] = current_price
        
        # Calculate percentage change from base (previous close)
        pct_change = ((current_price - base_price) / base_price) * 100
        
        # Ensure price doesn't drift too far from base
        if abs(pct_change) > 5:
            current_price = base_price * (1 + random.uniform(-0.05, 0.05))
            st.session_state.demo_prev_prices[ticker] = current_price
            pct_change = ((current_price - base_price) / base_price) * 100
        
        volume = random.randint(5000000, 100000000)
        
        data.append({
            "ticker": ticker,
            "price": round(current_price, 2),
            "prev_close": round(base_price, 2),
            "pct_change": round(pct_change, 2),
            "volume": volume,
            "currency": "USD",
        })
    
    return pd.DataFrame(data)


def display_monitoring_table(quotes_df: pd.DataFrame) -> None:
    """Display a compact monitoring table with real-time stock data and visual indicators."""
    if quotes_df.empty:
        return
    
    # Sort by percentage change (descending) for better monitoring
    display_df = quotes_df.copy()
    display_df = display_df.sort_values('pct_change', ascending=False)
    
    # Create styled dataframe for display
    st.subheader("📊 Real-time Stock Monitor")
    
    # Store previous prices for change detection (for visual indicators)
    if 'previous_prices' not in st.session_state:
        st.session_state.previous_prices = {}
    
    # Prepare display data with enhanced formatting
    monitor_data = []
    for _, row in display_df.iterrows():
        ticker = row['ticker']
        price = row['price']
        pct_change = row['pct_change']
        prev_close = row['prev_close']
        volume = row['volume']
        
        # Track price changes for visual indicators
        price_change_indicator = ""
        if ticker in st.session_state.previous_prices:
            prev_price = st.session_state.previous_prices[ticker]
            if pd.notna(price) and pd.notna(prev_price):
                if price > prev_price:
                    price_change_indicator = "📈"  # Rising
                elif price < prev_price:
                    price_change_indicator = "📉"  # Falling
                else:
                    price_change_indicator = "➡️"  # Stable
        st.session_state.previous_prices[ticker] = price
        
        # Format price with change indicator
        if pd.notna(price):
            price_str = f"{price_change_indicator} ${price:.2f}"
        else:
            price_str = "N/A"
        
        # Format percentage change with enhanced visual indicators
        if pd.notna(pct_change):
            if pct_change > 2:
                pct_str = f"🟢 🔼 {pct_change:+.2f}%"
                pct_style = "color: green; font-weight: bold;"
            elif pct_change > 0:
                pct_str = f"🟢 {pct_change:+.2f}%"
                pct_style = "color: green;"
            elif pct_change < -2:
                pct_str = f"🔴 🔽 {pct_change:+.2f}%"
                pct_style = "color: red; font-weight: bold;"
            elif pct_change < 0:
                pct_str = f"🔴 {pct_change:+.2f}%"
                pct_style = "color: red;"
            else:
                pct_str = f"⚪ {pct_change:.2f}%"
                pct_style = "color: gray;"
        else:
            pct_str = "N/A"
            pct_style = ""
        
        # Format previous close
        prev_close_str = f"${prev_close:.2f}" if pd.notna(prev_close) else "N/A"
        
        # Format volume
        if pd.notna(volume):
            if volume >= 1e6:
                volume_str = f"{volume/1e6:.2f}M"
            elif volume >= 1e3:
                volume_str = f"{volume/1e3:.2f}K"
            else:
                volume_str = f"{volume:.0f}"
        else:
            volume_str = "N/A"
        
        monitor_data.append({
            "Ticker": ticker,
            "Price": price_str,
            "Change %": pct_str,
            "Prev Close": prev_close_str,
            "Volume": volume_str,
            "_sort_pct": pct_change if pd.notna(pct_change) else -999,
            "_style": pct_style
        })
    
    # Create DataFrame and display
    monitor_df = pd.DataFrame(monitor_data)
    monitor_df = monitor_df.sort_values('_sort_pct', ascending=False)
    monitor_df = monitor_df.drop(columns=['_sort_pct', '_style'])
    
    # Display as a styled table with better formatting
    st.dataframe(
        monitor_df.set_index("Ticker"),
        use_container_width=True,
        height=min(500, len(monitor_df) * 40 + 50),  # Dynamic height
        hide_index=False
    )
    
    # Show update timestamp
    if st.session_state.last_update_time:
        update_time = st.session_state.last_update_time.strftime("%H:%M:%S")
        st.caption(f"🕐 Last updated: {update_time} | Total stocks: {len(monitor_df)}")
    
    st.divider()


def plot_vertical_bar(quotes_df: pd.DataFrame, metric: str, title: str, height: Optional[int] = None) -> None:
    """Plot a normal vertical bar chart: one bar per stock, side by side."""
    if quotes_df.empty:
        st.warning(f"No data to plot for {title}")
        return
    
    if "pct_change" not in quotes_df.columns or quotes_df["pct_change"].isna().all():
        st.warning("No percentage change data available for selected stocks")
        return
    
    valid_data = quotes_df[~quotes_df["pct_change"].isna()].copy()
    if valid_data.empty:
        st.warning("No valid percentage change data after filtering")
        return
    
    valid_data["price"] = valid_data["price"].fillna(pd.NA)
    valid_data = valid_data.sort_values("pct_change", ascending=True)
    
    tickers = valid_data["ticker"].tolist()
    pct_changes = valid_data["pct_change"].tolist()
    prices = valid_data["price"].tolist()
    
    colors = []
    for pct in pct_changes:
        if pct > 0:
            colors.append("#22c55e")
        elif pct < 0:
            colors.append("#ef4444")
        else:
            colors.append("#94a3b8")
    
    hover_texts = []
    for i, row in valid_data.iterrows():
        pct = row["pct_change"]
        price = row["price"]
        price_str = f"${price:.2f}" if pd.notna(price) else "N/A"
        hover_texts.append(f"{row['ticker']}<br>Change: {pct:+.2f}%<br>Price: {price_str}")
    
    fig = go.Figure()
    # Bar width = half of previous (0.28 -> 0.14); ~5px gap between bars (chart width ~460px)
    n_bars = len(tickers)
    gap_px = 5
    chart_width_px = 460
    # For ~5px gap: bar width = 1 - (n_bars-1)*gap_px/chart_width_px. Min 0.14 (half of previous 0.28).
    bar_width_5px = max(0.1, 1 - (n_bars - 1) * gap_px / chart_width_px) if n_bars >= 1 else 0.14
    bar_width = max(0.14, bar_width_5px)  # at least half-width; wider when needed for 5px gap
    fig.add_trace(go.Bar(
        x=tickers,
        y=pct_changes,
        width=bar_width,
        marker=dict(
            color=colors,
            line=dict(color="rgba(0,0,0,0.3)", width=1),
        ),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_texts,
        text=[f"{p:+.2f}%" for p in pct_changes],
        textposition="outside",
        textfont=dict(size=10),
    ))
    
    if height is None:
        chart_height = 180
    else:
        chart_height = height
    
    y_min = min(pct_changes) if pct_changes else -5
    y_max = max(pct_changes) if pct_changes else 5
    # Shorter y scale: less padding so bars have better length (tighter range)
    span = y_max - y_min if y_max != y_min else 1
    y_pad = max(0.3, span * 0.08)  # was 0.15, now 0.08 for shorter scale
    y_range = [y_min - y_pad, y_max + y_pad]
    if 0 < y_min or 0 > y_max:
        y_range = [min(y_range[0], 0), max(y_range[1], 0)]
    
    fig.update_layout(
        title=dict(
            text="Stock Performance - Daily Change (%)",
            font=dict(size=12, family="Arial"),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            title="",
            tickangle=-45,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="Change (%)",
            showgrid=True,
            gridcolor="rgba(128,128,128,0.3)",
            zeroline=True,
            zerolinecolor="rgba(0,0,0,0.5)",
            zerolinewidth=1,
            range=y_range,
        ),
        height=chart_height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=40, t=50, b=80),
        showlegend=False,
        bargap=0.05,
        bargroupgap=0,
    )
    
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Stock Dashboard", layout="wide")
    
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
    use_demo = st.sidebar.checkbox("🧪 Use Demo Data (for testing)", value=False, help="Enable to use mock data instead of real API calls. Perfect for development when rate limited!")
    
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
    
    # Track all stocks that have been used in any graph (for dropdown options)
    # Use a stable set that persists across reruns to keep options list stable
    if 'all_used_stocks_set' not in st.session_state:
        st.session_state.all_used_stocks_set = set()
    
    # Get current selections from all graphs
    current_used_stocks = set()
    for graph_idx, graph in enumerate(st.session_state.graphs):
        multiselect_key = f"graph_stocks_{graph_idx}"
        if multiselect_key in st.session_state:
            current_used_stocks.update(st.session_state[multiselect_key])
        else:
            current_used_stocks.update(graph.get("stocks", []))
    
    # Update persistent set (union to keep all previously used stocks)
    # Check if there are new stocks before updating cache
    new_stocks = current_used_stocks - st.session_state.all_used_stocks_set
    if new_stocks:
        st.session_state.all_used_stocks_set.update(new_stocks)
        # Rebuild cached list only when new stocks are added
        st.session_state.all_available_stocks_cached = sorted(list(set(POPULAR_STOCKS + list(st.session_state.all_used_stocks_set))))
    
    # Initialize cache if it doesn't exist
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
            st.session_state.graphs.append({
                "stocks": []
            })
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
    
    # Get all stocks that are in any graph (from session state for accuracy)
    all_stocks_in_graphs = set()
    for graph_idx, graph in enumerate(st.session_state.graphs):
        multiselect_key = f"graph_stocks_{graph_idx}"
        if multiselect_key in st.session_state:
            all_stocks_in_graphs.update(st.session_state[multiselect_key])
        else:
            all_stocks_in_graphs.update(graph.get("stocks", []))
    
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
    
    # Display monitoring table if we have data
    if not valid_quotes.empty:
        show_monitor = st.sidebar.checkbox(
            "📊 Show Monitoring Table",
            value=True,
            help="Display a compact table view of all stocks for easy monitoring"
        )
        if show_monitor:
            display_monitoring_table(valid_quotes)
    
    # Display graphs in a grid layout (2 columns for better viewport usage)
    if st.session_state.graphs:
        num_graphs = len(st.session_state.graphs)
        graphs_per_row = 2
        num_rows = (num_graphs + graphs_per_row - 1) // graphs_per_row
        
        # Display each graph in grid
        for row in range(num_rows):
            cols = st.columns(graphs_per_row)
            for col_idx in range(graphs_per_row):
                graph_idx = row * graphs_per_row + col_idx
                if graph_idx < num_graphs:
                    graph = st.session_state.graphs[graph_idx]
                    with cols[col_idx]:
                        with st.container():
                            # Graph header with stock selection and delete button
                            col_stocks, col_delete = st.columns([3, 1])
                            
                            with col_stocks:
                                multiselect_key = f"graph_stocks_{graph_idx}"
                                
                                # Initialize session state for this multiselect if it doesn't exist
                                if multiselect_key not in st.session_state:
                                    st.session_state[multiselect_key] = graph.get("stocks", [])
                                
                                # Get the current value from session state (this is the source of truth)
                                current_selection = st.session_state[multiselect_key]
                                
                                # Get stocks selected in OTHER graphs (exclusive selection)
                                stocks_in_other_graphs = set()
                                for other_idx, other_graph in enumerate(st.session_state.graphs):
                                    if other_idx != graph_idx:
                                        other_key = f"graph_stocks_{other_idx}"
                                        if other_key in st.session_state:
                                            stocks_in_other_graphs.update(st.session_state[other_key])
                                        else:
                                            stocks_in_other_graphs.update(other_graph.get("stocks", []))
                                
                                # Filter available stocks to exclude those in other graphs
                                # But always include currently selected stocks (so they don't disappear from dropdown)
                                available_stocks_for_this_graph = [
                                    stock for stock in all_available_stocks 
                                    if stock not in stocks_in_other_graphs or stock in current_selection
                                ]
                                
                                # Ensure current selection is maintained (in case of any edge cases)
                                for stock in current_selection:
                                    if stock not in available_stocks_for_this_graph:
                                        available_stocks_for_this_graph.append(stock)
                                
                                # Sort to keep order consistent
                                available_stocks_for_this_graph = sorted(available_stocks_for_this_graph)
                                
                                graph_stocks = st.multiselect(
                                    f"Graph {graph_idx + 1} - Select stocks",
                                    options=available_stocks_for_this_graph,
                                    default=current_selection,
                                    key=multiselect_key
                                )
                                
                                # Update graph state to match session state (multiselect updates session state automatically)
                                graph["stocks"] = st.session_state[multiselect_key]
                            
                            with col_delete:
                                st.markdown("<br>", unsafe_allow_html=True)
                                if st.button("Delete", key=f"delete_{graph_idx}", use_container_width=True):
                                    # Clean up session state for this graph
                                    multiselect_key = f"graph_stocks_{graph_idx}"
                                    if multiselect_key in st.session_state:
                                        del st.session_state[multiselect_key]
                                    st.session_state.graphs.pop(graph_idx)
                                    st.rerun()
                            
                            # Plot graph - use container to prevent flickering
                            graph_container = st.container()
                            with graph_container:
                                if graph_stocks:
                                    if valid_quotes.empty:
                                        # Check if we're currently fetching
                                        if 'needs_fetch' in locals() and needs_fetch and all_stocks_in_graphs:
                                            st.info(f"⏳ Fetching data for {', '.join(graph_stocks)}... This may take 10-30 seconds.")
                                            st.caption("💡 Tip: If this takes too long, try enabling 'Use Demo Data' in the sidebar for testing.")
                                        else:
                                            st.warning(f"⚠️ No data available. Click 'Refresh Data' to fetch.")
                                    else:
                                        graph_quotes = valid_quotes[valid_quotes["ticker"].isin(graph_stocks)].copy()
                                        if graph_quotes.empty:
                                            st.warning(f"⚠️ No data available for selected stocks: {', '.join(graph_stocks)}")
                                        else:
                                            plot_vertical_bar(graph_quotes, "pct_change", f"Graph {graph_idx + 1} - Stock Performance", height=180)
                                else:
                                    st.info("👆 Select stocks above to display graph")
                            
                            st.divider()
        
        # Debug view (collapsed)
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
