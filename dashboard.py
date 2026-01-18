import logging
import time
from typing import Dict, List, Optional
from datetime import datetime
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    """Fetch current quote data (price, % change, volume) for a list of tickers.
    Works even when market is closed by using previous close data."""
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
    is_market_open, _ = check_market_status()
    
    for t in tickers:
        ticker = yf.Ticker(t)
        last_price = None
        prev_close = None
        volume = None
        currency = "USD"
        
        try:
            # Method 1: Try info dict first (most reliable, works when market closed)
            try:
                info = ticker.info
                # Get current/regular market price
                last_price = (
                    info.get('currentPrice') or 
                    info.get('regularMarketPrice') or 
                    info.get('previousClose')  # Use previous close if market closed
                )
                prev_close = (
                    info.get('previousClose') or 
                    info.get('regularMarketPreviousClose')
                )
                volume = info.get('volume') or info.get('regularMarketVolume') or info.get('regularMarketVolume')
                currency = info.get('currency', 'USD')
                
                # If market is closed and we don't have current price, use previous close
                if (last_price is None or pd.isna(last_price)) and prev_close is not None:
                    last_price = prev_close
                    
            except Exception as e:
                logger.debug(f"Info method failed for {t}: {e}")
            
            # Method 2: Try fast_info if info didn't work
            if last_price is None or pd.isna(last_price):
                try:
                    fast_info = ticker.fast_info
                    last_price = fast_info.last_price
                    prev_close = fast_info.previous_close
                    volume = fast_info.last_volume
                    if hasattr(fast_info, 'currency'):
                        currency = fast_info.currency
                    
                    # If market closed and no current price, use previous close
                    if (last_price is None or pd.isna(last_price)) and prev_close is not None:
                        last_price = prev_close
                        
                except Exception as e:
                    logger.debug(f"Fast_info method failed for {t}: {e}")
            
            # Method 3: Try history as last resort (slower, more likely to rate limit)
            if last_price is None or pd.isna(last_price):
                try:
                    # Use 5d period to get more reliable data
                    hist = ticker.history(period="5d", interval="1d")
                    if not hist.empty:
                        last_price = hist['Close'].iloc[-1]
                        if len(hist) > 1:
                            prev_close = hist['Close'].iloc[-2]
                        else:
                            prev_close = last_price
                        volume = hist['Volume'].iloc[-1] if 'Volume' in hist.columns else None
                except Exception as e:
                    logger.debug(f"History method failed for {t}: {e}")
            
            # If still no price but we have previous close, use it
            if (last_price is None or pd.isna(last_price)) and prev_close is not None:
                last_price = prev_close
            
            # Calculate percentage change
            if last_price is not None and not pd.isna(last_price) and prev_close is not None and not pd.isna(prev_close) and prev_close != 0:
                pct_change = ((last_price - prev_close) / prev_close) * 100
            else:
                pct_change = 0.0
            
            data.append({
                "ticker": t.upper(),
                "price": last_price,
                "prev_close": prev_close,
                "pct_change": pct_change,
                "volume": volume,
                "currency": currency,
            })
            
        except YFRateLimitError as e:
            logger.warning(f"Rate limited for {t}")
            # Even if rate limited, try to get at least previous close from cached data
            try:
                info = ticker.info
                prev_close = info.get('previousClose')
                if prev_close:
                    data.append({
                        "ticker": t.upper(),
                        "price": prev_close,  # Use previous close as fallback
                        "prev_close": prev_close,
                        "pct_change": 0.0,
                        "volume": None,
                        "currency": info.get('currency', 'USD'),
                    })
                else:
                    raise
            except:
                mark_rate_limited()
                data.append({
                    "ticker": t.upper(),
                    "price": None,
                    "prev_close": None,
                    "pct_change": None,
                    "volume": None,
                    "currency": "N/A",
                })
        except Exception as e:
            logger.error(f"Error fetching data for {t}: {e}")
            data.append({
                "ticker": t.upper(),
                "price": None,
                "prev_close": None,
                "pct_change": None,
                "volume": None,
                "currency": "N/A",
            })
        
        # Optimized delay between requests - reduced for faster updates
        # Only delay if we have multiple tickers to avoid rate limiting
        if len(tickers) > 1 and t != tickers[-1]:
            time.sleep(1)  # Reduced from 3s to 1s for faster updates
    
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
    """Plot a segmented vertical bar chart with stocks grouped by percentage change ranges."""
    if quotes_df.empty:
        st.warning(f"No data to plot for {title}")
        return
    
    # Always use percentage change for the main visualization
    if "pct_change" not in quotes_df.columns or quotes_df["pct_change"].isna().all():
        st.warning("No percentage change data available for selected stocks")
        return
    
    valid_data = quotes_df[~quotes_df["pct_change"].isna()].copy()
    if valid_data.empty:
        st.warning("No valid percentage change data after filtering")
        return
    
    # Define percentage change ranges (slabs)
    ranges = [
        (float('-inf'), -5, "-5%+", '#8B0000'),      # Dark red
        (-5, -2, "-2% to -5%", '#DC143C'),           # Crimson
        (-2, 0, "0% to -2%", '#FF6347'),            # Tomato
        (0, 1, "+0% to +1%", '#FFD700'),            # Gold
        (1, 2, "+1% to +2%", '#FFA500'),            # Orange
        (2, 5, "+2% to +5%", '#32CD32'),            # Lime green
        (5, float('inf'), "+5%+", '#00FF00'),        # Bright green
    ]
    
    # Group stocks by percentage change ranges
    range_groups = {range_label: [] for _, _, range_label, _ in ranges}
    
    for _, row in valid_data.iterrows():
        pct = row["pct_change"]
        ticker = row["ticker"]
        price = row["price"]
        
        # Find which range this stock belongs to
        for min_pct, max_pct, range_label, _ in ranges:
            if min_pct == float('-inf'):
                if pct < max_pct:
                    range_groups[range_label].append({"ticker": ticker, "pct_change": pct, "price": price})
                    break
            elif max_pct == float('inf'):
                if pct >= min_pct:
                    range_groups[range_label].append({"ticker": ticker, "pct_change": pct, "price": price})
                    break
            else:
                if min_pct <= pct < max_pct:
                    range_groups[range_label].append({"ticker": ticker, "pct_change": pct, "price": price})
                    break
    
    # Create stacked vertical bar - all in same line
    fig = go.Figure()
    
    y_bottom = 0
    annotations = []
    
    # Build stacked segments from bottom to top - all at x=0 (same vertical line)
    for min_pct, max_pct, range_label, color in ranges:
        stocks_in_range = range_groups[range_label]
        if not stocks_in_range:
            continue
        
        # Calculate segment height (based on number of stocks, with min height)
        segment_height = max(5, len(stocks_in_range) * 2)  # At least 5 units, 2 per stock
        y_top = y_bottom + segment_height
        y_mid = (y_bottom + y_top) / 2
        
        # Create the segment - all at x=0 to keep in same vertical line
        fig.add_trace(go.Bar(
            x=[0],  # All segments at x=0 (same vertical line)
            y=[segment_height],
            base=[y_bottom],
            name=range_label,
            marker=dict(
                color=color,
                line=dict(color='black', width=2)
            ),
            hovertemplate=f"<b>{range_label}</b><br>" +
                         "<br>".join([f"{s['ticker']}: {s['pct_change']:.2f}% (${s['price']:.2f})" 
                                     for s in stocks_in_range]) +
                         "<extra></extra>",
            orientation='v',
            showlegend=False,
            width=0.25  # Narrower bar width
        ))
        
        # Add stock labels
        stock_labels = ", ".join([s['ticker'] for s in stocks_in_range])
        if len(stock_labels) > 30:
            # Split into multiple lines if too long
            tickers = [s['ticker'] for s in stocks_in_range]
            mid = len(tickers) // 2
            stock_labels = ", ".join(tickers[:mid]) + "<br>" + ", ".join(tickers[mid:])
        
        # Add annotations for stock names and range label
        annotations.append(dict(
            x=0.15,
            y=y_mid,
            text=stock_labels,
            showarrow=False,
            xref="paper",
            yref="y",
            font=dict(size=9, color='black'),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            align="left",
            valign="middle"
        ))
        
        annotations.append(dict(
            x=0.85,
            y=y_mid,
            text=range_label,
            showarrow=False,
            xref="paper",
            yref="y",
            font=dict(size=9, color='black', weight='bold'),
            align="right",
            valign="middle"
        ))
        
        # Add horizontal line at segment boundary
        if y_bottom > 0:
            fig.add_shape(
                type="line",
                x0=0.0, y0=y_bottom, x1=1.0, y1=y_bottom,
                line=dict(color="black", width=1.5, dash="dash"),
                xref="paper",
                yref="y"
            )
        
        y_bottom = y_top
    
    # Add upward arrow
    if y_bottom > 0:
        annotations.append(dict(
            x=0.5,
            y=y_bottom + 2,
            text="↑",
            showarrow=False,
            xref="paper",
            yref="y",
            font=dict(size=30, color='black'),
            align="center",
            valign="middle"
        ))
    
    # Calculate height: 20% of typical viewport (approximately 150px for 20% of 750px viewport)
    # Or use provided height parameter
    if height is None:
        # Default to 20% of screen - approximately 150-200px
        chart_height = 180
    else:
        chart_height = height
    
    fig.update_layout(
        title=dict(
            text="Stock Performance - Daily Change (%)",
            font=dict(size=12, family='Arial'),
            x=0.5,
            xanchor="center"
        ),
        xaxis=dict(
            title="",
            showticklabels=False,
            range=[-0.2, 0.2],  # Narrow range to keep bar centered
            showgrid=False,
            zeroline=False,
            fixedrange=True
        ),
        yaxis=dict(
            title="Percentage Change (%)",
            showgrid=True,
            gridcolor='rgba(128,128,128,0.3)',
            gridwidth=1,
            tickfont=dict(size=9),
        ),
        height=chart_height,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=80, r=80, t=50, b=50),
        showlegend=False,
        annotations=annotations,
        barmode='stack',  # Ensure proper stacking
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
    
    # Status bar - use empty containers to prevent flickering
    if 'status_containers' not in st.session_state:
        st.session_state.status_containers = {
            'status_col1': st.empty(),
            'status_col2': st.empty(),
            'status_col3': st.empty()
        }
    
    status_col1, status_col2, status_col3 = st.columns([2, 2, 1])
    with status_col1:
        status_placeholder1 = st.empty()
    with status_col2:
        status_placeholder2 = st.empty()
    with status_col3:
        status_placeholder3 = st.empty()
    
    # Update status bar only when data actually changes, not on every rerun
    if 'last_status_update' not in st.session_state or st.session_state.last_status_update != st.session_state.last_update_time:
        if st.session_state.last_update_time:
            time_ago = (datetime.now() - st.session_state.last_update_time).total_seconds()
            if time_ago < 60:
                status_placeholder1.caption(f"🟢 Last update: {int(time_ago)}s ago")
            else:
                status_placeholder1.caption(f"🟡 Last update: {int(time_ago/60)}m ago")
        else:
            status_placeholder1.caption("⚪ Waiting for first update...")
        
        if st.session_state.auto_refresh_enabled:
            status_placeholder2.caption(f"🔄 Auto-refresh: Every {st.session_state.refresh_interval}s")
        else:
            status_placeholder2.caption("⏸️ Auto-refresh: Disabled")
        
        status_placeholder3.caption(f"📊 Updates: {st.session_state.update_count}")
        st.session_state.last_status_update = st.session_state.last_update_time
    
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
        should_fetch = (stocks_set != st.session_state.last_fetched_stocks or 
                       st.session_state.cached_quotes.empty or
                       (st.session_state.auto_refresh_enabled and 
                        st.session_state.last_update_time and
                        (datetime.now() - st.session_state.last_update_time).total_seconds() >= st.session_state.refresh_interval))
        
        if stocks_to_fetch and should_fetch:
            try:
                # Fetch current data (this happens in background, UI remains responsive)
                if use_demo:
                    quotes = generate_demo_data(stocks_to_fetch)
                else:
                    quotes = fetch_current_quotes(stocks_to_fetch)
                
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
                logger.error(f"Error fetching data: {e}")
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
                                if graph_stocks and not valid_quotes.empty:
                                    graph_quotes = valid_quotes[valid_quotes["ticker"].isin(graph_stocks)].copy()
                                    if not graph_quotes.empty:
                                        plot_vertical_bar(graph_quotes, "pct_change", f"Graph {graph_idx + 1} - Stock Performance", height=180)
                            
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
