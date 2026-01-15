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

# Global rate limiting state
if 'last_rate_limit_time' not in st.session_state:
    st.session_state.last_rate_limit_time = None
if 'rate_limit_cooldown' not in st.session_state:
    st.session_state.rate_limit_cooldown = False
if 'graphs' not in st.session_state:
    st.session_state.graphs = []
if 'all_stocks' not in st.session_state:
    st.session_state.all_stocks = []


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


@st.cache_data(ttl=300)  # Cache for 5 minutes to reduce API calls
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
        
        # Longer delay between requests to avoid rate limiting
        if t != tickers[-1]:
            time.sleep(3)  # Increased delay to avoid rate limiting
    
    return pd.DataFrame(data)


def generate_demo_data(tickers: List[str]) -> pd.DataFrame:
    """Generate demo/mock data for testing when API is rate limited."""
    import random
    data = []
    
    # Base prices for common stocks (realistic current prices)
    base_prices = {
        "AAPL": 259.0, "MSFT": 479.0, "GOOGL": 328.0, "AMZN": 247.0,
        "META": 653.0, "NVDA": 875.0, "TSLA": 445.0, "NFLX": 89.0,
        "JPM": 329.0, "BAC": 58.0, "WFC": 96.0, "GS": 485.0,
        "JNJ": 168.0, "PFE": 28.0, "UNH": 545.0, "ABBV": 178.0,
        "WMT": 114.0, "HD": 385.0, "MCD": 295.0, "NKE": 95.0,
        "XOM": 118.0, "CVX": 152.0, "SPY": 580.0, "QQQ": 485.0,
    }
    
    for t in tickers:
        # Get base price or generate random
        base_price = base_prices.get(t, random.uniform(50, 500))
        
        # Add realistic variation (-3% to +3%)
        variation = random.uniform(-0.03, 0.03)
        price = base_price * (1 + variation)
        prev_close = base_price
        pct_change = variation * 100
        volume = random.randint(5000000, 100000000)
        
        data.append({
            "ticker": t.upper(),
            "price": round(price, 2),
            "prev_close": round(prev_close, 2),
            "pct_change": round(pct_change, 2),
            "volume": volume,
            "currency": "USD",
        })
    
    return pd.DataFrame(data)


def plot_vertical_bar(quotes_df: pd.DataFrame, metric: str, title: str) -> None:
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
            font=dict(size=11, color='black'),
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
            font=dict(size=11, color='black', weight='bold'),
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
            font=dict(size=40, color='black'),
            align="center",
            valign="middle"
        ))
    
    fig.update_layout(
        title=dict(
            text="Stock Performance - Daily Change (%)",
            font=dict(size=16, family='Arial'),
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
            tickfont=dict(size=11),
        ),
        height=max(400, y_bottom * 15 + 100),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=120, r=120, t=80, b=80),
        showlegend=False,
        annotations=annotations,
        barmode='stack',  # Ensure proper stacking
    )
    
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Stock Dashboard", layout="wide")
    
    st.title("Stock Dashboard")
    
    # Demo mode toggle
    use_demo = st.sidebar.checkbox("🧪 Use Demo Data (for testing)", value=False, help="Enable to use mock data instead of real API calls. Perfect for development when rate limited!")
    
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
    
    # Combine popular stocks with already added stocks
    all_available_stocks = sorted(list(set(POPULAR_STOCKS + st.session_state.all_stocks)))
    
    # Stock selection with searchable dropdown
    col1, col2, col3 = st.columns([3, 1.5, 1])
    
    with col1:
        selected_new_stocks = st.multiselect(
            "Add stocks",
            options=all_available_stocks,
            default=[],
            help="Search and select stocks to add"
        )
    
    with col2:
        custom_stock = st.text_input(
            "Custom ticker",
            value="",
            placeholder="e.g., CUSTOM",
            help="Add a custom ticker not in the list"
        )
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Add", use_container_width=True):
            new_stocks = []
            if selected_new_stocks:
                new_stocks.extend(selected_new_stocks)
            if custom_stock:
                custom = custom_stock.strip().upper()
                if custom and custom not in new_stocks:
                    new_stocks.append(custom)
            
            if new_stocks:
                for stock in new_stocks:
                    if stock not in st.session_state.all_stocks:
                        st.session_state.all_stocks.append(stock)
                st.rerun()
    
    # Display all stocks
    if st.session_state.all_stocks:
        selected_stocks = st.multiselect(
            "Select stocks to track",
            options=st.session_state.all_stocks,
            default=st.session_state.all_stocks,
        )
        
        if selected_stocks:
            # Check market status (only if not using demo)
            if not use_demo:
                is_open, market_msg = check_market_status()
                if not is_open:
                    st.info(f"ℹ️ {market_msg} - Showing previous close prices for testing")
            
            # Fetch current data
            if use_demo:
                quotes = generate_demo_data(selected_stocks)
                st.info("🧪 **Demo Mode Active** - Using mock data for testing")
            else:
                quotes = fetch_current_quotes(selected_stocks)
            
            valid_quotes = quotes[~quotes["price"].isna()].copy()
            
            # Show fetch status
            if valid_quotes.empty and not use_demo:
                failed_tickers = quotes[quotes["price"].isna()]["ticker"].tolist() if not quotes.empty else selected_stocks
                if failed_tickers:
                    st.warning(f"⚠️ Failed to fetch data for: {', '.join(failed_tickers)}")
                    st.info("💡 **Tip:** Enable '🧪 Use Demo Data' in the sidebar to continue developing with mock data!")
            
            if not valid_quotes.empty:
                # Show success message
                st.success(f"✅ Fetched data for {len(valid_quotes)} stock(s)")
                
                # Controls
                col_add, col_refresh, col_clear = st.columns([1, 1, 1])
                
                with col_add:
                    if st.button("+ Add Graph", use_container_width=True):
                        if 'graphs' not in st.session_state:
                            st.session_state.graphs = []
                        st.session_state.graphs.append({
                            "stocks": []
                        })
                        st.rerun()
                
                with col_refresh:
                    if st.button("🔄 Refresh Data", use_container_width=True):
                        fetch_current_quotes.clear()
                        st.rerun()
                
                with col_clear:
                    if st.button("🗑️ Clear Cache", use_container_width=True):
                        st.cache_data.clear()
                        st.success("Cache cleared!")
                        st.rerun()
                
                # Initialize graphs if empty
                if 'graphs' not in st.session_state or not st.session_state.graphs:
                    st.session_state.graphs = [{"stocks": selected_stocks.copy()}]
                
                # Get stocks not yet in any graph
                stocks_in_graphs = set()
                for graph in st.session_state.graphs:
                    stocks_in_graphs.update(graph.get("stocks", []))
                available_stocks = [s for s in selected_stocks if s not in stocks_in_graphs]
                
                # Display each graph
                for idx, graph in enumerate(st.session_state.graphs):
                    with st.container():
                        col_stocks, col_delete = st.columns([4, 1])
                        
                        with col_stocks:
                            graph_stocks = st.multiselect(
                                f"Graph {idx + 1} - Select stocks",
                                options=selected_stocks,
                                default=graph.get("stocks", []),
                                key=f"graph_stocks_{idx}"
                            )
                            graph["stocks"] = graph_stocks
                        
                        with col_delete:
                            st.markdown("<br>", unsafe_allow_html=True)
                            if st.button("Delete", key=f"delete_{idx}", use_container_width=True):
                                st.session_state.graphs.pop(idx)
                                st.rerun()
                        
                        # Plot graph
                        if graph_stocks:
                            graph_quotes = valid_quotes[valid_quotes["ticker"].isin(graph_stocks)].copy()
                            if not graph_quotes.empty:
                                plot_vertical_bar(graph_quotes, "pct_change", f"Graph {idx + 1} - Stock Performance")
                        
                        st.divider()
                
                # Add remaining stocks to new graph option
                if available_stocks:
                    if st.button(f"Add Graph for Remaining Stocks ({len(available_stocks)})", use_container_width=True):
                        st.session_state.graphs.append({
                            "stocks": available_stocks.copy()
                        })
                        st.rerun()
                
                # Debug view (collapsed)
                with st.expander("📊 View Raw Data", expanded=False):
                    display_quotes = valid_quotes.copy()
                    display_quotes["price"] = display_quotes["price"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
                    display_quotes["pct_change"] = display_quotes["pct_change"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
                    display_quotes["volume"] = display_quotes["volume"].apply(lambda x: f"{x/1e6:.2f}M" if pd.notna(x) and x >= 1e6 else f"{x/1e3:.2f}K" if pd.notna(x) and x >= 1e3 else "N/A" if pd.isna(x) else f"{x:.0f}")
                    st.dataframe(display_quotes.set_index("ticker"), use_container_width=True)
            else:
                # Show what we tried to fetch
                st.error(f"❌ No data available for selected stocks: {', '.join(selected_stocks)}")
                
                # Show raw quotes for debugging
                if not quotes.empty:
                    with st.expander("🔍 Debug: See what was fetched", expanded=True):
                        st.dataframe(quotes)
                        st.caption("If you see None/NaN values, the API may be rate-limited. Wait 30-60 seconds and click Refresh.")
                
                st.info("💡 **Troubleshooting:**")
                st.markdown("""
                - **Rate limiting**: Wait 30-60 seconds between requests, then click Refresh
                - **Market closed**: Previous close prices should still load (good for testing!)
                - **Invalid tickers**: Check that ticker symbols are correct (e.g., AAPL, not APPL)
                - **Try one stock at a time** if rate limiting persists
                """)
        else:
            st.info("Select stocks to track.")
    else:
        st.info("Add stocks to get started.")
    
    # Remove stocks option
    if st.session_state.all_stocks:
        with st.expander("Manage Stocks"):
            to_remove = st.multiselect(
                "Remove stocks",
                options=st.session_state.all_stocks,
            )
            if st.button("Remove Selected"):
                for stock in to_remove:
                    if stock in st.session_state.all_stocks:
                        st.session_state.all_stocks.remove(stock)
                    # Remove from all graphs
                    for graph in st.session_state.graphs:
                        if stock in graph["stocks"]:
                            graph["stocks"].remove(stock)
                st.rerun()


if __name__ == "__main__":
    main()
