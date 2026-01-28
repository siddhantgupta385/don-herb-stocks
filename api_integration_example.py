"""
Example implementations for different stock data APIs.
Choose one and integrate it into dashboard.py
"""

import requests
import pandas as pd
from typing import List, Dict, Optional
import streamlit as st


# ============================================================================
# OPTION 1: Alpha Vantage (Free tier available, good for testing)
# ============================================================================
def fetch_alpha_vantage_quotes(tickers: List[str], api_key: str) -> pd.DataFrame:
    """
    Fetch real-time quotes from Alpha Vantage.
    Free tier: 5 calls/minute, 500 calls/day
    Premium: Real-time data, higher limits
    """
    data = []
    
    for ticker in tickers:
        try:
            # Global Quote endpoint for real-time data
            url = f"https://www.alphavantage.co/query"
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": ticker,
                "apikey": api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            if "Global Quote" in result:
                quote = result["Global Quote"]
                price = float(quote.get("05. price", 0))
                prev_close = float(quote.get("08. previous close", 0))
                change = float(quote.get("09. change", 0))
                change_percent = float(quote.get("10. change percent", "0%").replace("%", ""))
                volume = int(quote.get("06. volume", 0))
                
                data.append({
                    "ticker": ticker.upper(),
                    "price": price,
                    "prev_close": prev_close,
                    "pct_change": change_percent,
                    "volume": volume,
                    "currency": "USD",
                })
        except Exception as e:
            print(f"Error fetching {ticker} from Alpha Vantage: {e}")
            data.append({
                "ticker": ticker.upper(),
                "price": None,
                "prev_close": None,
                "pct_change": None,
                "volume": None,
                "currency": "N/A",
            })
    
    return pd.DataFrame(data)


# ============================================================================
# OPTION 2: IEX Cloud (Recommended for budget-friendly real-time)
# ============================================================================
def fetch_iex_cloud_quotes(tickers: List[str], api_key: str) -> pd.DataFrame:
    """
    Fetch real-time quotes from IEX Cloud.
    Pricing: $9/month starter plan
    Real-time data, no rate limits on paid plans
    """
    data = []
    
    # IEX Cloud supports batch requests - fetch all at once
    tickers_str = ",".join(tickers)
    url = f"https://cloud.iexapis.com/stable/stock/market/batch"
    params = {
        "symbols": tickers_str,
        "types": "quote",
        "token": api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        result = response.json()
        
        for ticker in tickers:
            ticker_upper = ticker.upper()
            if ticker_upper in result and "quote" in result[ticker_upper]:
                quote = result[ticker_upper]["quote"]
                
                price = quote.get("latestPrice", quote.get("iexRealtimePrice"))
                prev_close = quote.get("previousClose", 0)
                change = quote.get("change", 0)
                change_percent = quote.get("changePercent", 0) * 100 if quote.get("changePercent") else 0
                volume = quote.get("latestVolume", quote.get("volume", 0))
                
                data.append({
                    "ticker": ticker_upper,
                    "price": price,
                    "prev_close": prev_close,
                    "pct_change": change_percent,
                    "volume": volume,
                    "currency": "USD",
                })
            else:
                data.append({
                    "ticker": ticker_upper,
                    "price": None,
                    "prev_close": None,
                    "pct_change": None,
                    "volume": None,
                    "currency": "N/A",
                })
    except Exception as e:
        print(f"Error fetching from IEX Cloud: {e}")
        # Return empty data for all tickers
        for ticker in tickers:
            data.append({
                "ticker": ticker.upper(),
                "price": None,
                "prev_close": None,
                "pct_change": None,
                "volume": None,
                "currency": "N/A",
            })
    
    return pd.DataFrame(data)


# ============================================================================
# OPTION 3: Polygon.io (Best for real-time with WebSocket)
# ============================================================================
def fetch_polygon_quotes(tickers: List[str], api_key: str) -> pd.DataFrame:
    """
    Fetch real-time quotes from Polygon.io.
    Pricing: $29/month starter
    Best for real-time data, supports WebSocket streaming
    """
    data = []
    
    # Polygon supports batch requests
    for ticker in tickers:
        try:
            # Get last trade (real-time)
            url = f"https://api.polygon.io/v2/last/trade/{ticker.upper()}"
            params = {"apikey": api_key}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            if result.get("status") == "OK" and "results" in result:
                trade = result["results"]
                price = trade.get("p", 0)  # Last price
                
                # Get previous close
                prev_url = f"https://api.polygon.io/v2/aggs/ticker/{ticker.upper()}/prev"
                prev_response = requests.get(prev_url, params=params, timeout=10)
                prev_result = prev_response.json()
                
                prev_close = 0
                if prev_result.get("status") == "OK" and "results" in prev_result:
                    prev_close = prev_result["results"][0].get("c", 0)  # Close price
                
                change_percent = ((price - prev_close) / prev_close * 100) if prev_close > 0 else 0
                
                data.append({
                    "ticker": ticker.upper(),
                    "price": price,
                    "prev_close": prev_close,
                    "pct_change": change_percent,
                    "volume": trade.get("s", 0),  # Size
                    "currency": "USD",
                })
            else:
                data.append({
                    "ticker": ticker.upper(),
                    "price": None,
                    "prev_close": None,
                    "pct_change": None,
                    "volume": None,
                    "currency": "N/A",
                })
        except Exception as e:
            print(f"Error fetching {ticker} from Polygon: {e}")
            data.append({
                "ticker": ticker.upper(),
                "price": None,
                "prev_close": None,
                "pct_change": None,
                "volume": None,
                "currency": "N/A",
            })
    
    return pd.DataFrame(data)


# ============================================================================
# OPTION 4: Twelve Data (Good balance)
# ============================================================================
def fetch_twelve_data_quotes(tickers: List[str], api_key: str) -> pd.DataFrame:
    """
    Fetch real-time quotes from Twelve Data.
    Pricing: $29/month
    Good balance of features and cost
    """
    data = []
    
    for ticker in tickers:
        try:
            url = f"https://api.twelvedata.com/price"
            params = {
                "symbol": ticker.upper(),
                "apikey": api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            if "price" in result:
                price = float(result["price"])
                
                # Get quote for previous close
                quote_url = f"https://api.twelvedata.com/quote"
                quote_params = {
                    "symbol": ticker.upper(),
                    "apikey": api_key
                }
                quote_response = requests.get(quote_url, params=quote_params, timeout=10)
                quote_result = quote_response.json()
                
                prev_close = float(quote_result.get("previous_close", 0))
                change_percent = float(quote_result.get("percent_change", 0))
                volume = int(quote_result.get("volume", 0))
                
                data.append({
                    "ticker": ticker.upper(),
                    "price": price,
                    "prev_close": prev_close,
                    "pct_change": change_percent,
                    "volume": volume,
                    "currency": "USD",
                })
            else:
                data.append({
                    "ticker": ticker.upper(),
                    "price": None,
                    "prev_close": None,
                    "pct_change": None,
                    "volume": None,
                    "currency": "N/A",
                })
        except Exception as e:
            print(f"Error fetching {ticker} from Twelve Data: {e}")
            data.append({
                "ticker": ticker.upper(),
                "price": None,
                "prev_close": None,
                "pct_change": None,
                "volume": None,
                "currency": "N/A",
            })
    
    return pd.DataFrame(data)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
"""
To use in dashboard.py:

1. Add API key to Streamlit secrets:
   Create .streamlit/secrets.toml:
   [api]
   iex_cloud_key = "your_api_key_here"
   # OR
   polygon_key = "your_api_key_here"
   # OR
   alpha_vantage_key = "your_api_key_here"

2. Replace fetch_current_quotes function:

import streamlit as st

@st.cache_data(ttl=10, show_spinner=False)
def fetch_current_quotes(tickers: List[str]) -> pd.DataFrame:
    # Get API key from secrets
    api_key = st.secrets.get("api", {}).get("iex_cloud_key")
    
    if not api_key:
        st.error("API key not configured. Please add it to .streamlit/secrets.toml")
        return pd.DataFrame()
    
    # Use IEX Cloud (or your chosen API)
    return fetch_iex_cloud_quotes(tickers, api_key)
"""
