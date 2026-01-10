#!/usr/bin/env python3
"""
Test script to demonstrate yfinance API calls.
This shows the actual HTTP requests being made server-side.
"""

import logging
import sys
from urllib.request import urlopen

# Enable detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enable HTTP debugging
import http.client
http.client.HTTPConnection.debuglevel = 1

print("=" * 60)
print("Testing yfinance API Calls")
print("=" * 60)
print("\nThis script makes server-side HTTP requests to Yahoo Finance.")
print("These requests happen in Python, NOT in the browser.\n")
print("Watch the output below to see the actual HTTP requests:\n")
print("-" * 60)

try:
    import yfinance as yf
    
    print("\n1. Fetching quote data for AAPL...")
    ticker = yf.Ticker("AAPL")
    info = ticker.fast_info
    
    print(f"\n✅ Success! Got data:")
    print(f"   Price: ${info.last_price}")
    print(f"   Previous Close: ${info.previous_close}")
    print(f"   Volume: {info.last_volume:,}")
    
    print("\n2. Fetching historical data...")
    hist = ticker.history(period="5d", interval="1d")
    print(f"\n✅ Success! Got {len(hist)} days of data")
    print(f"   Latest close: ${hist['Close'].iloc[-1]:.2f}")
    
    print("\n" + "=" * 60)
    print("These HTTP requests were made by Python (server-side)")
    print("They do NOT appear in browser HAR files!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nThis might be due to:")
    print("  - Rate limiting by Yahoo Finance")
    print("  - Network connectivity issues")
    print("  - yfinance library issues")
    sys.exit(1)
