# Why Yahoo Finance API Calls Don't Appear in Browser HAR Files

## The Issue

You're looking at a **browser HAR file** (`localhost.har`), which only captures network requests made **from the browser** (JavaScript/HTTP requests initiated by the web page).

## Where the API Calls Actually Happen

The `yfinance` library makes HTTP requests **server-side in Python**, not from the browser. Here's the flow:

```
Browser (HAR file)          Streamlit Server (Python)          Yahoo Finance API
     |                              |                                |
     |--- HTTP Request ------------>|                                |
     |   (Streamlit UI)              |                                |
     |                               |--- HTTP Request ------------->|
     |                               |   (yfinance library)           |
     |                               |                                |
     |                               |<-- JSON Response -------------|
     |                               |   (stock data)                 |
     |<-- HTML/JSON Response --------|                                |
     |   (rendered dashboard)        |                                |
```

## What You See in the HAR File

The HAR file only shows:
1. **Streamlit analytics** (webhooks.fivetran.com) - telemetry data
2. **Browser ↔ Streamlit server** communication
3. **NOT** the Python → Yahoo Finance API calls

## How to See the Actual API Calls

### Option 1: Check Streamlit Terminal Output
When you run `streamlit run dashboard.py`, the terminal will show:
- Python logs
- Any print statements
- Error messages from yfinance

### Option 2: Enable Python HTTP Logging
I've added logging to `dashboard.py` that will show:
- When API calls are made
- Which tickers are being fetched
- Success/failure of requests

### Option 3: Use Python's `http.client` Debugging
Add this to see raw HTTP requests:

```python
import http.client
http.client.HTTPConnection.debuglevel = 1
```

### Option 4: Use a Proxy Tool
Tools like:
- **mitmproxy** - intercept Python HTTP requests
- **Charles Proxy** - monitor all network traffic
- **Wireshark** - packet-level inspection

### Option 5: Check yfinance Source Code
The `yfinance` library makes requests to endpoints like:
- `query1.finance.yahoo.com`
- `query2.finance.yahoo.com`

You can inspect the library's source code to see exactly what URLs it calls.

## Quick Test

Run this in your terminal to see an API call happen:

```bash
python3 -c "import yfinance as yf; ticker = yf.Ticker('AAPL'); print(ticker.fast_info.last_price)"
```

This will make a real HTTP request to Yahoo Finance that you can monitor with network tools.

## Summary

- **Browser HAR** = Only browser-side requests (Streamlit UI ↔ Server)
- **Server-side requests** = Python (yfinance) → Yahoo Finance API
- **To see server requests** = Check terminal logs, use Python logging, or network monitoring tools
