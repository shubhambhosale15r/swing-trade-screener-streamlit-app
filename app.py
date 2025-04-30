import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from stocklist import *  # Ensure this file defines STOCK_UNIVERSE

# Configuration
PAGE_TITLE = "Swing Trade"
PAGE_ICON = "📈"
LOADING_TEXT = "Analyzing Stocks..."

# --- Basic App Setup ---
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    menu_items=None,
)

# --- Session State Initialization ---
def initialize_session_state():
    if 'view_universe_rankings' not in st.session_state:
        st.session_state.view_universe_rankings = False
    if 'view_recommended_stocks' not in st.session_state:
        st.session_state.view_recommended_stocks = False
    if 'analyze_button_clicked' not in st.session_state:
        st.session_state.analyze_button_clicked = False
    if 'view_high_momentum_stocks' not in st.session_state:
        st.session_state.view_high_momentum_stocks = False

initialize_session_state()

# --- CSS Styling ---
def inject_custom_css():
    st.markdown(f"""
        <style>
            /* ... existing CSS ... */
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# --- Header ---
def display_header():
    st.markdown(f"""
        <h1 style='text-align: center;'>{PAGE_ICON} {PAGE_TITLE}</h1>
        <div style="text-align: center; font-size: 1.2rem; color: #c0c0c0;">
            Select a stock universe and click buttons to analyze momentum.<br>
        </div>
    """, unsafe_allow_html=True)

display_header()

# --- Sidebar ---
def create_sidebar():
    with st.sidebar:
        universe_name = st.radio("Select Stock Universe", list(STOCK_UNIVERSE.keys()))
        selected_symbols = STOCK_UNIVERSE[universe_name]
        st.info(f"Selected: {universe_name}")

        if st.button("Analyze Stock Universe", key="analyze_stock_universe_sidebar"):
            st.session_state.analyze_button_clicked = True
            st.rerun()
        if st.button("Stock Universes Ranks", key="universe_ranks_sidebar"):
            st.session_state.view_universe_rankings = True
            st.rerun()
        if st.button("Recommended Stocks", key="recommended_stocks_sidebar"):
            st.session_state.view_recommended_stocks = True
            st.rerun()
        if st.button("High Momentum Stocks", key="high_momentum_stocks_sidebar"):
            st.session_state.view_high_momentum_stocks = True
            st.rerun()

    return universe_name, selected_symbols

stock_universe_name, selected_stocks = create_sidebar()

# --- NSE Session Helper ---
def create_nse_session(
    homepage_url: str = "https://www.nseindia.com",
    retries: int = 5,
    backoff_factor: float = 1.0,
    timeout: float = 5.0
) -> requests.Session:
    """
    Returns a requests.Session preloaded with cookies and realistic browser headers
    from the NSE homepage. Retries on non-200 responses with exponential back-off.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) "
            "Gecko/20100101 Firefox/109.0"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;"
            "q=0.9,image/avif,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-User": "?1",
        "TE": "trailers",
    }

    session = requests.Session()
    session.headers.update(headers)

    delay = backoff_factor
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(homepage_url, timeout=timeout)
            if resp.status_code == 200 and "html" in resp.headers.get("Content-Type", ""):
                return session
            else:
                time.sleep(delay)
                delay *= 2
        except requests.RequestException:
            time.sleep(delay)
            delay *= 2

    raise RuntimeError(f"Unable to establish NSE session after {retries} attempts.")

# --- Data Download via NSE Scraping ---
@st.cache_data(show_spinner=False)
def download_stock_data(ticker, start_date, end_date, retries=3):
    """
    Fetch daily OHLCV for `ticker` from NSE between start_date and end_date,
    using a browser-like session for cookies/headers.
    """
    start_str = start_date.strftime("%d-%m-%Y")
    end_str = end_date.strftime("%d-%m-%Y")

    url = "https://www.nseindia.com/api/historicalOR/generateSecurityWiseHistoricalData"
    params = {
        "from": start_str,
        "to": end_str,
        "symbol": ticker,
        "type": "priceVolumeDeliverable",
        "series": "ALL"
    }

    for _ in range(retries):
        try:
            session = create_nse_session()
            resp = session.get(url, params=params, timeout=10)
            resp.raise_for_status()

            payload = resp.json().get("data", [])
            if not payload:
                raise ValueError(f"No data for {ticker}")

            df = pd.DataFrame(payload)
            df['Date'] = pd.to_datetime(df['mTIMESTAMP'], dayfirst=True)
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)

            df.rename(columns={
                'CH_OPENING_PRICE': 'Open',
                'CH_TRADE_HIGH_PRICE': 'High',
                'CH_TRADE_LOW_PRICE': 'Low',
                'CH_CLOSING_PRICE': 'Close',
                'CH_TOT_TRADED_QTY': 'Volume'
            }, inplace=True)

            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            return df.reset_index()

        except Exception:
            time.sleep(1)
            continue

    return pd.DataFrame()

# Calculate returns over n trading days
def calculate_returns(df, period):
    if len(df) >= period:
        prices = df['Close'].values
        return ((prices[-1] - prices[-period]) / prices[-period]).item()
    return np.nan

# Analyze a universe and compute momentum for each ticker
def analyze_universe(name, symbols):
    end = datetime.today().date()
    start = end - timedelta(days=400)
    rows = []

    for t in symbols:
        df = download_stock_data(t, start, end)
        if df.empty: continue
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df[~df.index.duplicated()]
        df['Daily Return'] = df['Close'].pct_change()
        vol = df['Daily Return'].dropna().std() * np.sqrt(63)

        r3 = calculate_returns(df, 63)
        r1 = calculate_returns(df, 21)
        r0 = calculate_returns(df, 5)

        if vol and pd.notna(r3) and pd.notna(r1) and pd.notna(r0):
            mom = ((0.6*r3) + (0.3*r1) + (0.1*r0)) / vol
        else:
            mom = np.nan

        rows.append({
            "Ticker": t,
            "Momentum Score": mom,
            "3-Month Return (%)": r3*100 if pd.notna(r3) else np.nan,
            "1-Month Return (%)": r1*100 if pd.notna(r1) else np.nan,
            "1-Week Return (%)": r0*100 if pd.notna(r0) else np.nan,
            "Annualized Volatility": vol
        })

    df_res = pd.DataFrame(rows)
    avg_score = df_res["Momentum Score"].mean() if not df_res.empty else np.nan
    return df_res, avg_score

# Top universes by avg momentum
def get_top_universes_by_momentum():
    data = []
    for name, syms in STOCK_UNIVERSE.items():
        _, avg = analyze_universe(name, syms)
        data.append({"Stock Universe": name, "Average Momentum Score": avg})
    return pd.DataFrame(data).sort_values("Average Momentum Score", ascending=False)

# Top stocks in a universe
def get_top_stocks_from_universe(name, symbols):
    df, _ = analyze_universe(name, symbols)
    return df.sort_values("Momentum Score", ascending=False) if not df.empty else pd.DataFrame()

# Top 10 high momentum stocks overall
def get_top_momentum_stocks_overall():
    all_dfs = []
    for syms in STOCK_UNIVERSE.values():
        df, _ = analyze_universe(None, syms)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.dropna(subset=["Momentum Score"], inplace=True)
    combined.sort_values("Momentum Score", ascending=False, inplace=True)
    unique = combined.drop_duplicates(subset=["Ticker"], keep="first")
    return unique.head(10)

# --- Main App Logic ---
def main():
    # Analyze specific universe
    if st.session_state.analyze_button_clicked:
        st.subheader(f"Momentum Analysis: {stock_universe_name}")
        placeholder = st.empty()
        with placeholder:
            st.markdown(f"<div class='loading-container'>{LOADING_TEXT}</div>", unsafe_allow_html=True)
        df, _ = analyze_universe(stock_universe_name, selected_stocks)
        placeholder.empty()
        if not df.empty:
            df = df.sort_values("Momentum Score", ascending=False)
            st.dataframe(df.style.format({
                "3-Month Return (%)": "{:.2f}%",
                "1-Month Return (%)": "{:.2f}%",
                "1-Week Return (%)": "{:.2f}%",
                "Annualized Volatility": "{:.4f}",
                "Momentum Score": "{:.4f}"}), use_container_width=True)
        else:
            st.warning("No data available.")
        st.session_state.analyze_button_clicked = False

    # Recommended universes
    if st.session_state.view_recommended_stocks:
        st.subheader("Stock Universes Based on Momentum")
        loading = st.empty(); loading.markdown("<div class='loading-container'>...</div>", unsafe_allow_html=True)
        top_unis = get_top_universes_by_momentum()
        loading.empty()
        for _, row in top_unis.iterrows():
            st.markdown(f"### {row['Stock Universe']} ({row['Average Momentum Score']:.4f})")
            top5 = get_top_stocks_from_universe(row['Stock Universe'], STOCK_UNIVERSE[row['Stock Universe']])
            if not top5.empty:
                st.dataframe(top5[['Ticker','Momentum Score','3-Month Return (%)',
                                   '1-Month Return (%)','1-Week Return (%)']], height=300)
            else:
                st.warning(f"No data for {row['Stock Universe']}")
        st.session_state.view_recommended_stocks = False

    # Universe rankings
    if st.session_state.view_universe_rankings:
        st.subheader("Stock Universes Ranking")
        progress = st.progress(0)
        rows = []
        for i, (name, syms) in enumerate(STOCK_UNIVERSE.items()):
            _, avg = analyze_universe(name, syms)
            rows.append({"Stock Universe": name, "Average Momentum Score": avg})
            progress.progress((i+1)/len(STOCK_UNIVERSE))
        progress.empty()
        df_unis = pd.DataFrame(rows).sort_values("Average Momentum Score", ascending=False)
        st.dataframe(df_unis.style.format({'Average Momentum Score':'{:.4f}'}), height=400)
        c1,c2 = st.columns(2)
        best = df_unis.iloc[0]; worst = df_unis.iloc[-1]
        c1.metric("Highest Momentum Universe", best['Stock Universe'], f"{best['Average Momentum Score']:.4f}")
        c2.metric("Lowest Momentum Universe", worst['Stock Universe'], f"{worst['Average Momentum Score']:.4f}")
        st.session_state.view_universe_rankings = False

    # High Momentum Stocks Overall
    if st.session_state.view_high_momentum_stocks:
        st.subheader("Top 10 High Momentum Stocks (All Universes Combined)")
        loading = st.empty(); loading.markdown("<div class='loading-container'>Loading ...</div>", unsafe_allow_html=True)
        top10 = get_top_momentum_stocks_overall()
        loading.empty()
        if not top10.empty:
            st.dataframe(top10.style.format({
                "3-Month Return (%)": "{:.2f}%",
                "1-Month Return (%)": "{:.2f}%",
                "1-Week Return (%)": "{:.2f}%",
                "Annualized Volatility": "{:.4f}",
                "Momentum Score": "{:.4f}"}), use_container_width=True)
        else:
            st.warning("No data available across stock universes.")
        st.session_state.view_high_momentum_stocks = False

if __name__ == '__main__':
    main()
