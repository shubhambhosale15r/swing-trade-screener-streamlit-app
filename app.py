import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import tempfile
from datetime import datetime, timedelta
from fyers_apiv3 import fyersModel
from stocklist import STOCK_UNIVERSE
from stqdm import stqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

PAGE_TITLE = "Swing Trade"
PAGE_ICON = "📈"
LOADING_TEXT = "Analyzing Stocks..."

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    menu_items=None,
)

def initialize_session_state():
    if 'view_universe_rankings' not in st.session_state:
        st.session_state.view_universe_rankings = False
    if 'view_recommended_stocks' not in st.session_state:
        st.session_state.view_recommended_stocks = False
    if 'analyze_button_clicked' not in st.session_state:
        st.session_state.analyze_button_clicked = False
    if 'view_high_momentum_stocks' not in st.session_state:
        st.session_state.view_high_momentum_stocks = False
    if 'fyers_access_token' not in st.session_state:
        st.session_state.fyers_access_token = ""

initialize_session_state()

def inject_custom_css():
    st.markdown(f"""
        <style>
            .loading-container {{
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                padding: 2rem;
            }}
            .spinner {{
                border: 4px solid rgba(0, 0, 0, 0.1);
                border-radius: 50%;
                border-top: 4px solid #3498db;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin-bottom: 1rem;
            }}
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

def display_header():
    st.markdown(f"""
        <h1 style='text-align: center;'>{PAGE_ICON} {PAGE_TITLE}</h1>
        <div style="text-align: center; font-size: 1.2rem; color: #c0c0c0;">
            Select a stock universe and click buttons to analyze momentum.<br>
        </div>
    """, unsafe_allow_html=True)

display_header()

def create_sidebar():
    with st.sidebar:
        token_input = st.text_input("Enter Fyers Access Token", type="password", value=st.session_state.fyers_access_token)
        if token_input:
            st.session_state.fyers_access_token = token_input
        universe_name = st.radio("Select Stock Universe", list(STOCK_UNIVERSE.keys()))
        st.info(f"Selected: {universe_name}")
    
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Analyze Stock Universe", use_container_width=True):
                st.session_state.view_universe_rankings = False
                st.session_state.view_recommended_stocks = False
                st.session_state.view_high_momentum_stocks = False
                st.session_state.analyze_button_clicked = True
                st.rerun()
            
            if st.button("Stock Universes Ranks", use_container_width=True):
                st.session_state.analyze_button_clicked = False
                st.session_state.view_recommended_stocks = False
                st.session_state.view_high_momentum_stocks = False
                st.session_state.view_universe_rankings = True
                st.rerun()
                
        with col2:
            if st.button("Recommended Stocks", use_container_width=True):
                st.session_state.analyze_button_clicked = False
                st.session_state.view_universe_rankings = False
                st.session_state.view_high_momentum_stocks = False
                st.session_state.view_recommended_stocks = True
                st.rerun()
            
            if st.button("High Momentum Stocks", use_container_width=True):
                st.session_state.analyze_button_clicked = False
                st.session_state.view_universe_rankings = False
                st.session_state.view_recommended_stocks = False
                st.session_state.view_high_momentum_stocks = True
                st.rerun()

    return universe_name

stock_universe_name = create_sidebar()

def initialize_fyers():
    if st.session_state.fyers_access_token:
        # Create a proper log directory
        temp_dir = tempfile.gettempdir()
        log_dir = os.path.join(temp_dir, "fyers_logs")
        os.makedirs(log_dir, exist_ok=True)
        
        fyers = fyersModel.FyersModel(
            client_id="YOUR_CLIENT_ID",  # Replace with your actual client ID
            token=st.session_state.fyers_access_token,
            log_path=log_dir + os.sep  # Add trailing separator
        )
        return fyers
    else:
        st.error("Please enter your Fyers Access Token in the sidebar.")
        return None

fyers = initialize_fyers()

@st.cache_data(show_spinner=False)
def download_stock_data(ticker, start_date, end_date, retries=3):
    if fyers is None:
        return pd.DataFrame()

    symbol = f"NSE:{ticker}-EQ"
    all_data = []
    current_start = start_date

    while current_start <= end_date:
        current_end = min(current_start + timedelta(days=89), end_date)

        for _ in range(retries):
            try:
                data = {
                    "symbol": symbol",
                    "resolution": "D",
                    "date_format": "1",
                    "range_from": current_start.strftime("%Y-%m-%d"),
                    "range_to": current_end.strftime("%Y-%m-%d"),
                    "cont_flag": "1"
                }
                response = fyers.history(data)

                if response.get("error"):
                    print(f"Error response for {ticker}: {response['error']}")
                    break

                candles = response.get("candles", [])
                if not candles:
                    print(f"No candles returned for {ticker}")
                    break

                df_chunk = pd.DataFrame(candles, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
                df_chunk["Date"] = pd.to_datetime(df_chunk["timestamp"], unit="s")
                all_data.append(df_chunk)
                break
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
                time.sleep(1)
                continue

        current_start = current_end + timedelta(days=1)

    if not all_data:
        return pd.DataFrame()

    full_df = pd.concat(all_data, ignore_index=True)
    full_df = full_df.dropna()
    full_df["Date"] = pd.to_datetime(full_df["timestamp"], unit="s")
    full_df.set_index("Date", inplace=True)
    full_df.sort_index(inplace=True)
    full_df = full_df[["Open", "High", "Low", "Close", "Volume"]]
    return full_df.reset_index()

def calculate_returns(df, period):
    df = df.dropna(subset=['Close']).copy()
    df.sort_index(inplace=True)
    if len(df) < period:
        return np.nan
    try:
        return (df['Close'].iloc[-1] / df['Close'].iloc[-period]) - 1
    except Exception as e:
        print(f"Return calculation error: {e}")
        return np.nan

def process_symbol(t, start, end):
    try:
        df = download_stock_data(t, start, end)
        if df.empty:
            return None

        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        df['Daily Return'] = df['Close'].pct_change()
        vol = df['Daily Return'].dropna().std() * np.sqrt(63)

        r3 = calculate_returns(df, 63) if len(df) >= 63 else np.nan
        r1 = calculate_returns(df, 21) if len(df) >= 21 else np.nan
        r0 = calculate_returns(df, 5) if len(df) >= 5 else np.nan

        if vol and pd.notna(r3) and pd.notna(r1) and pd.notna(r0):
            mom = ((0.6 * r3) + (0.3 * r1) + (0.1 * r0)) / vol
        else:
            mom = np.nan

        return {
            "Ticker": t,
            "Momentum Score": mom,
            "3-Month Return (%)": r3 * 100 if pd.notna(r3) else np.nan,
            "1-Month Return (%)": r1 * 100 if pd.notna(r1) else np.nan,
            "1-Week Return (%)": r0 * 100 if pd.notna(r0) else np.nan,
            "Annualized Volatility": vol
        }
    except Exception as e:
        print(f"Error processing {t}: {e}")
        return None

@st.cache_data(show_spinner=False)
def analyze_universe(name, symbols):
    end = datetime.today().date()
    start = end - timedelta(days=400)
    rows = []

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks
        futures = {executor.submit(process_symbol, t, start, end): t for t in symbols}
        
        # Create progress bar
        progress_bar = stqdm(total=len(symbols), desc=f"Processing {name}")
        
        # Process completed tasks
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                rows.append(result)
            progress_bar.update(1)
        
        progress_bar.close()

    df_res = pd.DataFrame(rows)
    avg_score = df_res["Momentum Score"].mean() if not df_res.empty else np.nan
    return df_res, avg_score

def get_top_universes_by_momentum():
    data = []
    for name, syms in stqdm(STOCK_UNIVERSE.items(), desc="Processing Universes", leave=False):
        _, avg = analyze_universe(name, syms)
        data.append({"Stock Universe": name, "Average Momentum Score": avg})
    return pd.DataFrame(data).sort_values("Average Momentum Score", ascending=False)

def get_top_stocks_from_universe(name, symbols):
    df, _ = analyze_universe(name, symbols)
    return df.sort_values("Momentum Score", ascending=False) if not df.empty else pd.DataFrame()

def get_top_momentum_stocks_overall():
    all_dfs = []
    for name, syms in stqdm(STOCK_UNIVERSE.items(), desc="Processing All Universes", leave=False):
        df, _ = analyze_universe(name, syms)
        if not df.empty:
            all_dfs.append(df)
    if not all_dfs:
        return pd.DataFrame()
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.dropna(subset=["Momentum Score"], inplace=True)
    combined.sort_values("Momentum Score", ascending=False, inplace=True)
    unique = combined.drop_duplicates(subset=["Ticker"], keep="first")
    return unique.head(10)

def display_loading():
    return st.markdown(f"""
        <div class='loading-container'>
            <div class="spinner"></div>
            <div>{LOADING_TEXT}</div>
        </div>
    """, unsafe_allow_html=True)

def main():
    # Analyze Stock Universe section
    if st.session_state.analyze_button_clicked:
        st.subheader(f"Momentum Analysis: {stock_universe_name}")
        loading_placeholder = st.empty()
        
        # Only show loading when we start processing
        with loading_placeholder.container():
            display_loading()
        
        df, _ = analyze_universe(stock_universe_name, STOCK_UNIVERSE[stock_universe_name])
        
        # Clear loading animation
        loading_placeholder.empty()
        
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

    # Stock Universes Ranks section
    if st.session_state.view_universe_rankings:
        st.subheader("Stock Universes Rankings by Average Momentum")
        loading_placeholder = st.empty()
        
        with loading_placeholder.container():
            display_loading()
        
        top_unis = get_top_universes_by_momentum()
        
        loading_placeholder.empty()
        
        if not top_unis.empty:
            st.dataframe(top_unis.style.format({"Average Momentum Score": "{:.4f}"}), 
                         use_container_width=True)
        else:
            st.warning("No data available for universes ranking.")
        st.session_state.view_universe_rankings = False

    # Recommended Stocks section
    if st.session_state.view_recommended_stocks:
        st.subheader("Recommended Stocks (Top 5 from Top 3 Universes)")
        loading_placeholder = st.empty()
        
        with loading_placeholder.container():
            display_loading()
        
        top_unis = get_top_universes_by_momentum().head(3)
        
        loading_placeholder.empty()
        
        if top_unis.empty:
            st.warning("No universe data available.")
        else:
            for index, row in top_unis.iterrows():
                st.markdown(f"### {row['Stock Universe']} (Avg Score: {row['Average Momentum Score']:.4f})")
                
                universe_loading = st.empty()
                with universe_loading.container():
                    display_loading()
                
                top5 = get_top_stocks_from_universe(
                    row['Stock Universe'], 
                    STOCK_UNIVERSE[row['Stock Universe']]
                )
                
                universe_loading.empty()
                
                if not top5.empty:
                    st.dataframe(top5.head(5).style.format({
                        "3-Month Return (%)": "{:.2f}%",
                        "1-Month Return (%)": "{:.2f}%",
                        "1-Week Return (%)": "{:.2f}%",
                        "Annualized Volatility": "{:.4f}",
                        "Momentum Score": "{:.4f}"
                    }), use_container_width=True)
                else:
                    st.write(f"No stocks data for {row['Stock Universe']}")
        st.session_state.view_recommended_stocks = False

    # High Momentum Stocks section
    if st.session_state.view_high_momentum_stocks:
        st.subheader("Top 10 High Momentum Stocks (Across All Universes)")
        loading_placeholder = st.empty()
        
        with loading_placeholder.container():
            display_loading()
        
        top_momentum = get_top_momentum_stocks_overall()
        
        loading_placeholder.empty()
        
        if not top_momentum.empty:
            st.dataframe(top_momentum.style.format({
                "3-Month Return (%)": "{:.2f}%",
                "1-Month Return (%)": "{:.2f}%",
                "1-Week Return (%)": "{:.2f}%",
                "Annualized Volatility": "{:.4f}",
                "Momentum Score": "{:.4f}"
            }), use_container_width=True)
        else:
            st.warning("No high momentum data available.")
        st.session_state.view_high_momentum_stocks = False

# Add this at the very end of your script
if __name__ == "__main__":
    if not any([
        st.session_state.analyze_button_clicked,
        st.session_state.view_universe_rankings,
        st.session_state.view_recommended_stocks,
        st.session_state.view_high_momentum_stocks
    ]):
        st.write("Select an option from the sidebar to begin analysis")
    else:
        main()
