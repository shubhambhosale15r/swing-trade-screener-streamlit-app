import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from stocklist import *

# Configuration
PAGE_TITLE = "Swing Trade"
PAGE_ICON = "📈"
LOADING_TEXT = "Analyzing Stocks..."

# --- Basic App Setup ---
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    menu_items=None,  # Hides the top-right menu button
)

# --- Session State Initialization ---
def initialize_session_state():
    if 'view_universe_rankings' not in st.session_state:
        st.session_state.view_universe_rankings = False
    if 'view_recommended_stocks' not in st.session_state:
        st.session_state.view_recommended_stocks = False
    if 'analyze_button_clicked' not in st.session_state:
        st.session_state.analyze_button_clicked = False

initialize_session_state()

# --- CSS Styling ---
def inject_custom_css():
    st.markdown(f"""
        <style>
            /* General Body and Background */
            body {{
                background: var(--bg-color);
                color: var(--text-color);
                font-family: 'Inter', sans-serif;
                transition: all 0.3s ease-in-out;
            }}

            /* Hide Header and Footer */
            header, footer {{visibility: hidden;}}

            /* Color Scheme (Light Mode) */
            @media (prefers-color-scheme: light) {{
                :root {{
                    --bg-color: #f4f4f7;
                    --text-color: #222;
                    --card-bg: rgba(255, 255, 255, 0.8);
                    --border-color: #ddd;
                    --primary-color: #0078FF;
                    --secondary-color: #0056B3;
                    --hover-bg: rgba(0, 120, 255, 0.1);
                    --shadow-color: rgba(0, 0, 0, 0.1);
                }}
            }}

            /* Color Scheme (Dark Mode) */
            @media (prefers-color-scheme: dark) {{
                :root {{
                    --bg-color: #0F172A;
                    --text-color: #E5E7EB;
                    --card-bg: rgba(30, 41, 59, 0.9);
                    --border-color: #1E293B;
                    --primary-color: #64FFDA;
                    --secondary-color: #4ECDC4;
                    --hover-bg: rgba(100, 255, 218, 0.1);
                    --shadow-color: rgba(0, 0, 0, 0.5);
                }}
            }}

            /* Centered Content */
            .center-div {{
                display: flex;
                text-align: center;
                justify-content: center;
                width: 100%;
            }}

            /* Loading Animation */
            .loading-container {{
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 200px;
                text-align: center;
                background: var(--card-bg);
                border-radius: 12px;
                box-shadow: 0 6px 16px var(--shadow-color);
                margin: 20px 0;
            }}
            .loading-text {{
                color: var(--primary-color);
                font-size: 1.2rem;
                margin-top: 20px;
            }}

            /* Responsive Adjustments */
            @media (max-width: 768px) {{
                /* Example adjustment for smaller screens */
                h1 {{
                    font-size: 2em;  /* Reduce title size on smaller screens */
                }}
            }}
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# --- Title and Subtitle ---
def display_header():
    st.markdown(f"""
        <h1 style='text-align: center;'>{PAGE_ICON} {PAGE_TITLE}</h1>
        <div style="text-align: center; font-size: 1.2rem; color: #c0c0c0;">
            Select a stock universe then click on Analyse Stock Universe Button to Find momentum stocks in that specific sector/index<br><br>
            Click on Stock Universes Rank to get the sector/indices that are in momentum<br><br>
            Click on Recommended Stock to get stock from sector/indices which are in momentum<br><br>
        </div>
    """, unsafe_allow_html=True)

display_header()

# --- Sidebar ---
def create_sidebar():
    with st.sidebar:
        stock_universe_name = st.radio(
            "Select Stock Universe",
            list(STOCK_UNIVERSE.keys())
        )
        selected_stocks = STOCK_UNIVERSE[stock_universe_name]
        st.info(f"You have selected the {stock_universe_name} stock list.")

        # Add the "Analyze Stock Universe" Button
        if st.button("Analyze Stock Universe", key="analyze_stock_universe_sidebar"):
            st.session_state.analyze_button_clicked = True  # Set the state to True for analysis
            st.rerun()  # Rerun to trigger the analysis logic

        if st.button("Stock Universes Ranks", key="universe_ranks_sidebar"):
            st.session_state.view_universe_rankings = True
            st.rerun()

        # Add the "Recommended Stocks" Button
        if st.button("Recommended Stocks", key="recommended_stocks_sidebar"):
            st.session_state.view_recommended_stocks = True
            st.rerun()

    return stock_universe_name, selected_stocks

stock_universe_name, selected_stocks = create_sidebar()

# --- Data Download ---
@st.cache_data(show_spinner=False)
def download_stock_data(ticker, start_date, end_date, retries=3):
    for _ in range(retries):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if df.empty:
                raise ValueError(f"No data returned for {ticker}")
            df['Date'] = df.index
            df.reset_index(drop=True, inplace=True)
            return df
        except Exception as e:
            print(f"Retrying {ticker}: {e}")
    print(f"Failed to download data for {ticker} after {retries} retries")
    return pd.DataFrame()

# --- Calculate Returns ---
def calculate_returns(df, period):
    if len(df) >= period:
        close_prices = df['Close'].values
        end_price = close_prices[-1]
        start_price = close_prices[-period]
        return ((end_price - start_price) / start_price).item()
    else:
        return np.nan

# --- Analyze Universe ---
def analyze_universe(universe_name, universe_symbols):
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=400)
    results = []

    for ticker in universe_symbols:
        df = download_stock_data(ticker, start_date, end_date)
        if df.empty:
            continue

        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df[~df.index.duplicated()]

        # Use daily returns for volatility calculation
        df['Daily Return'] = df['Close'].pct_change()
        daily_volatility = df['Daily Return'].dropna().std() * np.sqrt(63)

        # Calculate returns using the new periods:
        # 3-month return (≈63 trading days), 1-month return (≈21 trading days), and 1-week return (≈5 trading days)
        return_3M = calculate_returns(df, 63)
        return_1M = calculate_returns(df, 21)
        return_1W = calculate_returns(df, 5)

        # Compute momentum score using the new returns and daily volatility
        if (daily_volatility != 0 and pd.notna(return_3M) and pd.notna(return_1M) and pd.notna(return_1W)):
            momentum_score = ((0.6 * return_3M) + (0.3 * return_1M) + (0.1 * return_1W)) / daily_volatility
        else:
            momentum_score = np.nan

        results.append({
            "Ticker": ticker,
            "Momentum Score": momentum_score,
            "3-Month Return (%)": return_3M * 100 if pd.notna(return_3M) else np.nan,
            "1-Month Return (%)": return_1M * 100 if pd.notna(return_1M) else np.nan,
            "1-Week Return (%)": return_1W * 100 if pd.notna(return_1W) else np.nan,
            "Annualized Volatility": daily_volatility
        })

    results_df = pd.DataFrame(results)
    avg_momentum_score = results_df["Momentum Score"].mean() if not results_df.empty else np.nan
    return results_df, avg_momentum_score

# --- Get Top Universes by Momentum ---
def get_top_universes_by_momentum():
    universe_results = []
    for name, symbols in STOCK_UNIVERSE.items():
        _, avg_momentum = analyze_universe(name, symbols)
        universe_results.append({
            "Stock Universe": name,
            "Average Momentum Score": avg_momentum
        })

    # Sort the universes by Average Momentum Score
    universe_df = pd.DataFrame(universe_results)
    universe_df.sort_values(by="Average Momentum Score", ascending=False, inplace=True)

    # Get the top 5 universes based on the momentum score
    top_universes = universe_df
    return top_universes

# --- Get Top Stocks from Universe ---
def get_top_stocks_from_universe(universe_name, universe_symbols):
    results_df, _ = analyze_universe(universe_name, universe_symbols)
    if not results_df.empty:
        # Sort by momentum score
        results_df.sort_values(by="Momentum Score", ascending=False, inplace=True)
        # Get the top 5 stocks by momentum score
        top_stocks = results_df
        return top_stocks
    else:
        return pd.DataFrame()

# --- Main Application ---
def main():
    # Analyse Stock Universe Button
    if st.session_state.get('analyze_button_clicked', False):
        st.subheader(f"Momentum Analysis for {stock_universe_name}")

        # Placeholder for analysis results
        analysis_placeholder = st.empty()

        try:
            # Show loading container
            with analysis_placeholder:
                st.markdown(f"""
                    <div class="loading-container">
                        <div class="lds-ripple"><div></div><div></div></div>
                        <p class="loading-text">{LOADING_TEXT}</p>
                    </div>
                """, unsafe_allow_html=True)

            # Perform analysis and display results
            results_df, _ = analyze_universe(stock_universe_name, selected_stocks)

            # Clear loading container
            analysis_placeholder.empty()

            if not results_df.empty:
                # Sort by Momentum Score
                results_df.sort_values(by="Momentum Score", ascending=False, inplace=True)

                # Format and display the dataframe
                styled_df = results_df.style.format({
                    "3-Month Return (%)": "{:.2f}%",
                    "1-Month Return (%)": "{:.2f}%",
                    "1-Week Return (%)": "{:.2f}%",
                    "Annualized Volatility": "{:.4f}",
                    "Momentum Score": "{:.4f}"
                })

                st.dataframe(styled_df, use_container_width=True)
            else:
                st.warning("No data available for the selected stock universe.")

        except Exception as e:
            analysis_placeholder.empty()
            st.error(f"An error occurred: {str(e)}")
        finally:
            st.session_state.analyze_button_clicked = False  # Reset the button state

    # Recommended Stock Button
    if st.session_state.get('view_recommended_stocks', False):
        # Show the loading animation when the "Recommended Stocks" button is clicked
        loading_container = st.empty()
        loading_container.markdown(f"""
            <div class="loading-container">
                <div class="lds-ripple"><div></div><div></div></div>
                <p class="loading-text">{LOADING_TEXT}</p>
            </div>
        """, unsafe_allow_html=True)

        try:
            # Wait for the data to be processed
            top_universes = get_top_universes_by_momentum()

            # Once data is ready, clear the loading animation
            loading_container.empty()

            st.subheader("Stock Universes Based on Momentum Score")

            for _, row in top_universes.iterrows():
                st.markdown(f"### {row['Stock Universe']} (Momentum Score: {row['Average Momentum Score']:.4f})")

                # Get and display the top 5 stocks from this universe
                universe_name = row['Stock Universe']
                universe_symbols = STOCK_UNIVERSE[universe_name]
                top_stocks = get_top_stocks_from_universe(universe_name, universe_symbols)

                if not top_stocks.empty:
                    st.dataframe(
                        top_stocks[['Ticker', 'Momentum Score', '3-Month Return (%)', '1-Month Return (%)',
                                    '1-Week Return (%)']],
                        height=300
                    )
                else:
                    st.warning(f"No stocks data available for {universe_name}.")

        except Exception as e:
            loading_container.empty()
            st.error(f"An error occurred: {str(e)}")
        finally:
            st.session_state.view_recommended_stocks = False  # Reset the state

    # Existing code to handle universe rankings or stock analysis
    if st.session_state.get('view_universe_rankings', False):
        st.subheader("Stock Universes Ranking by Average Momentum Score")
        universe_results = []
        progress_container = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            for idx, (name, symbols) in enumerate(STOCK_UNIVERSE.items()):
                with progress_container:
                    status_text.text(f"🔄 Analyzing {name}...")
                    _, avg_momentum = analyze_universe(name, symbols)
                    universe_results.append({
                        "Stock Universe": name,
                        "Average Momentum Score": avg_momentum
                    })
                    progress = (idx + 1) / len(STOCK_UNIVERSE)
                    progress_bar.progress(progress)

            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            progress_container.empty()

            # Display results
            if universe_results:
                universe_df = pd.DataFrame(universe_results)
                universe_df.sort_values(by="Average Momentum Score", ascending=False, inplace=True)

                st.dataframe(
                    universe_df.style
                    .format({'Average Momentum Score': '{:.4f}'}),
                    height=400
                )

                # Display metrics
                st.subheader("Summary")
                col1, col2 = st.columns(2)
                with col1:
                    best_universe = universe_df.iloc[0]
                    st.metric(
                        "Highest Momentum Universe",
                        best_universe["Stock Universe"],
                        f"{best_universe['Average Momentum Score']:.4f}"
                    )
                with col2:
                    worst_universe = universe_df.iloc[-1]
                    st.metric(
                        "Lowest Momentum Universe",
                        worst_universe["Stock Universe"],
                        f"{worst_universe['Average Momentum Score']:.4f}"
                    )

            # Reset the view state
            st.session_state.view_universe_rankings = False

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.session_state.view_universe_rankings = False
        finally:
            st.session_state.analyze_button_clicked = False

    if st.session_state.get('analyze_button_clicked', False):
        st.subheader(f"Momentum Analysis for {stock_universe_name}")

        # Placeholder for analysis results
        analysis_placeholder = st.empty()

        try:
            # Show loading container
            with analysis_placeholder:
                st.markdown(f"""
                    <div class="loading-container">
                        <div class="lds-ripple"><div></div><div></div></div>
                        <p class="loading-text">{LOADING_TEXT}</p>
                    </div>
                """, unsafe_allow_html=True)

            # Perform analysis and display results
            results_df, _ = analyze_universe(stock_universe_name, selected_stocks)

            # Clear loading container
            analysis_placeholder.empty()

            if not results_df.empty:
                # Sort by Momentum Score
                results_df.sort_values(by="Momentum Score", ascending=False, inplace=True)

                # Format and display the dataframe
                styled_df = results_df.style.format({
                    "3-Month Return (%)": "{:.2f}%",
                    "1-Month Return (%)": "{:.2f}%",
                    "1-Week Return (%)": "{:.2f}%",
                    "Annualized Volatility": "{:.4f}",
                    "Momentum Score": "{:.4f}"
                })

                st.dataframe(styled_df, use_container_width=True)
            else:
                st.warning("No data available for the selected stock universe.")

        except Exception as e:
            analysis_placeholder.empty()
            st.error(f"An error occurred: {str(e)}")
        finally:
            st.session_state.analyze_button_clicked = False

if __name__ == '__main__':
    main()
