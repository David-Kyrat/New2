import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Portfolio Performance", layout="wide")

@st.cache_data(ttl=3600)
def download_data(tickers, start_date, end_date):
    """Download historical data for multiple tickers in one batch call."""
    if not tickers:
        return None
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)
    if len(tickers) == 1:
        return pd.DataFrame({tickers[0]: data['Adj Close']})
    return data['Adj Close']

def calculate_cagr(portfolio_values):
    """Calculate Compound Annual Growth Rate."""
    days = (portfolio_values.index[-1] - portfolio_values.index[0]).days
    if days == 0:
        return 0.0
    years = days / 365.25
    total_return = portfolio_values.iloc[-1] / portfolio_values.iloc[0]
    return (total_return ** (1 / years) - 1) * 100

def calculate_volatility(portfolio_values):
    """Calculate annualized volatility."""
    returns = portfolio_values.pct_change().dropna()
    return returns.std() * np.sqrt(252) * 100

def calculate_max_drawdown(portfolio_values):
    """Calculate maximum drawdown."""
    cumulative_max = portfolio_values.cummax()
    drawdown = (portfolio_values - cumulative_max) / cumulative_max * 100
    return drawdown.min()

def calculate_sharpe_ratio(portfolio_values, risk_free_rate=0.0):
    """Calculate annualized Sharpe ratio."""
    returns = portfolio_values.pct_change().dropna()
    if len(returns) == 0:
        return 0.0
    excess_returns = returns - (risk_free_rate / 252)
    if excess_returns.std() == 0:
        return 0.0
    sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    return sharpe

def parse_portfolio_input(input_string):
    """Parse ticker:shares input string."""
    holdings = {}
    if not input_string.strip():
        return holdings
    
    pairs = [p.strip() for p in input_string.split(',')]
    for pair in pairs:
        if ':' in pair:
            ticker, shares = pair.split(':', 1)
            ticker = ticker.strip().upper()
            try:
                shares = float(shares.strip())
                if shares > 0:
                    holdings[ticker] = shares
            except ValueError:
                st.sidebar.warning(f"Invalid shares for {ticker}")
    return holdings

# Streamlit UI
st.title("📈 Portfolio Performance Tracker")

# Sidebar inputs
st.sidebar.header("Portfolio Configuration")

portfolio_input = st.sidebar.text_area(
    "Enter holdings (TICKER:SHARES)",
    value="AAPL:10, MSFT:5, GOOGL:3",
    height=100,
    help="Format: TICKER:SHARES, separated by commas. Example: AAPL:10, MSFT:5"
)

range_options = {
    "2 Weeks": 14,
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365
}

selected_range = st.sidebar.selectbox(
    "Time Range",
    options=list(range_options.keys()),
    index=3  # Default to 6 Months
)

# Parse portfolio
holdings = parse_portfolio_input(portfolio_input)

if holdings:
    # Calculate date range
    end_date = datetime.now()
    days = range_options[selected_range]
    start_date = end_date - timedelta(days=days)
    
    # Download data
    tickers = list(holdings.keys())
    
    with st.spinner(f"Downloading data for {', '.join(tickers)}..."):
        price_data = download_data(tickers, start_date, end_date)
    
    if price_data is not None and not price_data.empty:
        # Calculate portfolio value
        portfolio_value = pd.Series(0.0, index=price_data.index)
        
        for ticker, shares in holdings.items():
            if ticker in price_data.columns:
                portfolio_value += price_data[ticker] * shares
            else:
                st.warning(f"No data available for {ticker}")
        
        # Remove any NaN values
        portfolio_value = portfolio_value.dropna()
        
        if len(portfolio_value) > 0:
            # Calculate metrics
            initial_value = portfolio_value.iloc[0]
            final_value = portfolio_value.iloc[-1]
            total_return = ((final_value - initial_value) / initial_value) * 100
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Portfolio Value", f"${final_value:,.2f}")
            
            with col2:
                st.metric("Total Return", f"{total_return:+.2f}%")
            
            with col3:
                if len(portfolio_value) > 1:
                    cagr = calculate_cagr(portfolio_value)
                    st.metric("CAGR", f"{cagr:.2f}%")
                else:
                    st.metric("CAGR", "N/A")
            
            with col4:
                if len(portfolio_value) > 1:
                    volatility = calculate_volatility(portfolio_value)
                    st.metric("Volatility (Ann.)", f"{volatility:.2f}%")
                else:
                    st.metric("Volatility", "N/A")
            
            # Additional metrics
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                st.metric("Initial Value", f"${initial_value:,.2f}")
            
            with col6:
                if len(portfolio_value) > 1:
                    max_dd = calculate_max_drawdown(portfolio_value)
                    st.metric("Max Drawdown", f"{max_dd:.2f}%")
                else:
                    st.metric("Max Drawdown", "N/A")
            
            with col7:
                if len(portfolio_value) > 1:
                    sharpe = calculate_sharpe_ratio(portfolio_value)
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                else:
                    st.metric("Sharpe Ratio", "N/A")
            
            with col8:
                st.metric("Holdings", len(holdings))
            
            # Chart
            st.subheader(f"Portfolio Value - {selected_range}")
            
            chart_data = pd.DataFrame({
                'Portfolio Value': portfolio_value
            })
            
            st.line_chart(chart_data, height=500)
            
            # Show holdings breakdown
            with st.expander("📊 Holdings Breakdown"):
                holdings_data = []
                for ticker, shares in holdings.items():
                    if ticker in price_data.columns:
                        current_price = price_data[ticker].dropna().iloc[-1]
                        position_value = current_price * shares
                        weight = (position_value / final_value) * 100
                        holdings_data.append({
                            'Ticker': ticker,
                            'Shares': shares,
                            'Price': f"${current_price:.2f}",
                            'Value': f"${position_value:,.2f}",
                            'Weight': f"{weight:.2f}%"
                        })
                
                if holdings_data:
                    st.dataframe(
                        pd.DataFrame(holdings_data),
                        use_container_width=True,
                        hide_index=True
                    )
        else:
            st.error("No valid data available for the selected time range.")
    else:
        st.error("Failed to download data. Please check ticker symbols.")
else:
    st.info("👈 Enter your portfolio holdings in the sidebar to get started.")
    st.markdown("""
    ### How to use:
    1. Enter your holdings in the format `TICKER:SHARES`
    2. Separate multiple holdings with commas
    3. Select a time range
    4. View your portfolio performance!
    
    **Example:** `AAPL:10, MSFT:5, GOOGL:3`
    """)
