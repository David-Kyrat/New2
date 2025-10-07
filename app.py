import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

st.set_page_config(page_title="Portfolio Performance", layout="wide")

@st.cache_data(ttl=3600)
def download_data(
    tickers: List[str], 
    start_date: datetime, 
    end_date: datetime
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Download close prices for multiple tickers via yfinance.

    Downloads each ticker individually using yf.Ticker() to avoid rate limits.
    Implements retry logic with exponential backoff for rate limit errors.
    Uses auto_adjust=False and returns Close prices (equivalent to Adj Close).

    Args:
        tickers: List of ticker symbols (e.g., ['AAPL', 'MSFT']).
        start_date: Start date for historical data.
        end_date: End date for historical data.

    Returns:
        Tuple of (DataFrame with tickers as columns and dates as index or None, 
                 error message string or None).
    """
    if not tickers:
        return None, None
    
    max_retries = 3
    base_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            # Use individual Ticker objects for better reliability
            if len(tickers) == 1:
                # Single ticker
                ticker_obj = yf.Ticker(tickers[0])
                hist = ticker_obj.history(start=start_date, end=end_date, auto_adjust=False)
                if hist.empty:
                    return None, f"No data returned for {tickers[0]}. Please verify the ticker symbol."
                result = pd.DataFrame({tickers[0]: hist['Close']})
            else:
                # Multiple tickers - download individually to avoid rate limits
                all_data = {}
                for ticker in tickers:
                    ticker_obj = yf.Ticker(ticker)
                    hist = ticker_obj.history(start=start_date, end=end_date, auto_adjust=False)
                    if not hist.empty:
                        all_data[ticker] = hist['Close']
                    else:
                        # Try with batch download as fallback
                        time.sleep(0.5)  # Small delay between attempts
                
                if not all_data:
                    return None, "No data returned for any ticker. Please verify ticker symbols."
                
                result = pd.DataFrame(all_data)
            
            return result, None
            
        except Exception as e:
            error_msg = str(e)
            
            # Check if it's a rate limit error
            if 'Rate limit' in error_msg or 'Too Many Requests' in error_msg or '429' in error_msg:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                    time.sleep(delay)
                    continue
                else:
                    return None, (
                        "Yahoo Finance rate limit reached. Please wait a minute and try again. "
                        "Tip: Try using fewer tickers or a shorter time range."
                    )
            else:
                return None, f"Error downloading data: {error_msg}"
    
    return None, "Failed to download data after multiple retries."

def compute_cagr(portfolio_values: pd.Series) -> float:
    """Compute Compound Annual Growth Rate (CAGR).

    Computes annualized return assuming continuous compounding over the
    entire period spanned by portfolio_values.

    Args:
        portfolio_values: Time series of portfolio values with DatetimeIndex.

    Returns:
        CAGR as a percentage. Returns 0.0 if period is zero days.
    """
    days = (portfolio_values.index[-1] - portfolio_values.index[0]).days
    if days == 0:
        return 0.0
    years = days / 365.25
    total_return = portfolio_values.iloc[-1] / portfolio_values.iloc[0]
    return (total_return ** (1 / years) - 1) * 100

def compute_volatility(portfolio_values: pd.Series) -> float:
    """Compute annualized volatility (standard deviation of returns).

    Args:
        portfolio_values: Time series of portfolio values with DatetimeIndex.

    Returns:
        Annualized volatility as a percentage (daily std * √252).
    """
    returns = portfolio_values.pct_change().dropna()
    return returns.std() * np.sqrt(252) * 100

def compute_max_drawdown(portfolio_values: pd.Series) -> float:
    """Compute maximum drawdown as peak-to-trough decline.

    Args:
        portfolio_values: Time series of portfolio values with DatetimeIndex.

    Returns:
        Maximum drawdown as a percentage (negative value).
    """
    cumulative_max = portfolio_values.cummax()
    drawdown = (portfolio_values - cumulative_max) / cumulative_max * 100
    return drawdown.min()

def compute_sharpe_ratio(
    portfolio_values: pd.Series, 
    risk_free_rate: float = 0.0
) -> float:
    """Compute annualized Sharpe ratio (risk-adjusted return).

    Args:
        portfolio_values: Time series of portfolio values with DatetimeIndex.
        risk_free_rate: Annual risk-free rate as decimal (default 0.0).

    Returns:
        Annualized Sharpe ratio. Returns 0.0 if std is zero or no returns.
    """
    returns = portfolio_values.pct_change().dropna()
    if len(returns) == 0:
        return 0.0
    excess_returns = returns - (risk_free_rate / 252)
    if excess_returns.std() == 0:
        return 0.0
    sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    return sharpe

def parse_portfolio_input(input_string: str) -> Dict[str, float]:
    """Parse user input string into ticker-shares mapping.

    Expects format: 'TICKER:SHARES, TICKER2:SHARES2, ...'. Ignores invalid
    entries and displays warnings via Streamlit sidebar.

    Args:
        input_string: Raw user input (e.g., 'AAPL:10, MSFT:5').

    Returns:
        Dictionary mapping ticker symbols to share counts (positive floats only).
    """
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
        price_data, error = download_data(tickers, start_date, end_date)
    
    # Handle download errors
    if error:
        st.error(error)
        if 'rate limit' in error.lower():
            st.info("💡 **Tips to avoid rate limits:**\n"
                   "- Wait 30-60 seconds before retrying\n"
                   "- Try with fewer tickers\n"
                   "- Use a shorter time range\n"
                   "- The app caches data for 1 hour, so successful downloads won't need to re-fetch")
    elif price_data is not None and not price_data.empty:
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
            # Compute metrics
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
                    volatility = compute_volatility(portfolio_value)
                    st.metric("Volatility (Ann.)", f"{volatility:.2f}%")
                else:
                    st.metric("Volatility", "N/A")
            
            with col4:
                st.metric("Holdings", len(holdings))
            
            # Additional metrics
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                st.metric("Initial Value", f"${initial_value:,.2f}")
            
            with col6:
                if len(portfolio_value) > 1:
                    max_dd = compute_max_drawdown(portfolio_value)
                    st.metric("Max Drawdown", f"{max_dd:.2f}%")
                else:
                    st.metric("Max Drawdown", "N/A")
            
            # CAGR and Sharpe Ratio - only for 1 year+ data
            needs_yearly_data = selected_range != "1 Year"
            
            with col7:
                if not needs_yearly_data and len(portfolio_value) > 1:
                    cagr = compute_cagr(portfolio_value)
                    st.metric("CAGR", f"{cagr:.2f}%")
                else:
                    st.metric("CAGR", "—")
            
            with col8:
                if not needs_yearly_data and len(portfolio_value) > 1:
                    sharpe = compute_sharpe_ratio(portfolio_value)
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                else:
                    st.metric("Sharpe Ratio", "—")
            
            # Button to compute annualized metrics for short ranges
            if needs_yearly_data:
                st.info("💡 **CAGR and Sharpe Ratio** require at least 1 year of data for meaningful results.")
                if st.button("📊 Compute Annualized Metrics (Downloads 1 Year Data)", type="primary"):
                    with st.spinner("Downloading 1 year of data..."):
                        year_start = datetime.now() - timedelta(days=365)
                        year_end = datetime.now()
                        yearly_data, yearly_error = download_data(tickers, year_start, year_end)
                    
                    if yearly_error:
                        st.error(f"Failed to download yearly data: {yearly_error}")
                    elif yearly_data is not None and not yearly_data.empty:
                        # Calculate portfolio value for 1 year
                        yearly_portfolio = pd.Series(0.0, index=yearly_data.index)
                        for ticker, shares in holdings.items():
                            if ticker in yearly_data.columns:
                                yearly_portfolio += yearly_data[ticker] * shares
                        
                        yearly_portfolio = yearly_portfolio.dropna()
                        
                        if len(yearly_portfolio) > 1:
                            ann_cagr = compute_cagr(yearly_portfolio)
                            ann_sharpe = compute_sharpe_ratio(yearly_portfolio)
                            
                            st.success("✅ Annualized metrics computed over 1 year:")
                            metric_col1, metric_col2 = st.columns(2)
                            with metric_col1:
                                st.metric("CAGR (1 Year)", f"{ann_cagr:.2f}%")
                            with metric_col2:
                                st.metric("Sharpe Ratio (1 Year)", f"{ann_sharpe:.2f}")
                        else:
                            st.warning("Insufficient yearly data to compute metrics.")
                    else:
                        st.error("Failed to download yearly data.")
            
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
    st.info("👈 Enter your portfolio holdings in the sidebar to get started.")
    st.markdown("""
    ### How to use:
    1. Enter your holdings in the format `TICKER:SHARES`
    2. Separate multiple holdings with commas
    3. Select a time range
    4. View your portfolio performance!
    
    **Example:** `AAPL:10, MSFT:5, GOOGL:3`
    """)
