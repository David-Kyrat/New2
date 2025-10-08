import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import altair as alt
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
                # Normalize to date-only index to handle timezones
                hist.index = hist.index.tz_localize(None).normalize()
                result = pd.DataFrame({tickers[0]: hist['Close']})
            else:
                # Multiple tickers - download individually to avoid rate limits
                all_data = {}
                failed_tickers = []
                ticker_info = []
                
                for ticker in tickers:
                    ticker_obj = yf.Ticker(ticker)
                    hist = ticker_obj.history(start=start_date, end=end_date, auto_adjust=False)
                    if not hist.empty:
                        # Normalize to date-only index to handle timezone mismatches
                        hist.index = hist.index.tz_localize(None).normalize()
                        all_data[ticker] = hist['Close']
                        ticker_info.append(f"✓ {ticker}: {len(hist)} days")
                    else:
                        failed_tickers.append(ticker)
                        ticker_info.append(f"✗ {ticker}: No data")
                        time.sleep(0.5)  # Small delay between attempts
                
                if not all_data:
                    error_details = "\n".join(ticker_info)
                    return None, f"No data returned for any ticker.\n\nTicker Status:\n{error_details}\n\nPlease verify ticker symbols are correct."
                
                # Combine data and forward-fill to handle different trading days
                result = pd.DataFrame(all_data)
                
                # Log which tickers succeeded/failed
                if failed_tickers:
                    failed_str = ", ".join(failed_tickers)
                    return result, f"Warning: No data for {failed_str}. Proceeding with available tickers."
            
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
    "1 Year": 365,
    "2 Years": 730,
    "3 Years": 1095,
    "4 Years": 1460,
    "5 Years": 1825,
    "10 Years": 3650
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
    
    # Handle download errors and warnings
    if error and price_data is None:
        # Critical error - no data at all
        st.error(error)
        if 'rate limit' in error.lower():
            st.info("💡 **Tips to avoid rate limits:**\n"
                   "- Wait 30-60 seconds before retrying\n"
                   "- Try with fewer tickers\n"
                   "- Use a shorter time range\n"
                   "- The app caches data for 1 hour, so successful downloads won't need to re-fetch")
    elif price_data is not None and not price_data.empty:
        # Show warnings if some tickers failed but we have data
        if error and 'Warning' in error:
            st.warning(error)
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
            # Check if data range matches requested range
            actual_start = portfolio_value.index[0]
            actual_end = portfolio_value.index[-1]
            actual_days = (actual_end - actual_start).days
            requested_days = days
            
            # Show warning if data is significantly less than requested (allow 10% tolerance for weekends/holidays)
            if actual_days < requested_days * 0.7:  # Less than 70% of requested range
                days_short = requested_days - actual_days
                st.warning(
                    f"⚠️ **Limited Data Available**\n\n"
                    f"Requested: {selected_range} ({requested_days} days)\n\n"
                    f"Available: {actual_days} days (from {actual_start.strftime('%Y-%m-%d')} to {actual_end.strftime('%Y-%m-%d')})\n\n"
                    f"Data is {days_short} days shorter than requested. Some tickers may have limited historical data."
                )
            
            # Compute metrics
            initial_value = portfolio_value.iloc[0]
            final_value = portfolio_value.iloc[-1]
            total_return = ((final_value - initial_value) / initial_value) * 100
            dollar_change = final_value - initial_value
            
            # Display metrics
            portfolio_value_col, total_return_col, volatility_col = st.columns(3)
            
            with portfolio_value_col:
                st.metric("Portfolio Value", f"${final_value:,.2f}", 
                         delta=f"{total_return:+.2f}%", border=True)
            
            with total_return_col:
                st.metric("Total Return", f"{total_return:+.2f}%", border=True)
            
            with volatility_col:
                if len(portfolio_value) > 1:
                    volatility = compute_volatility(portfolio_value)
                    st.metric("Volatility (Ann.)", f"{volatility:.2f}%", border=True)
                else:
                    st.metric("Volatility", "N/A", border=True)
            
            # Additional metrics
            initial_value_col, max_dd_col, cagr_col, sharpe_col = st.columns(4)
            
            with initial_value_col:
                st.metric("Initial Value", f"${initial_value:,.2f}", 
                         delta=f"${dollar_change:+,.2f}", border=True)
            
            with max_dd_col:
                if len(portfolio_value) > 1:
                    max_dd = compute_max_drawdown(portfolio_value)
                    st.metric("Max Drawdown", f"{max_dd:.2f}%", border=True)
                else:
                    st.metric("Max Drawdown", "N/A", border=True)
            
            # CAGR and Sharpe Ratio - only for 1 year+ data
            needs_yearly_data = range_options[selected_range] < range_options["1 Year"]
            
            # Initialize session state for yearly metrics
            if 'yearly_cagr' not in st.session_state:
                st.session_state.yearly_cagr = None
            if 'yearly_sharpe' not in st.session_state:
                st.session_state.yearly_sharpe = None
            
            with cagr_col:
                if not needs_yearly_data and len(portfolio_value) > 1:
                    cagr = compute_cagr(portfolio_value)
                    st.metric("CAGR", f"{cagr:.2f}%", border=True)
                elif st.session_state.yearly_cagr is not None:
                    st.metric("CAGR", f"{st.session_state.yearly_cagr:.2f}%", 
                             help="Requires at least 1 year of data for meaningful results. Computed over 1 year.",
                             border=True)
                else:
                    st.metric("CAGR", "—",
                             help="Requires at least 1 year of data for meaningful results. Click button below to compute.",
                             border=True)
            
            with sharpe_col:
                if not needs_yearly_data and len(portfolio_value) > 1:
                    sharpe = compute_sharpe_ratio(portfolio_value)
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}", border=True)
                elif st.session_state.yearly_sharpe is not None:
                    st.metric("Sharpe Ratio", f"{st.session_state.yearly_sharpe:.2f}",
                             help="Requires at least 1 year of data for meaningful results. Computed over 1 year.",
                             border=True)
                else:
                    st.metric("Sharpe Ratio", "—",
                             help="Requires at least 1 year of data for meaningful results. Click button below to compute.",
                             border=True)
            
            # Button to compute annualized metrics for short ranges
            if needs_yearly_data:
                if st.button("📊 Compute Annualized Metrics (Downloads 1 Year Data)", type="secondary"):
                    with st.spinner("Downloading 1 year of data..."):
                        year_start = datetime.now() - timedelta(days=365)
                        year_end = datetime.now()
                        yearly_data, yearly_error = download_data(tickers, year_start, year_end)
                    
                    if yearly_error:
                        st.error(f"Failed to download yearly data: {yearly_error}")
                        st.session_state.yearly_cagr = None
                        st.session_state.yearly_sharpe = None
                    elif yearly_data is not None and not yearly_data.empty:
                        # Calculate portfolio value for 1 year
                        yearly_portfolio = pd.Series(0.0, index=yearly_data.index)
                        for ticker, shares in holdings.items():
                            if ticker in yearly_data.columns:
                                yearly_portfolio += yearly_data[ticker] * shares
                        
                        yearly_portfolio = yearly_portfolio.dropna()
                        
                        if len(yearly_portfolio) > 1:
                            st.session_state.yearly_cagr = compute_cagr(yearly_portfolio)
                            st.session_state.yearly_sharpe = compute_sharpe_ratio(yearly_portfolio)
                            st.rerun()
                        else:
                            st.warning("Insufficient yearly data to compute metrics.")
                            st.session_state.yearly_cagr = None
                            st.session_state.yearly_sharpe = None
                    else:
                        st.error("Failed to download yearly data.")
                        st.session_state.yearly_cagr = None
                        st.session_state.yearly_sharpe = None
            
            # Chart
            st.subheader(f"Portfolio Value - {selected_range}")
            
            # Prepare data for Altair chart
            chart_data = pd.DataFrame({
                'Date': portfolio_value.index,
                'Portfolio Value': portfolio_value.values
            })
            
            # Create Altair chart with custom y-axis and baseline
            base_chart = alt.Chart(chart_data).encode(
                x=alt.X('Date:T', title='Date'),
                y=alt.Y('Portfolio Value:Q', 
                       title='Portfolio Value ($)',
                       scale=alt.Scale(zero=False))
            )
            
            # Line for portfolio value
            line = base_chart.mark_line(color='#1f77b4', size=2).encode(
                tooltip=[
                    alt.Tooltip('Date:T', format='%Y-%m-%d'),
                    alt.Tooltip('Portfolio Value:Q', format='$,.2f')
                ]
            )
            
            # Horizontal line for initial investment
            initial_line = alt.Chart(pd.DataFrame({
                'y': [initial_value]
            })).mark_rule(color='gray', strokeDash=[5, 5], opacity=0.6).encode(
                y='y:Q',
                tooltip=alt.value(f'Initial Investment: ${initial_value:,.2f}')
            )
            
            # Combine charts
            chart = (line + initial_line).properties(
                height=500
            ).interactive()
            
            st.altair_chart(chart, use_container_width=True)
            
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
