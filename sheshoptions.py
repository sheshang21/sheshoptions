import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from scipy.stats import norm
import time

# Page config
st.set_page_config(
    page_title="Live Options Pricing Model",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    }
    .stDataFrame {
        background-color: rgba(30, 41, 59, 0.5);
    }
    h1 {
        color: #60a5fa;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.5);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

# Black-Scholes Functions
def black_scholes(S, K, T, r, sigma, option_type='call'):
    """Calculate Black-Scholes option price"""
    if T <= 0:
        if option_type == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """Calculate option Greeks"""
    if T <= 0:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Delta
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
    
    # Gamma
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Theta
    if option_type == 'call':
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                 r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                 r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    
    # Vega
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    
    # Rho
    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }

@st.cache_data(ttl=30)
def fetch_stock_data(symbol):
    """Fetch real-time stock data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period='1d', interval='1m')
        if not data.empty:
            current_price = data['Close'].iloc[-1]
            prev_close = ticker.info.get('previousClose', current_price)
            return {
                'price': current_price,
                'prev_close': prev_close,
                'change': current_price - prev_close,
                'change_pct': ((current_price - prev_close) / prev_close) * 100
            }
    except Exception as e:
        st.warning(f"Could not fetch data for {symbol}: {str(e)}")
    return None

def generate_option_chain(symbol, spot_price, risk_free_rate=0.065, implied_vol=0.30):
    """Generate option chain with various strikes and expiries"""
    
    # Generate strike prices
    strikes = [
        int(spot_price * 0.85),
        int(spot_price * 0.90),
        int(spot_price * 0.95),
        int(spot_price),
        int(spot_price * 1.05),
        int(spot_price * 1.10),
        int(spot_price * 1.15)
    ]
    
    # Generate expiry dates
    expiries = [
        (datetime.now() + timedelta(days=16), "30 Dec 25"),
        (datetime.now() + timedelta(days=47), "30 Jan 26"),
        (datetime.now() + timedelta(days=103), "27 Mar 26")
    ]
    
    options_data = []
    
    for exp_date, exp_label in expiries:
        T = (exp_date - datetime.now()).days / 365.0
        
        for strike in strikes:
            for opt_type in ['call', 'put']:
                # Calculate option price and Greeks
                theo_price = black_scholes(spot_price, strike, T, risk_free_rate, implied_vol, opt_type)
                
                if opt_type == 'call':
                    intrinsic = max(spot_price - strike, 0)
                else:
                    intrinsic = max(strike - spot_price, 0)
                
                time_value = theo_price - intrinsic
                greeks = calculate_greeks(spot_price, strike, T, risk_free_rate, implied_vol, opt_type)
                
                # Generate option code
                option_code = f"{symbol.replace('.NS', '').replace('.BO', '')} {exp_label} {strike} {'CE' if opt_type == 'call' else 'PE'}"
                
                options_data.append({
                    'Option Code': option_code,
                    'Underlying': symbol,
                    'Spot Price': spot_price,
                    'Strike': strike,
                    'Type': opt_type.upper(),
                    'Expiry': exp_label,
                    'Days to Expiry': (exp_date - datetime.now()).days,
                    'Theoretical Price': theo_price,
                    'Intrinsic Value': intrinsic,
                    'Time Value': time_value,
                    'Delta': greeks['delta'],
                    'Gamma': greeks['gamma'],
                    'Theta': greeks['theta'],
                    'Vega': greeks['vega'],
                    'Rho': greeks['rho'],
                    'ITM': (opt_type == 'call' and spot_price > strike) or (opt_type == 'put' and spot_price < strike)
                })
    
    return pd.DataFrame(options_data)

# Title
st.title("📈 Live Options Pricing Model")
st.markdown("Real-time options data with Black-Scholes pricing and Greeks")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Market selection
market = st.sidebar.selectbox(
    "Select Market",
    ["NSE (India)", "NYSE (US)", "NASDAQ (US)"]
)

# Define stocks for each market
market_stocks = {
    "NSE (India)": ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS'],
    "NYSE (US)": ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
    "NASDAQ (US)": ['NVDA', 'META', 'NFLX', 'AMD', 'INTC']
}

selected_stocks = market_stocks[market]

# Parameters
st.sidebar.subheader("Model Parameters")
risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 15.0, 6.5, 0.1) / 100
implied_vol = st.sidebar.slider("Implied Volatility (%)", 10.0, 100.0, 30.0, 1.0) / 100

# Auto-refresh
auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)

# Search box
st.subheader("🔍 Search Options")
search_query = st.text_input(
    "Search by option code (e.g., RELIANCE 30 Dec 25 1550 CE) or underlying symbol",
    placeholder="Enter search term..."
)

# Refresh button
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    refresh_button = st.button("🔄 Refresh Data", use_container_width=True)
with col2:
    st.write(f"Last update: {datetime.now().strftime('%H:%M:%S')}")

# Fetch and display data
st.subheader("📊 Options Chain")

all_options = []

with st.spinner("Fetching live data..."):
    for symbol in selected_stocks:
        stock_data = fetch_stock_data(symbol)
        
        if stock_data:
            # Generate options for this stock
            options_df = generate_option_chain(
                symbol, 
                stock_data['price'],
                risk_free_rate,
                implied_vol
            )
            all_options.append(options_df)

if all_options:
    # Combine all options
    combined_df = pd.concat(all_options, ignore_index=True)
    
    # Apply search filter
    if search_query:
        mask = (
            combined_df['Option Code'].str.contains(search_query, case=False, na=False) |
            combined_df['Underlying'].str.contains(search_query, case=False, na=False)
        )
        filtered_df = combined_df[mask]
    else:
        filtered_df = combined_df
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Options", len(filtered_df))
    with col2:
        st.metric("Call Options", len(filtered_df[filtered_df['Type'] == 'CALL']))
    with col3:
        st.metric("Put Options", len(filtered_df[filtered_df['Type'] == 'PUT']))
    with col4:
        st.metric("ITM Options", len(filtered_df[filtered_df['ITM']]))
    
    # Format and display the dataframe
    display_df = filtered_df.copy()
    
    # Format numeric columns
    display_df['Spot Price'] = display_df['Spot Price'].map('₹{:.2f}'.format)
    display_df['Strike'] = display_df['Strike'].map('₹{:.0f}'.format)
    display_df['Theoretical Price'] = display_df['Theoretical Price'].map('₹{:.2f}'.format)
    display_df['Intrinsic Value'] = display_df['Intrinsic Value'].map('₹{:.2f}'.format)
    display_df['Time Value'] = display_df['Time Value'].map('₹{:.2f}'.format)
    display_df['Delta'] = display_df['Delta'].map('{:.4f}'.format)
    display_df['Gamma'] = display_df['Gamma'].map('{:.4f}'.format)
    display_df['Theta'] = display_df['Theta'].map('{:.4f}'.format)
    display_df['Vega'] = display_df['Vega'].map('{:.4f}'.format)
    display_df['Rho'] = display_df['Rho'].map('{:.4f}'.format)
    
    # Drop ITM column for display (used for filtering only)
    display_df = display_df.drop('ITM', axis=1)
    
    st.dataframe(
        display_df,
        use_container_width=True,
        height=600
    )
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv,
        file_name=f"options_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
else:
    st.error("Unable to fetch data. Please check your internet connection and try again.")

# Information section
st.sidebar.markdown("---")
st.sidebar.subheader("📖 Understanding Greeks")
st.sidebar.markdown("""
- **Delta**: Rate of change of option price vs underlying
- **Gamma**: Rate of change of delta
- **Theta**: Time decay per day
- **Vega**: Sensitivity to volatility (per 1%)
- **Rho**: Sensitivity to interest rate (per 1%)
""")

st.sidebar.markdown("---")
st.sidebar.info("""
**Note**: Theoretical prices are calculated using the Black-Scholes model. 
Actual market prices may differ due to factors like liquidity, supply/demand, and market conditions.
""")

# Auto-refresh logic
if auto_refresh:
    time.sleep(30)
    st.rerun()
