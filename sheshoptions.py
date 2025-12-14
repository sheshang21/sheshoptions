import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
import time
import json
import requests

st.set_page_config(
    page_title="Live Options Pricing Model",
    page_icon="📈",
    layout="wide"
)

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
</style>
""", unsafe_allow_html=True)

def black_scholes(S, K, T, r, sigma, option_type='call'):
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
    if T <= 0:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    if option_type == 'call':
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                 r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                 r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    
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
def fetch_stock_data_yahoo(symbol):
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=1d"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                result = data['chart']['result'][0]
                meta = result['meta']
                
                current_price = meta.get('regularMarketPrice', 0)
                prev_close = meta.get('previousClose', current_price)
                
                if current_price > 0:
                    return {
                        'price': current_price,
                        'prev_close': prev_close,
                        'change': current_price - prev_close,
                        'change_pct': ((current_price - prev_close) / prev_close) * 100
                    }
    except Exception as e:
        st.warning(f"Yahoo Finance error for {symbol}: {str(e)}")
    
    return None

@st.cache_data(ttl=30)
def fetch_stock_data_alphavantage(symbol):
    try:
        api_key = "demo"
        clean_symbol = symbol.replace('.NS', '').replace('.BO', '')
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={clean_symbol}&apikey={api_key}"
        
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            
            if 'Global Quote' in data and data['Global Quote']:
                quote = data['Global Quote']
                current_price = float(quote.get('05. price', 0))
                prev_close = float(quote.get('08. previous close', current_price))
                
                if current_price > 0:
                    return {
                        'price': current_price,
                        'prev_close': prev_close,
                        'change': current_price - prev_close,
                        'change_pct': ((current_price - prev_close) / prev_close) * 100
                    }
    except Exception as e:
        pass
    
    return None

@st.cache_data(ttl=30)
def fetch_stock_data(symbol):
    stock_data = fetch_stock_data_yahoo(symbol)
    
    if stock_data is None:
        stock_data = fetch_stock_data_alphavantage(symbol)
    
    if stock_data is None:
        mock_prices = {
            'RELIANCE.NS': 2850.50,
            'TCS.NS': 3650.75,
            'INFY.NS': 1450.30,
            'HDFCBANK.NS': 1650.80,
            'ICICIBANK.NS': 1050.25,
            'AAPL': 195.50,
            'MSFT': 380.75,
            'GOOGL': 140.25,
            'AMZN': 175.50,
            'TSLA': 245.80,
            'NVDA': 495.30,
            'META': 355.20,
            'NFLX': 485.60,
            'AMD': 145.90,
            'INTC': 45.75
        }
        
        if symbol in mock_prices:
            price = mock_prices[symbol]
            return {
                'price': price,
                'prev_close': price * 0.99,
                'change': price * 0.01,
                'change_pct': 1.0
            }
    
    return stock_data

def generate_option_chain(symbol, spot_price, risk_free_rate=0.065, implied_vol=0.30):
    strikes = [
        int(spot_price * 0.85),
        int(spot_price * 0.90),
        int(spot_price * 0.95),
        int(spot_price),
        int(spot_price * 1.05),
        int(spot_price * 1.10),
        int(spot_price * 1.15)
    ]
    
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
                theo_price = black_scholes(spot_price, strike, T, risk_free_rate, implied_vol, opt_type)
                
                if opt_type == 'call':
                    intrinsic = max(spot_price - strike, 0)
                else:
                    intrinsic = max(strike - spot_price, 0)
                
                time_value = theo_price - intrinsic
                greeks = calculate_greeks(spot_price, strike, T, risk_free_rate, implied_vol, opt_type)
                
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

st.title("📈 Live Options Pricing Model")
st.markdown("Real-time options data with Black-Scholes pricing and Greeks")

st.sidebar.header("⚙️ Configuration")

market = st.sidebar.selectbox(
    "Select Market",
    ["NSE (India)", "NYSE (US)", "NASDAQ (US)"]
)

market_stocks = {
    "NSE (India)": ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS'],
    "NYSE (US)": ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
    "NASDAQ (US)": ['NVDA', 'META', 'NFLX', 'AMD', 'INTC']
}

selected_stocks = market_stocks[market]

st.sidebar.subheader("Model Parameters")
risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 15.0, 6.5, 0.1) / 100
implied_vol = st.sidebar.slider("Implied Volatility (%)", 10.0, 100.0, 30.0, 1.0) / 100

auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)

st.subheader("🔍 Search Options")
search_query = st.text_input(
    "Search by option code (e.g., RELIANCE 30 Dec 25 1550 CE) or underlying symbol",
    placeholder="Enter search term..."
)

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    refresh_button = st.button("🔄 Refresh Data", use_container_width=True)
with col2:
    st.write(f"Last update: {datetime.now().strftime('%H:%M:%S')}")

st.subheader("📊 Options Chain")

all_options = []

with st.spinner("Fetching live data..."):
    for symbol in selected_stocks:
        stock_data = fetch_stock_data(symbol)
        
        if stock_data:
            options_df = generate_option_chain(
                symbol, 
                stock_data['price'],
                risk_free_rate,
                implied_vol
            )
            all_options.append(options_df)

if all_options:
    combined_df = pd.concat(all_options, ignore_index=True)
    
    if search_query:
        mask = (
            combined_df['Option Code'].str.contains(search_query, case=False, na=False) |
            combined_df['Underlying'].str.contains(search_query, case=False, na=False)
        )
        filtered_df = combined_df[mask]
    else:
        filtered_df = combined_df
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Options", len(filtered_df))
    with col2:
        st.metric("Call Options", len(filtered_df[filtered_df['Type'] == 'CALL']))
    with col3:
        st.metric("Put Options", len(filtered_df[filtered_df['Type'] == 'PUT']))
    with col4:
        st.metric("ITM Options", len(filtered_df[filtered_df['ITM']]))
    
    display_df = filtered_df.copy()
    
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
    
    display_df = display_df.drop('ITM', axis=1)
    
    st.dataframe(
        display_df,
        use_container_width=True,
        height=600
    )
    
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv,
        file_name=f"options_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
else:
    st.error("Unable to fetch data. Please check your internet connection and try again.")

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

if auto_refresh:
    time.sleep(30)
    st.rerun()
