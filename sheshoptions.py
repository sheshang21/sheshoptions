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
                        'change_pct': ((current_price - prev_close) / prev_close) * 100,
                        'symbol': symbol
                    }
    except Exception as e:
        pass
    
    return None

def generate_option_chain(symbol, spot_price, risk_free_rate, implied_vol, num_strikes, strike_interval, num_expiries, expiry_start_days):
    strikes = []
    base_strike = int(spot_price)
    
    for i in range(-num_strikes // 2, num_strikes // 2 + 1):
        strike = base_strike + (i * strike_interval)
        if strike > 0:
            strikes.append(strike)
    
    expiries = []
    for i in range(num_expiries):
        days_ahead = expiry_start_days + (i * 30)
        exp_date = datetime.now() + timedelta(days=days_ahead)
        exp_label = exp_date.strftime("%d %b %y")
        expiries.append((exp_date, exp_label))
    
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
                
                option_code = f"{symbol.replace('.NS', '').replace('.BO', '').replace('.L', '')} {exp_label} {strike} {'CE' if opt_type == 'call' else 'PE'}"
                
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

st.title("📈 Live Options Pricing Model - Global Markets")
st.markdown("Real-time options pricing for any stock worldwide with Black-Scholes model")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔍 Enter Stock Symbols")
    
    symbols_input = st.text_area(
        "Enter stock symbols (one per line). Add market suffix for international stocks:",
        placeholder="Examples:\nRELIANCE.NS (NSE India)\nAAPL (US)\nTSLA (US)\n^NSEI (Nifty 50)\nVODAFONE.L (London)",
        height=150,
        help="NSE India: .NS | BSE India: .BO | London: .L | US stocks: no suffix"
    )
    
    symbols_list = [s.strip().upper() for s in symbols_input.split('\n') if s.strip()]

with col2:
    st.subheader("⚙️ Model Parameters")
    
    risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 15.0, 6.5, 0.1) / 100
    implied_vol = st.slider("Implied Volatility (%)", 10.0, 100.0, 30.0, 1.0) / 100
    
    st.subheader("📊 Option Chain Settings")
    
    num_strikes = st.slider("Number of Strike Prices", 5, 21, 11, 2, help="Total strike prices to generate")
    strike_interval = st.number_input("Strike Interval", min_value=1, max_value=1000, value=50, help="Price gap between strikes")
    num_expiries = st.slider("Number of Expiries", 1, 12, 3, help="Number of expiration dates")
    expiry_start_days = st.slider("First Expiry (days ahead)", 1, 90, 15, help="Days until first expiration")

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    generate_button = st.button("🚀 Generate Options Chain", use_container_width=True, type="primary")
with col2:
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

if 'options_data' not in st.session_state:
    st.session_state.options_data = None

if generate_button and symbols_list:
    all_options = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, symbol in enumerate(symbols_list):
        status_text.text(f"Fetching data for {symbol}...")
        
        stock_data = fetch_stock_data_yahoo(symbol)
        
        if stock_data:
            st.success(f"✓ {symbol}: ₹{stock_data['price']:.2f} ({stock_data['change_pct']:+.2f}%)")
            
            options_df = generate_option_chain(
                symbol, 
                stock_data['price'],
                risk_free_rate,
                implied_vol,
                num_strikes,
                strike_interval,
                num_expiries,
                expiry_start_days
            )
            all_options.append(options_df)
        else:
            st.warning(f"✗ Could not fetch data for {symbol}")
        
        progress_bar.progress((idx + 1) / len(symbols_list))
    
    progress_bar.empty()
    status_text.empty()
    
    if all_options:
        st.session_state.options_data = pd.concat(all_options, ignore_index=True)
        st.success(f"✓ Generated {len(st.session_state.options_data)} options!")
    else:
        st.error("No data could be fetched. Please check your symbols and try again.")

elif generate_button and not symbols_list:
    st.warning("⚠️ Please enter at least one stock symbol")

if st.session_state.options_data is not None and not st.session_state.options_data.empty:
    st.markdown("---")
    st.subheader("📊 Options Chain Data")
    
    search_query = st.text_input(
        "🔍 Search options (by code, underlying, strike, or type)",
        placeholder="e.g., RELIANCE 30 Dec 25 1550 CE"
    )
    
    filtered_df = st.session_state.options_data.copy()
    
    if search_query:
        mask = (
            filtered_df['Option Code'].str.contains(search_query, case=False, na=False) |
            filtered_df['Underlying'].str.contains(search_query, case=False, na=False) |
            filtered_df['Type'].str.contains(search_query, case=False, na=False) |
            filtered_df['Strike'].astype(str).str.contains(search_query, na=False)
        )
        filtered_df = filtered_df[mask]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Options", len(filtered_df))
    with col2:
        st.metric("Call Options", len(filtered_df[filtered_df['Type'] == 'CALL']))
    with col3:
        st.metric("Put Options", len(filtered_df[filtered_df['Type'] == 'PUT']))
    with col4:
        st.metric("ITM Options", len(filtered_df[filtered_df['ITM']]))
    with col5:
        st.metric("Unique Stocks", filtered_df['Underlying'].nunique())
    
    display_df = filtered_df.copy()
    
    currency_symbol = '₹' if any('.NS' in s or '.BO' in s for s in symbols_list) else '$'
    
    display_df['Spot Price'] = display_df['Spot Price'].map(f'{currency_symbol}{{:.2f}}'.format)
    display_df['Strike'] = display_df['Strike'].map(f'{currency_symbol}{{:.0f}}'.format)
    display_df['Theoretical Price'] = display_df['Theoretical Price'].map(f'{currency_symbol}{{:.2f}}'.format)
    display_df['Intrinsic Value'] = display_df['Intrinsic Value'].map(f'{currency_symbol}{{:.2f}}'.format)
    display_df['Time Value'] = display_df['Time Value'].map(f'{currency_symbol}{{:.2f}}'.format)
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
    st.info("👆 Enter stock symbols above and click 'Generate Options Chain' to get started!")
    
    st.markdown("### 📚 Example Symbols by Market:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🇮🇳 India (NSE)**
        - RELIANCE.NS
        - TCS.NS
        - INFY.NS
        - HDFCBANK.NS
        - ICICIBANK.NS
        - ^NSEI (Nifty 50)
        """)
    
    with col2:
        st.markdown("""
        **🇺🇸 United States**
        - AAPL (Apple)
        - MSFT (Microsoft)
        - GOOGL (Google)
        - TSLA (Tesla)
        - NVDA (Nvidia)
        - ^GSPC (S&P 500)
        """)
    
    with col3:
        st.markdown("""
        **🌍 Other Markets**
        - VODAFONE.L (London)
        - BMW.DE (Germany)
        - 7203.T (Toyota - Tokyo)
        - 0700.HK (Tencent - HK)
        - SAP (NYSE ADR)
        """)

st.sidebar.markdown("---")
st.sidebar.subheader("📖 Understanding Greeks")
st.sidebar.markdown("""
- **Delta**: Rate of change of option price vs underlying (0 to 1 for calls, -1 to 0 for puts)
- **Gamma**: Rate of change of delta (convexity)
- **Theta**: Time decay per day (negative for long options)
- **Vega**: Sensitivity to volatility per 1% change
- **Rho**: Sensitivity to interest rate per 1% change
""")

st.sidebar.markdown("---")
st.sidebar.subheader("💡 Tips")
st.sidebar.markdown("""
- Use market suffixes: .NS (India NSE), .BO (India BSE), .L (London)
- US stocks don't need suffix
- Adjust strike interval based on stock price
- Higher volatility = higher option prices
- ITM options have intrinsic value
""")

st.sidebar.markdown("---")
st.sidebar.info("""
**Note**: Theoretical prices use Black-Scholes model. 
Actual market prices may differ.

Data from Yahoo Finance.
""")
