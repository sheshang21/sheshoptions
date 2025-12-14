import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
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
    h1 {
        color: #60a5fa;
    }
</style>
""", unsafe_allow_html=True)

def black_scholes(S, K, T, r, sigma, option_type='call'):
    if T <= 0:
        return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    if T <= 0:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    if option_type == 'call':
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                 r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                 r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    
    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'rho': rho}

@st.cache_data(ttl=30)
def fetch_stock_price(symbol):
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=1d"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                meta = data['chart']['result'][0]['meta']
                return {
                    'price': meta.get('regularMarketPrice', 0),
                    'prev_close': meta.get('previousClose', 0),
                    'currency': meta.get('currency', 'USD')
                }
    except:
        pass
    return None

@st.cache_data(ttl=30)
def fetch_real_options_yahoo(symbol):
    """Fetch REAL options data from Yahoo Finance"""
    try:
        url = f"https://query2.finance.yahoo.com/v7/finance/options/{symbol}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'optionChain' in data and 'result' in data['optionChain']:
                result = data['optionChain']['result'][0]
                
                if 'options' not in result or not result['options']:
                    return None, None
                
                expiration_dates = result.get('expirationDates', [])
                quote = result.get('quote', {})
                spot_price = quote.get('regularMarketPrice', 0)
                
                options_data = []
                
                for option in result['options']:
                    exp_timestamp = option.get('expirationDate', 0)
                    exp_date = datetime.fromtimestamp(exp_timestamp)
                    exp_label = exp_date.strftime("%d %b %y")
                    
                    days_to_expiry = (exp_date - datetime.now()).days
                    
                    calls = option.get('calls', [])
                    puts = option.get('puts', [])
                    
                    for call in calls:
                        options_data.append({
                            'Option Code': f"{symbol} {exp_label} {call.get('strike', 0)} CE",
                            'Underlying': symbol,
                            'Spot Price': spot_price,
                            'Strike': call.get('strike', 0),
                            'Type': 'CALL',
                            'Expiry': exp_label,
                            'Days to Expiry': days_to_expiry,
                            'Last Price': call.get('lastPrice', 0),
                            'Bid': call.get('bid', 0),
                            'Ask': call.get('ask', 0),
                            'Volume': call.get('volume', 0),
                            'Open Interest': call.get('openInterest', 0),
                            'Implied Volatility': call.get('impliedVolatility', 0) * 100,
                            'In The Money': call.get('inTheMoney', False)
                        })
                    
                    for put in puts:
                        options_data.append({
                            'Option Code': f"{symbol} {exp_label} {put.get('strike', 0)} PE",
                            'Underlying': symbol,
                            'Spot Price': spot_price,
                            'Strike': put.get('strike', 0),
                            'Type': 'PUT',
                            'Expiry': exp_label,
                            'Days to Expiry': days_to_expiry,
                            'Last Price': put.get('lastPrice', 0),
                            'Bid': put.get('bid', 0),
                            'Ask': put.get('ask', 0),
                            'Volume': put.get('volume', 0),
                            'Open Interest': put.get('openInterest', 0),
                            'Implied Volatility': put.get('impliedVolatility', 0) * 100,
                            'In The Money': put.get('inTheMoney', False)
                        })
                
                return pd.DataFrame(options_data), expiration_dates
    except Exception as e:
        st.error(f"Error fetching options for {symbol}: {str(e)}")
    
    return None, None

def calculate_theoretical_and_greeks(df, risk_free_rate):
    """Add Black-Scholes theoretical prices and Greeks to real options data"""
    
    for idx, row in df.iterrows():
        S = row['Spot Price']
        K = row['Strike']
        T = row['Days to Expiry'] / 365.0
        sigma = row['Implied Volatility'] / 100 if row['Implied Volatility'] > 0 else 0.30
        opt_type = 'call' if row['Type'] == 'CALL' else 'put'
        
        theo_price = black_scholes(S, K, T, risk_free_rate, sigma, opt_type)
        intrinsic = max(S - K, 0) if opt_type == 'call' else max(K - S, 0)
        time_value = theo_price - intrinsic
        
        greeks = calculate_greeks(S, K, T, risk_free_rate, sigma, opt_type)
        
        df.at[idx, 'Theoretical Price'] = theo_price
        df.at[idx, 'Intrinsic Value'] = intrinsic
        df.at[idx, 'Time Value'] = time_value
        df.at[idx, 'Delta'] = greeks['delta']
        df.at[idx, 'Gamma'] = greeks['gamma']
        df.at[idx, 'Theta'] = greeks['theta']
        df.at[idx, 'Vega'] = greeks['vega']
        df.at[idx, 'Rho'] = greeks['rho']
    
    return df

st.title("📈 Live Options Pricing - Real Market Data")
st.markdown("Fetch REAL options data from exchanges worldwide via Yahoo Finance")

st.info("💡 **This tool fetches ACTUAL options contracts** traded on exchanges. Enter any symbol that has listed options (US stocks work best). Examples: AAPL, TSLA, SPY, QQQ, GOOGL, MSFT, AMZN, NVDA, etc.")

col1, col2 = st.columns([2, 1])

with col1:
    symbol_input = st.text_input(
        "🔍 Enter Stock Symbol (with options listed)",
        placeholder="e.g., AAPL, TSLA, SPY, QQQ, NFLX",
        help="Enter a stock symbol that has listed options. US stocks work best."
    ).upper().strip()

with col2:
    risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 15.0, 5.0, 0.1) / 100

fetch_button = st.button("🚀 Fetch Real Options Data", use_container_width=True, type="primary")

if 'options_df' not in st.session_state:
    st.session_state.options_df = None
if 'current_symbol' not in st.session_state:
    st.session_state.current_symbol = None

if fetch_button and symbol_input:
    with st.spinner(f"Fetching live options data for {symbol_input}..."):
        
        stock_data = fetch_stock_price(symbol_input)
        
        if stock_data and stock_data['price'] > 0:
            st.success(f"✓ {symbol_input} Stock Price: ${stock_data['price']:.2f} (Currency: {stock_data.get('currency', 'USD')})")
            
            options_df, expiration_dates = fetch_real_options_yahoo(symbol_input)
            
            if options_df is not None and not options_df.empty:
                options_df = calculate_theoretical_and_greeks(options_df, risk_free_rate)
                
                st.session_state.options_df = options_df
                st.session_state.current_symbol = symbol_input
                
                st.success(f"✓ Fetched {len(options_df)} REAL options contracts!")
                st.info(f"📅 Available expiration dates: {len(set(options_df['Expiry']))} different dates")
            else:
                st.error(f"❌ No options data found for {symbol_input}. This stock may not have listed options, or try a different symbol.")
        else:
            st.error(f"❌ Could not fetch stock price for {symbol_input}. Please check the symbol and try again.")

elif fetch_button and not symbol_input:
    st.warning("⚠️ Please enter a stock symbol")

if st.session_state.options_df is not None:
    st.markdown("---")
    st.subheader(f"📊 Options Chain: {st.session_state.current_symbol}")
    
    df = st.session_state.options_df
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "🔍 Search (by code, strike, expiry, or type)",
            placeholder="e.g., 150, CALL, Dec, CE"
        )
    
    with col2:
        filter_type = st.selectbox("Filter by Type", ["All", "CALL", "PUT"])
    
    filtered_df = df.copy()
    
    if search_query:
        mask = (
            filtered_df['Option Code'].str.contains(search_query, case=False, na=False) |
            filtered_df['Strike'].astype(str).str.contains(search_query, na=False) |
            filtered_df['Expiry'].str.contains(search_query, case=False, na=False) |
            filtered_df['Type'].str.contains(search_query, case=False, na=False)
        )
        filtered_df = filtered_df[mask]
    
    if filter_type != "All":
        filtered_df = filtered_df[filtered_df['Type'] == filter_type]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Options", len(filtered_df))
    with col2:
        st.metric("Calls", len(filtered_df[filtered_df['Type'] == 'CALL']))
    with col3:
        st.metric("Puts", len(filtered_df[filtered_df['Type'] == 'PUT']))
    with col4:
        st.metric("ITM Options", len(filtered_df[filtered_df['In The Money']]))
    with col5:
        avg_iv = filtered_df['Implied Volatility'].mean()
        st.metric("Avg IV", f"{avg_iv:.1f}%")
    
    display_cols = [
        'Option Code', 'Spot Price', 'Strike', 'Type', 'Expiry', 'Days to Expiry',
        'Last Price', 'Bid', 'Ask', 'Volume', 'Open Interest',
        'Theoretical Price', 'Intrinsic Value', 'Time Value',
        'Implied Volatility', 'Delta', 'Gamma', 'Theta', 'Vega'
    ]
    
    display_df = filtered_df[display_cols].copy()
    
    st.dataframe(
        display_df.style.format({
            'Spot Price': '${:.2f}',
            'Strike': '${:.0f}',
            'Last Price': '${:.2f}',
            'Bid': '${:.2f}',
            'Ask': '${:.2f}',
            'Theoretical Price': '${:.2f}',
            'Intrinsic Value': '${:.2f}',
            'Time Value': '${:.2f}',
            'Implied Volatility': '{:.1f}%',
            'Delta': '{:.4f}',
            'Gamma': '{:.4f}',
            'Theta': '{:.4f}',
            'Vega': '{:.4f}',
            'Volume': '{:,.0f}',
            'Open Interest': '{:,.0f}'
        }),
        use_container_width=True,
        height=600
    )
    
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Full Options Data as CSV",
        data=csv,
        file_name=f"{st.session_state.current_symbol}_options_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    st.markdown("### 📈 Options Analytics")
    
    tab1, tab2, tab3 = st.tabs(["By Expiry", "By Strike", "Volume Analysis"])
    
    with tab1:
        expiry_summary = filtered_df.groupby('Expiry').agg({
            'Option Code': 'count',
            'Volume': 'sum',
            'Open Interest': 'sum',
            'Implied Volatility': 'mean'
        }).round(2)
        expiry_summary.columns = ['Total Contracts', 'Total Volume', 'Total OI', 'Avg IV (%)']
        st.dataframe(expiry_summary, use_container_width=True)
    
    with tab2:
        strike_summary = filtered_df.groupby(['Strike', 'Type']).agg({
            'Last Price': 'first',
            'Volume': 'sum',
            'Open Interest': 'sum'
        }).round(2)
        st.dataframe(strike_summary, use_container_width=True)
    
    with tab3:
        top_volume = filtered_df.nlargest(10, 'Volume')[['Option Code', 'Volume', 'Open Interest', 'Last Price']]
        st.write("**Top 10 by Volume:**")
        st.dataframe(top_volume, use_container_width=True)

else:
    st.markdown("### 🌟 Popular Symbols with Active Options Markets:")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        **📱 Tech Giants**
        - AAPL (Apple)
        - MSFT (Microsoft)
        - GOOGL (Google)
        - AMZN (Amazon)
        - META (Facebook)
        """)
    
    with col2:
        st.markdown("""
        **🚗 Growth Stocks**
        - TSLA (Tesla)
        - NVDA (Nvidia)
        - AMD (AMD)
        - NFLX (Netflix)
        - COIN (Coinbase)
        """)
    
    with col3:
        st.markdown("""
        **📊 ETFs & Indices**
        - SPY (S&P 500)
        - QQQ (Nasdaq 100)
        - IWM (Russell 2000)
        - DIA (Dow Jones)
        - VIX (Volatility)
        """)
    
    with col4:
        st.markdown("""
        **💰 Finance**
        - JPM (JP Morgan)
        - BAC (Bank of America)
        - GS (Goldman Sachs)
        - V (Visa)
        - MA (Mastercard)
        """)

st.sidebar.markdown("---")
st.sidebar.subheader("📖 Data Fields Explained")
st.sidebar.markdown("""
**Market Data:**
- **Last Price**: Latest traded price
- **Bid/Ask**: Current market quotes
- **Volume**: Contracts traded today
- **Open Interest**: Total open contracts
- **Implied Volatility**: Market's volatility expectation

**Calculated Values:**
- **Theoretical Price**: Black-Scholes fair value
- **Intrinsic Value**: Immediate exercise value
- **Time Value**: Premium above intrinsic
- **Greeks**: Risk sensitivities
""")

st.sidebar.markdown("---")
st.sidebar.info("""
**Data Source:** Yahoo Finance (Real-time market data)

**Note:** Options data availability depends on the exchange and symbol. US stocks have the most comprehensive options markets.
""")
