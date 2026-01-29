import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


portfolio_holdings = {
    'BHARTI.NS': {'quantity': 4430, 'name': 'Bharti Airtel', 'purchase_price': 2095, 'currency': 'INR', 'sector': 'TMT'},
    'ASML.AS': {'quantity': 79, 'name': 'ASML Holding', 'purchase_price': 895, 'currency': 'EUR', 'sector': 'TMT'},
    'FTK.DE': {'quantity': 2710, 'name': 'Flatexdegiro', 'purchase_price': 33, 'currency': 'EUR', 'sector': 'FIG'},
    'SREN.SW': {'quantity': 273, 'name': 'Swiss Re', 'purchase_price': 148, 'currency': 'CHF', 'sector': 'FIG'},
    'WAL': {'quantity': 1250, 'name': 'Western Alliance Bancorp', 'purchase_price': 79, 'currency': 'USD', 'sector': 'FIG'},
    'AIR.PA': {'quantity': 374, 'name': 'Airbus', 'purchase_price': 209, 'currency': 'EUR', 'sector': 'Industrials'},
    '1211.HK': {'quantity': 7944, 'name': 'BYD Company', 'purchase_price': 98, 'currency': 'HKD', 'sector': 'Consumer Goods'},
    'KAP.SW': {'quantity': 1976, 'name': 'Kape Technologies', 'purchase_price': 53, 'currency': 'CHF', 'sector': 'TMT'},
    'BSL.AX': {'quantity': 2077, 'name': 'BlueScope Steel', 'purchase_price': 22, 'currency': 'AUD', 'sector': 'PUI'},
    'DHER.DE': {'quantity': 2411, 'name': 'Delivery Hero', 'purchase_price': 20, 'currency': 'EUR', 'sector': 'Consumer Goods'},
    'MDLZ': {'quantity': 874, 'name': 'Mondelez International', 'purchase_price': 56, 'currency': 'USD', 'sector': 'Consumer Goods'},
    'ALC.SW': {'quantity': 682, 'name': 'Alcon', 'purchase_price': 59, 'currency': 'CHF', 'sector': 'Healthcare'}
}
# Currency pairs for conversion to CHF (base currency)
currency_pairs = {
    'USD': 'USDCHF=X',
    'EUR': 'EURCHF=X',
    'INR': 'INRCHF=X',
    'HKD': 'HKDCHF=X',
    'AUD': 'AUDCHF=X',
    'CHF': None  # Already in CHF
    }
        


# Page configuration
st.set_page_config(
    page_title="Portfolio Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for deep blue sidebar and white background
st.markdown("""
    <style>
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #030C30;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Radio button styling in sidebar */
    [data-testid="stSidebar"] .st-emotion-cache-1gulkj5 {
        color: white !important;
    }
    
    /* Main content background */
    .main {
        background-color: white;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] label {
        color: white !important;
    }
    
    /* Radio buttons */
    [data-testid="stSidebar"] [data-baseweb="radio"] {
        background-color: transparent;
    }
    
    [data-testid="stSidebar"] [data-baseweb="radio"] > div {
        color: white !important;
    }
    
    /* Main content buttons styling */
    div.stButton > button {
        height: 120px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border: 2px solid #e0e2e6;
        color: #0F1D64;
        font-size: 16px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    div.stButton > button:hover {
        background-color: #0F1D64;
        color: white;
        border-color: #0F1D64;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(15, 29, 100, 0.3);
    }
    
    /* Sidebar buttons styling */
    [data-testid="stSidebar"] div.stButton > button {
        height: auto;
        background-color: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        color: white;
    }
    
    [data-testid="stSidebar"] div.stButton > button:hover {
        background-color: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.5);
        transform: none;
        box-shadow: none;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")

# Add logo at the top of sidebar (if logo file exists)
try:
    st.sidebar.image("Screenshot 2026-01-29 at 17.22.15.png", use_container_width=True)
    st.sidebar.markdown("---")
except:
    # If logo doesn't exist, show placeholder text
    st.sidebar.markdown("### Portfolio Dashboard")
    st.sidebar.markdown("---")

# Home option
if st.sidebar.button("🏠 Home", use_container_width=True):
    st.session_state.main_page = "Home"

# Initialize session state if not exists
if 'main_page' not in st.session_state:
    st.session_state.main_page = "Home"

st.sidebar.markdown("---")
st.sidebar.subheader("Sectors")

# Sector buttons
if st.sidebar.button("📱 TMT", use_container_width=True):
    st.session_state.main_page = "TMT Sector"
if st.sidebar.button("🏦 FIG", use_container_width=True):
    st.session_state.main_page = "FIG Sector"
if st.sidebar.button("🏭 Industrials", use_container_width=True):
    st.session_state.main_page = "Industrials Sector"
if st.sidebar.button("⚡ PUI", use_container_width=True):
    st.session_state.main_page = "PUI Sector"
if st.sidebar.button("🛒 Consumer Goods", use_container_width=True):
    st.session_state.main_page = "Consumer Goods Sector"
if st.sidebar.button("🏥 Healthcare", use_container_width=True):
    st.session_state.main_page = "Healthcare Sector"

st.sidebar.markdown("---")

# Adding Tool button
if st.sidebar.button("🔧 Adding Tool", use_container_width=True):
    st.session_state.main_page = "Adding Tool"

main_page = st.session_state.main_page

# =============================================================================
# HOME PAGE
# =============================================================================
if main_page == "Home":
    st.title("🏠 Portfolio Dashboard - Home")
    
    # Initialize home tab session state
    if 'home_tab' not in st.session_state:
        st.session_state.home_tab = "Generic Summary"
    
    # Sub-navigation with clickable boxes
    st.markdown("### Select Analysis Type")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📈 Generic Summary\n\nKey metrics, performance vs MSCI World, portfolio treemap", 
                     key="gen_summary", use_container_width=True):
            st.session_state.home_tab = "Generic Summary"
    
    with col2:
        if st.button("🏗️ Portfolio Structure\n\nSector, geographical, and asset distribution", 
                     key="portfolio_struct", use_container_width=True):
            st.session_state.home_tab = "Portfolio Structure Analysis"
    
    with col3:
        if st.button("🔮 Forecast\n\nMonte Carlo, analyst targets, DCF analysis", 
                     key="forecast", use_container_width=True):
            st.session_state.home_tab = "Forecast"
    
    st.markdown("---")
    home_tab = st.session_state.home_tab
    
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # HOME - Generic Summary
    # -------------------------------------------------------------------------
    if home_tab == "Generic Summary":
        st.header("📈 Generic Summary")
        
        # Date range selection
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=pd.to_datetime('2024-11-06'),  # Purchase date
                help="Select the start date for analysis"
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=pd.to_datetime('today'),
                help="Select the end date for analysis"
            )
        
        if st.button("📊 Generate Analysis", type="primary"):
            if start_date >= end_date:
                st.error("Start date must be before end date.")
            else:
                try:
                    import yfinance as yf
                    import numpy as np
                    
                    with st.spinner('Fetching market data and exchange rates...'):
                        # Fetch MSCI World ETF as proxy (using URTH or ACWI)
                        msci_world = yf.download('URTH', start=start_date, end=end_date, progress=False)
                        
                        # Fetch exchange rates for the entire period
                        exchange_rates = {}
                        for currency, pair in currency_pairs.items():
                            if pair:
                                fx_data = yf.download(pair, start=start_date, end=end_date, progress=False)
                                exchange_rates[currency] = fx_data
                            else:
                                exchange_rates[currency] = None  # CHF
                        
                        # Fetch portfolio holdings data
                        portfolio_data = {}
                        initial_prices = {}
                        current_prices = {}
                        
                        for ticker, info in portfolio_holdings.items():
                            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                            if not data.empty:
                                portfolio_data[ticker] = data
                                # Handle multi-column or single-column Close data
                                if isinstance(data['Close'], pd.DataFrame):
                                    close_prices = data['Close'].iloc[:, 0]
                                else:
                                    close_prices = data['Close']
                                initial_prices[ticker] = float(close_prices.iloc[0])
                                current_prices[ticker] = float(close_prices.iloc[-1])
                    
                    # Helper function to get exchange rate for a specific date
                    def get_fx_rate(currency, date, exchange_rates):
                        if currency == 'CHF':
                            return 1.0
                        fx_data = exchange_rates[currency]
                        if fx_data is None or fx_data.empty:
                            # Fallback rates if data not available
                            fallback = {'USD': 0.88, 'EUR': 0.93, 'INR': 0.0104, 'HKD': 0.113, 'AUD': 0.57}
                            return fallback.get(currency, 1.0)
                        
                        # Get the exchange rate for the date or closest previous date
                        if isinstance(fx_data['Close'], pd.DataFrame):
                            fx_close = fx_data['Close'].iloc[:, 0]
                        else:
                            fx_close = fx_data['Close']
                        
                        available_dates = fx_close.index[fx_close.index <= date]
                        if len(available_dates) > 0:
                            return float(fx_close.loc[available_dates[-1]])
                        else:
                            return float(fx_close.iloc[0]) if len(fx_close) > 0 else 1.0
                    # =============================================================
                    # CALCULATE PORTFOLIO METRICS (Sharpe, Alpha, Beta)
                    # =============================================================
                    
                    st.subheader("📊 Portfolio Risk Metrics")
                    
                    with st.spinner('Calculating Sharpe ratio, Alpha, and Beta...'):
                        # Create daily portfolio values in CHF
                        all_dates = msci_world.index
                        portfolio_values = pd.Series(index=all_dates, dtype=float)
                        
                        for date in all_dates:
                            daily_value = 0
                            for ticker, info in portfolio_holdings.items():
                                if ticker in portfolio_data and not portfolio_data[ticker].empty:
                                    stock_data = portfolio_data[ticker]
                                    # Get stock price for this date or closest previous
                                    if isinstance(stock_data['Close'], pd.DataFrame):
                                        stock_close = stock_data['Close'].iloc[:, 0]
                                    else:
                                        stock_close = stock_data['Close']
                                    
                                    available_dates = stock_close.index[stock_close.index <= date]
                                    if len(available_dates) > 0:
                                        stock_price = float(stock_close.loc[available_dates[-1]])
                                    else:
                                        stock_price = initial_prices[ticker]
                                    
                                    # Convert to CHF
                                    fx_rate = get_fx_rate(info['currency'], date, exchange_rates)
                                    stock_value_chf = stock_price * info['quantity'] * fx_rate
                                    daily_value += stock_value_chf
                            
                            portfolio_values[date] = daily_value
                        
                        # Calculate daily returns
                        portfolio_returns = portfolio_values.pct_change().dropna()
                        
                        if isinstance(msci_world['Close'], pd.DataFrame):
                            msci_close = msci_world['Close'].iloc[:, 0]
                        else:
                            msci_close = msci_world['Close']
                        msci_returns = msci_close.pct_change().dropna()
                        
                        # Align dates
                        common_dates = portfolio_returns.index.intersection(msci_returns.index)
                        portfolio_returns_aligned = portfolio_returns.loc[common_dates]
                        msci_returns_aligned = msci_returns.loc[common_dates]
                        
                        # Calculate metrics
                        # Sharpe Ratio (assuming risk-free rate of 2% annually, or ~0.008% daily)
                        risk_free_rate_daily = 0.02 / 252  # Approx 252 trading days per year
                        excess_returns = portfolio_returns_aligned - risk_free_rate_daily
                        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() != 0 else 0
                        
                        # Beta and Alpha using linear regression
                        from scipy import stats
                        # 1. Ensure data is clean and aligne
                        # We combine them into a temporary DataFrame to drop rows where either has a NaN
                        regression_data = pd.DataFrame({
                            'msci': msci_returns_aligned,
                            'portfolio': portfolio_returns_aligned}).dropna()
                        # 2. Check if we still have enough data points after cleaning
                        if len(regression_data) > 2:
                            # Use the cleaned columns
                            slope, intercept, r_value, p_value, std_err = stats.linregress(
                                regression_data['msci'], 
                                regression_data['portfolio'])
                            beta = slope
                            # Geometric annualization for Alpha
                            alpha_annual = (1 + intercept)**252 - 1 
                        else:
                                # Fallback if there isn't enough overlapping data
                                beta = 0.0
                                alpha_annual = 0.0
                        
                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                label="Sharpe Ratio",
                                value=f"{sharpe_ratio:.2f}",
                                help=r"""**Formula:** $Sharpe = \frac{R_p - R_f}{\sigma_p}$
                                Where:
                                    - $R_p$: Expected portfolio return
                                    - $R_f$: Risk-free rate
                                    - $\sigma_p$: Standard deviation of portfolio excess return
                                    Higher is better. >1 is good, >2 is very good.""")
                        with col2:
                            st.metric(
                                label="Beta (vs MSCI World)",
                                value=f"{beta:.2f}",
                                help=r"""
                                **Formula:** $\beta = \frac{Cov(R_p, R_m)}{Var(R_m)}$
                                Measures sensitivity to the market. 
                                - 1.0: Moves with the market
                                - >1.0: More volatile (Aggressive)
                                - <1.0: Less volatile (Defensive)""")
                        with col3:
                            st.metric(
                                label="Alpha (Annualized)",
                                value=f"{alpha_annual*100:.2f}%",
                                help=r"""
                                **Formula:** $\alpha = R_p - [R_f + \beta(R_m - R_f)]$
                                Represents the value added by the manager relative to the benchmark return.
                                """)
                       
                        # Additional context
                        st.info(f"""
                        **Interpretation:**
                        - **Sharpe Ratio ({sharpe_ratio:.2f})**: {'Excellent' if sharpe_ratio > 2 else 'Good' if sharpe_ratio > 1 else 'Moderate' if sharpe_ratio > 0 else 'Poor'} risk-adjusted performance
                        - **Beta ({beta:.2f})**: Portfolio is {'more volatile than' if beta > 1 else 'less volatile than' if beta < 1 else 'as volatile as'} the market
                        - **Alpha ({alpha_annual*100:.2f}%)**: {'Outperforming' if alpha_annual > 0 else 'Underperforming'} the benchmark by {abs(alpha_annual*100):.2f}% annually
                        """)
                    
                    st.markdown("---")
                    
                    # =============================================================
                    # PERFORMANCE COMPARISON CHART
                    # =============================================================
                    st.subheader("📈 Portfolio vs MSCI World Performance")
                    
                    # Calculate portfolio daily values in CHF
                    if isinstance(msci_world['Close'], pd.DataFrame):
                        msci_close = msci_world['Close'].iloc[:, 0]
                    else:
                        msci_close = msci_world['Close']
                    
                    dates = msci_close.index
                    portfolio_values_chf = []
                    
                    # Calculate initial portfolio value in CHF using purchase prices
                    initial_portfolio_value_chf = 0
                    first_date = dates[0]
                    for ticker, info in portfolio_holdings.items():
                        fx_rate = get_fx_rate(info['currency'], first_date, exchange_rates)
                        value_chf = info['purchase_price'] * info['quantity'] * fx_rate
                        initial_portfolio_value_chf += value_chf
                    
                    for date in dates:
                        daily_value_chf = 0
                        for ticker, info in portfolio_holdings.items():
                            if ticker in portfolio_data:
                                data = portfolio_data[ticker]
                                if date in data.index:
                                    if isinstance(data['Close'], pd.DataFrame):
                                        price = float(data['Close'].iloc[:, 0].loc[date])
                                    else:
                                        price = float(data['Close'].loc[date])
                                else:
                                    # Use last available price
                                    available_dates = data.index[data.index <= date]
                                    if len(available_dates) > 0:
                                        last_date = available_dates[-1]
                                        if isinstance(data['Close'], pd.DataFrame):
                                            price = float(data['Close'].iloc[:, 0].loc[last_date])
                                        else:
                                            price = float(data['Close'].loc[last_date])
                                    else:
                                        price = info['purchase_price']
                                
                                # Convert to CHF
                                fx_rate = get_fx_rate(info['currency'], date, exchange_rates)
                                value_chf = price * info['quantity'] * fx_rate
                                daily_value_chf += value_chf
                        
                        portfolio_values_chf.append(daily_value_chf)
                    
                    # Normalize to base 100
                    base_portfolio_value = portfolio_values_chf[0] 
                    portfolio_normalized = [v / base_portfolio_value * 100 for v in portfolio_values_chf]
                    msci_normalized = (msci_close / msci_close.iloc[0] * 100).values
                    
                    # Create comparison chart
                    fig_performance = go.Figure()
                    
                    fig_performance.add_trace(go.Scatter(
                        x=dates,
                        y=portfolio_normalized,
                        mode='lines',
                        name='Your Portfolio',
                        line=dict(color='#0F1D64', width=3),
                        hovertemplate='<b>Portfolio</b><br>Date: %{x}<br>Index: %{y:.2f}<extra></extra>'
                    ))
                    
                    fig_performance.add_trace(go.Scatter(
                        x=dates,
                        y=msci_normalized,
                        mode='lines',
                        name='MSCI World (URTH)',
                        line=dict(color='#FF6B6B', width=2),
                        hovertemplate='<b>MSCI World</b><br>Date: %{x}<br>Index: %{y:.2f}<extra></extra>'
                    ))
                    
                    fig_performance.update_layout(
                        title="Portfolio Performance vs MSCI World (Base 100)",
                        xaxis_title="Date",
                        yaxis_title="Index Value (Base 100)",
                        hovermode='x unified',
                        height=500,
                        template="plotly_white",
                        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                    )
                    
                    st.plotly_chart(fig_performance, use_container_width=True)
                    
                    # Performance metrics
                    portfolio_return = ((portfolio_values_chf[-1] - initial_portfolio_value_chf) / initial_portfolio_value_chf * 100)
                    msci_return = ((msci_close.iloc[-1] - msci_close.iloc[0]) / msci_close.iloc[0] * 100)
                    outperformance = portfolio_return - msci_return
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Portfolio Return", f"{portfolio_return:.2f}%")
                    with col2:
                        st.metric("MSCI World Return", f"{msci_return:.2f}%")
                    with col3:
                        st.metric("Outperformance", f"{outperformance:.2f}%", 
                                 delta=f"{outperformance:.2f}%")
                    with col4:
                        st.metric("Portfolio Value", f"CHF {portfolio_values_chf[-1]:,.2f}")
                    
                    # =============================================================
                    # PORTFOLIO TREEMAP
                    # =============================================================
                    st.subheader("🗺️ Portfolio Composition Treemap")
                    
                    # Calculate weights and performance for each holding
                    treemap_data = []
                    final_date = dates[-1]
                    
                    for ticker, info in portfolio_holdings.items():
                        if ticker in current_prices:
                            # Get current FX rate
                            current_fx = get_fx_rate(info['currency'], final_date, exchange_rates)
                            # Get initial FX rate (at purchase/start date)
                            initial_fx = get_fx_rate(info['currency'], dates[0], exchange_rates)
                            
                            current_value_chf = current_prices[ticker] * info['quantity'] * current_fx
                            initial_value_chf = info['purchase_price'] * info['quantity'] * initial_fx
                            
                            weight = (current_value_chf / portfolio_values_chf[-1] * 100)
                            performance = ((current_value_chf - initial_value_chf) / initial_value_chf * 100)
                            
                            treemap_data.append({
                                'ticker': ticker,
                                'name': info['name'],
                                'weight': weight,
                                'performance': performance,
                                'value': current_value_chf,
                                'currency': info['currency']
                            })
                    
                    treemap_df = pd.DataFrame(treemap_data)
                    
                    # Create treemap
                    fig_treemap = px.treemap(
                        treemap_df,
                        path=[px.Constant("Portfolio"), 'name'],
                        values='weight',
                        color='performance',
                        color_continuous_scale='RdYlGn',
                        color_continuous_midpoint=0,
                        hover_data={
                            'weight': ':.2f',
                            'performance': ':.2f',
                            'value': ':,.2f',
                            'currency': True
                        },
                        labels={
                            'weight': 'Weight (%)',
                            'performance': 'Performance (%)',
                            'value': 'Value (CHF)',
                            'currency': 'Currency'
                        }
                    )
                    
                    fig_treemap.update_traces(
                        textposition='middle center',
                        texttemplate='<b>%{label}</b><br>%{value:.1f}%',
                        hovertemplate='<b>%{label}</b><br>Weight: %{customdata[0]:.2f}%<br>Performance: %{customdata[1]:.2f}%<br>Value: CHF %{customdata[2]:,.2f}<br>Currency: %{customdata[3]}<extra></extra>'
                    )
                    
                    fig_treemap.update_layout(
                        height=600,
                        margin=dict(t=50, l=0, r=0, b=0)
                    )
                    
                    st.plotly_chart(fig_treemap, use_container_width=True)
                    
                    st.markdown("""
                    **Treemap Legend:**
                    - **Size of square**: Weight in portfolio (%)
                    - **Color**: Performance over selected period (Green = positive, Red = negative)
                    - **Hover**: See detailed information for each holding
                    - **Note**: All values converted to CHF using daily exchange rates
                    """)
                    
                    # Holdings table
                    st.subheader("📋 Holdings Details")
                    holdings_table = treemap_df.copy()
                    holdings_table['Performance (%)'] = holdings_table['performance'].apply(lambda x: f"{x:.2f}%")
                    holdings_table['Weight (%)'] = holdings_table['weight'].apply(lambda x: f"{x:.2f}%")
                    holdings_table['Value (CHF)'] = holdings_table['value'].apply(lambda x: f"CHF {x:,.2f}")
                    holdings_table = holdings_table[['name', 'ticker', 'currency', 'Weight (%)', 'Performance (%)', 'Value (CHF)']]
                    holdings_table.columns = ['Company', 'Ticker', 'Currency', 'Weight', 'Performance', 'Value']
                    holdings_table = holdings_table.sort_values('Performance', ascending=False)
                    
                    st.dataframe(holdings_table, use_container_width=True, hide_index=True)
                    
                except ImportError:
                    st.error("⚠️ Required libraries not installed. Please install: `pip install yfinance plotly pandas`")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.info("If you're having issues with specific tickers, they may need adjustment for Yahoo Finance compatibility.")

    # HOME - Portfolio Structure Analysis
    # -------------------------------------------------------------------------
    elif home_tab == "Portfolio Structure Analysis":
        st.header("🏗️ Portfolio Structure Analysis")
                
        # Country mapping based on exchange/ticker
        country_mapping = {
            'BHARTI.NS': 'India',
            'ASML.AS': 'Netherlands',
            'FTK.DE': 'Germany',
            'SREN.SW': 'Switzerland',
            'WAL': 'United States',
            'AIR.PA': 'France',
            '1211.HK': 'China',
            'KAP.SW': 'Switzerland',
            'BSL.AX': 'Australia',
            'DHER.DE': 'Germany',
            'MDLZ': 'United States',
            'ALC.SW': 'Switzerland'
        }
        
        # Region mapping (EMEA, Americas, APAC)
        region_mapping = {
            'India': 'APAC',
            'Netherlands': 'EMEA',
            'Germany': 'EMEA',
            'Switzerland': 'EMEA',
            'United States': 'Americas',
            'France': 'EMEA',
            'China': 'APAC',
            'Australia': 'APAC'
        }
        
        # ISO country codes for map visualization
        country_iso_codes = {
            'India': 'IND',
            'Netherlands': 'NLD',
            'Germany': 'DEU',
            'Switzerland': 'CHE',
            'United States': 'USA',
            'France': 'FRA',
            'China': 'CHN',
            'Australia': 'AUS'
        }
        
        try:
            import yfinance as yf
            import numpy as np
            
            with st.spinner('Loading portfolio structure...'):
                # Fetch current prices and market cap data
                current_prices = {}
                exchange_rates_current = {}
                company_info = {}
                
                # Get current exchange rates
                for currency, pair in currency_pairs.items():
                    if pair:
                        fx_data = yf.Ticker(pair).history(period='1d')
                        if not fx_data.empty:
                            exchange_rates_current[currency] = float(fx_data['Close'].iloc[-1])
                        else:
                            # Fallback rates
                            exchange_rates_current[currency] = {'USD': 0.88, 'EUR': 0.93, 'INR': 0.0104, 'HKD': 0.113, 'AUD': 0.57}[currency]
                    else:
                        exchange_rates_current[currency] = 1.0
                
                # Fetch current stock prices and company info
                for ticker, info in portfolio_holdings.items():
                    try:
                        ticker_obj = yf.Ticker(ticker)
                        ticker_data = ticker_obj.history(period='1d')
                        
                        if not ticker_data.empty:
                            current_prices[ticker] = float(ticker_data['Close'].iloc[-1])
                        else:
                            current_prices[ticker] = info['purchase_price']
                        
                        # Try to get market cap from yfinance
                        try:
                            ticker_info = ticker_obj.info
                            market_cap = ticker_info.get('marketCap', 0)
                            company_info[ticker] = {'market_cap': market_cap}
                        except:
                            company_info[ticker] = {'market_cap': 0}
                    except:
                        current_prices[ticker] = info['purchase_price']
                        company_info[ticker] = {'market_cap': 0}
            
            # Helper function to classify market cap
            def classify_market_cap(market_cap_usd):
                if market_cap_usd == 0:
                    return 'Unknown'
                elif market_cap_usd >= 10e9:  # >= $10B
                    return 'Large'
                elif market_cap_usd >= 2e9:   # >= $2B
                    return 'Mid'
                else:
                    return 'Small'
            
            # Calculate portfolio values in CHF
            holdings_analysis = []
            total_value_chf = 0
            
            for ticker, info in portfolio_holdings.items():
                if ticker in current_prices:
                    fx_rate = exchange_rates_current[info['currency']]
                    value_chf = current_prices[ticker] * info['quantity'] * fx_rate
                    total_value_chf += value_chf
                    
                    # Get country and derive region
                    country = country_mapping.get(ticker, 'Unknown')
                    region = region_mapping.get(country, 'Unknown')
                    
                    # Get market cap classification
                    market_cap_usd = company_info.get(ticker, {}).get('market_cap', 0)
                    market_cap_class = classify_market_cap(market_cap_usd)
                    
                    holdings_analysis.append({
                        'ticker': ticker,
                        'name': info['name'],
                        'sector': info['sector'],
                        'country': country,
                        'country_iso': country_iso_codes.get(country, ''),
                        'region': region,
                        'market_cap': market_cap_class,
                        'currency': info['currency'],
                        'value_chf': value_chf,
                        'weight': 0  # Will calculate after total
                    })
            
            # Calculate weights
            for holding in holdings_analysis:
                holding['weight'] = (holding['value_chf'] / total_value_chf * 100)
            
            df_analysis = pd.DataFrame(holdings_analysis)
            
            # =============================================================
            # SECTOR DISTRIBUTION
            # =============================================================
            st.subheader("📊 Sector Distribution")
            
            sector_allocation = df_analysis.groupby('sector')['weight'].sum().reset_index()
            sector_allocation.columns = ['Sector', 'Weight (%)']
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Pie chart with blue color scheme
                blue_colors = ['#0F1D64', '#1E3A8A', '#3B82F6', '#60A5FA', '#93C5FD', '#DBEAFE']
                
                fig_sector = px.pie(
                    sector_allocation,
                    values='Weight (%)',
                    names='Sector',
                    title='Portfolio Allocation by Sector',
                    color_discrete_sequence=blue_colors,
                    hole=0.4
                )
                fig_sector.update_traces(textposition='inside', textinfo='percent+label')
                fig_sector.update_layout(height=400)
                st.plotly_chart(fig_sector, use_container_width=True)
            
            with col2:
                st.markdown("### Sector Breakdown")
                sector_allocation_sorted = sector_allocation.sort_values('Weight (%)', ascending=False)
                for _, row in sector_allocation_sorted.iterrows():
                    st.metric(row['Sector'], f"{row['Weight (%)']:.1f}%")
            
            # =============================================================
            # GEOGRAPHICAL DISTRIBUTION
            # =============================================================
            st.subheader("🌍 Geographical Distribution")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Geographic map by country
                country_allocation = df_analysis.groupby(['country', 'country_iso'])['weight'].sum().reset_index()
                country_allocation.columns = ['Country', 'ISO', 'Weight (%)']
                
                fig_map = px.choropleth(
                    country_allocation,
                    locations='ISO',
                    color='Weight (%)',
                    hover_name='Country',
                    hover_data={'ISO': False, 'Weight (%)': ':.2f'},
                    color_continuous_scale=[
                        [0, '#DBEAFE'],
                        [0.25, '#93C5FD'],
                        [0.5, '#60A5FA'],
                        [0.75, '#3B82F6'],
                        [1, '#0F1D64']
                    ],
                    title='Geographic Distribution by Country'
                )
                fig_map.update_geos(
                    showcountries=True,
                    countrycolor="lightgray",
                    showcoastlines=True,
                    projection_type='natural earth'
                )
                fig_map.update_layout(height=450, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig_map, use_container_width=True)
            
            with col2:
                # Regional breakdown table
                st.markdown("### Regional Allocation")
                region_allocation = df_analysis.groupby('region')['weight'].sum().reset_index()
                region_allocation.columns = ['Region', 'Weight (%)']
                region_allocation = region_allocation.sort_values('Weight (%)', ascending=False)
                
                for _, row in region_allocation.iterrows():
                    st.metric(row['Region'], f"{row['Weight (%)']:.1f}%")
                
                st.markdown("---")
                st.markdown("### Top Countries")
                country_allocation_table = country_allocation[['Country', 'Weight (%)']].sort_values('Weight (%)', ascending=False)
                st.dataframe(
                    country_allocation_table.style.format({'Weight (%)': '{:.1f}%'}),
                    use_container_width=True,
                    hide_index=True
                )
            
            # =============================================================
            # ADDITIONAL ANALYSIS
            # =============================================================
            st.subheader("📈 Additional Analysis")
            
            # Market Cap Distribution
            col1, col2 = st.columns(2)
            
            with col1:
                market_cap_allocation = df_analysis.groupby('market_cap')['weight'].sum().reset_index()
                market_cap_allocation.columns = ['Market Cap', 'Weight (%)']
                
                # Ensure proper ordering
                cap_order = ['Large', 'Mid', 'Small', 'Unknown']
                market_cap_allocation['Market Cap'] = pd.Categorical(
                    market_cap_allocation['Market Cap'], 
                    categories=cap_order, 
                    ordered=True
                )
                market_cap_allocation = market_cap_allocation.sort_values('Market Cap')
                
                fig_market_cap = px.bar(
                    market_cap_allocation,
                    x='Market Cap',
                    y='Weight (%)',
                    title='Market Cap Distribution',
                    color='Weight (%)',
                    color_continuous_scale=[
                        [0, '#DBEAFE'],
                        [0.5, '#3B82F6'],
                        [1, '#0F1D64']
                    ]
                )
                fig_market_cap.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig_market_cap, use_container_width=True)
            
            with col2:
                # Currency Exposure
                currency_allocation = df_analysis.groupby('currency')['weight'].sum().reset_index()
                currency_allocation.columns = ['Currency', 'Weight (%)']
                currency_allocation = currency_allocation.sort_values('Weight (%)', ascending=False)
                
                fig_currency = px.pie(
                    currency_allocation,
                    values='Weight (%)',
                    names='Currency',
                    title='Currency Exposure',
                    color_discrete_sequence=blue_colors
                )
                fig_currency.update_traces(textposition='inside', textinfo='percent+label')
                fig_currency.update_layout(height=350)
                st.plotly_chart(fig_currency, use_container_width=True)
            
            # =============================================================
            # CONCENTRATION METRICS
            # =============================================================
            st.subheader("🎯 Concentration Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            # Top holdings concentration
            df_sorted = df_analysis.sort_values('weight', ascending=False)
            top_5_concentration = df_sorted.head(5)['weight'].sum()
            top_10_concentration = df_sorted.head(10)['weight'].sum()
            
            # Calculate Herfindahl-Hirschman Index (HHI)
            hhi = (df_analysis['weight'] ** 2).sum()
            
            with col1:
                st.metric("Top 5 Holdings", f"{top_5_concentration:.1f}%")
            
            with col2:
                st.metric("Top 10 Holdings", f"{top_10_concentration:.1f}%")
            
            with col3:
                st.metric("HHI Index", f"{hhi:.0f}")
                if hhi < 1000:
                    st.caption("✅ Well diversified")
                elif hhi < 1800:
                    st.caption("⚠️ Moderately concentrated")
                else:
                    st.caption("🔴 Highly concentrated")
            
            # Top Holdings Table
            st.markdown("### Top 10 Holdings by Weight")
            top_holdings = df_sorted[['name', 'sector', 'country', 'weight']].head(10).copy()
            top_holdings.columns = ['Company', 'Sector', 'Country', 'Weight (%)']
            top_holdings['Weight (%)'] = top_holdings['Weight (%)'].apply(lambda x: f"{x:.2f}%")
            st.dataframe(top_holdings, use_container_width=True, hide_index=True)
            
            # Summary Statistics
            st.subheader("📋 Portfolio Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Holdings", len(df_analysis))
            
            with col2:
                st.metric("Total Value", f"CHF {total_value_chf:,.2f}")
            
            with col3:
                st.metric("Sectors Covered", df_analysis['sector'].nunique())
            
            with col4:
                st.metric("Countries", df_analysis['country'].nunique())
            
        except ImportError:
            st.error("⚠️ Required libraries not installed. Please install: `pip install yfinance plotly pandas`")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    
    # -------------------------------------------------------------------------
    # HOME - Forecast
    # -------------------------------------------------------------------------
    elif home_tab == "Forecast":
        st.header("🔮 Portfolio Forecast")
        st.markdown("""
        ### This page will include:
        
        **Monte Carlo Simulation:**
        - Probabilistic portfolio performance projections
        - Multiple scenario analysis (bull/base/bear cases)
        - Confidence intervals (5th, 50th, 95th percentiles)
        - Expected portfolio value distribution over time
        
        **Analyst Consensus Forecast:**
        - Aggregate average stock price targets from analysts
        - Weighted portfolio performance forecast
        - Upside/downside potential analysis
        - Time horizon: 12-month forward looking
        
        **DCF (Discounted Cash Flow) Analysis:**
        - Input parameters:
          - Forecasted Free Cash Flows (per holding)
          - WACC (Weighted Average Cost of Capital) adjustment
          - Terminal value assumptions
        - Fair value calculation for portfolio holdings
        - Comparison: Current price vs DCF-based fair value
        - Aggregated portfolio intrinsic value
        """)
        
        # Placeholder for future content
        st.info("📊 Forecast models and simulations will be implemented here")

# =============================================================================
# SECTOR PAGES (TMT, FIG, Industrials, PUI, Consumer Goods, Healthcare)
# =============================================================================
elif main_page in ["TMT Sector", "FIG Sector", "Industrials Sector", 
                    "PUI Sector", "Consumer Goods Sector", "Healthcare Sector"]:
    
    sector_name = main_page.replace(" Sector", "")
    st.title(f"🏢 {sector_name} Sector Analysis")
    
    # Filter holdings for this sector
    sector_holdings = {k: v for k, v in portfolio_holdings.items() if v['sector'] == sector_name}
    
    if not sector_holdings:
        st.warning(f"No holdings found in {sector_name} sector.")
    else:
        # Initialize sector tab session state
        if 'sector_tab' not in st.session_state:
            st.session_state.sector_tab = "Performance Analysis"
        
        # Sub-navigation with clickable boxes
        st.markdown("### Select Analysis Type")
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"📈 Performance Analysis", 
                 help="Compare to sector benchmark, alpha, returns",
                 key=f"perf_{sector_name}", 
                 use_container_width=True):
                    st.session_state.sector_tab = "Performance Analysis"
                    st.rerun()  # Forces the app to refresh and show the Performance tab

        with col2:
            if st.button(f"💰 Financial Analysis", 
                 help="Valuation ratios, profitability, sector comparisons",
                 key=f"fin_{sector_name}", 
                 use_container_width=True):
                    st.session_state.sector_tab = "Financial Analysis"
                    st.rerun()
    
        st.markdown("---")
        sector_tab = st.session_state.sector_tab
        
        # -------------------------------------------------------------------------
        # SECTOR - Performance Analysis
        # -------------------------------------------------------------------------
        if sector_tab == "Performance Analysis":
            st.header(f"📈 {sector_name} - Performance Analysis")
            
            # Sector benchmark mapping
            sector_benchmarks = {
                'TMT': 'XLK',  # Technology Select Sector SPDR Fund
                'FIG': 'XLF',  # Financial Select Sector SPDR Fund
                'Industrials': 'XLI',  # Industrial Select Sector SPDR Fund
                'PUI': 'XLB',  # Materials Select Sector SPDR Fund (closest to PUI)
                'Consumer Goods': 'XLP',  # Consumer Staples Select Sector SPDR Fund
                'Healthcare': 'XLV'  # Health Care Select Sector SPDR Fund
            }
            
            benchmark_ticker = sector_benchmarks.get(sector_name, 'URTH')
            
            # Date range selection
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=pd.to_datetime('2024-11-06'),
                    key=f"start_{sector_name}"
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=pd.to_datetime('today'),
                    key=f"end_{sector_name}"
                )
            
            if st.button("📊 Generate Performance Analysis", type="primary", key=f"gen_{sector_name}"):
                if start_date >= end_date:
                    st.error("Start date must be before end date.")
                else:
                    try:
                        import yfinance as yf
                        import numpy as np
                        from scipy import stats
                        
                        with st.spinner(f'Fetching {sector_name} sector data and benchmark...'):
                            # Fetch benchmark
                            benchmark_data = yf.download(benchmark_ticker, start=start_date, end=end_date, progress=False)
                            
                            # Fetch exchange rates
                            exchange_rates = {}
                            for currency, pair in currency_pairs.items():
                                if pair:
                                    fx_data = yf.download(pair, start=start_date, end=end_date, progress=False)
                                    exchange_rates[currency] = fx_data
                                else:
                                    exchange_rates[currency] = None
                            
                            # Fetch sector holdings data
                            sector_data = {}
                            initial_prices = {}
                            current_prices = {}
                            
                            for ticker, info in sector_holdings.items():
                                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                                if not data.empty:
                                    sector_data[ticker] = data
                                    if isinstance(data['Close'], pd.DataFrame):
                                        close_prices = data['Close'].iloc[:, 0]
                                    else:
                                        close_prices = data['Close']
                                    initial_prices[ticker] = float(close_prices.iloc[0])
                                    current_prices[ticker] = float(close_prices.iloc[-1])
                        
                        # Helper function for FX rates
                        def get_fx_rate(currency, date, exchange_rates):
                            if currency == 'CHF':
                                return 1.0
                            fx_data = exchange_rates[currency]
                            if fx_data is None or fx_data.empty:
                                fallback = {'USD': 0.88, 'EUR': 0.93, 'INR': 0.0104, 'HKD': 0.113, 'AUD': 0.57}
                                return fallback.get(currency, 1.0)
                            
                            if isinstance(fx_data['Close'], pd.DataFrame):
                                fx_close = fx_data['Close'].iloc[:, 0]
                            else:
                                fx_close = fx_data['Close']
                            
                            available_dates = fx_close.index[fx_close.index <= date]
                            if len(available_dates) > 0:
                                return float(fx_close.loc[available_dates[-1]])
                            else:
                                return float(fx_close.iloc[0]) if len(fx_close) > 0 else 1.0
                        
                        # Calculate sector portfolio values
                        all_dates = benchmark_data.index
                        sector_values = pd.Series(index=all_dates, dtype=float)
                        
                        for date in all_dates:
                            daily_value = 0
                            for ticker, info in sector_holdings.items():
                                if ticker in sector_data and not sector_data[ticker].empty:
                                    stock_data = sector_data[ticker]
                                    if isinstance(stock_data['Close'], pd.DataFrame):
                                        stock_close = stock_data['Close'].iloc[:, 0]
                                    else:
                                        stock_close = stock_data['Close']
                                    
                                    available_dates = stock_close.index[stock_close.index <= date]
                                    if len(available_dates) > 0:
                                        stock_price = float(stock_close.loc[available_dates[-1]])
                                    else:
                                        stock_price = initial_prices[ticker]
                                    
                                    fx_rate = get_fx_rate(info['currency'], date, exchange_rates)
                                    stock_value_chf = stock_price * info['quantity'] * fx_rate
                                    daily_value += stock_value_chf
                            
                            sector_values[date] = daily_value
                        
                        # Calculate returns
                        sector_returns = sector_values.pct_change().dropna()
                        
                        if isinstance(benchmark_data['Close'], pd.DataFrame):
                            benchmark_close = benchmark_data['Close'].iloc[:, 0]
                        else:
                            benchmark_close = benchmark_data['Close']
                        benchmark_returns = benchmark_close.pct_change().dropna()
                        
                        # Align dates
                        common_dates = sector_returns.index.intersection(benchmark_returns.index)
                        sector_returns_aligned = sector_returns.loc[common_dates]
                        benchmark_returns_aligned = benchmark_returns.loc[common_dates]
                        
                        # =============================================================
                        # KEY PERFORMANCE METRICS
                        # =============================================================
                        st.subheader("🎯 Key Performance Metrics")
                        
                        # Calculate metrics
                        total_return_sector = ((sector_values.iloc[-1] / sector_values.iloc[0]) - 1) * 100
                        total_return_benchmark = ((benchmark_close.iloc[-1] / benchmark_close.iloc[0]) - 1) * 100
                        
                        # Sharpe Ratio
                        risk_free_rate_daily = 0.02 / 252
                        excess_returns = sector_returns_aligned - risk_free_rate_daily
                        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() != 0 else 0
                        
                        # Beta and Alpha
                        if len(sector_returns_aligned) > 1:
                            slope, intercept, r_value, p_value, std_err = stats.linregress(
                                benchmark_returns_aligned, sector_returns_aligned
                            )
                            beta = slope
                            alpha_annual = intercept * 252
                        else:
                            beta = 0
                            alpha_annual = 0
                        
                        # Volatility
                        volatility_sector = sector_returns_aligned.std() * np.sqrt(252) * 100
                        volatility_benchmark = benchmark_returns_aligned.std() * np.sqrt(252) * 100
                        
                        # Maximum Drawdown
                        cumulative_returns = (1 + sector_returns_aligned).cumprod()
                        running_max = cumulative_returns.expanding().max()
                        drawdown = (cumulative_returns - running_max) / running_max
                        max_drawdown = drawdown.min() * 100
                        
                        # Information Ratio
                        active_returns = sector_returns_aligned - benchmark_returns_aligned
                        tracking_error = active_returns.std() * np.sqrt(252)
                        information_ratio = (active_returns.mean() * 252) / tracking_error if tracking_error != 0 else 0
                        
                        # Display metrics in columns
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Total Return (Sector)",
                                f"{total_return_sector:.2f}%",
                                delta=f"{total_return_sector - total_return_benchmark:.2f}% vs benchmark"
                            )
                            st.metric(
                                "Sharpe Ratio",
                                f"{sharpe_ratio:.2f}",
                                help="Risk-adjusted return"
                            )
                        
                        with col2:
                            st.metric(
                                "Total Return (Benchmark)",
                                f"{total_return_benchmark:.2f}%"
                            )
                            st.metric(
                                "Beta",
                                f"{beta:.2f}",
                                help="Volatility vs sector benchmark"
                            )
                        
                        with col3:
                            st.metric(
                                "Volatility (Sector)",
                                f"{volatility_sector:.2f}%",
                                help="Annualized volatility"
                            )
                            st.metric(
                                "Max Drawdown",
                                f"{max_drawdown:.2f}%",
                                help="Largest peak-to-trough decline"
                            )
                        
                        with col4:
                            st.metric(
                                "Volatility (Benchmark)",
                                f"{volatility_benchmark:.2f}%"
                            )
                            st.metric(
                                "Information Ratio",
                                f"{information_ratio:.2f}",
                                help="Active return per unit of tracking error"
                            )
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "Alpha (Annualized)",
                                f"{alpha_annual*100:.2f}%",
                                help="Excess return vs benchmark"
                            )
                        with col2:
                            st.metric(
                                "Tracking Error",
                                f"{tracking_error*100:.2f}%",
                                help="Standard deviation of active returns"
                            )
                        
                        # =============================================================
                        # PERFORMANCE COMPARISON CHART
                        # =============================================================
                        import plotly.graph_objects as go

                        st.subheader(f"📊 {sector_name} vs {benchmark_ticker} Performance")

                        # Normalize to 100
                        sector_normalized = (sector_values / sector_values.iloc[0]) * 100
                        benchmark_normalized = (benchmark_close / benchmark_close.iloc[0]) * 100

                        # Create plotly figure
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=sector_normalized.index, y=sector_normalized, 
                                                 name=f'{sector_name} Portfolio', mode='lines'))
                        fig.add_trace(go.Scatter(x=benchmark_normalized.index, y=benchmark_normalized,
                                                 name=f'{benchmark_ticker} Benchmark', mode='lines'))

                        # Update x-axis based on time range
                        fig.update_xaxes(
                            rangeslider_visible=False,
                            rangeselector=dict(
                                buttons=list([
                                    dict(count=1, label="1M", step="month", stepmode="backward"),
                                    dict(count=6, label="6M", step="month", stepmode="backward"),
                                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                                    dict(step="all", label="All")
                                    ])
                                )
                            )

                        fig.update_layout(yaxis_title="Normalized Value (Base=100)", xaxis_title="Date")
                        st.plotly_chart(fig, use_container_width=True)                        
                        # =============================================================
                        # HOLDINGS PERFORMANCE BREAKDOWN
                        # =============================================================
                        st.subheader("🏆 Holdings Performance Breakdown")
                        
                        holdings_performance = []
                        for ticker, info in sector_holdings.items():
                            if ticker in sector_data:
                                stock_return = ((current_prices[ticker] / initial_prices[ticker]) - 1) * 100
                                
                                # Calculate contribution to portfolio
                                initial_value = initial_prices[ticker] * info['quantity']
                                current_value = current_prices[ticker] * info['quantity']
                                
                                # Convert to CHF for contribution calculation
                                fx_rate_initial = get_fx_rate(info['currency'], sector_values.index[0], exchange_rates)
                                fx_rate_current = get_fx_rate(info['currency'], sector_values.index[-1], exchange_rates)
                                
                                initial_value_chf = initial_value * fx_rate_initial
                                current_value_chf = current_value * fx_rate_current
                                
                                weight = (initial_value_chf / sector_values.iloc[0]) * 100
                                contribution = ((current_value_chf - initial_value_chf) / sector_values.iloc[0]) * 100
                                
                                holdings_performance.append({
                                    'Ticker': ticker,
                                    'Name': info['name'],
                                    'Return (%)': stock_return,
                                    'Weight (%)': weight,
                                    'Contribution (%)': contribution
                                })
                        
                        perf_df = pd.DataFrame(holdings_performance).sort_values('Return (%)', ascending=False)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Best Performers**")
                            st.dataframe(
                                perf_df.head(3).style.format({
                                    'Return (%)': '{:.2f}%',
                                    'Weight (%)': '{:.2f}%',
                                    'Contribution (%)': '{:.2f}%'
                                }),
                                hide_index=True
                            )
                        
                        with col2:
                            st.markdown("**Worst Performers**")
                            st.dataframe(
                                perf_df.tail(3).style.format({
                                    'Return (%)': '{:.2f}%',
                                    'Weight (%)': '{:.2f}%',
                                    'Contribution (%)': '{:.2f}%'
                                }),
                                hide_index=True
                            )
                        
                        st.markdown("**All Holdings**")
                        st.dataframe(
                            perf_df.style.format({
                                'Return (%)': '{:.2f}%',
                                'Weight (%)': '{:.2f}%',
                                'Contribution (%)': '{:.2f}%'
                            }).background_gradient(subset=['Return (%)'], cmap='RdYlGn'),
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        # =============================================================
                        # RISK METRICS
                        # =============================================================
                        st.subheader("⚠️ Risk Metrics")
                        
                        # Value at Risk (95% confidence)
                        var_95 = np.percentile(sector_returns_aligned, 5) * 100
                        
                        # Correlation matrix
                        if len(sector_holdings) > 1:
                            returns_dict = {}
                            for ticker, info in sector_holdings.items():
                                if ticker in sector_data:
                                    stock_data = sector_data[ticker]
                                    if isinstance(stock_data['Close'], pd.DataFrame):
                                        stock_close = stock_data['Close'].iloc[:, 0]
                                    else:
                                        stock_close = stock_data['Close']
                                    returns_dict[info['name']] = stock_close.pct_change().dropna()
                            
                            returns_df = pd.DataFrame(returns_dict)
                            correlation_matrix = returns_df.corr()
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric(
                                    "Value at Risk (95%)",
                                    f"{var_95:.2f}%",
                                    help="Potential loss on worst 5% of days"
                                )
                            
                            with col2:
                                avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
                                st.metric(
                                    "Average Correlation",
                                    f"{avg_correlation:.2f}",
                                    help="Average correlation between holdings"
                                )
                            
                            st.markdown("**Correlation Matrix**")
                            st.dataframe(
                                correlation_matrix.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1).format("{:.2f}"),
                                use_container_width=True
                            )
                        
                    except Exception as e:
                        st.error(f"Error generating analysis: {str(e)}")
        
# -------------------------------------------------------------------------
# SECTOR - Financial Analysis
# -------------------------------------------------------------------------
elif sector_tab == "Financial Analysis":
    st.header(f"💰 {sector_name} - Financial Analysis")
    
    if st.button("📊 Generate Financial Analysis", type="primary", key=f"fin_gen_{sector_name}"):
        try:
            import yfinance as yf
            import plotly.graph_objects as go
            
            with st.spinner(f'Fetching financial data for {sector_name} holdings...'):
                financial_data = []
                
                for ticker, info in sector_holdings.items():
                    try:
                        stock = yf.Ticker(ticker)
                        stock_info = stock.info
                        
                        # Get cash flow data for operating cash flow ratio
                        try:
                            cash_flow = stock.cash_flow
                            balance_sheet = stock.balance_sheet
                            
                            if not cash_flow.empty and not balance_sheet.empty:
                                operating_cash_flow = cash_flow.loc['Operating Cash Flow'].iloc[0] if 'Operating Cash Flow' in cash_flow.index else None
                                current_liabilities = balance_sheet.loc['Current Liabilities'].iloc[0] if 'Current Liabilities' in balance_sheet.index else None
                                
                                if operating_cash_flow and current_liabilities and current_liabilities != 0:
                                    ocf_ratio = operating_cash_flow / current_liabilities
                                else:
                                    ocf_ratio = None
                            else:
                                ocf_ratio = None
                        except:
                            ocf_ratio = None
                        
                        financial_data.append({
                            'Ticker': ticker,
                            'Name': info['name'],
                            'Market Cap': stock_info.get('marketCap', None),
                            'P/E Ratio': stock_info.get('trailingPE', None),
                            'Forward P/E': stock_info.get('forwardPE', None),
                            'P/B Ratio': stock_info.get('priceToBook', None),
                            'Dividend Yield (%)': stock_info.get('dividendYield', None) * 100 if stock_info.get('dividendYield') else None,
                            'Profit Margin (%)': stock_info.get('profitMargins', None) * 100 if stock_info.get('profitMargins') else None,
                            'ROE (%)': stock_info.get('returnOnEquity', None) * 100 if stock_info.get('returnOnEquity') else None,
                            'ROA (%)': stock_info.get('returnOnAssets', None) * 100 if stock_info.get('returnOnAssets') else None,
                            'Debt/Equity': stock_info.get('debtToEquity', None),
                            'Current Ratio': stock_info.get('currentRatio', None),
                            'Revenue Growth (%)': stock_info.get('revenueGrowth', None) * 100 if stock_info.get('revenueGrowth') else None,
                            'Beta': stock_info.get('beta', None),
                            'OCF Ratio': ocf_ratio
                        })
                    except Exception as e:
                        st.warning(f"Could not fetch data for {ticker}: {str(e)}")
                
                if financial_data:
                    fin_df = pd.DataFrame(financial_data)
                    
                    # =============================================================
                    # KEY RATIOS - BAR CHARTS WITH SECTOR AVERAGE
                    # =============================================================
                    st.subheader("📊 Key Financial Ratios")
                    
                    # Define the ratios to chart
                    ratio_charts = [
                        {'column': 'P/E Ratio', 'title': 'Price-to-Earnings Ratio', 'format': '.2f'},
                        {'column': 'P/B Ratio', 'title': 'Price-to-Book Ratio', 'format': '.2f'},
                        {'column': 'Debt/Equity', 'title': 'Debt-to-Equity Ratio', 'format': '.2f'},
                        {'column': 'OCF Ratio', 'title': 'Operating Cash Flow Ratio', 'format': '.2f'}
                    ]
                    
                    # Create 2x2 grid for charts
                    col1, col2 = st.columns(2)
                    
                    for idx, ratio_info in enumerate(ratio_charts):
                        column = ratio_info['column']
                        title = ratio_info['title']
                        
                        # Filter out None/NaN values
                        chart_data = fin_df[['Name', column]].copy()
                        chart_data = chart_data[chart_data[column].notna()]
                        
                        if not chart_data.empty:
                            # Calculate sector average
                            sector_avg = chart_data[column].mean()
                            
                            # Create bar chart
                            fig = go.Figure()
                            
                            # Add bars
                            fig.add_trace(go.Bar(
                                x=chart_data['Name'],
                                y=chart_data[column],
                                name=title,
                                marker_color='lightblue',
                                text=chart_data[column].round(2),
                                textposition='outside'
                            ))
                            
                            # Add sector average line
                            fig.add_trace(go.Scatter(
                                x=chart_data['Name'],
                                y=[sector_avg] * len(chart_data),
                                mode='lines',
                                name=f'Sector Avg: {sector_avg:.2f}',
                                line=dict(color='red', width=2, dash='dot'),
                                showlegend=True
                            ))
                            
                            fig.update_layout(
                                title=title,
                                xaxis_title="Company",
                                yaxis_title=column,
                                height=400,
                                showlegend=True,
                                hovermode='x unified'
                            )
                            
                            # Display in alternating columns
                            if idx % 2 == 0:
                                with col1:
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                with col2:
                                    st.plotly_chart(fig, use_container_width=True)
                        else:
                            if idx % 2 == 0:
                                with col1:
                                    st.info(f"No data available for {title}")
                            else:
                                with col2:
                                    st.info(f"No data available for {title}")
                    
                    # =============================================================
                    # COMPREHENSIVE SUMMARY TABLE
                    # =============================================================
                    st.subheader("📋 Comprehensive Financial Summary")
                    
                    # Format function that handles None values
                    def format_value(val, decimal_places=2):
                        if val is None or pd.isna(val):
                            return 'N/A'
                        try:
                            return f'{float(val):.{decimal_places}f}'
                        except (ValueError, TypeError):
                            return 'N/A'
                    
                    def format_market_cap(val):
                        if val is None or pd.isna(val):
                            return 'N/A'
                        try:
                            val = float(val)
                            if val >= 1e12:
                                return f'${val/1e12:.2f}T'
                            elif val >= 1e9:
                                return f'${val/1e9:.2f}B'
                            elif val >= 1e6:
                                return f'${val/1e6:.2f}M'
                            else:
                                return f'${val:,.0f}'
                        except (ValueError, TypeError):
                            return 'N/A'
                    
                    # Create display columns
                    display_cols = ['Ticker', 'Name', 'Market Cap', 'P/E Ratio', 'Forward P/E', 'P/B Ratio',
                                   'Debt/Equity', 'OCF Ratio', 'Current Ratio', 'Profit Margin (%)', 
                                   'ROE (%)', 'ROA (%)', 'Dividend Yield (%)', 'Revenue Growth (%)', 'Beta']
                    
                    summary_display = fin_df[display_cols].copy()
                    
                    # Apply formatting
                    formatted_df = summary_display.style.format({
                        'Market Cap': format_market_cap,
                        'P/E Ratio': lambda x: format_value(x, 2),
                        'Forward P/E': lambda x: format_value(x, 2),
                        'P/B Ratio': lambda x: format_value(x, 2),
                        'Debt/Equity': lambda x: format_value(x, 2),
                        'OCF Ratio': lambda x: format_value(x, 2),
                        'Current Ratio': lambda x: format_value(x, 2),
                        'Profit Margin (%)': lambda x: format_value(x, 2),
                        'ROE (%)': lambda x: format_value(x, 2),
                        'ROA (%)': lambda x: format_value(x, 2),
                        'Dividend Yield (%)': lambda x: format_value(x, 2),
                        'Revenue Growth (%)': lambda x: format_value(x, 2),
                        'Beta': lambda x: format_value(x, 2)
                    })
                    
                    st.dataframe(formatted_df, hide_index=True, use_container_width=True)
                    
                    # =============================================================
                    # SECTOR STATISTICS
                    # =============================================================
                    st.subheader("🔍 Sector Statistics")
                    
                    # Calculate sector averages
                    numeric_cols = ['P/E Ratio', 'P/B Ratio', 'Debt/Equity', 'OCF Ratio',
                                   'Profit Margin (%)', 'ROE (%)', 'Dividend Yield (%)', 'Revenue Growth (%)']
                    
                    summary_stats = []
                    for col in numeric_cols:
                        values = pd.to_numeric(fin_df[col], errors='coerce').dropna()
                        if len(values) > 0:
                            summary_stats.append({
                                'Metric': col,
                                'Average': values.mean(),
                                'Median': values.median(),
                                'Min': values.min(),
                                'Max': values.max(),
                                'Std Dev': values.std()
                            })
                    
                    if summary_stats:
                        stats_df = pd.DataFrame(summary_stats)
                        st.dataframe(
                            stats_df.style.format({
                                'Average': '{:.2f}',
                                'Median': '{:.2f}',
                                'Min': '{:.2f}',
                                'Max': '{:.2f}',
                                'Std Dev': '{:.2f}'
                            }),
                            hide_index=True,
                            use_container_width=True
                        )
                    
                else:
                    st.warning("No financial data could be retrieved for this sector.")
        
        except Exception as e:
            st.error(f"Error generating financial analysis: {str(e)}")# ADDING TOOL PAGE
# =============================================================================
elif main_page == "Adding Tool":
    st.title("🔧 Portfolio Addition Simulator")
    
    st.markdown("""
    ### This tool will allow you to:
    
    **Stock Addition Simulation:**
    - Input a stock ticker to analyze
    - Specify allocation amount or percentage
    - Select position size relative to current portfolio
    
    **Impact Analysis:**
    - **Portfolio Composition Changes:**
      - New sector allocation breakdown
      - Updated geographical distribution
      - Changes in market cap exposure
    
    - **Performance Impact:**
      - Historical performance simulation (if stock was added X months ago)
      - Expected return impact based on analyst forecasts
      - Risk-adjusted return changes (Sharpe Ratio adjustment)
    
    - **Risk Metrics Changes:**
      - Portfolio Beta adjustment
      - Volatility impact
      - Correlation with existing holdings
      - Diversification benefit/cost
    
    - **Financial Ratios Impact:**
      - Weighted average P/E, P/B, P/S changes
      - Portfolio-level profitability metrics adjustment
      - Debt and financial health metrics
    
    **Comparison View:**
    - Side-by-side comparison: Current Portfolio vs Portfolio with New Addition
    - Visual charts showing before/after metrics
    - Recommendation engine: "Add", "Consider", or "Avoid" based on portfolio fit
    
    **What-If Scenarios:**
    - Test multiple stocks simultaneously
    - Compare different allocation sizes
    - Rebalancing suggestions after addition
    """)
    
    # Placeholder for future content
    st.info("🛠️ Stock addition simulation tool will be implemented here")
    
    # Add some interactive placeholder elements
    st.subheader("Quick Preview (Coming Soon)")
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Enter Stock Ticker", placeholder="e.g., AAPL, MSFT, NESN.SW")
        st.number_input("Allocation Amount (CHF)", min_value=0, value=10000, step=1000)
    with col2:
        st.selectbox("Allocation Method", ["Fixed Amount", "Percentage of Portfolio", "Equal Weight"])
        st.button("Run Simulation", type="primary", disabled=True)

# =============================================================================
# FOOTER
# =============================================================================
st.sidebar.markdown("---")
st.sidebar.info("""
**Portfolio Dashboard v1.0**

Navigate through different sections to analyze your investment portfolio comprehensively.
""")
