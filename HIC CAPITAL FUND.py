# -------------------------------------------------------------------------
        # SECTOR - Company Specific Analysis
        # -------------------------------------------------------------------------
        if sector_tab == "Company Specific":
            st.header(f"🏢 {sector_name} - Company Specific Analysis")
            
            # Company selection dropdown
            company_options = {info['name']: ticker for ticker, info in sector_holdings.items()}
            selected_company_name = st.selectbox(
                "Select Company to Analyze",
                options=list(company_options.keys()),
                key=f"company_select_{sector_name}"
            )
            
            if selected_company_name:
                selected_ticker = company_options[selected_company_name]
                company_info = sector_holdings[selected_ticker]
                
                st.subheader(f"📊 {selected_company_name} ({selected_ticker})")
                
                # Display company key info
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Sector", company_info['sector'])
                with col2:
                    st.metric("Currency", company_info['currency'])
                with col3:
                    st.metric("Quantity Held", f"{company_info['quantity']:,}")
                with col4:
                    # Display purchase price from portfolio_holdings (fetched at startup)
                    purchase_price = company_info.get('purchase_price', 0.0)
                    st.metric("Purchase Price", f"{purchase_price:.2f}")
                
                st.markdown("---")
                
                # Date range selection for stock price chart
                st.subheader("📈 Stock Price Analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    # Parse the purchase_date string to datetime for date input default
                    purchase_datetime = pd.to_datetime(company_info['purchase_date'])
                    chart_start_date = st.date_input(
                        "Chart Start Date",
                        value=purchase_datetime.date() - pd.Timedelta(days=30),
                        key=f"chart_start_{selected_ticker}"
                    )
                with col2:
                    chart_end_date = st.date_input(
                        "Chart End Date",
                        value=pd.to_datetime('today'),
                        key=f"chart_end_{selected_ticker}"
                    )
                
                # Fetch analyst data for additional context (but use Target_price from dictionary)
                try:
                    stock = yf.Ticker(selected_ticker)
                    stock_info = stock.info
                    analyst_target = stock_info.get('targetMeanPrice', None)
                    num_analysts = stock_info.get('numberOfAnalystOpinions', 0)
                    recommendation = stock_info.get('recommendationKey', 'N/A').upper()
                except:
                    analyst_target = None
                    num_analysts = 0
                    recommendation = 'N/A'
                
                # Get target price from dictionary
                target_price = company_info.get('Target_price', None)
                
                if st.button("📊 Generate Stock Analysis", type="primary", key=f"gen_stock_{selected_ticker}"):
                    if chart_start_date >= chart_end_date:
                        st.error("Start date must be before end date.")
                    else:
                        try:
                            with st.spinner(f'Fetching stock data for {selected_company_name}...'):
                                # Fetch historical stock data
                                stock_data = yf.download(selected_ticker, start=chart_start_date, end=chart_end_date, progress=False)
                                
                                if stock_data.empty:
                                    st.error(f"No data available for {selected_ticker} in the selected date range.")
                                else:
                                    # Extract Close prices
                                    if isinstance(stock_data['Close'], pd.DataFrame):
                                        close_prices = stock_data['Close'].iloc[:, 0]
                                    else:
                                        close_prices = stock_data['Close']
                                    
                                    # Create the stock price chart
                                    fig = go.Figure()
                                    
                                    # Main stock price line
                                    fig.add_trace(go.Scatter(
                                        x=close_prices.index,
                                        y=close_prices,
                                        mode='lines',
                                        name=f'{selected_ticker} Price',
                                        line=dict(color='#0F1D64', width=2),
                                        hovertemplate='<b>Date</b>: %{x}<br><b>Price</b>: %{y:.2f}<extra></extra>'
                                    ))
                                    
                                    # Add vertical line for purchase date using shapes (no annotation)
                                    purchase_date_str = company_info['purchase_date']
                                    purchase_date_obj = pd.to_datetime(purchase_date_str)
                                    purchase_price_display = company_info.get('purchase_price', 0.0)
                                    
                                    if chart_start_date <= purchase_date_obj.date() <= chart_end_date:
                                        # Add vertical line using shapes
                                        fig.add_shape(
                                            type="line",
                                            x0=purchase_date_str,
                                            x1=purchase_date_str,
                                            y0=0,
                                            y1=1,
                                            yref="paper",
                                            line=dict(color="green", width=2, dash="dot")
                                        )
                                        
                                        # Add invisible trace for legend entry
                                        fig.add_trace(go.Scatter(
                                            x=[None],
                                            y=[None],
                                            mode='lines',
                                            name=f'Purchase Date: {purchase_date_str} (Price: {purchase_price_display:.2f})',
                                            line=dict(color='green', width=2, dash='dot'),
                                            showlegend=True
                                        ))
                                    
                                    # Add horizontal line for target price using shapes (from dictionary)
                                    if target_price and target_price > 0:
                                        fig.add_shape(
                                            type="line",
                                            x0=0,
                                            x1=1,
                                            xref="paper",
                                            y0=target_price,
                                            y1=target_price,
                                            line=dict(color="red", width=2, dash="dash")
                                        )
                                        
                                        # Add invisible trace for legend entry
                                        fig.add_trace(go.Scatter(
                                            x=[None],
                                            y=[None],
                                            mode='lines',
                                            name=f'Target Price: {target_price:.2f}',
                                            line=dict(color='red', width=2, dash='dash'),
                                            showlegend=True
                                        ))
                                    
                                    # Update layout
                                    fig.update_layout(
                                        title=f'{selected_company_name} ({selected_ticker}) - Stock Price History',
                                        xaxis_title='Date',
                                        yaxis_title=f'Price ({company_info["currency"]})',
                                        hovermode='x unified',
                                        height=500,
                                        template='plotly_white',
                                        showlegend=True,
                                        legend=dict(
                                            yanchor="top",
                                            y=0.99,
                                            xanchor="left",
                                            x=0.01
                                        )
                                    )
                                    
                                    # Add range slider
                                    fig.update_xaxes(
                                        rangeslider_visible=True,
                                        rangeselector=dict(
                                            buttons=list([
                                                dict(count=1, label="1M", step="month", stepmode="backward"),
                                                dict(count=3, label="3M", step="month", stepmode="backward"),
                                                dict(count=6, label="6M", step="month", stepmode="backward"),
                                                dict(count=1, label="1Y", step="year", stepmode="backward"),
                                                dict(step="all", label="All")
                                            ])
                                        )
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Performance metrics
                                    st.subheader("📊 Performance Metrics")
                                    
                                    current_price = float(close_prices.iloc[-1])
                                    purchase_price_calc = company_info.get('purchase_price', 0.0)
                                    
                                    if purchase_price_calc > 0:
                                        total_return = ((current_price - purchase_price_calc) / purchase_price_calc) * 100
                                    else:
                                        total_return = 0.0
                                    
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    with col1:
                                        st.metric(
                                            "Current Price",
                                            f"{current_price:.2f} {company_info['currency']}",
                                            delta=f"{total_return:.2f}%"
                                        )
                                    
                                    with col2:
                                        position_value = current_price * company_info['quantity']
                                        st.metric(
                                            "Position Value",
                                            f"{position_value:,.2f} {company_info['currency']}"
                                        )
                                    
                                    with col3:
                                        gain_loss = (current_price - purchase_price_calc) * company_info['quantity']
                                        st.metric(
                                            "Total Gain/Loss",
                                            f"{gain_loss:,.2f} {company_info['currency']}",
                                            delta=f"{total_return:.2f}%"
                                        )
                                    
                                    with col4:
                                        if target_price and target_price > 0:
                                            upside = ((target_price - current_price) / current_price) * 100
                                            st.metric(
                                                "Upside to Target",
                                                f"{upside:.2f}%",
                                                delta=f"{target_price:.2f} target"
                                            )
                                        else:
                                            st.metric("Upside to Target", "N/A")
                                    
                                    # Target Price & Analyst Information
                                    st.markdown("---")
                                    st.subheader("🎯 Price Targets")
                                    
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    with col1:
                                        if target_price and target_price > 0:
                                            st.metric("Your Target Price", f"{target_price:.2f} {company_info['currency']}")
                                        else:
                                            st.metric("Your Target Price", "Not Set")
                                    
                                    with col2:
                                        if analyst_target:
                                            st.metric("Analyst Consensus", f"{analyst_target:.2f} {company_info['currency']}")
                                        else:
                                            st.metric("Analyst Consensus", "N/A")
                                    
                                    with col3:
                                        if analyst_target:
                                            st.metric("Number of Analysts", num_analysts)
                                        else:
                                            st.metric("Number of Analysts", "N/A")
                                    
                                    with col4:
                                        st.metric("Recommendation", recommendation)
                                    
                                    # Investment Thesis
                                    st.markdown("---")
                                    st.subheader("💡 Investment Thesis")
                                    
                                    thesis_text = company_info.get('thesis', 'No thesis available')
                                    st.info(thesis_text)
                                    
                                    # DCF Parameters (if available)
                                    st.markdown("---")
                                    st.subheader("💰 DCF Valuation Parameters")
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.write("**WACC:**", f"{company_info.get('WACC', 'N/A')}%")
                                    
                                    with col2:
                                        st.write("**Cash Flow Projections:**")
                                        for i in range(1, 6):
                                            cf_value = company_info.get(f'CF_{i}', 'N/A')
                                            st.write(f"Year {i}: {cf_value}")
                                    
                        except Exception as e:
                            st.error(f"Error generating stock analysis: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
