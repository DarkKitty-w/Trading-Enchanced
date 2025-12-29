import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from core.database import Database
import numpy as np
from typing import Dict, List, Optional
import asyncio


def display_strategy_data(strategy_id: str, strategy_name: str, data: Dict):
    """Display data for a single strategy."""
    
    # --- KPI ROW ---
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cash = data.get('cash', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">AVAILABLE CASH</div>
            <div class="metric-value">${cash:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Calculate Total Equity
        positions_value = sum([p['quantity'] * p['entry_price'] for p in data.get('positions', [])])
        total_equity = cash + positions_value
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">TOTAL EQUITY</div>
            <div class="metric-value">${total_equity:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Calculate Realized PnL from closed positions
        realized_pnl = 0
        for metric in data.get('metrics', []):
            if 'total_pnl' in metric and metric['total_pnl'] is not None:
                realized_pnl += metric['total_pnl']
        
        pnl_class = "positive" if realized_pnl >= 0 else "negative"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">REALIZED P&L</div>
            <div class="metric-value {pnl_class}">${realized_pnl:+,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Calculate Win Rate
        winning_trades = [t for t in data.get('trades', []) if t.get('side') == 'SELL']
        total_closed = len(winning_trades)
        win_rate = 0
        if total_closed > 0:
            profitable_trades = 0
            for t in winning_trades:
                profit = t.get('profit')
                if profit is not None and profit > 0:
                    profitable_trades += 1
            win_rate = (profitable_trades / total_closed * 100)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">WIN RATE</div>
            <div class="metric-value">{win_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    # --- TABS VIEW ---
    tab_pos, tab_trades, tab_history, tab_analysis = st.tabs([
        "🛡️ POSITIONS", 
        "📜 TRADES", 
        "📈 PERFORMANCE",
        "🔍 ANALYSIS"
    ])
    
    # TAB 1: POSITIONS
    with tab_pos:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Open Positions")
            positions = data.get('positions', [])
            
            if positions:
                df_positions = pd.DataFrame(positions)
                
                # Calculate unrealized PnL
                unrealized_pnls = []
                for pos in positions:
                    # For now, use entry price as current price (in real system, fetch live prices)
                    current_price = pos['entry_price']  # This should be replaced with live price
                    if pos['side'] == 'LONG':
                        unrealized = (current_price - pos['entry_price']) * pos['quantity']
                    else:
                        unrealized = (pos['entry_price'] - current_price) * pos['quantity']
                    unrealized_pnls.append(unrealized)
                
                df_positions['unrealized_pnl'] = unrealized_pnls
                df_positions['unrealized_pnl_pct'] = (df_positions['unrealized_pnl'] / 
                                                     (df_positions['entry_price'] * df_positions['quantity'])) * 100
                
                # Format display
                display_cols = ['symbol', 'side', 'quantity', 'entry_price', 'unrealized_pnl', 'unrealized_pnl_pct']
                st.dataframe(
                    df_positions[display_cols].style.format({
                        'quantity': '{:.6f}',
                        'entry_price': '${:.2f}',
                        'unrealized_pnl': '${:+,.2f}',
                        'unrealized_pnl_pct': '{:+.2f}%'
                    }).apply(
                        lambda x: ['background-color: rgba(16, 185, 129, 0.1)' if val > 0 else 
                                 'background-color: rgba(239, 68, 68, 0.1)' for val in x],
                        subset=['unrealized_pnl', 'unrealized_pnl_pct']
                    ),
                    use_container_width=True,
                    height=300
                )
            else:
                st.info("No active positions.")
        
        with col2:
            st.subheader("Position Summary")
            if positions:
                total_position_value = sum([p['quantity'] * p['entry_price'] for p in positions])
                avg_position_size = total_position_value / len(positions) if positions else 0
                
                st.metric("Total Positions", len(positions))
                st.metric("Total Exposure", f"${total_position_value:,.2f}")
                st.metric("Avg Position Size", f"${avg_position_size:,.2f}")
                
                # Side distribution
                long_count = len([p for p in positions if p['side'] == 'LONG'])
                short_count = len([p for p in positions if p['side'] == 'SHORT'])
                
                fig = px.pie(
                    values=[long_count, short_count],
                    names=['LONG', 'SHORT'],
                    title='Position Sides',
                    color_discrete_sequence=['#10b981', '#ef4444']
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#94a3b8'),
                    showlegend=True,
                    height=250
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: TRADES
    with tab_trades:
        trades = data.get('trades', [])
        
        if trades:
            df_trades = pd.DataFrame(trades)
            df_trades['executed_at'] = pd.to_datetime(df_trades['executed_at'])
            
            # Add profit/loss coloring
            def color_profit(val):
                if val is None:
                    return ''
                try:
                    val_float = float(val)
                    if val_float > 0:
                        return 'color: #10b981'
                    elif val_float < 0:
                        return 'color: #ef4444'
                except (ValueError, TypeError):
                    return ''
                return ''
            
            # Display trades
            display_cols = ['executed_at', 'symbol', 'side', 'quantity', 'price', 'fees']
            if 'profit' in df_trades.columns:
                display_cols.append('profit')
            
            # Handle profit column if it exists
            if 'profit' in df_trades.columns:
                df_trades['profit'] = pd.to_numeric(df_trades['profit'], errors='coerce')
            
            st.dataframe(
                df_trades[display_cols].sort_values('executed_at', ascending=False).style.format({
                    'quantity': '{:.6f}',
                    'price': '${:.2f}',
                    'fees': '${:.2f}',
                    'profit': '${:+,.2f}' if 'profit' in df_trades.columns else '${:.2f}',
                    'executed_at': lambda t: t.strftime('%Y-%m-%d %H:%M')
                }).applymap(color_profit, subset=['profit'] if 'profit' in df_trades.columns else []),
                use_container_width=True,
                height=400
            )
            
            # Trade statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                buy_trades = len([t for t in trades if t['side'] == 'BUY'])
                sell_trades = len([t for t in trades if t['side'] == 'SELL'])
                st.metric("Buy Trades", buy_trades)
                st.metric("Sell Trades", sell_trades)
            
            with col2:
                avg_trade_size = df_trades['quantity'].mean() if not df_trades.empty else 0
                avg_trade_price = df_trades['price'].mean() if not df_trades.empty else 0
                st.metric("Avg Trade Size", f"{avg_trade_size:.4f}")
                st.metric("Avg Price", f"${avg_trade_price:.2f}")
            
            with col3:
                total_fees = df_trades['fees'].sum() if 'fees' in df_trades.columns else 0
                st.metric("Total Fees", f"${total_fees:.2f}")
            
            # Trade timeline visualization
            st.subheader("Trade Timeline")
            if len(df_trades) > 1:
                fig = px.scatter(
                    df_trades, 
                    x='executed_at', 
                    y='price', 
                    color='side',
                    size='quantity',
                    title='Trade Execution History',
                    color_discrete_map={'BUY': '#10b981', 'SELL': '#ef4444'},
                    hover_data=['quantity', 'symbol', 'fees']
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#94a3b8'),
                    xaxis_title="Date",
                    yaxis_title="Price ($)"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trade history available.")
    
    # TAB 3: PERFORMANCE
    with tab_history:
        history = data.get('history', [])
        
        if history:
            df_history = pd.DataFrame(history)
            df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
            
            # Equity curve
            st.subheader("Equity Curve")
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_history['timestamp'],
                y=df_history['total_equity'],
                mode='lines',
                name='Equity',
                line=dict(color='#3b82f6', width=3)
            ))
            
            # Add initial capital line
            if not df_history.empty:
                initial_equity = df_history['total_equity'].iloc[0]
                fig.add_hline(
                    y=initial_equity,
                    line_dash="dash",
                    line_color="green",
                    opacity=0.5,
                    annotation_text="Initial Capital"
                )
            
            fig.update_layout(
                title='Portfolio Equity Over Time',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8'),
                xaxis_title="Date",
                yaxis_title="Equity ($)",
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance metrics table
            st.subheader("Performance Metrics")
            
            if not df_history.empty:
                # Calculate metrics
                initial = df_history['total_equity'].iloc[0]
                final = df_history['total_equity'].iloc[-1]
                total_return = ((final - initial) / initial * 100)
                
                # Calculate drawdown
                df_history['peak'] = df_history['total_equity'].cummax()
                df_history['drawdown'] = (df_history['total_equity'] - df_history['peak']) / df_history['peak'] * 100
                max_drawdown = df_history['drawdown'].min()
                
                # Display metrics
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                with metrics_col1:
                    st.metric("Total Return", f"{total_return:.2f}%")
                    st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
                with metrics_col2:
                    # Calculate daily returns (simplified)
                    if len(df_history) > 1:
                        daily_returns = df_history['total_equity'].pct_change().dropna()
                        sharpe_ratio = (daily_returns.mean() / daily_returns.std() * np.sqrt(365)) if daily_returns.std() > 0 else 0
                        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                    else:
                        st.metric("Sharpe Ratio", "N/A")
                with metrics_col3:
                    if len(df_history) > 1:
                        daily_returns = df_history['total_equity'].pct_change().dropna()
                        winning_days = len([r for r in daily_returns if r > 0])
                        total_days = len(daily_returns)
                        win_rate = (winning_days / total_days * 100) if total_days > 0 else 0
                        st.metric("Win Rate (Days)", f"{win_rate:.1f}%")
                    else:
                        st.metric("Win Rate (Days)", "N/A")
        else:
            st.info("No performance history available.")
    
    # TAB 4: ANALYSIS
    with tab_analysis:
        st.subheader("Strategy Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk metrics
            st.metric("Risk per Trade", f"{data.get('risk_per_trade', '2')}%")
            st.metric("Max Drawdown Limit", f"{data.get('max_drawdown_limit', '15')}%")
            st.metric("Consecutive Loss Limit", f"{data.get('max_consecutive_losses', '5')}")
        
        with col2:
            # Current market exposure
            positions = data.get('positions', [])
            total_exposure = sum([p['quantity'] * p['entry_price'] for p in positions])
            cash = data.get('cash', 0)
            total_equity = cash + total_exposure
            exposure_pct = (total_exposure / total_equity * 100) if total_equity > 0 else 0
            
            st.metric("Market Exposure", f"{exposure_pct:.1f}%")
            st.metric("Cash Reserve", f"${cash:,.2f}")
            
            # Exposure gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=exposure_pct,
                title={'text': "Exposure %"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#3b82f6"},
                    'steps': [
                        {'range': [0, 50], 'color': "rgba(16, 185, 129, 0.2)"},
                        {'range': [50, 80], 'color': "rgba(245, 158, 11, 0.2)"},
                        {'range': [80, 100], 'color': "rgba(239, 68, 68, 0.2)"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8'),
                height=250
            )
            st.plotly_chart(fig, use_container_width=True)


# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="PHOENIX TRADING DASHBOARD",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional dashboard
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        color: #e2e8f0;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid #475569;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 6px 12px -2px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px -4px rgba(0, 0, 0, 0.4);
    }
    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #ffffff;
        margin-top: 8px;
    }
    .metric-label {
        font-size: 14px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .positive {
        color: #10b981 !important;
    }
    .negative {
        color: #ef4444 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 41, 59, 0.5);
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        font-weight: 600;
    }
    
    /* Dataframe styling */
    .dataframe {
        background-color: rgba(15, 23, 42, 0.7) !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #334155;
    }
    
    /* Auto-refresh indicator */
    .refresh-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .refresh-active {
        background-color: #10b981;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# Initialize Database Connection
@st.cache_resource
def init_database():
    try:
        return Database()
    except Exception as e:
        st.error(f"❌ Database Connection Failed: {e}")
        st.stop()

db = init_database()

# --- 2. SIDEBAR: CONTROLS & FILTERS ---
st.sidebar.title("📡 PHOENIX CONTROL PANEL")

# Auto-refresh toggle
auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh", value=True, help="Auto-refresh data every 30 seconds")
refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 10, 300, 30)

# Time range selector
st.sidebar.markdown("---")
st.sidebar.subheader("📅 Time Range")
time_range = st.sidebar.selectbox(
    "Select Time Range",
    ["Last 24 hours", "Last 7 days", "Last 30 days", "All time", "Custom"]
)

if time_range == "Custom":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End Date", datetime.now())

# Strategy selection
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Strategy Selection")

# Fetch Strategies
try:
    strategies_res = db.client.table("strategies").select("id, name, created_at").execute()
    strategies = strategies_res.data
except Exception as e:
    st.sidebar.error(f"Failed to fetch strategies: {e}")
    strategies = []

if not strategies:
    st.sidebar.warning("⚠️ No strategies found.")
    st.sidebar.info("Run a strategy script to register one.")
    selected_strategy_id = None
    selected_strategy_name = None
else:
    # Create mapping for dropdown
    strategy_options = [s['name'] for s in strategies]
    selected_strategy_name = st.sidebar.selectbox(
        "SELECT STRATEGY",
        options=strategy_options,
        index=0 if strategy_options else None
    )
    
    # Get selected strategy ID
    selected_strategy = next((s for s in strategies if s['name'] == selected_strategy_name), None)
    selected_strategy_id = selected_strategy['id'] if selected_strategy else None
    
    # Multi-strategy selection for portfolio view
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Portfolio View")
    show_portfolio = st.sidebar.checkbox("Show Portfolio Summary", value=False)
    
    if show_portfolio:
        selected_strategies = st.sidebar.multiselect(
            "Select Strategies for Portfolio",
            options=strategy_options,
            default=[selected_strategy_name] if selected_strategy_name else []
        )
    else:
        selected_strategies = [selected_strategy_name]

# System status
st.sidebar.markdown("---")
st.sidebar.subheader("🖥️ System Status")
status_col1, status_col2 = st.sidebar.columns(2)
with status_col1:
    if auto_refresh:
        st.markdown('<div class="refresh-indicator refresh-active"></div> Live', unsafe_allow_html=True)
    else:
        st.markdown('⏸️ Paused')
with status_col2:
    st.metric("Strategies", len(strategies))

# --- 3. MAIN DASHBOARD LOGIC ---

def fetch_strategy_data(strategy_id: str) -> Dict:
    """Fetch all data for a strategy."""
    data = {}
    
    try:
        # 1. Cash
        data['cash'] = db.get_strategy_cash(strategy_id)
        
        # 2. Positions
        data['positions'] = db.get_strategy_positions(strategy_id, status="OPEN")
        data['closed_positions'] = db.get_strategy_positions(strategy_id, status="CLOSED")
        
        # 3. Trades
        data['trades'] = db.get_strategy_trades(strategy_id)
        
        # 4. Portfolio History
        data['history'] = db.get_portfolio_history(strategy_id, limit=1000)
        
        # 5. Performance Metrics
        metrics_res = db.client.table("performance_metrics")\
            .select("*")\
            .eq("strategy_id", strategy_id)\
            .order("calculated_at", desc=True)\
            .limit(100)\
            .execute()
        data['metrics'] = metrics_res.data
        
    except Exception as e:
        st.error(f"❌ Error fetching data: {e}")
    
    return data

def calculate_portfolio_metrics(strategies_data: Dict[str, Dict]) -> Dict:
    """Calculate aggregate portfolio metrics."""
    total_equity = 0
    total_cash = 0
    total_positions_value = 0
    total_pnl = 0
    total_trades = 0
    
    for strategy_id, data in strategies_data.items():
        total_cash += data.get('cash', 0)
        total_trades += len(data.get('trades', []))
        
        # Calculate positions value (simplified - using entry price)
        for pos in data.get('positions', []):
            total_positions_value += pos['quantity'] * pos['entry_price']
        
        # Get latest equity from history
        if data.get('history'):
            total_equity += data['history'][-1].get('total_equity', 0)
        
        # Sum PnL from metrics
        for metric in data.get('metrics', []):
            if 'total_pnl' in metric and metric['total_pnl'] is not None:
                total_pnl += metric['total_pnl']
    
    return {
        'total_equity': total_equity,
        'total_cash': total_cash,
        'total_positions_value': total_positions_value,
        'total_pnl': total_pnl,
        'total_trades': total_trades,
        'strategy_count': len(strategies_data)
    }

# --- 4. DASHBOARD HEADER ---
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.title("🚀")
with col2:
    st.title("PHOENIX TRADING DASHBOARD")
with col3:
    current_time = datetime.utcnow()
    st.markdown(f"<div style='text-align: right; color: #94a3b8; margin-top: 20px;'>{current_time.strftime('%Y-%m-%d %H:%M:%S UTC')}</div>", unsafe_allow_html=True)

# Auto-refresh logic
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

# --- 5. PORTFOLIO VIEW OR SINGLE STRATEGY VIEW ---
if show_portfolio and len(selected_strategies) > 1:
    # PORTFOLIO VIEW
    st.header("📊 PORTFOLIO SUMMARY")
    
    # Fetch data for all selected strategies
    strategies_data = {}
    for strat_name in selected_strategies:
        strat = next((s for s in strategies if s['name'] == strat_name), None)
        if strat:
            strategies_data[strat['id']] = fetch_strategy_data(strat['id'])
    
    if strategies_data:
        # Calculate portfolio metrics
        portfolio_metrics = calculate_portfolio_metrics(strategies_data)
        
        # Display portfolio KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">TOTAL EQUITY</div>
                <div class="metric-value">${portfolio_metrics['total_equity']:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            pnl_class = "positive" if portfolio_metrics['total_pnl'] >= 0 else "negative"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">TOTAL P&L</div>
                <div class="metric-value {pnl_class}">${portfolio_metrics['total_pnl']:+,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">ACTIVE STRATEGIES</div>
                <div class="metric-value">{portfolio_metrics['strategy_count']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">TOTAL TRADES</div>
                <div class="metric-value">{portfolio_metrics['total_trades']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Strategy performance comparison chart
        st.subheader("📈 Strategy Performance Comparison")
        
        # Prepare data for comparison chart
        comparison_data = []
        for strat_name in selected_strategies:
            strat = next((s for s in strategies if s['name'] == strat_name), None)
            if strat and strat['id'] in strategies_data:
                data = strategies_data[strat['id']]
                if data.get('history'):
                    latest_equity = data['history'][-1].get('total_equity', 0)
                    initial_cash = data.get('cash', 0) + sum(
                        p['quantity'] * p['entry_price'] for p in data.get('positions', [])
                    )
                    return_pct = ((latest_equity - initial_cash) / initial_cash * 100) if initial_cash > 0 else 0
                    
                    comparison_data.append({
                        'Strategy': strat_name,
                        'Equity': latest_equity,
                        'Return %': return_pct,
                        'Trades': len(data.get('trades', [])),
                        'Positions': len(data.get('positions', []))
                    })
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            
            # Create comparison charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(df_comparison, x='Strategy', y='Return %', 
                            title='Strategy Returns (%)',
                            color='Return %',
                            color_continuous_scale='RdYlGn')
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#94a3b8')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(df_comparison, x='Trades', y='Return %',
                                size='Equity', color='Strategy',
                                title='Trades vs Returns',
                                hover_data=['Strategy', 'Return %', 'Trades', 'Positions'])
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#94a3b8')
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Individual strategy tabs
        st.subheader("📋 Individual Strategy Details")
        strategy_tabs = st.tabs(selected_strategies)
        
        for i, strat_name in enumerate(selected_strategies):
            with strategy_tabs[i]:
                strat = next((s for s in strategies if s['name'] == strat_name), None)
                if strat:
                    display_strategy_data(strat['id'], strat_name, strategies_data[strat['id']])
    
elif selected_strategy_id:
    # SINGLE STRATEGY VIEW
    st.header(f"📊 {selected_strategy_name}")
    
    # Fetch strategy data
    with st.spinner(f"Loading data for {selected_strategy_name}..."):
        strategy_data = fetch_strategy_data(selected_strategy_id)
    
    if not strategy_data:
        st.warning("No data available for this strategy.")
        st.stop()
    
    # Display strategy data
    display_strategy_data(selected_strategy_id, selected_strategy_name, strategy_data)
else:
    st.info("👈 Please select a strategy from the sidebar to begin.")


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.9em;'>"
    "🚀 Phoenix Trading System • v1.0 • "
    f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    "</div>",
    unsafe_allow_html=True
)
