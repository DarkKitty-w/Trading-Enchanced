import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from database import Database
import numpy as np
from typing import Dict, List, Optional
import calendar

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="PHOENIX TRADING DASHBOARD",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Global Theme */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(15, 23, 42) 0%, rgb(8, 15, 30) 90%);
        color: #e2e8f0;
    }
    
    /* Metrics Cards */
    .metric-container {
        display: flex;
        flex-direction: row;
        gap: 1rem;
        flex-wrap: wrap;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(148, 163, 184, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 24px;
        flex: 1;
        min-width: 200px;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(56, 189, 248, 0.3);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    .metric-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 4px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        opacity: 0.5;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(180deg, #fff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 8px;
    }
    .metric-label {
        font-size: 13px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    .metric-delta {
        font-size: 14px;
        font-weight: 600;
        margin-top: 4px;
    }
    .positive { color: #34d399 !important; }
    .negative { color: #f87171 !important; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
        padding-bottom: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 41, 59, 0.3);
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 500;
        color: #94a3b8;
        border: 1px solid transparent;
        transition: all 0.2s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(59, 130, 246, 0.1);
        color: #fff;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(59, 130, 246, 0.15) !important;
        border-color: rgba(59, 130, 246, 0.3) !important;
        color: #60a5fa !important;
    }
    
    /* Auto-refresh indicator */
    .refresh-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
        box-shadow: 0 0 10px currentColor;
    }
    .refresh-active {
        background-color: #10b981;
        color: #10b981;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.2); }
        100% { opacity: 1; transform: scale(1); }
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
st.sidebar.markdown("### 📡 PHOENIX CONTROL")

# Auto-refresh toggle
auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh", value=True, help="Auto-refresh data")
refresh_interval = st.sidebar.slider("Refresh interval (s)", 10, 300, 30)

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
st.sidebar.subheader("🎯 Strategy")

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
        "Select Strategy",
        options=strategy_options,
        index=0 if strategy_options else None
    )
    
    # Get selected strategy ID
    selected_strategy = next((s for s in strategies if s['name'] == selected_strategy_name), None)
    selected_strategy_id = selected_strategy['id'] if selected_strategy else None
    
    # Multi-strategy selection for portfolio view
    st.sidebar.markdown("---")
    show_portfolio = st.sidebar.checkbox("📊 Portfolio View", value=False)
    
    if show_portfolio:
        selected_strategies = st.sidebar.multiselect(
            "Filter Portfolio",
            options=strategy_options,
            default=[selected_strategy_name] if selected_strategy_name else []
        )
    else:
        selected_strategies = [selected_strategy_name]

# System status
st.sidebar.markdown("---")
st.sidebar.caption("SYSTEM STATUS")
status_col1, status_col2 = st.sidebar.columns(2)
with status_col1:
    if auto_refresh:
        st.markdown('<div class="refresh-indicator refresh-active"></div> Live', unsafe_allow_html=True)
    else:
        st.markdown('⏸️ Paused')
with status_col2:
    st.markdown(f"**{len(strategies)}** Strategies")

# --- 3. DATA FETCHING (CACHED) ---

@st.cache_data(ttl=15, show_spinner=False)
def fetch_strategy_data(strategy_id: str) -> Dict:
    """Fetch all data for a strategy with caching to prevent DB spam."""
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
        # Return empty structure on failure to prevent crash
        return {'cash': 0, 'positions': [], 'trades': [], 'history': [], 'metrics': []}
    
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
        else:
            total_equity += (data.get('cash', 0) + total_positions_value)
        
        # Sum PnL from metrics
        for metric in data.get('metrics', []):
            if 'total_pnl' in metric:
                total_pnl += metric['total_pnl']
    
    return {
        'total_equity': total_equity,
        'total_cash': total_cash,
        'total_positions_value': total_positions_value,
        'total_pnl': total_pnl,
        'total_trades': total_trades,
        'strategy_count': len(strategies_data)
    }

# --- 4. HELPER PLOTTING FUNCTIONS ---

def plot_underwater(df_history):
    """Generates a Drawdown (Underwater) plot."""
    if df_history.empty:
        return go.Figure()

    df = df_history.copy()
    df['peak'] = df['total_equity'].cummax()
    df['drawdown'] = (df['total_equity'] - df['peak']) / df['peak'] * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['drawdown'],
        fill='tozeroy',
        mode='lines',
        line=dict(color='#ef4444', width=1),
        name='Drawdown'
    ))
    fig.update_layout(
        title='Drawdown (%)',
        yaxis_title='Drawdown %',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8'),
        height=250,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    return fig

def plot_pnl_distribution(trades):
    """Generates a histogram of Trade PnL."""
    if not trades:
        return go.Figure()
        
    df = pd.DataFrame(trades)
    if 'profit' not in df.columns:
        return go.Figure()
        
    fig = px.histogram(
        df, 
        x="profit", 
        nbins=20, 
        title="PnL Distribution",
        color_discrete_sequence=['#3b82f6']
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8'),
        xaxis_title="Profit/Loss ($)",
        yaxis_title="Count",
        height=250,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    return fig

# --- 5. MAIN DASHBOARD UI ---

col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.markdown("## 🚀 PHOENIX")
with col3:
    current_time = datetime.utcnow()
    st.markdown(f"<div style='text-align: right; color: #64748b; font-family: monospace;'>{current_time.strftime('%Y-%m-%d %H:%M:%S UTC')}</div>", unsafe_allow_html=True)

# Auto-refresh logic
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

# --- DISPLAY LOGIC ---

def display_strategy_data(strategy_id: str, strategy_name: str, data: Dict):
    """Display data for a single strategy."""
    
    # --- KPI ROW ---
    cash = data.get('cash', 0)
    
    # Calculate Total Equity
    positions = data.get('positions', [])
    positions_value = sum([p['quantity'] * p['entry_price'] for p in positions])
    total_equity = cash + positions_value
    
    # Realized PnL
    realized_pnl = 0
    # Prefer summing actual trade profits if available, else use metrics snapshots
    trades = data.get('trades', [])
    if trades and 'profit' in trades[0]:
         realized_pnl = sum([t.get('profit', 0) for t in trades])
    elif data.get('metrics'):
        # Fallback to last metric snapshot
        last_metric = data['metrics'][0]
        realized_pnl = last_metric.get('total_pnl', 0)
    
    pnl_color = "positive" if realized_pnl >= 0 else "negative"
    
    # Win Rate
    closed_trades = [t for t in trades if t.get('side') == 'SELL'] # Assuming sell closes
    winning_trades = [t for t in closed_trades if t.get('profit', 0) > 0]
    win_rate = (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0
    
    # HTML Metric Cards
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-card">
            <div class="metric-label">Total Equity</div>
            <div class="metric-value">${total_equity:,.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Realized P&L</div>
            <div class="metric-value {pnl_color}">${realized_pnl:+,.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Available Cash</div>
            <div class="metric-value">${cash:,.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Win Rate</div>
            <div class="metric-value">{win_rate:.1f}%</div>
            <div class="metric-delta">{len(closed_trades)} Trades</div>
        </div>
    </div>
    <br>
    """, unsafe_allow_html=True)
    
    # --- TABS VIEW ---
    tab_pos, tab_trades, tab_history, tab_analysis = st.tabs([
        "🛡️ POSITIONS", 
        "📜 TRADES", 
        "📈 PERFORMANCE",
        "🔍 ANALYTICS"
    ])
    
    # TAB 1: POSITIONS
    with tab_pos:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Active Holdings")
            if positions:
                df_positions = pd.DataFrame(positions)
                
                # Calculate unrealized PnL (Mocked using entry price)
                # NOTE: In a real scenario, fetch live price here.
                df_positions['current_price'] = df_positions['entry_price'] 
                df_positions['market_value'] = df_positions['quantity'] * df_positions['entry_price']
                
                # Format for display
                display_df = df_positions[['symbol', 'side', 'quantity', 'entry_price', 'market_value', 'opened_at']].copy()
                display_df['opened_at'] = pd.to_datetime(display_df['opened_at']).dt.strftime('%m-%d %H:%M')
                
                st.dataframe(
                    display_df.style.format({
                        'quantity': '{:.4f}',
                        'entry_price': '${:.2f}',
                        'market_value': '${:,.2f}'
                    }),
                    use_container_width=True,
                    height=300
                )
            else:
                st.info("No active positions currently open.")
        
        with col2:
            st.subheader("Exposure Breakdown")
            if positions:
                # Allocation Pie Chart
                fig = px.pie(
                    df_positions, 
                    values='market_value', 
                    names='symbol',
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.Teal
                )
                fig.update_layout(
                    showlegend=True,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#94a3b8'),
                    margin=dict(t=0, b=0, l=0, r=0),
                    height=250
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary Stats
                total_exposure = sum(p['quantity'] * p['entry_price'] for p in positions)
                leverage = total_exposure / total_equity if total_equity > 0 else 0
                st.metric("Total Market Value", f"${total_exposure:,.2f}")
                st.metric("Gross Leverage", f"{leverage:.2f}x")
    
    # TAB 2: TRADES
    with tab_trades:
        if trades:
            df_trades = pd.DataFrame(trades)
            df_trades['executed_at'] = pd.to_datetime(df_trades['executed_at'])
            
            # Cumulative PnL Chart (if profit column exists)
            if 'profit' in df_trades.columns:
                df_trades = df_trades.sort_values('executed_at')
                df_trades['cum_pnl'] = df_trades['profit'].cumsum()
                
                fig_cum = px.area(
                    df_trades, x='executed_at', y='cum_pnl',
                    title='Cumulative Realized PnL',
                    line_shape='hv'
                )
                fig_cum.update_traces(line_color='#10b981', fillcolor='rgba(16, 185, 129, 0.1)')
                fig_cum.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#94a3b8'),
                    yaxis_title="Profit ($)",
                    height=300
                )
                st.plotly_chart(fig_cum, use_container_width=True)

            # Table
            st.markdown("##### Trade History")
            display_cols = ['executed_at', 'symbol', 'side', 'quantity', 'price', 'fees']
            if 'profit' in df_trades.columns:
                display_cols.append('profit')
                
            st.dataframe(
                df_trades.sort_values('executed_at', ascending=False)[display_cols].style.format({
                    'price': '${:.2f}',
                    'profit': '${:+.2f}' if 'profit' in df_trades.columns else '{}',
                    'fees': '${:.2f}',
                    'executed_at': lambda x: x.strftime('%Y-%m-%d %H:%M')
                }).applymap(
                    lambda v: 'color: #34d399' if v > 0 else 'color: #f87171', 
                    subset=['profit'] if 'profit' in df_trades.columns else []
                ),
                use_container_width=True,
                height=400
            )
        else:
            st.info("No trade history available yet.")
    
    # TAB 3: PERFORMANCE
    with tab_history:
        history = data.get('history', [])
        
        if history:
            df_history = pd.DataFrame(history)
            df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
            
            # Main Equity Curve
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_history['timestamp'],
                y=df_history['total_equity'],
                mode='lines',
                name='Equity',
                line=dict(color='#3b82f6', width=2),
                fill='tozeroy',
                fillcolor='rgba(59, 130, 246, 0.1)'
            ))
            
            fig.update_layout(
                title='Portfolio Value Growth',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8'),
                xaxis_title="Time",
                yaxis_title="Equity ($)",
                hovermode='x unified',
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
            
            col_dd, col_heat = st.columns([1, 1])
            
            with col_dd:
                # Underwater Plot
                st.plotly_chart(plot_underwater(df_history), use_container_width=True)
                
            with col_heat:
                # Monthly Returns (Simulated from daily data)
                if len(df_history) > 1:
                    df_daily = df_history.set_index('timestamp').resample('D').last().ffill()
                    df_daily['pct_change'] = df_daily['total_equity'].pct_change() * 100
                    
                    # Create a heatmap matrix (Month vs Year)
                    df_daily['year'] = df_daily.index.year
                    df_daily['month'] = df_daily.index.month
                    pivot = df_daily.pivot_table(index='year', columns='month', values='pct_change', aggfunc='sum')
                    
                    fig_heat = px.imshow(
                        pivot, 
                        labels=dict(x="Month", y="Year", color="Return %"),
                        x=[calendar.month_abbr[i] for i in range(1, 13)],
                        color_continuous_scale='RdYlGn',
                        text_auto='.1f'
                    )
                    fig_heat.update_layout(
                        title="Monthly Returns (%)",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#94a3b8'),
                        height=250
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("Not enough history to generate performance charts.")
    
    # TAB 4: ANALYTICS
    with tab_analysis:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Win/Loss Analysis")
            if trades and 'profit' in df_trades.columns:
                st.plotly_chart(plot_pnl_distribution(trades), use_container_width=True)
                
                wins = df_trades[df_trades['profit'] > 0]['profit']
                losses = df_trades[df_trades['profit'] < 0]['profit']
                
                avg_win = wins.mean() if not wins.empty else 0
                avg_loss = losses.mean() if not losses.empty else 0
                
                c1, c2 = st.columns(2)
                c1.metric("Avg Win", f"${avg_win:.2f}")
                c2.metric("Avg Loss", f"${avg_loss:.2f}")
                
        with col2:
            st.markdown("##### Risk Gauges")
            # Calculate Risk Metrics
            if history:
                df_h = pd.DataFrame(history)
                peak = df_h['total_equity'].max()
                current = df_h['total_equity'].iloc[-1]
                dd = (peak - current) / peak * 100 if peak > 0 else 0
                
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = dd,
                    title = {'text': "Current Drawdown %"},
                    gauge = {
                        'axis': {'range': [0, 20]},
                        'bar': {'color': "#ef4444"},
                        'steps': [
                            {'range': [0, 5], 'color': "rgba(16, 185, 129, 0.1)"},
                            {'range': [5, 10], 'color': "rgba(245, 158, 11, 0.1)"}
                        ]
                    }
                ))
                fig_gauge.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#94a3b8'),
                    height=250,
                    margin=dict(l=20, r=20, t=30, b=20)
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

# --- PORTFOLIO OR SINGLE VIEW LOGIC ---

if show_portfolio and len(selected_strategies) > 0:
    # PORTFOLIO VIEW
    st.header("📊 Portfolio Aggregate")
    
    # Fetch data
    strategies_data = {}
    with st.spinner("Aggregating portfolio data..."):
        for strat_name in selected_strategies:
            strat = next((s for s in strategies if s['name'] == strat_name), None)
            if strat:
                strategies_data[strat['id']] = fetch_strategy_data(strat['id'])
    
    if strategies_data:
        metrics = calculate_portfolio_metrics(strategies_data)
        
        # Portfolio KPIs
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Portfolio Equity", f"${metrics['total_equity']:,.2f}")
        col2.metric("Total Cash", f"${metrics['total_cash']:,.2f}")
        col3.metric("Total PnL", f"${metrics['total_pnl']:+,.2f}")
        col4.metric("Active Strategies", metrics['strategy_count'])
        
        st.markdown("---")
        
        # Comparison Charts
        st.subheader("Strategy Comparison")
        comp_data = []
        for s_id, data in strategies_data.items():
            name = next(s['name'] for s in strategies if s['id'] == s_id)
            equity = data['history'][-1]['total_equity'] if data['history'] else 0
            trades_count = len(data['trades'])
            comp_data.append({'Name': name, 'Equity': equity, 'Trades': trades_count})
            
        if comp_data:
            df_comp = pd.DataFrame(comp_data)
            c1, c2 = st.columns(2)
            
            with c1:
                fig = px.bar(df_comp, x='Name', y='Equity', title="Equity by Strategy", color='Equity', color_continuous_scale='Bluered')
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8'))
                st.plotly_chart(fig, use_container_width=True)
                
            with c2:
                fig = px.pie(df_comp, values='Trades', names='Name', title="Trade Activity Volume")
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8'))
                st.plotly_chart(fig, use_container_width=True)

elif selected_strategy_id:
    # SINGLE STRATEGY VIEW
    st.markdown(f"### 🔭 {selected_strategy_name}")
    
    with st.spinner(f"Syncing data for {selected_strategy_name}..."):
        strategy_data = fetch_strategy_data(selected_strategy_id)
    
    if strategy_data:
        display_strategy_data(selected_strategy_id, selected_strategy_name, strategy_data)
    else:
        st.error("Failed to load strategy data.")

else:
    st.info("👈 Please select a strategy to begin.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #475569; font-size: 0.8em;'>"
    "PHOENIX TRADING SYSTEM v2.0 • "
    f"Current Session: {uuid.uuid4().hex[:8]}"
    "</div>",
    unsafe_allow_html=True
)

import uuid # Added for the footer session ID
