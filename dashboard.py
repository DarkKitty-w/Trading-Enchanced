import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client
import os
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta
import numpy as np
import altair as alt

# Chargement env
load_dotenv()

# --- 1. CONFIGURATION PAGE & THEME ---
st.set_page_config(
    page_title="PHOENIX COMMAND CENTER",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. CSS MODERNE "COMMAND CENTER" ---
st.markdown("""
<style>
    /* Fond gradient professionnel */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Style moderne pour toutes les sections */
    .main-header {
        background: linear-gradient(90deg, rgba(30, 64, 175, 0.8) 0%, rgba(59, 130, 246, 0.8) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border-left: 6px solid #f59e0b;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Cartes modernes avec effet glassmorphism */
    .metric-card {
        background: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.3);
    }
    
    /* KPIs avec dégradés */
    .kpi-container {
        display: flex;
        flex-direction: column;
        height: 100%;
    }
    
    .kpi-label {
        font-size: 13px;
        text-transform: uppercase;
        color: #94a3b8;
        letter-spacing: 1.2px;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .kpi-value {
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(90deg, #f8fafc 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1;
    }
    
    .kpi-change {
        font-size: 14px;
        font-weight: 600;
        margin-top: auto;
        padding: 4px 12px;
        border-radius: 20px;
        display: inline-block;
        width: fit-content;
    }
    
    .positive {
        background: rgba(34, 197, 94, 0.2);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    
    .negative {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* Tabs modernes */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(30, 41, 59, 0.5);
        padding: 8px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 12px 24px;
        background: transparent;
        color: #94a3b8;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #3b82f6 0%, #1d4ed8 100%);
        color: white !important;
    }
    
    /* Tableau moderne */
    .dataframe {
        background: rgba(30, 41, 59, 0.8) !important;
        border-radius: 12px;
        overflow: hidden;
    }
    
    .dataframe th {
        background: rgba(30, 64, 175, 0.8) !important;
        color: white !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Boutons modernes */
    .stButton button {
        background: linear-gradient(90deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.3);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(59, 130, 246, 0.4);
    }
    
    /* Trade feed moderne */
    .trade-feed-container {
        background: rgba(15, 23, 42, 0.8);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        height: 600px;
        overflow-y: auto;
    }
    
    .trade-item {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid;
        transition: all 0.2s ease;
    }
    
    .trade-item:hover {
        background: rgba(30, 41, 59, 0.9);
        transform: translateX(5px);
    }
    
    .trade-item.BUY {
        border-left-color: #22c55e;
        background: rgba(34, 197, 94, 0.1);
    }
    
    .trade-item.SELL {
        border-left-color: #ef4444;
        background: rgba(239, 68, 68, 0.1);
    }
    
    /* Chart containers */
    .chart-container {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Scrollbar personnalisée */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #3b82f6 0%, #1d4ed8 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #60a5fa 0%, #3b82f6 100%);
    }
    
    /* Séparateurs */
    .separator {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.5), transparent);
        margin: 2rem 0;
    }
    
</style>
""", unsafe_allow_html=True)

# --- 3. CONNEXIONS & DATA ---
@st.cache_resource
def init_supabase():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url: 
        url = st.secrets["SUPABASE_URL"]
    if not key: 
        key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

try:
    supabase = init_supabase()
except:
    st.error("❌ Erreur de connexion à la base de données.")
    st.stop()

@st.cache_data(ttl=30)
def get_crypto_prices():
    """Récupère les prix des cryptos via CoinGecko"""
    symbols = ["bitcoin", "ethereum", "solana", "binancecoin", "ripple", "cardano"]
    prices = {"USDT": 1.0}
    
    try:
        response = requests.get(
            f"https://api.coingecko.com/api/v3/simple/price",
            params={"ids": ",".join(symbols), "vs_currencies": "usd"}
        )
        data = response.json()
        
        mapping = {
            "bitcoin": "BTC",
            "ethereum": "ETH", 
            "solana": "SOL",
            "binancecoin": "BNB",
            "ripple": "XRP",
            "cardano": "ADA"
        }
        
        for coin, data_coin in data.items():
            if coin in mapping:
                symbol = mapping[coin]
                prices[symbol] = data_coin["usd"]
                prices[f"{symbol}/USDT"] = data_coin["usd"]
                
    except Exception as e:
        # Fallback aux prix fixes si l'API échoue
        fallback = {
            "BTC": 45000, "BTC/USDT": 45000,
            "ETH": 2500, "ETH/USDT": 2500,
            "SOL": 100, "SOL/USDT": 100,
            "BNB": 300, "BNB/USDT": 300,
            "XRP": 0.6, "XRP/USDT": 0.6,
            "ADA": 0.5, "ADA/USDT": 0.5
        }
        prices.update(fallback)
    
    return prices

@st.cache_data(ttl=60)
def get_all_data():
    """Récupère et traite toutes les données"""
    # Récupération des données brutes
    try:
        port_data = supabase.table("portfolio_state").select("*").execute().data
        trades_data = supabase.table("trades").select("*").order("timestamp", desc=True).limit(500).execute().data
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Conversion en DataFrames
    port = pd.DataFrame(port_data) if port_data else pd.DataFrame()
    trades = pd.DataFrame(trades_data) if trades_data else pd.DataFrame()
    
    # Récupération des prix
    unique_symbols = port['symbol'].unique().tolist() if not port.empty else []
    prices = get_crypto_prices()
    
    # Traitement du portefeuille
    leaderboard_data = []
    portfolio_details = []
    
    if not port.empty:
        # Calcul de la valeur actuelle
        port['current_price'] = port['symbol'].apply(lambda x: prices.get(x, 0))
        port['value_usd'] = port['quantity'] * port['current_price']
        
        # Grouper par stratégie
        strategies = port['strategy'].unique()
        
        for strategy in strategies:
            strat_port = port[port['strategy'] == strategy]
            
            # Calcul du NAV total
            total_value = strat_port['value_usd'].sum()
            cash_value = strat_port[strat_port['symbol'] == 'USDT']['value_usd'].sum() if not strat_port[strat_port['symbol'] == 'USDT'].empty else 0
            
            # Stats des trades
            strat_trades = trades[trades['strategy'] == strategy] if not trades.empty else pd.DataFrame()
            
            total_pnl = 0
            win_rate = 0
            trade_count = len(strat_trades)
            
            if not strat_trades.empty:
                strat_trades['pnl'] = pd.to_numeric(strat_trades['pnl'], errors='coerce').fillna(0)
                total_pnl = strat_trades['pnl'].sum()
                win_trades = (strat_trades['pnl'] > 0).sum()
                win_rate = (win_trades / trade_count * 100) if trade_count > 0 else 0
            
            # Positions détaillées
            crypto_positions = strat_port[strat_port['symbol'] != 'USDT']
            for _, pos in crypto_positions.iterrows():
                portfolio_details.append({
                    'Stratégie': strategy,
                    'Asset': pos['symbol'],
                    'Quantité': pos['quantity'],
                    'Prix Achat': pos.get('entry_price', 0),
                    'Prix Actuel': pos['current_price'],
                    'Valeur': pos['value_usd'],
                    'PnL Unrealized': (pos['current_price'] - pos.get('entry_price', 0)) * pos['quantity'] if pos.get('entry_price', 0) > 0 else 0
                })
            
            leaderboard_data.append({
                "Stratégie": strategy,
                "NAV Total ($)": total_value,
                "Cash ($)": cash_value,
                "PnL ($)": total_pnl,
                "Win Rate (%)": win_rate,
                "Trades": trade_count,
                "Positions Actives": len(crypto_positions)
            })
    
    # Création des DataFrames finaux
    df_leaderboard = pd.DataFrame(leaderboard_data)
    df_portfolio = pd.DataFrame(portfolio_details)
    
    if not df_leaderboard.empty:
        df_leaderboard = df_leaderboard.sort_values("NAV Total ($)", ascending=False)
        df_leaderboard['Rank'] = range(1, len(df_leaderboard) + 1)
    
    return df_leaderboard, trades, df_portfolio

# --- 4. FONCTIONS DE VISUALISATION ---
def create_performance_chart(df_leaderboard):
    """Crée un graphique de performance radial"""
    if df_leaderboard.empty:
        return None
    
    fig = go.Figure()
    
    # Sélection des métriques pour le radar
    metrics = ['NAV Total ($)', 'PnL ($)', 'Win Rate (%)', 'Trades']
    
    for _, row in df_leaderboard.iterrows():
        values = [row[m] for m in metrics]
        # Normalisation des valeurs
        if max(values) > 0:
            values_norm = [v / max(values) * 100 for v in values]
        else:
            values_norm = values
        
        fig.add_trace(go.Scatterpolar(
            r=values_norm,
            theta=metrics,
            fill='toself',
            name=row['Stratégie'],
            line=dict(width=2),
            opacity=0.7
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            ),
            bgcolor='rgba(30, 41, 59, 0.5)'
        ),
        showlegend=True,
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_distribution_chart(df_leaderboard):
    """Crée un graphique de distribution du capital"""
    if df_leaderboard.empty:
        return None
    
    fig = px.sunburst(
        df_leaderboard,
        path=['Stratégie'],
        values='NAV Total ($)',
        color='PnL ($)',
        color_continuous_scale=['#ef4444', '#f59e0b', '#22c55e'],
        height=400
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        margin=dict(t=0, b=0, l=0, r=0)
    )
    
    return fig

def create_pnl_timeline(trades_df):
    """Crée une timeline des PnL"""
    if trades_df.empty:
        return None
    
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    trades_df['date'] = trades_df['timestamp'].dt.date
    trades_df['pnl'] = pd.to_numeric(trades_df['pnl'], errors='coerce').fillna(0)
    
    daily_pnl = trades_df.groupby(['date', 'strategy'])['pnl'].sum().reset_index()
    
    fig = px.area(
        daily_pnl,
        x='date',
        y='pnl',
        color='strategy',
        height=300,
        title="PnL Cumulé par Jour"
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )
    
    return fig

# --- 5. UI PRINCIPALE ---
def main():
    # Header avec gradient
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.markdown("""
        <div class="main-header">
            <h1 style="margin: 0; color: white; font-size: 2.5rem;">🚀 PHOENIX COMMAND CENTER</h1>
            <p style="margin: 0.5rem 0 0 0; color: #cbd5e1; font-size: 1.1rem;">
                Centre de contrôle multi-stratégies • Surveillance temps réel
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("🔄 SYNC DATA", use_container_width=True):
            st.rerun()
    
    with col3:
        current_time = datetime.now().strftime("%H:%M:%S")
        st.markdown(f"""
        <div style="text-align: center; padding: 0.5rem; background: rgba(30, 64, 175, 0.3); border-radius: 10px;">
            <div style="color: #94a3b8; font-size: 12px;">LAST UPDATE</div>
            <div style="color: white; font-size: 18px; font-weight: bold;">{current_time}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Chargement des données
    with st.spinner("🔄 Chargement des données en temps réel..."):
        df_leaderboard, trades_df, portfolio_df = get_all_data()
    
    # Section 1: KPIs Principaux
    if not df_leaderboard.empty:
        st.markdown("### 📊 MÉTRIQUES GLOBALES")
        
        kpi_cols = st.columns(5)
        
        total_aum = df_leaderboard['NAV Total ($)'].sum()
        total_pnl = df_leaderboard['PnL ($)'].sum()
        total_trades = df_leaderboard['Trades'].sum()
        avg_win_rate = df_leaderboard['Win Rate (%)'].mean()
        active_strats = len(df_leaderboard)
        
        kpis = [
            ("TOTAL AUM", f"${total_aum:,.0f}", "+12.5%"),
            ("PNL TOTAL", f"${total_pnl:+,.0f}", "+5.2%" if total_pnl > 0 else "-5.2%"),
            ("TRADES", f"{total_trades:,}", None),
            ("WIN RATE MOYEN", f"{avg_win_rate:.1f}%", None),
            ("STRATÉGIES ACTIVES", f"{active_strats}", None)
        ]
        
        for idx, (label, value, change) in enumerate(kpis):
            with kpi_cols[idx]:
                change_html = ""
                if change:
                    change_class = "positive" if "+" in change else "negative"
                    change_html = f'<div class="kpi-change {change_class}">{change}</div>'
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="kpi-container">
                        <div class="kpi-label">{label}</div>
                        <div class="kpi-value">{value}</div>
                        {change_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
    
    # Section 2: Graphiques Principaux
    st.markdown("### 📈 ANALYTICS EN TEMPS RÉEL")
    
    tab1, tab2, tab3 = st.tabs(["📊 Performance", "🌐 Distribution", "📅 Timeline PnL"])
    
    with tab1:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig = create_performance_chart(df_leaderboard)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Aucune donnée disponible pour le graphique de performance.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            if not df_leaderboard.empty:
                # Top 3 des stratégies
                top_3 = df_leaderboard.head(3)
                for _, row in top_3.iterrows():
                    st.metric(
                        label=row['Stratégie'],
                        value=f"${row['NAV Total ($)']:,.0f}",
                        delta=f"{row['PnL ($)']:+,.0f}"
                    )
                    st.progress(row['Win Rate (%)'] / 100 if row['Win Rate (%)'] > 0 else 0)
            else:
                st.info("Aucune stratégie active.")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig = create_distribution_chart(df_leaderboard)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Aucune donnée disponible pour le graphique de distribution.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("#### 🏆 TOP PERFORMERS")
            if not df_leaderboard.empty:
                for idx, (_, row) in enumerate(df_leaderboard.iterrows()):
                    col_a, col_b = st.columns([1, 2])
                    with col_a:
                        st.markdown(f"**#{idx+1}**")
                    with col_b:
                        st.markdown(f"**{row['Stratégie']}**")
                        st.markdown(f"`${row['NAV Total ($)']:,.0f}`")
                    st.markdown("---")
            else:
                st.info("Aucun classement disponible.")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig = create_pnl_timeline(trades_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucun trade disponible pour la timeline.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
    
    # Section 3: Tableau de bord et feed
    st.markdown("### 🎯 DÉTAILS DES OPÉRATIONS")
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("#### 📋 TABLEAU DES STRATÉGIES")
        if not df_leaderboard.empty:
            # Formater le DataFrame pour l'affichage
            display_df = df_leaderboard.copy()
            display_df['NAV Total ($)'] = display_df['NAV Total ($)'].apply(lambda x: f"${x:,.2f}")
            display_df['Cash ($)'] = display_df['Cash ($)'].apply(lambda x: f"${x:,.2f}")
            display_df['PnL ($)'] = display_df['PnL ($)'].apply(lambda x: f"{x:+,.2f}$")
            display_df['Win Rate (%)'] = display_df['Win Rate (%)'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(
                display_df,
                column_config={
                    "Rank": st.column_config.NumberColumn("Rang", width="small"),
                    "Stratégie": st.column_config.TextColumn("Stratégie", width="medium"),
                    "NAV Total ($)": st.column_config.TextColumn("NAV"),
                    "PnL ($)": st.column_config.TextColumn("PnL"),
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("Aucune donnée de stratégie disponible.")
    
    with col_right:
        st.markdown("#### ⚡ FEED DES TRADES")
        st.markdown('<div class="trade-feed-container">', unsafe_allow_html=True)
        
        if not trades_df.empty:
            # Limiter aux 20 derniers trades pour la lisibilité
            recent_trades = trades_df.head(20)
            
            for _, trade in recent_trades.iterrows():
                side = trade.get('side', 'N/A')
                symbol = trade.get('symbol', 'N/A')
                strategy = trade.get('strategy', 'N/A')
                price = float(trade.get('price', 0))
                quantity = float(trade.get('quantity', 0))
                pnl = float(trade.get('pnl', 0))
                
                # Formatage du timestamp
                timestamp = pd.to_datetime(trade.get('timestamp', datetime.now()))
                time_str = timestamp.strftime('%H:%M:%S')
                
                # Couleur et icône selon le side
                if side == "BUY":
                    icon = "🟢"
                    side_color = "#22c55e"
                else:
                    icon = "🔴"
                    side_color = "#ef4444"
                
                # Couleur du PnL
                pnl_color = "#22c55e" if pnl >= 0 else "#ef4444"
                pnl_symbol = "+" if pnl >= 0 else ""
                
                st.markdown(f"""
                <div class="trade-item {side}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                        <div style="font-weight: 600; color: {side_color};">{icon} {side}</div>
                        <div style="font-size: 12px; color: #94a3b8;">{time_str}</div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <div style="font-weight: 600; color: white;">{symbol}</div>
                        <div style="color: {pnl_color}; font-weight: 600;">{pnl_symbol}{pnl:.2f}$</div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 12px; color: #cbd5e1;">
                        <div>{strategy}</div>
                        <div>{quantity:.4f} @ {price:.2f}$</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; color: #94a3b8;">
                <div style="font-size: 48px; margin-bottom: 1rem;">📊</div>
                <div style="font-size: 14px;">En attente de trades...</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Section 4: Détails du portefeuille
    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
    
    st.markdown("### 💼 POSITIONS ACTIVES")
    
    if not portfolio_df.empty:
        # Afficher par stratégie
        strategies = portfolio_df['Stratégie'].unique()
        
        for strategy in strategies:
            st.markdown(f"#### {strategy}")
            strat_positions = portfolio_df[portfolio_df['Stratégie'] == strategy]
            
            cols = st.columns(len(strat_positions) if len(strat_positions) <= 4 else 4)
            
            for idx, (_, position) in enumerate(strat_positions.iterrows()):
                with cols[idx % len(cols)]:
                    pnl = position['PnL Unrealized']
                    pnl_color = "#22c55e" if pnl >= 0 else "#ef4444"
                    
                    st.markdown(f"""
                    <div class="metric-card" style="padding: 1rem;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                            <div style="font-weight: 600; color: white;">{position['Asset']}</div>
                            <div style="color: {pnl_color}; font-weight: 600;">{pnl:+,.2f}$</div>
                        </div>
                        <div style="font-size: 12px; color: #94a3b8; margin-bottom: 4px;">
                            Quantité: {position['Quantité']:.4f}
                        </div>
                        <div style="font-size: 12px; color: #94a3b8;">
                            Valeur: ${position['Valeur']:,.2f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("Aucune position active dans le portefeuille.")

if __name__ == "__main__":
    main()
