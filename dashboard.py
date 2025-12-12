import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client
import os
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Chargement env local
load_dotenv()

# --- 1. CONFIGURATION DE LA PAGE (WIDE MODE) ---
st.set_page_config(
    page_title="PHOENIX | Hedge Fund Monitor",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. STYLE CSS MODERN (DARK GLASSMORPHISM) ---
st.markdown("""
<style>
    /* Fond global */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Style des cartes KPIs */
    .metric-container {
        background-color: #262730;
        border: 1px solid #333;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        text-align: center;
    }
    .metric-label {
        font-size: 14px;
        color: #A0A0A0;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #FFFFFF;
    }
    .metric-delta {
        font-size: 14px;
        font-weight: 500;
    }
    .positive { color: #00CC96; }
    .negative { color: #EF553B; }
    
    /* Style des tables */
    .stDataFrame {
        border: 1px solid #333;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. CONNEXIONS (SUPABASE & API PRIX) ---
@st.cache_resource
def init_supabase():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url: url = st.secrets["SUPABASE_URL"]
    if not key: key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

try:
    supabase = init_supabase()
except:
    st.error("❌ Erreur connexion Supabase")
    st.stop()

@st.cache_data(ttl=60) # Cache de 60 secondes pour ne pas spammer
def get_current_prices(symbols):
    """Récupère les prix actuels pour valoriser le portfolio"""
    # Mapping Symbol -> CoinLore ID (Simplifié pour l'exemple)
    mapping = {"BTC": "90", "ETH": "80", "SOL": "48543", "BNB": "2710", "XRP": "58"}
    prices = {}
    
    # On met USDT à 1$ par défaut
    prices["USDT"] = 1.0
    
    # Pour les autres, on fait un appel rapide (optionnel, sinon on met 0)
    # Ici on simule ou on fait un appel groupé si possible.
    # Pour la rapidité du dashboard, on va se baser sur le fait que l'utilisateur veut voir le CASH surtout.
    return prices

def get_data():
    """Récupère les données brutes"""
    portfolio = pd.DataFrame(supabase.table("portfolio_state").select("*").execute().data)
    trades = pd.DataFrame(supabase.table("trades").select("*").order("timestamp", desc=True).limit(500).execute().data)
    return portfolio, trades

# --- 4. LOGIQUE MÉTIER ---
with st.spinner("Chargement du Cockpit PHOENIX..."):
    df_port, df_trade = get_data()

# Traitement des données
if not df_port.empty:
    df_port['quantity'] = df_port['quantity'].astype(float)
    
    # Séparation Cash / Crypto
    df_cash = df_port[df_port['symbol'] == 'USDT'].copy()
    df_crypto = df_port[df_port['symbol'] != 'USDT'].copy()
    
    total_cash = df_cash['quantity'].sum()
    active_positions = len(df_crypto)
    
else:
    total_cash = 0
    active_positions = 0
    df_cash = pd.DataFrame()

# Calcul PnL depuis les trades
total_pnl = 0
win_rate = 0
if not df_trade.empty:
    df_trade['pnl'] = df_trade['pnl'].astype(float)
    df_trade['timestamp'] = pd.to_datetime(df_trade['timestamp'])
    total_pnl = df_trade['pnl'].sum()
    
    wins = len(df_trade[df_trade['pnl'] > 0])
    if len(df_trade) > 0:
        win_rate = (wins / len(df_trade)) * 100

# --- 5. INTERFACE GRAPHIQUE (LAYOUT) ---

# HEADER
c1, c2 = st.columns([5, 1])
with c1:
    st.title("🦅 PHOENIX TERMINAL")
    st.markdown(f"**Status:** `ONLINE` • **Update:** `{datetime.now().strftime('%H:%M:%S')}`")
with c2:
    if st.button("RERUN ↻", use_container_width=True):
        st.rerun()

st.markdown("---")

# ROW 1 : BIG KPI CARDS (HTML Custom)
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

def display_card(col, label, value, delta=None, is_currency=True):
    with col:
        color_class = "positive" if delta and delta >= 0 else "negative"
        delta_html = f'<span class="metric-delta {color_class}">{delta:+.2f}$</span>' if delta is not None else ""
        val_str = f"{value:,.2f} $" if is_currency else f"{value}"
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val_str}</div>
            {delta_html}
        </div>
        """, unsafe_allow_html=True)

display_card(kpi1, "Liquidité Totale (USDT)", total_cash)
display_card(kpi2, "PnL Réalisé (Session)", total_pnl, delta=total_pnl)
display_card(kpi3, "Positions Actives", active_positions, is_currency=False)
display_card(kpi4, "Win Rate", f"{win_rate:.1f} %", is_currency=False)

st.write("") # Spacer

# ROW 2 : GRAPHIQUES MODERNES
g1, g2 = st.columns([2, 1])

with g1:
    st.subheader("📈 Performance Cumulée")
    if not df_trade.empty:
        # Création de la courbe de PnL cumulé
        df_chart = df_trade.sort_values("timestamp").copy()
        df_chart['cumulative_pnl'] = df_chart['pnl'].cumsum()
        
        fig = px.area(
            df_chart, x="timestamp", y="cumulative_pnl",
            template="plotly_dark",
            height=350
        )
        # Customisation "Neon Style"
        fig.update_traces(line_color='#00CC96', fillcolor='rgba(0, 204, 150, 0.1)')
        fig.update_layout(
            xaxis_title=None, yaxis_title="Gain (USDT)",
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("En attente de données de trading...")

with g2:
    st.subheader("🍰 Allocation Cash")
    if not df_cash.empty:
        # Donut Chart pour voir qui a le cash
        fig_pie = px.pie(
            df_cash, values='quantity', names='strategy',
            template="plotly_dark",
            hole=0.6, height=350,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_pie.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            margin=dict(l=0, r=0, t=20, b=50),
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Portefeuille vide.")

# ROW 3 : DÉTAILS STRATÉGIES & TRADES
st.markdown("---")
t1, t2 = st.columns([1, 1])

with t1:
    st.subheader("💼 Portefeuille (Hedge Fund Mode)")
    if not df_port.empty:
        # On affiche un tableau propre trié
        df_display = df_port[['strategy', 'symbol', 'quantity']].copy()
        # Filtrer les poussières
        df_display = df_display[df_display['quantity'] > 0.0001]
        
        st.dataframe(
            df_display,
            use_container_width=True,
            column_config={
                "strategy": "Stratégie",
                "symbol": "Crypto",
                "quantity": st.column_config.NumberColumn("Quantité", format="%.5f")
            },
            hide_index=True,
            height=300
        )
    else:
        st.write("Aucune position.")

with t2:
    st.subheader("⚡ Dernières Activités")
    if not df_trade.empty:
        # Tableau des trades stylisé
        df_logs = df_trade[['timestamp', 'strategy', 'symbol', 'side', 'price', 'pnl']].head(50)
        
        st.dataframe(
            df_logs,
            use_container_width=True,
            column_config={
                "timestamp": st.column_config.DatetimeColumn("Heure", format="HH:mm"),
                "side": "Action",
                "pnl": st.column_config.NumberColumn("PnL", format="$ %.2f"),
                "price": st.column_config.NumberColumn("Prix", format="$ %.2f")
            },
            hide_index=True,
            height=300
        )
    else:
        st.write("Aucun trade récent.")