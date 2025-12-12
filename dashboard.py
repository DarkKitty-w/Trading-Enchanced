import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client
import os
import requests
from dotenv import load_dotenv
from datetime import datetime

# Chargement env local
load_dotenv()

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="PHOENIX | Hedge Fund Monitor",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# IDs CoinLore pour les prix
COIN_MAPPING = {
    "BTC": "90", "BTC/USDT": "90",
    "ETH": "80", "ETH/USDT": "80",
    "SOL": "48543", "SOL/USDT": "48543",
    "BNB": "2710", "BNB/USDT": "2710",
    "XRP": "58", "XRP/USDT": "58",
    "ADA": "257", "ADA/USDT": "257",
    "DOGE": "2", "DOGE/USDT": "2",
    "USDT": "518", "USDT/USDT": "518"
}

# --- 2. CSS DARK MODE ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    .metric-container {
        background-color: #262730; border: 1px solid #333;
        padding: 15px; border-radius: 10px; text-align: center;
    }
    .metric-value { font-size: 24px; font-weight: bold; color: #FFF; }
    .metric-label { font-size: 12px; color: #AAA; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

# --- 3. CONNEXIONS ---
@st.cache_resource
def init_supabase():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    # Fallback Streamlit Secrets
    if not url: url = st.secrets["SUPABASE_URL"]
    if not key: key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

try:
    supabase = init_supabase()
except:
    st.error("❌ Erreur connexion Supabase. Vérifiez vos clés.")
    st.stop()

@st.cache_data(ttl=60)
def get_market_prices(symbols_in_portfolio):
    """Récupère les prix actuels via CoinLore pour valoriser le portfolio"""
    prices = {"USDT": 1.0}
    
    # On identifie les IDs nécessaires
    ids_to_fetch = []
    for sym in symbols_in_portfolio:
        clean = sym.replace("/USDT", "")
        if clean in COIN_MAPPING and clean != "USDT":
            ids_to_fetch.append(COIN_MAPPING[clean])
    
    if not ids_to_fetch:
        return prices
        
    # Appel API Groupé (Limit 50 ids par call)
    ids_str = ",".join(ids_to_fetch)
    url = f"https://api.coinlore.net/api/ticker/?id={ids_str}"
    
    try:
        resp = requests.get(url, timeout=5)
        data = resp.json()
        for item in data:
            symbol = item['symbol'] # Ex: BTC
            price = float(item['price_usd'])
            # Mapping inverse ou direct
            prices[symbol] = price
            prices[f"{symbol}/USDT"] = price
    except Exception as e:
        print(f"Erreur Prix: {e}")
        
    return prices

def get_data():
    """Récupère Portfolio + Trades"""
    # On récupère la table portfolio PLATE (SQL)
    port_data = supabase.table("portfolio_state").select("*").execute().data
    trade_data = supabase.table("trades").select("*").order("timestamp", desc=True).limit(100).execute().data
    return pd.DataFrame(port_data), pd.DataFrame(trade_data)

# --- 4. LOGIQUE PRINCIPALE ---
with st.spinner("Synchronisation des marchés..."):
    df_port, df_trade = get_data()

if df_port.empty:
    st.warning("📭 Portefeuille vide ou base de données inaccessible.")
    st.stop()

# Nettoyage des types
df_port['quantity'] = df_port['quantity'].astype(float)
symbols_present = df_port['symbol'].unique().tolist()

# Récupération Prix Réels
real_prices = get_market_prices(symbols_present)

# Calcul de la VALEUR USD pour chaque ligne
def calculate_value(row):
    sym = row['symbol']
    qty = row['quantity']
    # Prix trouvé ou 0.0 si inconnu
    price = real_prices.get(sym, real_prices.get(sym.replace("/USDT", ""), 0.0))
    return qty * price

df_port['value_usd'] = df_port.apply(calculate_value, axis=1)

# --- AGRÉGATIONS ---
# 1. Total Global
total_liquidity = df_port['value_usd'].sum()

# 2. Cash vs Crypto
cash_val = df_port[df_port['symbol'] == 'USDT']['value_usd'].sum()
crypto_val = total_liquidity - cash_val
crypto_pct = (crypto_val / total_liquidity * 100) if total_liquidity > 0 else 0

# 3. Par Stratégie (Hedge Fund View)
# On s'assure que la colonne 'strategy' existe (cas de la migration)
if 'strategy' in df_port.columns:
    df_strat = df_port.groupby('strategy')['value_usd'].sum().reset_index()
else:
    # Fallback si vieux schema
    df_port['strategy'] = 'Legacy'
    df_strat = df_port.groupby('strategy')['value_usd'].sum().reset_index()

# 4. PnL Session
session_pnl = 0
if not df_trade.empty:
    df_trade['pnl'] = df_trade['pnl'].astype(float)
    session_pnl = df_trade['pnl'].sum()

# --- 5. INTERFACE ---

# HEADER
c1, c2 = st.columns([6, 1])
with c1:
    st.title("🦅 PHOENIX | Hedge Fund Cockpit")
    st.caption(f"Valorisation en Temps Réel • {len(df_port)} Positions Actives")
with c2:
    if st.button("RERUN ↻"): st.rerun()

st.divider()

# KPI CARDS
k1, k2, k3, k4 = st.columns(4)

def kpi(col, label, val, color=None):
    col.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color: {color if color else '#FFF'}">{val}</div>
    </div>
    """, unsafe_allow_html=True)

kpi(k1, "VALEUR TOTALE (NAV)", f"{total_liquidity:,.2f} $", "#00CC96")
kpi(k2, "CASH DISPONIBLE", f"{cash_val:,.2f} $")
kpi(k3, "EXPOSITION CRYPTO", f"{crypto_pct:.1f} %", "#F5B700")
kpi(k4, "PNL RÉALISÉ", f"{session_pnl:+.2f} $", "#EF553B" if session_pnl < 0 else "#00CC96")

st.write("")

# GRAPHIQUES
g1, g2 = st.columns([1, 1])

with g1:
    st.subheader("🍰 Allocation par Stratégie")
    fig_strat = px.pie(df_strat, values='value_usd', names='strategy', hole=0.5, template="plotly_dark")
    fig_strat.update_traces(textinfo='percent+label')
    st.plotly_chart(fig_strat, use_container_width=True)

with g2:
    st.subheader("📊 Composition du Portefeuille")
    # On groupe par Symbole (hors USDT) pour voir l'expo Crypto globale
    df_crypto_only = df_port[df_port['symbol'] != 'USDT']
    if not df_crypto_only.empty:
        df_sym = df_crypto_only.groupby('symbol')['value_usd'].sum().reset_index()
        fig_sym = px.bar(df_sym, x='symbol', y='value_usd', color='symbol', template="plotly_dark")
        st.plotly_chart(fig_sym, use_container_width=True)
    else:
        st.info("Portefeuille 100% Cash.")

# TABLEAU DÉTAILLÉ
st.subheader("💼 Détail des Positions (Hedge Fund Mode)")
st.dataframe(
    df_port[['strategy', 'symbol', 'quantity', 'value_usd']].sort_values('strategy'),
    column_config={
        "strategy": "Stratégie",
        "symbol": "Actif",
        "quantity": st.column_config.NumberColumn("Quantité", format="%.5f"),
        "value_usd": st.column_config.NumberColumn("Valeur ($)", format="$ %.2f"),
    },
    use_container_width=True,
    hide_index=True
)

# DERNIERS TRADES
if not df_trade.empty:
    st.subheader("⚡ Derniers Trades")
    st.dataframe(
        df_trade[['timestamp', 'strategy', 'symbol', 'side', 'price', 'pnl']].head(50),
        use_container_width=True,
        hide_index=True
    )
