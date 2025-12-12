import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client
import os
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Chargement env
load_dotenv()

# --- 1. CONFIGURATION PAGE & THEME ---
st.set_page_config(
    page_title="PHOENIX ARENA",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. CSS "ARENA STYLE" ---
st.markdown("""
<style>
    /* Fond sombre global */
    .stApp { background-color: #0e1117; }
    
    /* Style du Feed de droite */
    .trade-feed {
        height: 600px;
        overflow-y: auto;
        padding-right: 10px;
    }
    .trade-card {
        background-color: #1c1f26;
        border-left: 4px solid #3b82f6;
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 4px;
        font-size: 13px;
    }
    .trade-card.SELL { border-left-color: #ef4444; } /* Rouge pour Vente */
    .trade-card.BUY { border-left-color: #22c55e; }  /* Vert pour Achat */
    
    .trade-header { font-weight: bold; color: #e5e7eb; display: flex; justify-content: space-between; }
    .trade-meta { color: #9ca3af; font-size: 11px; margin-bottom: 5px; }
    .trade-pnl { font-weight: bold; float: right; }
    .win { color: #22c55e; }
    .loss { color: #ef4444; }

    /* Style KPIs */
    .kpi-box {
        background-color: #1c1f26;
        border: 1px solid #2d3748;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    .kpi-label { font-size: 12px; text-transform: uppercase; color: #6b7280; letter-spacing: 1px; }
    .kpi-val { font-size: 24px; font-weight: 800; color: #f3f4f6; }
    
</style>
""", unsafe_allow_html=True)

# --- 3. CONNEXIONS & DATA ---
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
    st.error("❌ Erreur Supabase.")
    st.stop()

@st.cache_data(ttl=30) # Refresh prix toutes les 30s
def get_prices(symbols):
    """Récupère les prix CoinLore"""
    # Mapping manuel ID (à compléter si besoin)
    ids = {"BTC": "90", "ETH": "80", "SOL": "48543", "BNB": "2710", "XRP": "58", "ADA": "257"}
    prices = {"USDT": 1.0}
    
    # Extraction des IDs à fetcher
    id_list = []
    for s in symbols:
        clean = s.replace("/USDT", "")
        if clean in ids: id_list.append(ids[clean])
    
    if id_list:
        try:
            resp = requests.get(f"https://api.coinlore.net/api/ticker/?id={','.join(id_list)}")
            for item in resp.json():
                prices[item['symbol']] = float(item['price_usd'])
                prices[f"{item['symbol']}/USDT"] = float(item['price_usd'])
        except: pass
    return prices

def get_data():
    """Récupère et croise toutes les données"""
    # 1. Raw Data
    port = pd.DataFrame(supabase.table("portfolio_state").select("*").execute().data)
    trades = pd.DataFrame(supabase.table("trades").select("*").order("timestamp", desc=True).limit(200).execute().data)
    
    # 2. Prices
    unique_syms = port['symbol'].unique() if not port.empty else []
    prices = get_prices(unique_syms)
    
    # 3. Process Portfolio (NAV par Stratégie)
    leaderboard = []
    
    if not port.empty:
        # On calcule la valeur USD de chaque ligne
        port['current_price'] = port['symbol'].apply(lambda x: prices.get(x, 0))
        port['value_usd'] = port['quantity'] * port['current_price']
        
        # Groupement par Stratégie
        strats = port['strategy'].unique()
        for s in strats:
            df_s = port[port['strategy'] == s]
            
            total_val = df_s['value_usd'].sum()
            cash = df_s[df_s['symbol'] == 'USDT']['value_usd'].sum() if not df_s[df_s['symbol'] == 'USDT'].empty else 0
            
            # Stats Trading
            s_trades = trades[trades['strategy'] == s] if not trades.empty else pd.DataFrame()
            win_rate = 0
            pnl = 0
            count = 0
            
            if not s_trades.empty:
                s_trades['pnl'] = s_trades['pnl'].astype(float)
                pnl = s_trades['pnl'].sum()
                count = len(s_trades)
                wins = len(s_trades[s_trades['pnl'] > 0])
                win_rate = (wins / count * 100) if count > 0 else 0
                
            leaderboard.append({
                "Stratégie": s,
                "NAV Total ($)": total_val,
                "Cash ($)": cash,
                "PnL ($)": pnl,
                "Win Rate (%)": win_rate,
                "Trades": count
            })
            
    df_lb = pd.DataFrame(leaderboard)
    if not df_lb.empty:
        # Tri par NAV décroissant (Le gagnant en haut)
        df_lb = df_lb.sort_values("NAV Total ($)", ascending=False)
        
    return df_lb, trades, port

# --- 4. UI CONSTRUCTION ---

# Header
c1, c2 = st.columns([6, 1])
with c1:
    st.title("PHOENIX ARENA 🏟️")
    st.caption("Live Strategy Competition • Hedge Fund Mode")
with c2:
    if st.button("REFRESH ⚡"): st.rerun()

# Chargement
with st.spinner("Synchronisation de l'arène..."):
    df_leaderboard, df_trades, df_port = get_data()

# KPI GLOBAUX
if not df_leaderboard.empty:
    top_strat = df_leaderboard.iloc[0]['Stratégie']
    total_aum = df_leaderboard['NAV Total ($)'].sum()
    total_pnl = df_leaderboard['PnL ($)'].sum()
    
    k1, k2, k3, k4 = st.columns(4)
    def kpi(col, label, val, color="white"):
        col.markdown(f"""<div class="kpi-box"><div class="kpi-label">{label}</div><div class="kpi-val" style="color:{color}">{val}</div></div>""", unsafe_allow_html=True)
        
    kpi(k1, "TOTAL AUM (NAV)", f"${total_aum:,.0f}")
    kpi(k2, "TOP STRATÉGIE", top_strat, "#3b82f6") # Bleu
    kpi(k3, "PNL AGRÉGÉ", f"${total_pnl:+.2f}", "#22c55e" if total_pnl >=0 else "#ef4444")
    kpi(k4, "POSITIONS ACTIVES", len(df_port[df_port['symbol']!='USDT']), "#f59e0b")

st.markdown("---")

# MAIN LAYOUT : GAUCHE (CHART + TABLE) | DROITE (FEED)
col_main, col_feed = st.columns([3, 1])

with col_main:
    # 1. CHART DE COMPARAISON (Comme l'image 2)
    st.subheader("📈 Performance Relative")
    
    # Ici on triche un peu : comme on n'a pas l'historique NAV par stratégie dans une table simple pour l'instant,
    # on affiche la répartition actuelle. 
    # (Pour avoir la courbe temporelle exacte type Alpha Arena, il faut historiser le NAV chaque heure en DB)
    if not df_leaderboard.empty:
        fig = px.bar(
            df_leaderboard, 
            x="Stratégie", 
            y="NAV Total ($)", 
            color="Stratégie",
            template="plotly_dark",
            title="Capital Actuel par Stratégie",
            text_auto='.2s'
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350)
        st.plotly_chart(fig, use_container_width=True)

    # 2. LEADERBOARD TABLE (Comme l'image 1)
    st.subheader("🏆 Classement des Stratégies")
    if not df_leaderboard.empty:
        st.dataframe(
            df_leaderboard,
            column_config={
                "Stratégie": st.column_config.TextColumn("Stratégie", width="medium"),
                "NAV Total ($)": st.column_config.ProgressColumn(
                    "Capital", 
                    format="$%.2f",
                    min_value=0, 
                    max_value=max(df_leaderboard["NAV Total ($)"])
                ),
                "Win Rate (%)": st.column_config.NumberColumn("Win Rate", format="%.1f %%"),
                "PnL ($)": st.column_config.NumberColumn("PnL Net", format="$%.2f")
            },
            hide_index=True,
            use_container_width=True
        )

with col_feed:
    st.subheader("⚡ Live Feed")
    st.markdown('<div class="trade-feed">', unsafe_allow_html=True)
    
    if not df_trades.empty:
        for idx, row in df_trades.iterrows():
            side = row['side']
            symbol = row['symbol']
            strat = row['strategy']
            price = float(row['price'])
            qty = float(row['quantity'])
            pnl = float(row['pnl'])
            time_str = pd.to_datetime(row['timestamp']).strftime('%H:%M')
            
            # Icone & Couleur
            icon = "🟢" if side == "BUY" else "🔴"
            css_class = side
            
            pnl_html = ""
            if side == "SELL":
                color = "win" if pnl >= 0 else "loss"
                pnl_html = f'<span class="trade-pnl {color}">{pnl:+.2f}$</span>'
            
            st.markdown(f"""
            <div class="trade-card {css_class}">
                <div class="trade-meta">{time_str} • {strat}</div>
                <div class="trade-header">
                    <span>{icon} {side} {symbol}</span>
                    {pnl_html}
                </div>
                <div>{qty:.4f} @ {price:.2f}$</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("En attente de trades...")
        
    st.markdown('</div>', unsafe_allow_html=True)
