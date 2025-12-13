import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from supabase import create_client
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

# --- 1. CONFIGURATION PAGE & CSS "BLOOMBERG STYLE" ---
st.set_page_config(
    page_title="Phoenix Fund Commander",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* FOND & TYPO */
    .stApp { background-color: #000000; font-family: 'Roboto Mono', monospace; }
    
    /* LES CARTES DE DONNÉES */
    div[data-testid="stMetric"] {
        background-color: #111111;
        border: 1px solid #333;
        border-radius: 4px;
        padding: 10px;
    }
    div[data-testid="stMetric"]:hover { border-color: #555; }
    
    /* COULEURS TEXTE */
    h1, h2, h3, h4 { color: #e0e0e0 !important; font-family: 'Inter', sans-serif; font-weight: 700; }
    p, span { color: #888 !important; }
    
    /* GRAPHIQUES */
    .js-plotly-plot .plotly .modebar { display: none !important; }
    
    /* TABLEAUX */
    .dataframe { font-size: 12px; font-family: 'Roboto Mono', monospace; }
</style>
""", unsafe_allow_html=True)

# --- 2. DATA LAYER ---
load_dotenv()
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

@st.cache_resource
def init_db():
    if not url or not key: return None
    return create_client(url, key)

supabase = init_db()

def get_data():
    if not supabase: return pd.DataFrame(), pd.DataFrame()
    
    # 1. Trades (Historique PnL)
    res_t = supabase.table('trades').select("*").execute()
    df_trades = pd.DataFrame(res_t.data) if res_t.data else pd.DataFrame()
    
    # 2. Portfolio (Positions Actuelles)
    res_p = supabase.table('portfolio_state').select("*").execute()
    df_port = pd.DataFrame(res_p.data) if res_p.data else pd.DataFrame()
    
    if not df_trades.empty:
        df_trades['timestamp'] = pd.to_datetime(df_trades['timestamp'])
        df_trades = df_trades.sort_values('timestamp')
        
    return df_port, df_trades

df_port, df_trades = get_data()

# --- 3. MOTEUR DE CALCUL "MULTI-FLUX" ---

def calculate_multi_strategy_equity(trades_df, initial_virtual_allocation=1000):
    """
    Crée une série temporelle de Net Worth POUR CHAQUE STRATÉGIE.
    On suppose que chaque stratégie commence avec 'initial_virtual_allocation' (ex: 1000$)
    pour qu'on puisse comparer leur évolution sur une base commune.
    """
    if trades_df.empty: return pd.DataFrame()

    strategies = trades_df['strategy'].unique()
    all_series = []

    for strat in strategies:
        # Filtrer les trades de CETTE stratégie
        strat_df = trades_df[trades_df['strategy'] == strat].copy()
        strat_df = strat_df.sort_values('timestamp')
        
        running_equity = initial_virtual_allocation
        history = []
        
        # Point de départ (T-1 minute)
        start_time = strat_df['timestamp'].iloc[0] - timedelta(minutes=1)
        history.append({'timestamp': start_time, 'net_worth': running_equity, 'strategy': strat})
        
        for _, row in strat_df.iterrows():
            # Net Worth = Capital précédent + PnL du trade - Frais
            pnl = row.get('pnl', 0)
            fee = row.get('fee', 0)
            running_equity += (pnl - fee)
            
            history.append({
                'timestamp': row['timestamp'],
                'net_worth': running_equity,
                'strategy': strat
            })
            
        all_series.append(pd.DataFrame(history))
    
    if not all_series: return pd.DataFrame()
    return pd.concat(all_series)

# --- 4. DASHBOARD UI ---

st.title("🦅 PHOENIX // FUND COMMANDER")
st.markdown("---")

if df_trades.empty:
    st.error("SYSTEM STATUS: WAITING FOR DATA (Lancez main.py)")
    st.stop()

# --- PARTIE 1 : LE GRAPHIQUE ULTIME (COMPARATIF NET WORTH) ---
st.subheader("💹 Net Worth Evolution by Strategy (Base 1000$)")
st.caption("Ce graphique montre comment chaque stratégie fait fructifier son capital indépendamment.")

equity_df = calculate_multi_strategy_equity(df_trades, initial_virtual_allocation=1000)

if not equity_df.empty:
    fig = px.line(
        equity_df, 
        x="timestamp", 
        y="net_worth", 
        color="strategy",
        color_discrete_sequence=px.colors.qualitative.Bold, # Couleurs bien distinctes
        template="plotly_dark"
    )
    
    fig.update_layout(
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#0e0e0e',
        xaxis_title="",
        yaxis_title="Net Worth ($)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            y=1.02,
            x=0.5,
            xanchor="center"
        ),
        xaxis=dict(showgrid=False, gridcolor='#333'),
        yaxis=dict(showgrid=True, gridcolor='#222')
    )
    
    # Ligne de référence (Break even)
    fig.add_hline(y=1000, line_width=1, line_dash="dash", line_color="gray")
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Calcul des courbes en cours...")

st.markdown("---")

# --- PARTIE 2 : KPI PER STRATEGY (TABLEAU DE BORD) ---
st.subheader("⚡ Performance Matrix")

strategies = df_trades['strategy'].unique()
cols = st.columns(len(strategies))

for idx, strat in enumerate(strategies):
    # Données spécifiques
    strat_trades = df_trades[df_trades['strategy'] == strat]
    
    # Calculs
    total_pnl = strat_trades['pnl'].sum() - strat_trades['fee'].sum()
    nb_trades = len(strat_trades)
    win_rate = len(strat_trades[strat_trades['pnl'] > 0]) / nb_trades * 100 if nb_trades > 0 else 0
    
    # Couleur dynamique pour le header
    header_color = "green" if total_pnl > 0 else "red"
    
    with cols[idx]:
        st.markdown(f"""
        <div style="border-top: 3px solid {header_color}; background: #161b22; padding: 15px; border-radius: 5px;">
            <h4 style="margin:0; color: white;">{strat}</h4>
            <div style="font-size: 24px; font-weight: bold; color: {'#4ade80' if total_pnl >= 0 else '#f87171'}; margin: 10px 0;">
                {total_pnl:+.2f} $
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 12px; color: #8b949e;">
                <span>Trades: {nb_trades}</span>
                <span>Win: {win_rate:.0f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Petit graph Sparkline en dessous (Distribution PnL)
        fig_spark = px.bar(strat_trades, x='timestamp', y='pnl')
        fig_spark.update_layout(
            showlegend=False, 
            height=60, 
            margin=dict(l=0,r=0,t=0,b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        fig_spark.update_traces(marker_color=['#4ade80' if x > 0 else '#f87171' for x in strat_trades['pnl']])
        st.plotly_chart(fig_spark, use_container_width=True, config={'displayModeBar': False})

st.markdown("---")

# --- PARTIE 3 : POSITIONS ACTUELLES (SECTORIEL) ---
c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("📋 Positions Ouvertes (Live Assets)")
    if not df_port.empty:
        # On nettoie l'affichage
        view_df = df_port[df_port['symbol'] != 'USDT'].copy()
        if not view_df.empty:
            view_df = view_df[['strategy', 'symbol', 'quantity', 'entry_price', 'updated_at']]
            # Calcul Valeur Estimée (Prix Entrée * Qty car pas de prix live ici)
            view_df['Initial Value ($)'] = view_df['quantity'] * view_df['entry_price']
            
            st.dataframe(
                view_df.style.background_gradient(subset=['Initial Value ($)'], cmap='Greens'),
                use_container_width=True,
                height=300
            )
        else:
            st.info("Aucune position active. Le fonds est 100% Cash.")

with c2:
    st.subheader("🧨 Control Room")
    st.warning("Zone de maintenance")
    if st.button("🔴 PURGE TOTAL DB", use_container_width=True):
        try:
            supabase.table('trades').delete().neq('id', '0').execute()
            supabase.table('portfolio_state').delete().neq('symbol', '0').execute()
            st.success("Base de données nettoyée.")
            st.rerun()
        except:
            st.error("Erreur de purge.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("ℹ️ Note : Le 'Net Worth' par stratégie est simulé sur une base de 1000$ de départ pour permettre la comparaison.")
