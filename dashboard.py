import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from supabase import create_client
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

# --- 1. CONFIGURATION INITIALE ---
st.set_page_config(
    page_title="Phoenix Strategy Analyzer",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

# --- 2. STYLE CSS "DARK PRO" ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    div[data-testid="stMetric"] {
        background-color: #1a1c24; border: 1px solid #2d303e;
        padding: 10px; border-radius: 8px;
    }
    h1, h2, h3 { color: #ffffff; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap;
        background-color: #1a1c24; border-radius: 4px 4px 0px 0px;
        gap: 1px; padding-top: 10px; padding-bottom: 10px; color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2d303e; border-bottom: 2px solid #00ff7f;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. CONNEXION DB ---
@st.cache_resource
def init_connection():
    if not url or not key: return None
    return create_client(url, key)

supabase = init_connection()

# --- 4. RÉCUPÉRATION DES DONNÉES ---
def get_data():
    if not supabase: return pd.DataFrame(), pd.DataFrame()
    
    # Récupération Portfolio (Positions actuelles)
    res_port = supabase.table('portfolio_state').select("*").execute()
    df_port = pd.DataFrame(res_port.data) if res_port.data else pd.DataFrame()
    
    # Récupération Trades (Historique)
    res_trades = supabase.table('trades').select("*").execute()
    df_trades = pd.DataFrame(res_trades.data) if res_trades.data else pd.DataFrame()
    
    return df_port, df_trades

df_portfolio, df_trades = get_data()

# --- 5. CALCULS PAR STRATÉGIE ---
def process_strategy_curve(trades_df, strategy_name, initial_capital=1000):
    """Calcule la courbe d'équité pour UNE stratégie spécifique"""
    if trades_df.empty: return pd.DataFrame()
    
    # Filtrer pour la stratégie demandée
    strat_trades = trades_df[trades_df['strategy'] == strategy_name].copy()
    if strat_trades.empty: return pd.DataFrame()
    
    strat_trades['timestamp'] = pd.to_datetime(strat_trades['timestamp'])
    strat_trades = strat_trades.sort_values('timestamp')
    
    running_pnl = 0
    history = []
    
    # Point de départ
    history.append({'timestamp': strat_trades['timestamp'].iloc[0] - timedelta(minutes=1), 'equity': 0})
    
    # On reconstitue le PnL cumulé (Somme des 'pnl' enregistrés en base)
    # Note: Votre Metrics.py enregistre le PnL réalisé dans la colonne 'pnl' de la table trades
    for idx, row in strat_trades.iterrows():
        pnl = row.get('pnl', 0)
        running_pnl += pnl
        history.append({'timestamp': row['timestamp'], 'equity': running_pnl})
        
    return pd.DataFrame(history)

# --- 6. INTERFACE ---

st.title("🦅 Phoenix : Analyse Sectorielle")
st.caption("Séparation stricte des performances par stratégie")

if df_trades.empty:
    st.error("⚠️ Aucune donnée de trade trouvée. Avez-vous lancé le bot ?")
    st.stop()

# Liste des stratégies uniques trouvées dans l'historique
strategies_list = df_trades['strategy'].unique().tolist()

# === ONGLET 1 : LA COURSE (COMPARAISON) ===
tab1, tab2 = st.tabs(["🏆 Comparatif Global", "🔍 Détail par Stratégie"])

with tab1:
    st.subheader("Performance Relative (Qui gagne ?)")
    
    fig_comp = go.Figure()
    
    for strat in strategies_list:
        df_curve = process_strategy_curve(df_trades, strat)
        if not df_curve.empty:
            fig_comp.add_trace(go.Scatter(
                x=df_curve['timestamp'], 
                y=df_curve['equity'], 
                mode='lines', 
                name=strat,
                line=dict(width=2)
            ))
            
    fig_comp.update_layout(
        template="plotly_dark",
        yaxis_title="Gains/Pertes Cumulés ($)",
        height=500,
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0.5, xanchor="center"),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_comp, use_container_width=True)

# === ONGLET 2 : LE MICROSCOPE (DÉTAIL) ===
with tab2:
    col_sel, col_blank = st.columns([1, 2])
    with col_sel:
        selected_strat = st.selectbox("📂 Choisir la Stratégie à analyser :", strategies_list)
    
    st.markdown("---")
    
    # Filtrage des données pour la stratégie choisie
    strat_trades = df_trades[df_trades['strategy'] == selected_strat]
    
    if not strat_trades.empty:
        # Calculs KPIs
        total_pnl = strat_trades['pnl'].sum()
        win_trades = len(strat_trades[strat_trades['pnl'] > 0])
        total_count = len(strat_trades)
        win_rate = (win_trades / total_count * 100) if total_count > 0 else 0
        
        # Affichage Métriques
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Gains Net (PnL)", f"${total_pnl:,.2f}", delta_color="normal")
        c2.metric("Nombre de Trades", total_count)
        c3.metric("Win Rate", f"{win_rate:.1f}%")
        
        # Graphique Spécifique (Area Chart)
        df_curve_strat = process_strategy_curve(df_trades, selected_strat)
        
        fig_strat = px.area(
            df_curve_strat, x='timestamp', y='equity', 
            title=f"Courbe de Profit : {selected_strat}",
            template="plotly_dark"
        )
        fig_strat.update_traces(line_color='#00ff7f', fillcolor='rgba(0, 255, 127, 0.1)')
        fig_strat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_strat, use_container_width=True)
        
        # Positions Actuelles de CETTE stratégie
        st.subheader(f"Positions Actives : {selected_strat}")
        if not df_portfolio.empty:
            strat_positions = df_portfolio[
                (df_portfolio['strategy'] == selected_strat) & 
                (df_portfolio['symbol'] != 'USDT') &
                (df_portfolio['quantity'] > 0)
            ]
            
            if not strat_positions.empty:
                # Joli tableau
                display_df = strat_positions[['symbol', 'quantity', 'entry_price', 'updated_at']].copy()
                display_df.columns = ['Crypto', 'Quantité', 'Prix Entrée', 'Dernière MàJ']
                st.dataframe(display_df, use_container_width=True)
            else:
                st.info(f"Aucune position ouverte pour {selected_strat} en ce moment.")
    else:
        st.warning("Pas encore de trades pour cette stratégie.")

# --- SIDEBAR : BOUTON RESET ---
with st.sidebar:
    st.markdown("### ⚠️ Zone Danger")
    if st.button("VIDER TOUTE LA BASE DE DONNÉES", type="primary"):
        try:
            supabase.table('trades').delete().neq('id', '0').execute()
            supabase.table('portfolio_state').delete().neq('symbol', '0').execute()
            st.toast("Base de données effacée ! Rechargez la page.", icon="💥")
        except Exception as e:
            st.error(f"Erreur: {e}")
