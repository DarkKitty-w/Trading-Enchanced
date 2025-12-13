import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

# --- 1. CONFIGURATION PAGE & THEME ---
st.set_page_config(
    page_title="PHOENIX STRATEGY COMMANDER",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chargement env
load_dotenv()

# --- 2. CSS MODERNE (Style "Command Center") ---
st.markdown("""
<style>
    /* Fond sombre global */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Style des onglets (Tabs) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border-radius: 4px;
        color: white;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
    }

    /* Cartes de métriques */
    div[data-testid="stMetric"] {
        background-color: #1a1c24;
        border: 1px solid #334155;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Titres */
    h1, h2, h3 { color: #f8fafc; }
    p { color: #cbd5e1; }
</style>
""", unsafe_allow_html=True)

# --- 3. CONNEXION DB ---
@st.cache_resource
def init_connection():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        st.error("⚠️ Identifiants Supabase manquants (.env)")
        st.stop()
    return create_client(url, key)

supabase = init_connection()

# --- 4. RÉCUPÉRATION DES DONNÉES ---
def get_data():
    """Récupère et nettoie toutes les données nécessaires"""
    try:
        # 1. Trades
        trades_resp = supabase.table('trades').select("*").execute()
        trades_df = pd.DataFrame(trades_resp.data)
        if not trades_df.empty:
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            trades_df = trades_df.sort_values('timestamp')

        # 2. Portfolio (Positions actuelles)
        port_resp = supabase.table('portfolio_state').select("*").execute()
        port_df = pd.DataFrame(port_resp.data)
        if not port_df.empty:
            port_df['updated_at'] = pd.to_datetime(port_df['updated_at'])
            # Garder seulement la dernière entrée pour chaque paire (Asset + Strategy)
            port_df = port_df.sort_values('updated_at').groupby(['symbol', 'strategy']).tail(1)
            # Retirer les quantités nulles (positions fermées)
            port_df = port_df[port_df['quantity'] > 0]

        return trades_df, port_df
    except Exception as e:
        st.error(f"Erreur DB: {e}")
        return pd.DataFrame(), pd.DataFrame()

# --- 5. CALCULS PAR STRATÉGIE ---
def calculate_strategy_curve(trades_df, strategy_name):
    """Calcule la courbe de PnL cumulé pour une stratégie donnée"""
    if trades_df.empty: return pd.DataFrame()
    
    # Filtrer par stratégie
    strat_trades = trades_df[trades_df['strategy'] == strategy_name].copy()
    if strat_trades.empty: return pd.DataFrame()
    
    # On calcule le PnL cumulé au fil du temps
    # Note: On suppose que la colonne 'pnl' est remplie dans la DB lors de la vente.
    # Si 'pnl' est 0 (ex: achat), le cumul ne bouge pas.
    strat_trades['cumulative_pnl'] = strat_trades['pnl'].cumsum()
    
    return strat_trades[['timestamp', 'cumulative_pnl']]

# --- 6. INTERFACE ---

# Sidebar
with st.sidebar:
    st.title("🦅 Phoenix Control")
    st.write("Filtres globaux")
    if st.button("🔄 Actualiser Données"):
        st.rerun()
    
    st.markdown("---")
    st.warning("⚠️ Zone Danger")
    if st.button("💥 RESET DATABASE (Tout effacer)"):
        supabase.table('trades').delete().neq('id', '0').execute()
        supabase.table('portfolio_state').delete().neq('symbol', '0').execute()
        st.success("Base de données vidée !")
        st.rerun()

# Chargement
trades_df, port_df = get_data()

st.title("📊 Performance par Stratégie")

# Liste des stratégies détectées
if not trades_df.empty:
    strategies = trades_df['strategy'].unique().tolist()
else:
    strategies = []
    
# Ajout d'une vue "Globale"
all_tabs = ["🔍 VUE D'ENSEMBLE"] + strategies
tabs = st.tabs(all_tabs)

# --- ONGLET 1 : VUE D'ENSEMBLE (COMPARATIF) ---
with tabs[0]:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("⚔️ Comparaison des Performances")
        if not trades_df.empty:
            # Création d'un graphique multiline
            fig = go.Figure()
            
            for strat in strategies:
                curve = calculate_strategy_curve(trades_df, strat)
                if not curve.empty:
                    fig.add_trace(go.Scatter(
                        x=curve['timestamp'], 
                        y=curve['cumulative_pnl'],
                        mode='lines',
                        name=strat,
                        line=dict(width=2)
                    ))
            
            fig.update_layout(
                template="plotly_dark",
                xaxis_title="Temps",
                yaxis_title="Profit Cumulé ($)",
                height=500,
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucun trade enregistré pour le moment.")

    with col2:
        st.subheader("🏆 Classement")
        if not trades_df.empty:
            leaderboard = trades_df.groupby('strategy')['pnl'].sum().sort_values(ascending=False).reset_index()
            leaderboard.columns = ['Stratégie', 'Profit Total ($)']
            
            # Affichage joli avec couleur conditionnelle
            st.dataframe(
                leaderboard.style.format({'Profit Total ($)': '{:.2f}'})
                .background_gradient(cmap='RdYlGn', subset=['Profit Total ($)']),
                use_container_width=True,
                hide_index=True
            )

# --- ONGLETS DYNAMIQUES : DÉTAIL PAR STRATÉGIE ---
for i, strat in enumerate(strategies):
    with tabs[i + 1]: # +1 car le 0 est la vue d'ensemble
        st.markdown(f"## Détails: {strat}")
        
        # Filtres spécifiques à la stratégie
        strat_trades = trades_df[trades_df['strategy'] == strat]
        strat_positions = port_df[port_df['strategy'] == strat] if not port_df.empty else pd.DataFrame()
        
        # Métriques Rapides
        total_pnl = strat_trades['pnl'].sum()
        win_rate = (len(strat_trades[strat_trades['pnl'] > 0]) / len(strat_trades)) * 100 if len(strat_trades) > 0 else 0
        nb_trades = len(strat_trades)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("💰 Profit Net", f"${total_pnl:+.2f}", delta_color="normal")
        m2.metric("🎯 Win Rate", f"{win_rate:.1f}%")
        m3.metric("🔄 Total Trades", nb_trades)
        
        st.divider()
        
        # Courbe Spécifique (Area Chart)
        curve = calculate_strategy_curve(trades_df, strat)
        if not curve.empty:
            fig_strat = px.area(
                curve, x='timestamp', y='cumulative_pnl',
                title=f"Courbe d'Équité - {strat}",
                template="plotly_dark"
            )
            fig_strat.update_traces(line_color='#3b82f6', fillcolor="rgba(59, 130, 246, 0.2)")
            st.plotly_chart(fig_strat, use_container_width=True)
            
        # Positions Actives (Cartes comme demandé)
        st.subheader("📦 Positions Actives")
        if not strat_positions.empty:
            cols = st.columns(4)
            for idx, row in strat_positions.iterrows():
                # Calcul PnL latent approximatif (si on avait le prix actuel, ici on simule)
                # entry_price = row['entry_price']
                # current_price = ... (Besoin de fetcher le prix live pour être précis)
                
                with cols[idx % 4]:
                    st.markdown(f"""
                    <div style="background-color: #1e293b; padding: 15px; border-radius: 8px; border-left: 5px solid #3b82f6;">
                        <h4 style="margin:0; color:white;">{row['symbol']}</h4>
                        <p style="margin:5px 0; color:#94a3b8; font-size:12px;">Qté: {row['quantity']:.4f}</p>
                        <p style="margin:0; color:#e2e8f0; font-weight:bold;">Entrée: ${row['entry_price']:.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Aucune position ouverte pour cette stratégie.")

        # Historique Récent
        st.subheader("📜 Derniers Trades")
        st.dataframe(
            strat_trades[['timestamp', 'side', 'symbol', 'price', 'quantity', 'pnl']]
            .sort_values('timestamp', ascending=False)
            .head(10)
            .style.format({
                'price': '{:.4f}', 
                'quantity': '{:.4f}', 
                'pnl': '{:+.2f}'
            })
            .applymap(lambda v: 'color: #ef4444' if v < 0 else 'color: #22c55e' if v > 0 else '', subset=['pnl']),
            use_container_width=True
        )
