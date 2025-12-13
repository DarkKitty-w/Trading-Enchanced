import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from supabase import create_client
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import numpy as np

# --- 1. CONFIGURATION PAGE & CSS PREMIUM ---
st.set_page_config(
    page_title="Phoenix Terminal | Hedge Fund Edition",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Injection CSS pour le look "Financial Terminal"
st.markdown("""
<style>
    /* RESET ET FOND */
    .stApp {
        background-color: #0b0e11; /* Noir profond */
        font-family: 'Inter', sans-serif;
    }
    
    /* CARTES DE MÉTRIQUES (KPIS) */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #161b22, #0d1117);
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: #58a6ff;
    }
    
    /* TITRES ET TEXTES */
    h1, h2, h3 { color: #f0f6fc !important; font-weight: 600; }
    p, label { color: #8b949e !important; }
    
    /* ONGLETS (TABS) STYLISÉS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #161b22;
        border-radius: 6px;
        color: #8b949e;
        border: 1px solid #30363d;
        padding: 0 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #238636 !important; /* Vert GitHub/Finance */
        color: white !important;
        border-color: #238636 !important;
    }
    
    /* GRAPHIQUES PLOTLY */
    .js-plotly-plot .plotly .modebar { display: none !important; } /* Cacher la barre d'outils moche */
</style>
""", unsafe_allow_html=True)

# --- 2. CONNEXION ET DATA ---
load_dotenv()
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

@st.cache_resource
def init_db():
    if not url or not key: return None
    return create_client(url, key)

supabase = init_db()

def get_data():
    """Récupère et nettoie toutes les données nécessaires"""
    if not supabase: return pd.DataFrame(), pd.DataFrame()
    
    # Historique Trades
    res_t = supabase.table('trades').select("*").execute()
    df_trades = pd.DataFrame(res_t.data) if res_t.data else pd.DataFrame()
    
    # État Actuel Portfolio
    res_p = supabase.table('portfolio_state').select("*").execute()
    df_port = pd.DataFrame(res_p.data) if res_p.data else pd.DataFrame()
    
    # Nettoyage
    if not df_trades.empty:
        df_trades['timestamp'] = pd.to_datetime(df_trades['timestamp'])
        df_trades = df_trades.sort_values('timestamp')
        
    return df_port, df_trades

df_port, df_trades = get_data()

# --- 3. MOTEUR DE CALCUL FINANCIER ---

def calculate_net_worth_series(trades_df, initial_capital=1000):
    """
    Reconstruit l'évolution précise de la 'Net Worth' (Valeur Nette).
    Net Worth = Capital Initial + PnL Réalisé cumulé au fil du temps.
    """
    if trades_df.empty:
        return pd.DataFrame({'timestamp': [datetime.now()], 'net_worth': [initial_capital]})
    
    # On ne garde que les trades qui ont un PnL (généralement les ventes)
    # Si votre bot log le PnL à la vente, on utilise ça.
    history = []
    current_equity = initial_capital
    
    # Point de départ
    history.append({'timestamp': trades_df['timestamp'].iloc[0] - timedelta(minutes=1), 'net_worth': initial_capital})
    
    for _, row in trades_df.iterrows():
        pnl = row.get('pnl', 0)
        # On ajoute le PnL au capital seulement s'il est non nul (Trade fermé)
        # Ou on considère que les frais ('fee') réduisent le capital à chaque trade
        fee = row.get('fee', 0)
        
        current_equity += pnl - fee # Net Worth = PnL - Frais
        
        history.append({
            'timestamp': row['timestamp'],
            'net_worth': current_equity,
            'strategy': row.get('strategy', 'Global')
        })
        
    return pd.DataFrame(history)

# --- 4. DASHBOARD INTERFACE ---

st.title("🦅 Phoenix Terminal")
st.markdown("système de trading algorithmique multi-stratégies")
st.markdown("---")

if df_trades.empty:
    st.warning("⚠️ En attente de données... Lancez le bot (main.py) pour voir le terminal s'animer.")
    st.stop()

# --- CALCULS GLOBAUX ---
net_worth_df = calculate_net_worth_series(df_trades)
current_net_worth = net_worth_df.iloc[-1]['net_worth']
initial_cap = 1000
total_pnl = current_net_worth - initial_cap
pnl_pct = (total_pnl / initial_cap) * 100

total_trades = len(df_trades)
winning_trades = len(df_trades[df_trades['pnl'] > 0])
win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

# --- HEADER METRICS (GLOBAL) ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("💰 Net Worth (Valeur Totale)", f"${current_net_worth:,.2f}", f"{pnl_pct:+.2f}%")
c2.metric("📊 PnL Absolu", f"${total_pnl:+,.2f}", delta_color="normal")
c3.metric("🎯 Win Rate Global", f"{win_rate:.1f}%")
c4.metric("🔄 Volume Trades", total_trades)

st.markdown("---")

# --- ONGLETS PRINCIPAUX ---
tab_global, tab_strategy, tab_data = st.tabs(["🌍 Vue d'Ensemble (Global)", "🔬 Analyse Stratégie (X-Ray)", "💾 Data & Reset"])

# ==============================================================================
# ONGLET 1 : VUE GLOBALE (NET WORTH & ALLOCATION)
# ==============================================================================
with tab_global:
    col_chart, col_pie = st.columns([2, 1])
    
    with col_chart:
        st.subheader("📈 Évolution de la Valeur Nette (Net Worth)")
        # Graphique Area Chart Pro
        fig_equity = px.area(
            net_worth_df, 
            x='timestamp', 
            y='net_worth',
            template="plotly_dark",
        )
        fig_equity.update_traces(
            line_color='#00ff7f', # Vert Matrix
            fillcolor='rgba(0, 255, 127, 0.05)'
        )
        fig_equity.update_layout(
            height=450,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="",
            yaxis_title="USD",
            hovermode="x unified"
        )
        st.plotly_chart(fig_equity, use_container_width=True)
        
    with col_pie:
        st.subheader("🍰 Allocation Actuelle")
        if not df_port.empty:
            # On filtre USDT pour voir l'exposition crypto
            crypto_port = df_port[df_port['symbol'] != 'USDT'].copy()
            crypto_port['value'] = crypto_port['quantity'] * crypto_port['entry_price']
            crypto_port = crypto_port[crypto_port['value'] > 1] # Filtre poussière
            
            if not crypto_port.empty:
                fig_pie = px.pie(
                    crypto_port, 
                    values='value', 
                    names='symbol',
                    hole=0.6,
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                fig_pie.update_layout(
                    showlegend=True,
                    legend=dict(orientation="h", y=-0.1),
                    margin=dict(t=0, b=0, l=0, r=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=350
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Petit tableau résumé en dessous
                exp_total = crypto_port['value'].sum()
                cash = current_net_worth - exp_total
                st.progress(min(100, int((exp_total/current_net_worth)*100)), text=f"Exposition Marché: {exp_total:.0f}$ / Cash: {cash:.0f}$")

            else:
                st.info("Portefeuille 100% Cash (USDT)")
        else:
            st.warning("Données portefeuille indisponibles")

    # Comparatif Performance Stratégies (Bar Chart)
    st.subheader("🏆 Performance par Stratégie (PnL Réalisé)")
    strat_perf = df_trades.groupby('strategy')['pnl'].sum().reset_index().sort_values('pnl', ascending=False)
    
    fig_bar = px.bar(
        strat_perf, x='pnl', y='strategy', orientation='h',
        text='pnl',
        color='pnl',
        color_continuous_scale=['#ff4b4b', '#2d303e', '#00ff7f'] # Rouge -> Gris -> Vert
    )
    fig_bar.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="PnL ($)",
        yaxis_title="",
        height=300
    )
    fig_bar.update_traces(texttemplate='%{text:+.2f}$', textposition='outside')
    st.plotly_chart(fig_bar, use_container_width=True)


# ==============================================================================
# ONGLET 2 : ANALYSE STRATÉGIE (LE DÉTAIL)
# ==============================================================================
with tab_strategy:
    strategies = df_trades['strategy'].unique()
    selected_strat = st.selectbox("🔍 Sélectionnez une Stratégie à analyser :", strategies)
    
    # Filtrage
    strat_data = df_trades[df_trades['strategy'] == selected_strat].copy()
    
    # Calculs Spécifiques
    strat_pnl = strat_data['pnl'].sum()
    strat_fees = strat_data['fee'].sum()
    strat_count = len(strat_data)
    strat_wins = len(strat_data[strat_data['pnl'] > 0])
    strat_wr = (strat_wins / strat_count * 100) if strat_count > 0 else 0
    
    # Ligne de métriques
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PnL Net", f"${strat_pnl:+.2f}", f"Frais: -{strat_fees:.2f}$")
    c2.metric("Win Rate", f"{strat_wr:.1f}%")
    c3.metric("Trades Total", strat_count)
    
    # Profit Factor
    gains = strat_data[strat_data['pnl'] > 0]['pnl'].sum()
    pertes = abs(strat_data[strat_data['pnl'] < 0]['pnl'].sum())
    pf = (gains / pertes) if pertes > 0 else 99.0
    c4.metric("Profit Factor", f"{pf:.2f}")
    
    col_graph, col_dist = st.columns([2, 1])
    
    with col_graph:
        st.subheader(f"Courbe de Performance : {selected_strat}")
        # Recalcul de la courbe equity JUSTE pour cette strat
        strat_equity = [0]
        for p in strat_data['pnl']:
            strat_equity.append(strat_equity[-1] + p)
            
        fig_strat = px.line(
            x=list(range(len(strat_equity))), 
            y=strat_equity,
            labels={'x': 'Nombre de Trades', 'y': 'PnL Cumulé ($)'}
        )
        fig_strat.update_traces(line_color='#58a6ff', line_width=3) # Bleu Tech
        fig_strat.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_strat.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_strat, use_container_width=True)
        
    with col_dist:
        st.subheader("Distribution des Gains/Pertes")
        fig_hist = px.histogram(
            strat_data, x="pnl", nbins=20,
            color_discrete_sequence=['#8b949e']
        )
        fig_hist.update_layout(
            template="plotly_dark", 
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Montant ($)",
            yaxis_title="Fréquence",
            showlegend=False
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Tableau des derniers trades de cette stratégie
    st.subheader("Derniers Trades")
    display_cols = ['timestamp', 'symbol', 'side', 'price', 'quantity', 'pnl']
    st.dataframe(
        strat_data[display_cols].sort_values('timestamp', ascending=False).head(10),
        use_container_width=True,
        hide_index=True
    )

# ==============================================================================
# ONGLET 3 : DATA & RESET
# ==============================================================================
with tab_data:
    c1, c2 = st.columns([1, 3])
    
    with c1:
        st.error("⚠️ ZONE DANGER")
        st.markdown("Ceci supprimera définitivement tout l'historique.")
        if st.button("🧨 RÉINITIALISER LA BASE DE DONNÉES", type="primary", use_container_width=True):
            try:
                # Suppression via Supabase
                supabase.table('trades').delete().neq('id', '00000').execute()
                supabase.table('portfolio_state').delete().neq('symbol', '00000').execute()
                # On remet le cash à 1000 USDT (Optionnel, dépend de votre main.py)
                st.success("Base de données vidée avec succès ! Relancez le Dashboard.")
            except Exception as e:
                st.error(f"Erreur: {e}")
                
    with c2:
        st.info("ℹ️ Logs Système Brut")
        st.dataframe(df_trades.sort_values('timestamp', ascending=False), use_container_width=True)
