import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from supabase import create_client, Client
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import numpy as np

# --- CONFIGURATION DE LA PAGE (DOIT ÊTRE LA PREMIÈRE COMMANDE) ---
st.set_page_config(
    page_title="Phoenix Pro Terminal",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chargement des variables d'environnement
load_dotenv()
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

# Connexion Supabase sécurisée
@st.cache_resource
def init_connection():
    if not url or not key:
        st.error("Erreur: Identifiants Supabase manquants dans le fichier .env")
        st.stop()
    return create_client(url, key)

supabase = init_connection()

# --- CSS PROFESSIONNEL "DARK FINANCE" ---
st.markdown("""
<style>
    /* Fond global */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Style des conteneurs de métriques (Cartes) */
    div[data-testid="stMetric"] {
        background-color: #1a1c24;
        border: 1px solid #2d303e;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Couleurs des textes */
    .css-10trblm { color: #e0e0e0; }
    h1, h2, h3 { color: #ffffff; font-family: 'Roboto', sans-serif; }
    
    /* Style des tableaux */
    .dataframe { font-size: 14px; }
    
    /* Suppression des marges inutiles de Streamlit */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 10px; background: #0e1117; }
    ::-webkit-scrollbar-thumb { background: #2d303e; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# --- FONCTIONS DE RÉCUPÉRATION DE DONNÉES ---

def get_current_portfolio():
    """Récupère l'état ACTUEL du portefeuille"""
    try:
        # On suppose qu'il y a une table 'portfolio_current' ou on prend le dernier snapshot
        response = supabase.table('portfolio_state').select("*").execute()
        data = response.data
        if not data: return pd.DataFrame()
        
        # Astuce: si la table contient tout l'historique, on prend les entrées les plus récentes par actif
        df = pd.DataFrame(data)
        df['updated_at'] = pd.to_datetime(df['updated_at'])
        # On garde uniquement la dernière ligne pour chaque crypto + stratégie
        latest_portfolio = df.sort_values('updated_at').groupby(['symbol', 'strategy']).tail(1)
        return latest_portfolio
    except Exception as e:
        st.error(f"Erreur DB Portfolio: {e}")
        return pd.DataFrame()

def get_trades_history(days_lookback=30):
    """Récupère l'historique des trades pour la PnL Curve"""
    try:
        date_cutoff = (datetime.now() - timedelta(days=days_lookback)).isoformat()
        response = supabase.table('trades')\
            .select("*")\
            .gte('timestamp', date_cutoff)\
            .order('timestamp', desc=False)\
            .execute() # desc=False pour avoir l'ordre chronologique
        data = response.data
        if not data: return pd.DataFrame()
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Erreur DB Trades: {e}")
        return pd.DataFrame()

# --- FONCTIONS DE CALCULS ---

def calculate_equity_curve(trades_df, initial_capital=1000):
    """
    Reconstruit la courbe de valeur totale du portefeuille basée sur les trades réalisés.
    C'est la fameuse courbe que vous voulez.
    """
    if trades_df.empty:
        # Retourne une ligne plate si pas de trades
        return pd.DataFrame({'timestamp': [datetime.now()], 'total_equity': [initial_capital]})

    df = trades_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    # On calcule le PnL réalisé (Realized PnL) pour chaque trade VENTE
    # Note: Ceci est une simplification. Idéalement, il faudrait une table d'historique de snapshots.
    df['trade_pnl'] = 0.0
    # On suppose que le 'price' d'un SELL est le prix de sortie, et qu'on a besoin du prix d'entrée pour le PnL.
    # Si votre DB trades a déjà une colonne 'pnl', utilisez-la directement.
    # Sinon, une approximation est de tracker le cashflow :
    
    running_capital = initial_capital
    equity_history = []

    # Ajout du point de départ
    start_date = df['timestamp'].iloc[0] - timedelta(minutes=1)
    equity_history.append({'timestamp': start_date, 'total_equity': initial_capital})

    # Simulation simple de l'évolution du capital (cashflow)
    for index, row in df.iterrows():
        cost = row['price'] * row['quantity']
        if row['side'] == 'BUY':
            running_capital -= cost
        elif row['side'] == 'SELL':
            running_capital += cost
        
        equity_history.append({
            'timestamp': row['timestamp'],
            'total_equity': running_capital
        })
        
    equity_df = pd.DataFrame(equity_history)
    
    # Lissage pour le graphique si plusieurs trades à la même seconde
    equity_df = equity_df.groupby('timestamp')['total_equity'].last().reset_index()
    return equity_df

def color_pnl(val):
    """Fonction pour colorer le PnL dans les tableaux Pandas"""
    color = '#00ff7f' if val > 0 else '#ff4b4b' if val < 0 else '#e0e0e0'
    return f'color: {color}; font-weight: bold;'

# --- SIDEBAR (Filtres & Contrôles) ---
with st.sidebar:
    st.title("🦅 Contrôles Phoenix")
    st.markdown("---")
    
    # Filtre de temps pour les graphiques
    time_range = st.selectbox(
        "📅 Période d'analyse",
        options=["7 Derniers Jours", "30 Derniers Jours", "Tout l'historique"],
        index=1
    )
    
    lookback_days = 30
    if time_range == "7 Derniers Jours": lookback_days = 7
    elif time_range == "Tout l'historique": lookback_days = 365 * 5 # Grand nombre

    st.markdown("---")
    if st.button("🔄 Rafraîchir les données", use_container_width=True):
        # st.experimental_rerun() est obsolète, on utilise st.rerun()
        st.rerun()
        
    st.caption(f"Dernière MàJ: {datetime.now().strftime('%H:%M:%S')}")

# --- MAIN DASHBOARD ---

st.title("📊 Phoenix Pro Terminal")
st.markdown("---")

# 1. CHARGEMENT DES DONNÉES
portfolio_df = get_current_portfolio()
trades_df = get_trades_history(lookback_days)

# 2. CALCUL DES KIs (Key Performance Indicators)
total_value_usdt = 1000.0 # Valeur par défaut (Capital de départ)
total_pnl_abs = 0.0
total_pnl_pct = 0.0
active_positions_count = 0

if not portfolio_df.empty:
    # Isoler le cash USDT
    usdt_row = portfolio_df[portfolio_df['symbol'] == 'USDT']
    usdt_balance = usdt_row['quantity'].sum() if not usdt_row.empty else 0
    
    # Calculer la valeur des positions cryptos (Quantité * Prix d'entrée moyen)
    # Note: Pour un vrai total, il faudrait le prix ACTUEL du marché, pas le prix d'entrée.
    # On utilise le prix d'entrée comme approximation ici, ou on devrait fetcher les prix live.
    crypto_positions = portfolio_df[portfolio_df['symbol'] != 'USDT'].copy()
    crypto_value_at_entry = (crypto_positions['quantity'] * crypto_positions['entry_price']).sum()
    active_positions_count = len(crypto_positions[crypto_positions['quantity'] > 0])

    # Valeur Totale Estimée (Cash + Valeur d'achat des cryptos)
    # ATTENTION: C'est une approximation si on n'a pas les prix live.
    # Le mieux est d'utiliser la dernière valeur de la courbe d'équité si elle est fiable.
    total_value_usdt = usdt_balance + crypto_value_at_entry
    
    # Si on a une courbe d'équité, utilisons sa dernière valeur qui est plus juste sur le cashflow
    equity_curve = calculate_equity_curve(trades_df, initial_capital=1000)
    if not equity_curve.empty:
         total_value_usdt = equity_curve.iloc[-1]['total_equity']

    initial_capital = 1000.0 # À configurer ou récupérer de la DB
    total_pnl_abs = total_value_usdt - initial_capital
    total_pnl_pct = (total_pnl_abs / initial_capital) * 100 if initial_capital > 0 else 0

# 3. AFFICHAGE DES MÉTRIQUES (Top Row)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="💰 Valeur Totale du Portefeuille",
        value=f"${total_value_usdt:,.2f}",
        delta=f"{total_pnl_abs:+.2f}$ (Global)",
        delta_color="normal" # Le delta gère la couleur auto
    )

with col2:
    st.metric(
        label="📈 PnL Total (%)",
        value=f"{total_pnl_pct:+.2f}%",
        delta_color="normal" if total_pnl_pct >= 0 else "inverse"
    )
    
with col3:
    nb_trades = len(trades_df) if not trades_df.empty else 0
    st.metric(label="🔄 Trades Exécutés (Période)", value=nb_trades)

with col4:
    st.metric(label="⚡ Positions Actives", value=active_positions_count)

st.markdown("---")

# 4. LA "MASTER CURVE" (Courbe d'Équité)
st.subheader("📈 Évolution de la Valeur Totale (Equity Curve)")

if not trades_df.empty:
    equity_df = calculate_equity_curve(trades_df)
    
    # Création du graphique Area Chart professionnel
    fig = px.area(
        equity_df, 
        x='timestamp', 
        y='total_equity',
        template='plotly_dark',
    )

    # Customisation Poussée du Graphique
    fig.update_traces(
        line=dict(color='#00ff7f', width=3), # Ligne verte fluo
        fillcolor='rgba(0, 255, 127, 0.1)'   # Remplissage vert transparent
    )
    
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Valeur en USDT",
        hovermode="x unified",
        height=500,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)', # Fond transparent pour s'intégrer au dashboard
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=False,
            rangeslider=dict(visible=True), # Ajout du slider en bas
            type="date"
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor='#2d303e', # Grille subtile
            tickprefix="$"
        )
    )

    # Ligne de référence du capital de départ
    fig.add_hline(y=1000, line_dash="dash", line_color="gray", annotation_text="Capital Départ")
    
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("En attente de trades pour générer la courbe de performance...")

# 5. TABLEAU DES POSITIONS DÉTAILLÉ
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📋 Positions Actuelles et Récentes")
    if not portfolio_df.empty:
        # Préparation du tableau
        display_df = portfolio_df[portfolio_df['symbol'] != 'USDT'].copy()
        
        if not display_df.empty:
            # Calcul d'un PnL Latent approximatif (si on n'a pas les prix live)
            # Ici, on affiche juste les colonnes importantes.
            display_df = display_df[['updated_at', 'symbol', 'strategy', 'quantity', 'entry_price']]
            
            # Renommage pour l'affichage
            display_df.columns = ['Dernière MàJ', 'Crypto', 'Stratégie', 'Quantité', 'Prix Entrée Moy.']
            
            # Formatage des nombres
            display_df['Prix Entrée Moy.'] = display_df['Prix Entrée Moy.'].map('${:,.4f}'.format)
            display_df['Quantité'] = display_df['Quantité'].map('{:,.4f}'.format)

            # Affichage du tableau stylisé
            st.dataframe(
                display_df.sort_values('Dernière MàJ', ascending=False),
                use_container_width=True,
                height=300,
                hide_index=True
            )
        else:
            st.info("Aucune position crypto active.")
    else:
        st.warning("Impossible de charger le portefeuille.")

# 6. (Optionnel) Un petit camembert de répartition
with col_right:
    st.subheader("Exposition du Portefeuille")
    if not portfolio_df.empty:
        # Calcul simple de la répartition basée sur les coûts d'entrée
        allocation_df = portfolio_df.copy()
        allocation_df['value'] = allocation_df['quantity'] * allocation_df['entry_price']
        # Pour USDT, la valeur est juste la quantité
        allocation_df.loc[allocation_df['symbol'] == 'USDT', 'value'] = allocation_df.loc[allocation_df['symbol'] == 'USDT', 'quantity']
        
        # On ne garde que ce qui a une valeur positive
        allocation_df = allocation_df[allocation_df['value'] > 0.01]

        if not allocation_df.empty:
            fig_pie = px.pie(
                allocation_df, 
                values='value', 
                names='symbol',
                template='plotly_dark',
                hole=0.4 # Donut chart
            )
            fig_pie.update_traces(textinfo='percent+label')
            fig_pie.update_layout(
                margin=dict(l=20, r=20, t=0, b=20),
                 paper_bgcolor='rgba(0,0,0,0)',
                 showlegend=False
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
             st.info("Portefeuille vide.")
