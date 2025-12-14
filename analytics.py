import os
import logging
import pandas as pd
from typing import Dict, List, Any, Optional

# Configuration Matplotlib pour environnement Headless (Serveurs/Docker)
# Doit être fait AVANT d'importer pyplot
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Plotly pour l'interactif
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger("PhoenixAnalytics")

class AnalyticsVisualizer:
    """
    Moteur de visualisation PURE.
    
    Principes :
    1. AUCUN calcul financier (pas de pct_change, pas de cummax).
    2. Reçoit des métriques prêtes à l'emploi (provenant de metrics.py).
    3. Utilise l'API Orientée Objet de Matplotlib pour éviter les effets de bord.
    """

    def __init__(self, output_dir: str = "portfolio_logs_phoenix"):
        self.output_dir = output_dir
        self.charts_dir = os.path.join(self.output_dir, "charts")
        os.makedirs(self.charts_dir, exist_ok=True)
        
        # Style visuel cohérent
        self.colors = {
            'primary': '#1f77b4',  # Bleu
            'secondary': '#ff7f0e', # Orange
            'success': '#2ca02c',   # Vert
            'danger': '#d62728',    # Rouge
            'neutral': '#7f7f7f',   # Gris
            'grid': '#e6e6e6'
        }

    def generate_static_report(
        self, 
        equity_curve: pd.Series, 
        drawdown_curve: pd.Series, 
        benchmark_curve: Optional[pd.Series] = None,
        daily_returns: Optional[pd.Series] = None,
        filename: str = "performance_summary.png"
    ) -> str:
        """
        Génère une image statique résumant la performance.
        Attend des Séries temporelles alignées et pré-calculées.
        """
        if equity_curve.empty:
            logger.warning("Courbe d'équité vide, pas de graphique généré.")
            return ""

        # Création de la Figure en mode OO (pas de plt.figure global)
        fig = plt.figure(figsize=(12, 10), constrained_layout=True)
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1])

        # 1. Equity Curve
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(equity_curve.index, equity_curve.values, label='Stratégie', color=self.colors['primary'], linewidth=1.5)
        
        if benchmark_curve is not None and not benchmark_curve.empty:
            # On s'assure que le benchmark est aligné temporellement
            ax1.plot(benchmark_curve.index, benchmark_curve.values, label='Benchmark', color=self.colors['neutral'], linestyle='--', alpha=0.7)
            
        ax1.set_title("Évolution du Capital (Equity)", fontsize=12, fontweight='bold')
        ax1.set_ylabel("Capital ($)")
        ax1.legend(loc='upper left')
        ax1.grid(True, color=self.colors['grid'], linestyle='--')

        # 2. Drawdown (Doit être fourni en négatif ou positif selon convention, ici on affiche tel quel)
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.fill_between(drawdown_curve.index, drawdown_curve.values, 0, color=self.colors['danger'], alpha=0.3, label='Drawdown')
        ax2.plot(drawdown_curve.index, drawdown_curve.values, color=self.colors['danger'], linewidth=1)
        ax2.set_title("Drawdown", fontsize=10)
        ax2.set_ylabel("%")
        ax2.grid(True, color=self.colors['grid'], linestyle=':')

        # 3. Daily Returns (Histogramme ou Barplot)
        ax3 = fig.add_subplot(gs[2])
        if daily_returns is not None and not daily_returns.empty:
            ax3.bar(daily_returns.index, daily_returns.values, color=self.colors['secondary'], width=0.8, alpha=0.8)
            ax3.set_title("Rendements Quotidiens", fontsize=10)
            ax3.set_ylabel("%")
            ax3.axhline(0, color='black', linewidth=0.5)
        else:
            ax3.text(0.5, 0.5, "Données Returns manquantes", ha='center')

        # Formatage des dates sur l'axe X (commun)
        fig.autofmt_xdate()
        
        # Sauvegarde
        output_path = os.path.join(self.charts_dir, filename)
        try:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"📊 Rapport statique généré : {output_path}")
        except Exception as e:
            logger.error(f"❌ Erreur lors de la sauvegarde du graphique : {e}")
        finally:
            # NETTOYAGE CRITIQUE : Fermer explicitement la figure pour libérer la RAM
            plt.close(fig)
            
        return output_path

    def generate_interactive_chart(
        self, 
        ohlcv_df: pd.DataFrame, 
        trades: List[Dict[str, Any]], 
        indicators: Dict[str, pd.Series] = {},
        filename: str = "interactive_chart.html"
    ) -> str:
        """
        Génère un graphique interactif Plotly.
        Séparé de toute logique de calcul d'indicateurs.
        """
        if ohlcv_df.empty:
            return ""

        # Création des subplots (Prix + Volume)
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.05, 
            row_heights=[0.7, 0.3]
        )

        # 1. Chandeliers
        fig.add_trace(go.Candlestick(
            x=ohlcv_df.index,
            open=ohlcv_df['open'],
            high=ohlcv_df['high'],
            low=ohlcv_df['low'],
            close=ohlcv_df['close'],
            name='Prix'
        ), row=1, col=1)

        # 2. Indicateurs Techniques (Doivent être passés déjà calculés)
        colors = ['blue', 'orange', 'purple', 'brown']
        idx = 0
        for name, series in indicators.items():
            if series is not None and not series.empty:
                # Aligner la série avec l'index OHLCV si nécessaire
                series = series.reindex(ohlcv_df.index)
                fig.add_trace(go.Scatter(
                    x=series.index, 
                    y=series.values, 
                    mode='lines', 
                    name=name,
                    line=dict(width=1, color=colors[idx % len(colors)])
                ), row=1, col=1)
                idx += 1

        # 3. Marqueurs de Trades
        # On sépare les achats et les ventes pour le style
        buys_x, buys_y = [], []
        sells_x, sells_y = [], []
        
        for trade in trades:
            # On suppose que 'timestamp' est une string ISO ou datetime
            t_date = pd.to_datetime(trade.get('timestamp'))
            price = trade.get('price')
            
            if trade.get('side') == 'BUY':
                buys_x.append(t_date)
                buys_y.append(price)
            elif trade.get('side') == 'SELL':
                sells_x.append(t_date)
                sells_y.append(price)

        if buys_x:
            fig.add_trace(go.Scatter(
                x=buys_x, y=buys_y,
                mode='markers',
                marker=dict(symbol='triangle-up', size=10, color='green'),
                name='Achat'
            ), row=1, col=1)
            
        if sells_x:
            fig.add_trace(go.Scatter(
                x=sells_x, y=sells_y,
                mode='markers',
                marker=dict(symbol='triangle-down', size=10, color='red'),
                name='Vente'
            ), row=1, col=1)

        # 4. Volume
        fig.add_trace(go.Bar(
            x=ohlcv_df.index,
            y=ohlcv_df['volume'],
            name='Volume',
            marker_color='rgba(100, 100, 100, 0.5)'
        ), row=2, col=1)

        # Layout
        fig.update_layout(
            title="Analyse Détaillée - Phoenix Bot",
            yaxis_title="Prix",
            xaxis_rangeslider_visible=False,
            height=800,
            template="plotly_dark"
        )

        output_path = os.path.join(self.charts_dir, filename)
        try:
            fig.write_html(output_path)
            logger.info(f"🌐 Graphique interactif généré : {output_path}")
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde Plotly : {e}")

        return output_path

    def generate_metrics_heatmap(self, correlation_matrix: pd.DataFrame, filename: str = "correlations.png") -> str:
        """
        Visualise une matrice de corrélation (pure visualization).
        """
        if correlation_matrix.empty:
            return ""
            
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        
        cax = ax.matshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        fig.colorbar(cax)
        
        # Labels
        ticks = range(len(correlation_matrix.columns))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='left')
        ax.set_yticklabels(correlation_matrix.columns)
        
        ax.set_title("Corrélation entre Actifs/Stratégies")
        
        output_path = os.path.join(self.charts_dir, filename)
        try:
            fig.savefig(output_path, bbox_inches='tight')
        finally:
            plt.close(fig)
            
        return output_path
