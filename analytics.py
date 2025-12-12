import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any
import logging

# Mode Headless pour serveur
matplotlib.use('Agg')

logger = logging.getLogger("PhoenixAnalytics")

class AdvancedChartGenerator:
    """
    Générateur de graphiques optimisé.
    Utilise le Downsampling et WebGL pour éviter le lag sur les gros datasets.
    """
    
    def __init__(self, output_dir="portfolio_logs_phoenix"):
        self.output_dir = output_dir
        os.makedirs(os.path.join(self.output_dir, "charts"), exist_ok=True)
        self.setup_matplotlib_styles()
        
    def setup_matplotlib_styles(self):
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except OSError:
            plt.style.use('ggplot') 
            
        self.colors = {
            'primary': '#2E86AB', 'secondary': '#A23B72', 'success': '#18A558',
            'danger': '#F24236', 'warning': '#F5B700', 'info': '#4FB0C6',
            'dark': '#2F2F2F', 'light': '#F8F9FA'
        }
        self.palette = [
            '#2E86AB', '#A23B72', '#18A558', '#F24236', '#F5B700',
            '#4FB0C6', '#6A4C93', '#FF6B6B', '#4ECDC4', '#45B7D1'
        ]

    def _downsample_data(self, data: List[Dict], target_points: int = 1000) -> pd.DataFrame:
        """
        Réduit intelligemment le nombre de points pour l'affichage.
        Si on a 50 000 minutes, on résume à ~1000 points pour ne pas tuer le navigateur.
        """
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        if len(df) <= target_points:
            return df
            
        # Facteur de réduction
        factor = len(df) // target_points
        
        # On prend 1 ligne sur N (slicing simple et rapide)
        # Pour une courbe de capital, c'est souvent suffisant et plus rapide que le resample temporel
        return df.iloc[::factor].copy()

    def create_comprehensive_dashboard(self, all_results: Dict[str, Any]):
        """Génère le PNG statique (Haute résolution, pas de limite de points)"""
        if not all_results:
            return None

        try:
            fig = plt.figure(figsize=(24, 20))
            gs = GridSpec(5, 4, figure=fig, hspace=0.4, wspace=0.3)
            
            # --- BLOC STATIQUE MATPLOTLIB (Identique à avant, car PNG ne lag pas) ---
            ax1 = fig.add_subplot(gs[0, :2])
            self._plot_strategy_performance(ax1, all_results)
            ax2 = fig.add_subplot(gs[0, 2:])
            self._plot_risk_metrics(ax2, all_results)
            ax3 = fig.add_subplot(gs[1, :])
            self._plot_portfolio_evolution(ax3, all_results) # PNG supporte bcp de points
            ax4 = fig.add_subplot(gs[2, :2])
            self._plot_drawdown_analysis(ax4, all_results)
            ax5 = fig.add_subplot(gs[2, 2:])
            ax5.axis('off') # Heatmap désactivée
            ax6 = fig.add_subplot(gs[3, :2])
            self._plot_return_distribution(ax6, all_results)
            ax7 = fig.add_subplot(gs[3, 2:])
            self._plot_optimal_allocation(ax7, all_results)
            ax8 = fig.add_subplot(gs[4, :])
            self._plot_trading_metrics(ax8, all_results)
            
            plt.suptitle('PHOENIX DASHBOARD - STATIQUE', fontsize=20, fontweight='bold', y=0.95)
            filename = os.path.join(self.output_dir, "charts", "phoenix_dashboard_static.png")
            plt.savefig(filename, dpi=150, bbox_inches='tight') # DPI réduit pour vitesse
            plt.close()
            
            # Génération Interactif OPTIMISÉ
            self.create_interactive_dashboard(all_results)
            return filename
            
        except Exception as e:
            logger.error(f"Erreur PNG: {e}")
            return None

    def create_interactive_dashboard(self, all_results: Dict[str, Any]):
        """Génère le HTML interactif (Optimisé WebGL + Downsampling)"""
        try:
            strategies = list(all_results.keys())
            if not strategies: return None
                
            fig = make_subplots(
                rows=3, cols=2,
                specs=[[{'type': 'xy'}, {'type': 'xy'}],
                       [{'type': 'xy'}, {'type': 'polar'}],
                       [{'type': 'box'}, {'type': 'xy'}]],
                subplot_titles=('Performance', 'Évolution (Optimisé)', 'Drawdowns', 'Radar Risque', 'Distribution', 'Frontière'),
                vertical_spacing=0.1
            )

            # 1. Bar Chart (Léger par nature)
            for i, strat in enumerate(strategies):
                res = all_results[strat].get('results', {})
                fig.add_trace(go.Bar(
                    name=strat, x=[strat], y=[res.get('Return', 0)],
                    marker_color=self.palette[i % len(self.palette)], showlegend=False
                ), row=1, col=1)

            # 2. Line Chart Evolution (OPTIMISÉ SCATTERGL + DOWNSAMPLE)
            for i, (strat, data) in enumerate(all_results.items()):
                hist = data.get('portfolio_history', [])
                
                # OPTIMISATION MAJEURE ICI
                df_light = self._downsample_data(hist, target_points=800)
                
                if not df_light.empty:
                    # Scattergl utilise le GPU, beaucoup plus fluide
                    fig.add_trace(go.Scattergl(
                        x=df_light['timestamp'], y=df_light['value'], name=strat,
                        line=dict(color=self.palette[i % len(self.palette)], width=1.5),
                        mode='lines' # Pas de markers pour alléger
                    ), row=1, col=2)

            # 3. Drawdowns (Optimisé)
            for i, strat in enumerate(strategies):
                hist = all_results[strat].get('portfolio_history', [])
                df_light = self._downsample_data(hist, target_points=800)
                
                if not df_light.empty:
                    vals = df_light['value'].values
                    peak = np.maximum.accumulate(vals)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        dd = (vals - peak) / peak * 100
                        dd = np.nan_to_num(dd)
                    
                    fig.add_trace(go.Scattergl(
                        x=df_light['timestamp'], y=dd, name=f'DD {strat}',
                        line=dict(color=self.palette[i % len(self.palette)], width=1),
                        fill='tozeroy'
                    ), row=2, col=1)

            # 4. Radar Chart (Léger)
            metrics = ['Sharpe', 'Sortino', 'Win Rate', 'Calmar']
            for i, strat in enumerate(strategies):
                res = all_results[strat].get('results', {})
                vals = [
                    max(res.get('Sharpe Ratio', 0), 0),
                    max(res.get('Sortino Ratio', 0), 0),
                    res.get('Win Rate', 0) * 3, 
                    min(max(res.get('Calmar Ratio', 0), 0), 5)
                ]
                vals += vals[:1]
                fig.add_trace(go.Scatterpolar(
                    r=vals, theta=metrics + [metrics[0]], fill='toself', name=strat,
                    line=dict(color=self.palette[i % len(self.palette)])
                ), row=2, col=2)

            # 5. Box Plot (Léger car agrégé)
            for i, (strat, data) in enumerate(all_results.items()):
                hist = data.get('portfolio_history', [])
                if hist:
                    # On garde un peu plus de points pour la distribution mais pas tout
                    df_dist = self._downsample_data(hist, target_points=2000)
                    rets = df_dist['value'].pct_change().dropna() * 100
                    fig.add_trace(go.Box(
                        y=rets, name=strat, marker_color=self.palette[i % len(self.palette)]
                    ), row=3, col=1)

            # 6. Scatter
            for i, strat in enumerate(strategies):
                res = all_results[strat].get('results', {})
                fig.add_trace(go.Scatter(
                    x=[(res.get('Return', 0)/res.get('Sharpe Ratio', 1)) if res.get('Sharpe Ratio') else 0],
                    y=[res.get('Return', 0)],
                    mode='markers+text', name=strat, text=[strat],
                    marker=dict(size=12, color=self.palette[i % len(self.palette)])
                ), row=3, col=2)

            fig.update_layout(height=1000, title_text="PHOENIX DASHBOARD (FAST MODE)", template="plotly_dark")
            
            filename = os.path.join(self.output_dir, "charts", "phoenix_dashboard_interactive.html")
            fig.write_html(filename)
            logger.info(f"🌐 Dashboard interactif généré : {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Erreur HTML: {e}")
            return None

    # --- MÉTHODES MATPLOTLIB (Inchangées car PNG statique) ---
    def _plot_strategy_performance(self, ax, all_results):
        strategies = list(all_results.keys())
        returns = [all_results[s]['results'].get('Return', 0) for s in strategies]
        ax.bar(strategies, returns, color=self.colors['primary'])
        ax.set_title('Performance (%)')

    def _plot_portfolio_evolution(self, ax, all_results):
        for i, (strat, data) in enumerate(all_results.items()):
            hist = data.get('portfolio_history', [])
            if hist:
                vals = [x['value'] for x in hist]
                ax.plot(vals, label=strat, color=self.palette[i%len(self.palette)])
        ax.legend()
        ax.set_title('Valeur Portefeuille')

    def _plot_risk_metrics(self, ax, all_results):
        ax.text(0.5, 0.5, "Radar Chart (Voir HTML)", ha='center')

    def _plot_drawdown_analysis(self, ax, all_results):
        ax.text(0.5, 0.5, "Drawdown (Voir HTML)", ha='center')

    def _plot_return_distribution(self, ax, all_results):
        ax.text(0.5, 0.5, "Distribution (Voir HTML)", ha='center')

    def _plot_optimal_allocation(self, ax, all_results):
        ax.text(0.5, 0.5, "Frontière (Voir HTML)", ha='center')

    def _plot_trading_metrics(self, ax, all_results):
        strategies = list(all_results.keys())
        trades = [all_results[s]['results'].get('Trades', 0) for s in strategies]
        ax.bar(strategies, trades, color=self.colors['dark'])
        ax.set_title('Trades')