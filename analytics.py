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

# --- CONFIGURATION CRITIQUE ---
# Force Matplotlib à ne pas utiliser de fenêtre graphique (X11/Tkinter)
# Indispensable pour que ça marche sur GitHub Actions / Serveurs
matplotlib.use('Agg')

logger = logging.getLogger("PhoenixAnalytics")

class AdvancedChartGenerator:
    """
    Générateur de rapports graphiques Haute Performance.
    Génère :
    1. Un rapport Statique (PNG) pour l'archivage/email.
    2. Un rapport Interactif (HTML) optimisé pour le Web.
    """
    
    def __init__(self, output_dir="portfolio_logs_phoenix"):
        self.output_dir = output_dir
        # Création du dossier charts s'il n'existe pas
        os.makedirs(os.path.join(self.output_dir, "charts"), exist_ok=True)
        self.setup_style()
        
    def setup_style(self):
        """Configuration des couleurs et du style"""
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except OSError:
            plt.style.use('ggplot') 
            
        # Palette de couleurs "Phoenix"
        self.palette = [
            '#2E86AB', # Bleu
            '#A23B72', # Magenta
            '#18A558', # Vert
            '#F24236', # Rouge
            '#F5B700', # Jaune
            '#4FB0C6', # Cyan
            '#6A4C93', # Violet
            '#FF6B6B', # Saumon
        ]

    def _downsample_data(self, data: List[Dict], target_points: int = 1000) -> pd.DataFrame:
        """
        OPTIMISATION : Réduit le nombre de points pour l'affichage Web.
        Transforme une liste de dicts en DataFrame allégé.
        """
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        
        # Conversion Timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        # Si peu de données, on renvoie tout
        if len(df) <= target_points:
            return df
            
        # Sinon on garde 1 point tous les N (Slicing)
        # C'est beaucoup plus rapide que le resample() statistique
        factor = len(df) // target_points
        return df.iloc[::factor].copy()

    def create_comprehensive_dashboard(self, all_results: Dict[str, Any]):
        """
        Point d'entrée principal.
        Génère les deux versions du dashboard.
        """
        if not all_results:
            logger.warning("⚠️ Pas de données pour générer les graphiques.")
            return

        # 1. Version PNG (Statique, Haute Résolution)
        png_path = self._generate_static_png(all_results)
        
        # 2. Version HTML (Interactive, Légère)
        html_path = self._generate_interactive_html(all_results)
        
        return png_path, html_path

    def _generate_static_png(self, all_results):
        """Génère l'image statique via Matplotlib"""
        try:
            fig = plt.figure(figsize=(20, 15))
            # Grille : 4 rangées, 2 colonnes
            gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.2)
            
            # A. Performance (Bar Chart)
            ax1 = fig.add_subplot(gs[0, :])
            self._plot_mpl_performance(ax1, all_results)
            
            # B. Évolution Portefeuille (Line Chart)
            ax2 = fig.add_subplot(gs[1, :])
            self._plot_mpl_evolution(ax2, all_results)
            
            # C. Drawdowns (Area Chart)
            ax3 = fig.add_subplot(gs[2, 0])
            self._plot_mpl_drawdown(ax3, all_results)
            
            # D. Distribution (Box Plot)
            ax4 = fig.add_subplot(gs[2, 1])
            self._plot_mpl_distribution(ax4, all_results)
            
            # E. Risk/Reward (Scatter)
            ax5 = fig.add_subplot(gs[3, :])
            self._plot_mpl_risk_reward(ax5, all_results)
            
            # Titre & Save
            plt.suptitle('PHOENIX TRADING REPORT', fontsize=20, fontweight='bold', y=0.92)
            filename = os.path.join(self.output_dir, "charts", "phoenix_report.png")
            plt.savefig(filename, dpi=100, bbox_inches='tight')
            plt.close()
            return filename
            
        except Exception as e:
            logger.error(f"❌ Erreur PNG: {e}")
            return None

    def _generate_interactive_html(self, all_results):
        """Génère le dashboard interactif Plotly (Optimisé WebGL)"""
        try:
            strategies = list(all_results.keys())
            
            fig = make_subplots(
                rows=3, cols=2,
                specs=[[{'type': 'xy'}, {'type': 'xy'}],      # Bar / Line
                       [{'type': 'xy'}, {'type': 'polar'}],   # Drawdown / Radar
                       [{'type': 'box'}, {'type': 'xy'}]],    # Box / Scatter
                subplot_titles=(
                    'Rendement Total (%)', 'Évolution Capital (Optimisé)',
                    'Drawdowns', 'Profil de Risque',
                    'Distribution des Gains', 'Frontière Efficiente'
                ),
                vertical_spacing=0.1
            )

            for i, strat in enumerate(strategies):
                res = all_results[strat].get('results', {})
                hist = all_results[strat].get('portfolio_history', [])
                color = self.palette[i % len(self.palette)]
                
                # --- 1. Bar Chart (Simple) ---
                fig.add_trace(go.Bar(
                    name=strat, x=[strat], y=[res.get('Return', 0) * 100], # En %
                    marker_color=color, showlegend=False
                ), row=1, col=1)

                # --- 2. Line Chart (OPTIMISÉ SCATTERGL) ---
                # On downsample pour ne pas tuer le navigateur
                df_light = self._downsample_data(hist, target_points=800)
                if not df_light.empty:
                    fig.add_trace(go.Scattergl(
                        x=df_light['timestamp'], y=df_light['value'],
                        name=strat, mode='lines',
                        line=dict(color=color, width=2)
                    ), row=1, col=2)

                # --- 3. Drawdown Area (Optimisé) ---
                if not df_light.empty:
                    vals = df_light['value'].values
                    peak = np.maximum.accumulate(vals)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        dd = (vals - peak) / peak * 100
                        dd = np.nan_to_num(dd)
                    
                    fig.add_trace(go.Scattergl(
                        x=df_light['timestamp'], y=dd,
                        name=f"DD {strat}",
                        line=dict(color=color, width=1),
                        fill='tozeroy'
                    ), row=2, col=1)

                # --- 4. Radar Chart ---
                metrics = ['Sharpe Ratio', 'Sortino Ratio', 'Win Rate', 'Calmar Ratio']
                # Normalisation sommaire pour l'affichage (éviter les échelles écrasées)
                vals = [
                    max(res.get('Sharpe Ratio', 0), 0),
                    max(res.get('Sortino Ratio', 0), 0),
                    res.get('Win Rate', 0) * 5,  # Boost visuel pour le Winrate
                    min(max(res.get('Calmar Ratio', 0), 0), 5) # Cap à 5
                ]
                vals += vals[:1] # Fermer la boucle
                
                fig.add_trace(go.Scatterpolar(
                    r=vals, theta=metrics + [metrics[0]],
                    fill='toself', name=strat,
                    line=dict(color=color), showlegend=False
                ), row=2, col=2)

                # --- 5. Box Plot (Distribution) ---
                # On utilise un peu plus de points pour la distribution
                df_dist = self._downsample_data(hist, target_points=2000)
                if not df_dist.empty:
                    rets = df_dist['value'].pct_change().dropna() * 100
                    fig.add_trace(go.Box(
                        y=rets, name=strat, marker_color=color, showlegend=False
                    ), row=3, col=1)

                # --- 6. Scatter (Risk/Reward) ---
                ret = res.get('Return', 0) * 100
                sharpe = res.get('Sharpe Ratio', 0)
                # Volatilité estimée via Sharpe (Vol = Ret / Sharpe)
                vol = (ret / sharpe) if sharpe > 0.1 else 0
                
                fig.add_trace(go.Scatter(
                    x=[vol], y=[ret], mode='markers+text',
                    text=[strat], name=strat, textposition="top center",
                    marker=dict(size=12, color=color)
                ), row=3, col=2)

            # Mise en page sombre
            fig.update_layout(
                height=1000, 
                title_text="PHOENIX INTERACTIVE DASHBOARD", 
                template="plotly_dark",
                margin=dict(l=20, r=20, t=60, b=20)
            )
            
            filename = os.path.join(self.output_dir, "charts", "phoenix_interactive.html")
            fig.write_html(filename)
            logger.info(f"🌐 HTML généré : {filename}")
            return filename

        except Exception as e:
            logger.error(f"❌ Erreur HTML: {e}")
            return None

    # --- MÉTHODES MATPLOTLIB (INTERNES) ---
    
    def _plot_mpl_performance(self, ax, results):
        names = list(results.keys())
        vals = [results[n]['results'].get('Return', 0) * 100 for n in names]
        colors = [self.palette[i % len(self.palette)] for i in range(len(names))]
        ax.bar(names, vals, color=colors)
        ax.set_title('Rendement Total (%)')
        ax.grid(True, alpha=0.3)

    def _plot_mpl_evolution(self, ax, results):
        for i, (name, data) in enumerate(results.items()):
            hist = data.get('portfolio_history', [])
            if hist:
                vals = [x['value'] for x in hist]
                ax.plot(vals, label=name, color=self.palette[i % len(self.palette)], linewidth=2)
        ax.set_title('Évolution du Capital')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_mpl_drawdown(self, ax, results):
        for i, (name, data) in enumerate(results.items()):
            hist = data.get('portfolio_history', [])
            if hist:
                vals = np.array([x['value'] for x in hist])
                peak = np.maximum.accumulate(vals)
                with np.errstate(divide='ignore', invalid='ignore'):
                    dd = (vals - peak) / peak * 100
                ax.plot(dd, color=self.palette[i % len(self.palette)], alpha=0.8)
        ax.set_title('Drawdowns (%)')
        ax.grid(True, alpha=0.3)

    def _plot_mpl_distribution(self, ax, results):
        data_to_plot = []
        labels = []
        for name, data in results.items():
            hist = data.get('portfolio_history', [])
            if hist:
                vals = pd.Series([x['value'] for x in hist])
                rets = vals.pct_change().dropna() * 100
                data_to_plot.append(rets)
                labels.append(name)
        if data_to_plot:
            ax.boxplot(data_to_plot, labels=labels)
        ax.set_title('Distribution des Rendements')

    def _plot_mpl_risk_reward(self, ax, results):
        for i, (name, data) in enumerate(results.items()):
            res = data.get('results', {})
            ret = res.get('Return', 0) * 100
            sharpe = res.get('Sharpe Ratio', 0)
            vol = (ret / sharpe) if sharpe > 0.1 else 0
            
            ax.scatter(vol, ret, s=100, label=name, color=self.palette[i % len(self.palette)])
            ax.text(vol, ret+0.2, name, fontsize=9, ha='center')
            
        ax.set_xlabel('Volatilité (Risque)')
        ax.set_ylabel('Rendement (%)')
        ax.set_title('Frontière Efficiente')
        ax.grid(True, alpha=0.3)
