import numpy as np
import pandas as pd
from typing import List, Dict, Any, Protocol
from dataclasses import dataclass

# ==============================================================================
# 1. CONTRATS DE DONNÉES (INTERFACES STRICTES)
# ==============================================================================

@dataclass
class PortfolioSnapshot:
    """
    Représente l'état du portefeuille à un instant T.
    Doit impérativement contenir 'timestamp' et 'total_equity'.
    """
    timestamp: Any  # datetime ou str iso
    total_equity: float

@dataclass
class CompletedTrade:
    """
    Représente un trade TERMINE (Round-Trip) avec un résultat PnL.
    Ce n'est pas juste un ordre d'exécution, mais le bilan entrée/sortie.
    """
    trade_id: str
    symbol: str
    pnl: float          # Profit/Perte en USD
    return_pct: float   # Rendement en % (ex: 0.05 pour 5%)
    duration: float     # Durée en secondes/minutes
    side: str           # 'LONG' ou 'SHORT'

# ==============================================================================
# 2. MOTEUR DE CALCUL (PURE FUNCTIONS)
# ==============================================================================

class FinancialMetrics:
    """
    Moteur de calcul financier pur.
    
    Principes :
    1. Stateless : Pas d'état interne.
    2. Input Strict : Attend des Listes d'Objets typés ou des Series Pandas.
    3. Fail-Fast : Pas de .get(), pas de valeurs par défaut magiques.
    """

    @staticmethod
    def calculate_portfolio_metrics(snapshots: List[PortfolioSnapshot], risk_free_rate: float = 0.0) -> Dict[str, float]:
        """
        Calcule les métriques globales basées sur la courbe d'équité (Time-Weighted).
        
        Args:
            snapshots: Liste chronologique de PortfolioSnapshot.
        """
        if not snapshots or len(snapshots) < 2:
            return FinancialMetrics._empty_portfolio_metrics()

        # 1. Extraction Vectorisée (Fail-fast si attribut manquant)
        equity_curve = pd.Series(
            data=[s.total_equity for s in snapshots],
            index=pd.to_datetime([s.timestamp for s in snapshots])
        )
        
        # Nettoyage et tri
        equity_curve = equity_curve.sort_index()
        
        # 2. Calcul des variations
        initial_equity = equity_curve.iloc[0]
        final_equity = equity_curve.iloc[-1]
        
        # Returns (variations relatives)
        returns = equity_curve.pct_change().dropna()
        
        if returns.empty:
            return FinancialMetrics._empty_portfolio_metrics()

        # 3. Calculs Mathématiques
        total_return = (final_equity - initial_equity) / initial_equity
        
        # Max Drawdown
        drawdown_series = FinancialMetrics._compute_drawdown_series(equity_curve)
        max_drawdown = drawdown_series.min()  # Valeur négative (ex: -0.15)

        # Ratios (Annualisation approximative basée sur la fréquence des snapshots)
        # On suppose ici des snapshots haute fréquence, on normalise via std dev
        sharpe = FinancialMetrics._compute_sharpe_ratio(returns, risk_free_rate)
        sortino = FinancialMetrics._compute_sortino_ratio(returns, risk_free_rate)
        calmar = FinancialMetrics._compute_calmar_ratio(total_return, max_drawdown)
        
        volatility = returns.std()

        return {
            "initial_equity": initial_equity,
            "final_equity": final_equity,
            "net_profit": final_equity - initial_equity,
            "total_return_pct": total_return,
            "max_drawdown_pct": max_drawdown,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "volatility": volatility
        }

    @staticmethod
    def calculate_trade_metrics(trades: List[CompletedTrade]) -> Dict[str, float]:
        """
        Calcule les métriques basées sur les trades (Trade-Weighted).
        """
        if not trades:
            return FinancialMetrics._empty_trade_metrics()

        # 1. Extraction Vectorisée
        # Ici on accède directement aux attributs .pnl, .return_pct
        pnls = np.array([t.pnl for t in trades])
        returns = np.array([t.return_pct for t in trades])
        
        total_trades = len(trades)
        winning_trades = pnls[pnls > 0]
        losing_trades = pnls[pnls <= 0]
        
        n_wins = len(winning_trades)
        n_losses = len(losing_trades)
        
        # 2. Win Rate
        win_rate = n_wins / total_trades if total_trades > 0 else 0.0
        
        # 3. Profit Factor
        gross_profit = winning_trades.sum() if n_wins > 0 else 0.0
        gross_loss = abs(losing_trades.sum()) if n_losses > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
        
        # 4. Averages
        avg_pnl = pnls.mean()
        avg_win = winning_trades.mean() if n_wins > 0 else 0.0
        avg_loss = losing_trades.mean() if n_losses > 0 else 0.0

        # 5. Expectancy (Espérance mathématique par trade)
        # Formule : (WinRate * AvgWin) - (LossRate * AvgLoss)
        loss_rate = 1.0 - win_rate
        # AvgLoss est négatif ici, donc on l'ajoute ou on le soustrait selon convention.
        # Ici avg_loss est signé négatif.
        expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)

        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "avg_pnl": avg_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "expectancy": expectancy,
            "best_trade": pnls.max(),
            "worst_trade": pnls.min()
        }

    # ==========================================================================
    # MÉTHODES MATHÉMATIQUES INTERNES (VECTORISÉES)
    # ==========================================================================

    @staticmethod
    def _compute_drawdown_series(equity_curve: pd.Series) -> pd.Series:
        """Calcule la série de Drawdown (en négatif)."""
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        return drawdown

    @staticmethod
    def _compute_sharpe_ratio(returns: pd.Series, risk_free: float) -> float:
        """Ratio de Sharpe simplifié (sans annualisation complexe pour rester neutre)."""
        if returns.empty or returns.std() == 0:
            return 0.0
        excess_returns = returns - risk_free
        return excess_returns.mean() / returns.std()

    @staticmethod
    def _compute_sortino_ratio(returns: pd.Series, risk_free: float) -> float:
        """Ratio de Sortino (ne pénalise que la volatilité négative)."""
        if returns.empty:
            return 0.0
        excess_returns = returns - risk_free
        downside_returns = excess_returns[excess_returns < 0]
        
        if downside_returns.empty or downside_returns.std() == 0:
            return 0.0 if excess_returns.mean() <= 0 else 100.0 # Valeur arbitraire haute si aucun risque baissier
            
        downside_deviation = downside_returns.std()
        return excess_returns.mean() / downside_deviation

    @staticmethod
    def _compute_calmar_ratio(total_return: float, max_drawdown: float) -> float:
        """Ratio de Calmar (Retour total / Max Drawdown absolu)."""
        if max_drawdown == 0:
            return 0.0
        return total_return / abs(max_drawdown)

    # ==========================================================================
    # VALEURS PAR DÉFAUT
    # ==========================================================================

    @staticmethod
    def _empty_portfolio_metrics() -> Dict[str, float]:
        return {
            "initial_equity": 0.0, "final_equity": 0.0, "net_profit": 0.0,
            "total_return_pct": 0.0, "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0, "sortino_ratio": 0.0, "calmar_ratio": 0.0,
            "volatility": 0.0
        }

    @staticmethod
    def _empty_trade_metrics() -> Dict[str, float]:
        return {
            "total_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
            "gross_profit": 0.0, "gross_loss": 0.0, "avg_pnl": 0.0,
            "expectancy": 0.0, "best_trade": 0.0, "worst_trade": 0.0
        }
