import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional

class FinancialMetrics:
    """
    Bibliothèque de calculs financiers et de gestion de risque.
    Centralise les formules mathématiques utilisées par le Backtester et le Dashboard.
    """

    @staticmethod
    def calculate_sharpe_ratio(portfolio_history: List[Dict], risk_free_rate: float = 0.0, periods_per_year: int = 365*24*60) -> float:
        """
        Calcule le ratio de Sharpe (Rendement excédentaire / Volatilité).
        
        Args:
            portfolio_history: Liste de dicts avec clé 'value'
            risk_free_rate: Taux sans risque annuel (par défaut 0)
            periods_per_year: Facteur d'annualisation (par défaut minutes -> annuel)
        """
        if not portfolio_history or len(portfolio_history) < 2:
            return 0.0
        
        try:
            values = [p.get('value', 0) for p in portfolio_history if 'value' in p]
            returns = pd.Series(values).pct_change().dropna()
            
            if returns.std() == 0:
                return 0.0
            
            # Ajustement du taux sans risque pour la période (minute)
            rf_per_period = risk_free_rate / periods_per_year
            
            excess_returns = returns - rf_per_period
            sharpe = excess_returns.mean() / returns.std()
            
            # Annualisation (Racine carrée du temps)
            annualized_sharpe = sharpe * np.sqrt(periods_per_year)
            return annualized_sharpe
            
        except Exception:
            return 0.0

    @staticmethod
    def calculate_sortino_ratio(portfolio_history: List[Dict], target_return: float = 0.0, periods_per_year: int = 365*24*60) -> float:
        """
        Calcule le ratio de Sortino.
        Similaire au Sharpe, mais ne pénalise que la volatilité NÉGATIVE (Downside deviation).
        C'est souvent plus pertinent pour les cryptos.
        """
        if not portfolio_history or len(portfolio_history) < 2:
            return 0.0
        
        try:
            values = [p.get('value', 0) for p in portfolio_history if 'value' in p]
            returns = pd.Series(values).pct_change().dropna()
            
            # On ne garde que les rendements inférieurs à la cible (les "mauvais" mouvements)
            downside_returns = returns[returns < target_return]
            
            if downside_returns.empty or downside_returns.std() == 0:
                # Si aucune perte, le ratio est théoriquement infini -> on cap à une valeur haute
                return 10.0 if returns.mean() > 0 else 0.0
            
            expected_return = returns.mean()
            sortino = (expected_return - target_return) / downside_returns.std()
            
            # Annualisation
            return sortino * np.sqrt(periods_per_year)
            
        except Exception:
            return 0.0

    @staticmethod
    def calculate_total_value(portfolio: List[Dict], strategies: Dict = None) -> float:
        """
        Calcule la valeur totale estimée des positions en cours.
        Utilisé par main.py pour le suivi de performance.
        """
        total_value = 0.0
        
        if not portfolio:
            return 0.0

        for position in portfolio:
            # Récupération sécurisée de la quantité et du prix
            qty = float(position.get('quantity', 0.0))
            
            # On essaie de prendre le 'current_price' (prix actuel) s'il a été mis à jour,
            # sinon on se rabat sur 'entry_price' ou 'price'
            price = float(position.get('current_price', position.get('entry_price', position.get('price', 0.0))))
            
            total_value += qty * price
            
        return total_value

    @staticmethod
    def calculate_max_drawdown(values: List[float]) -> float:
        """
        Calcule la perte maximale historique (Max Drawdown) depuis un sommet.
        Retourne une valeur positive (ex: 0.20 pour -20%).
        """
        if not values or len(values) < 2:
            return 0.0
        
        try:
            np_values = np.array(values)
            # Maximum cumulatif jusqu'à chaque point
            peak = np.maximum.accumulate(np_values)
            
            # Drawdown à chaque point (Valeur actuelle / Pic précédent - 1)
            drawdowns = (np_values - peak) / peak
            
            # Le Max Drawdown est le minimum (le plus négatif) de cette série
            return abs(np.min(drawdowns))
            
        except Exception:
            return 0.0

    @staticmethod
    def calculate_calmar_ratio(total_return: float, max_drawdown: float) -> float:
        """
        Calcule le ratio Calmar (Rendement Total / Max Drawdown).
        Indique si le rendement vaut le risque de crash encouru.
        """
        if max_drawdown == 0:
            return 0.0 if total_return <= 0 else 20.0 # Cap de sécurité
        return total_return / max_drawdown

    @staticmethod
    def calculate_ulcer_index(values: List[float]) -> float:
        """
        Calcule l'Ulcer Index.
        Mesure le stress de l'investisseur (prend en compte la profondeur ET la durée des baisses).
        """
        if not values or len(values) < 2:
            return 0.0
        
        try:
            np_values = np.array(values)
            peak = np.maximum.accumulate(np_values)
            drawdowns = (np_values - peak) / peak
            
            # Moyenne quadratique des drawdowns
            squared_drawdowns = drawdowns ** 2
            return np.sqrt(squared_drawdowns.mean())
        except Exception:
            return 0.0

    @staticmethod
    def calculate_win_rate(trade_log: List[Dict]) -> float:
        """Calcule le pourcentage de trades gagnants (0.0 à 1.0)"""
        if not trade_log:
            return 0.0
        
        winning_trades = [t for t in trade_log if t.get('pnl', 0) > 0]
        return len(winning_trades) / len(trade_log)

    @staticmethod
    def get_comprehensive_stats(portfolio_history: List[Dict], trade_log: List[Dict]) -> Dict:
        """
        Génère un rapport statistique complet.
        Utilisé par le bot en fin de session et par le dashboard.
        """
        if not portfolio_history:
            return {
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'total_trades': 0
            }

        values = [p.get('value', 0) for p in portfolio_history if 'value' in p]
        if not values: 
            return {}

        initial_value = values[0]
        final_value = values[-1]
        
        # Rendement absolu
        total_return = (final_value - initial_value) / initial_value if initial_value > 0 else 0
        
        # Calculs de risque
        max_dd = FinancialMetrics.calculate_max_drawdown(values)
        sharpe = FinancialMetrics.calculate_sharpe_ratio(portfolio_history)
        sortino = FinancialMetrics.calculate_sortino_ratio(portfolio_history)
        
        # Calculs de trading
        win_rate = FinancialMetrics.calculate_win_rate(trade_log)
        avg_profit = np.mean([t['pnl'] for t in trade_log]) if trade_log else 0.0

        stats = {
            'initial_capital': initial_value,
            'final_capital': final_value,
            'net_profit': final_value - initial_value,
            'total_return': total_return,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': FinancialMetrics.calculate_calmar_ratio(total_return, max_dd),
            'ulcer_index': FinancialMetrics.calculate_ulcer_index(values),
            'win_rate': win_rate,
            'total_trades': len(trade_log),
            'average_profit_per_trade': avg_profit
        }
        return stats
