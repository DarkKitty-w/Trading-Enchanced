import pandas as pd
import numpy as np
from typing import List, Dict, Any

class FinancialMetrics:
    """
    Moteur de calcul des métriques financières.
    Pure functions: Input Data -> Output Metrics.
    Conçu pour travailler avec les dictionnaires générés par Backtester.
    """

    @staticmethod
    def calculate(portfolio_history: List[Dict], trades_log: List[Dict], risk_free_rate: float = 0.0) -> Dict[str, Any]:
        """
        Calcule l'ensemble des métriques de performance.
        
        Args:
            portfolio_history: Liste de dicts {'timestamp', 'equity', ...}
            trades_log: Liste de dicts {'profit', 'return_pct', ...}
            risk_free_rate: Taux sans risque (pour Sharpe)
            
        Returns:
            Dictionnaire contenant Sharpe, Sortino, Drawdown, Win Rate, etc.
        """
        results = {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "total_trades": len(trades_log),
            "profit_factor": 0.0,
            "avg_trade_return": 0.0
        }

        # --- 1. Analyse Portfolio (Equity Curve) ---
        if not portfolio_history:
            return results

        df = pd.DataFrame(portfolio_history)
        
        # Harmonisation des clés (equity vs total_equity)
        if 'equity' not in df.columns:
            if 'total_equity' in df.columns:
                df['equity'] = df['total_equity']
            else:
                return results # Impossible de calculer
        
        # Calcul des retours périodiques
        df['pct_change'] = df['equity'].pct_change().fillna(0)
        
        # 1. Total Return
        initial_equity = df['equity'].iloc[0]
        final_equity = df['equity'].iloc[-1]
        if initial_equity > 0:
            results['total_return'] = (final_equity - initial_equity) / initial_equity

        # 2. Max Drawdown
        df['cummax'] = df['equity'].cummax()
        df['drawdown'] = (df['equity'] - df['cummax']) / df['cummax']
        results['max_drawdown'] = df['drawdown'].min() # Sera négatif

        # 3. Sharpe Ratio
        mean_return = df['pct_change'].mean()
        std_return = df['pct_change'].std()
        
        if std_return > 0:
            # Note: Sans connaître la fréquence exacte (1h, 4h, 1d), 
            # on fournit un Sharpe par période (ex: par bougie).
            # Pour annualiser, il faudrait multiplier par sqrt(N_periodes_par_an).
            # Ici, on normalise simplement par la racine de l'échantillon pour la stabilité.
            results['sharpe_ratio'] = (mean_return - risk_free_rate) / std_return * np.sqrt(len(df))

        # 4. Sortino Ratio (Downside Deviation)
        negative_returns = df.loc[df['pct_change'] < 0, 'pct_change']
        if not negative_returns.empty:
            downside_std = negative_returns.std()
            if downside_std > 0:
                 results['sortino_ratio'] = (mean_return - risk_free_rate) / downside_std * np.sqrt(len(df))

        # --- 2. Analyse des Trades ---
        if trades_log:
            trades_df = pd.DataFrame(trades_log)
            
            # Win Rate
            winners = trades_df[trades_df['profit'] > 0]
            losers = trades_df[trades_df['profit'] <= 0]
            
            results['win_rate'] = len(winners) / len(trades_df)
            
            # Profit Factor
            gross_profit = winners['profit'].sum()
            gross_loss = abs(losers['profit'].sum())
            
            if gross_loss > 0:
                results['profit_factor'] = gross_profit / gross_loss
            else:
                results['profit_factor'] = float('inf') if gross_profit > 0 else 0.0
                
            # Avg Return
            results['avg_trade_return'] = trades_df['return_pct'].mean()

        return results