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
        """
        results = {
            "total_return": 0.0,
            "total_return_pct": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "profit_factor": 0.0,
            "avg_win_loss_ratio": 0.0
        }

        # --- 1. Analyse Portfolio (Equity Curve) ---
        if not portfolio_history:
            return results

        df = pd.DataFrame(portfolio_history)
        if 'equity' not in df.columns:
            if 'total_equity' in df.columns:
                df['equity'] = df['total_equity']
            else:
                return results

        if df.empty or len(df) < 2:
            return results

        df['pct_change'] = df['equity'].pct_change().fillna(0)
        
        initial_equity = df['equity'].iloc[0]
        final_equity = df['equity'].iloc[-1]
        if initial_equity > 0:
            results['total_return'] = (final_equity - initial_equity) / initial_equity
            results['total_return_pct'] = results['total_return'] * 100

        df['cummax'] = df['equity'].cummax()
        df['drawdown'] = (df['equity'] - df['cummax']) / df['cummax']
        results['max_drawdown'] = df['drawdown'].min()

        mean_return = df['pct_change'].mean()
        std_return = df['pct_change'].std()
        
        if std_return > 0:
            # Assuming daily returns for annualization
            annualization_factor = np.sqrt(252) # Assuming 252 trading days in a year
            results['sharpe_ratio'] = (mean_return / std_return) * annualization_factor

        negative_returns = df.loc[df['pct_change'] < 0, 'pct_change']
        if not negative_returns.empty:
            downside_std = negative_returns.std()
            if downside_std > 0:
                 results['sortino_ratio'] = (mean_return / downside_std) * annualization_factor

        # --- 2. Analyse des Trades ---
        if trades_log:
            # Filter for SELL trades, which contain the profit information for a round trip
            sell_trades = [t for t in trades_log if t.get('side') == 'SELL' and 'profit' in t]
            results['total_trades'] = len(sell_trades)
            
            if sell_trades:
                trades_df = pd.DataFrame(sell_trades)
                
                winners = trades_df[trades_df['profit'] > 0]
                losers = trades_df[trades_df['profit'] <= 0]
                
                if not trades_df.empty:
                    results['win_rate'] = len(winners) / len(trades_df)
                
                gross_profit = winners['profit'].sum()
                gross_loss = abs(losers['profit'].sum())
                
                if gross_loss > 0:
                    results['profit_factor'] = gross_profit / gross_loss
                else:
                    results['profit_factor'] = float('inf') if gross_profit > 0 else 1.0

                if not winners.empty and not losers.empty:
                    avg_win = gross_profit / len(winners)
                    avg_loss = gross_loss / len(losers)
                    if avg_loss > 0:
                        results['avg_win_loss_ratio'] = avg_win / avg_loss
                elif not winners.empty:
                    results['avg_win_loss_ratio'] = float('inf')
        
        return results