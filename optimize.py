import optuna
import json
import logging
import numpy as np
from typing import Dict, Any, List
import pandas as pd
import time
import sys
import argparse
from datetime import datetime
import textwrap
import asyncio
from dataclasses import dataclass
from scipy import stats

# Import du AdaptiveBacktester et des Stratégies
from backtest import AdaptiveBacktester
from strategies.strategies import (
    MeanReversion, 
    MA_Enhanced, 
    Momentum_Enhanced, 
    MeanReversion_Pro, 
    MA_Momentum_Hybrid, 
    Volatility_Regime_Adaptive
)

# --- Configuration logging minimaliste ---
logging.basicConfig(level=logging.WARNING, format='%(message)s')
logger = logging.getLogger("PhoenixOptimizer")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==============================================================================
# DATACLASS FOR SCORING METRICS
# ==============================================================================

@dataclass
class StrategyMetrics:
    """Container for strategy performance metrics"""
    sharpe_ratio: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_win_loss_ratio: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0
    volatility: float = 0.0
    
    @property
    def has_sufficient_data(self) -> bool:
        """Check if we have enough trades for valid evaluation"""
        return self.total_trades >= 5
    
    @property
    def is_profitable(self) -> bool:
        """Check if strategy is profitable"""
        return self.total_return > 0
    
    @property
    def is_acceptable_risk(self) -> bool:
        """Check if risk metrics are acceptable"""
        return (abs(self.max_drawdown) < 0.5 and  # Less than 50% drawdown
                self.sharpe_ratio > -1.0 and      # Not catastrophically bad
                self.total_trades > 0)            # At least some trades

# ==============================================================================
# OPTIMIZATION SCORING SYSTEM
# ==============================================================================

class ScoringSystem:
    """Advanced scoring system for strategy optimization"""
    
    @staticmethod
    def calculate_score(metrics: StrategyMetrics) -> float:
        """
        Calculate comprehensive strategy score with multiple components.
        
        Components (normalized to 0-1 range where possible):
        1. Risk-Adjusted Return (Sharpe/Sortino) - 30%
        2. Profitability & Growth - 25%
        3. Risk Management (Drawdown) - 20%
        4. Trade Quality (Win Rate, Profit Factor) - 15%
        5. Statistical Significance - 10%
        """
        
        if not metrics.has_sufficient_data:
            return -100.0  # Heavy penalty for insufficient trades
        
        if not metrics.is_acceptable_risk:
            return -50.0   # Penalty for unacceptable risk
        
        # Component 1: Risk-Adjusted Return (0-30 points)
        # Use Sortino ratio if available, otherwise Sharpe
        risk_adj_return = metrics.sortino_ratio if hasattr(metrics, 'sortino_ratio') and metrics.sortino_ratio != 0 else metrics.sharpe_ratio
        
        # Normalize Sharpe/Sortino: Good strategies > 1, Great > 2
        risk_score = ScoringSystem._normalize_risk_adj_return(risk_adj_return)
        
        # Component 2: Profitability & Growth (0-25 points)
        # Annualized return if we have timeframe info, otherwise use total return
        return_score = ScoringSystem._calculate_return_score(metrics.total_return)
        
        # Component 3: Risk Management (0-20 points)
        # Penalize drawdowns heavily
        risk_mgmt_score = ScoringSystem._calculate_risk_score(metrics.max_drawdown)
        
        # Component 4: Trade Quality (0-15 points)
        trade_quality_score = ScoringSystem._calculate_trade_quality_score(
            metrics.win_rate, 
            metrics.profit_factor,
            metrics.avg_win_loss_ratio
        )
        
        # Component 5: Statistical Significance (0-10 points)
        # Reward strategies with more trades (but not too many to avoid overfitting)
        stat_score = ScoringSystem._calculate_statistical_score(metrics.total_trades)
        
        # Combined weighted score
        total_score = (
            risk_score * 0.30 +
            return_score * 0.25 +
            risk_mgmt_score * 0.20 +
            trade_quality_score * 0.15 +
            stat_score * 0.10
        )
        
        # Apply bonuses/penalties
        total_score = ScoringSystem._apply_modifiers(total_score, metrics)
        
        return total_score
    
    @staticmethod
    def _normalize_risk_adj_return(value: float) -> float:
        """Normalize Sharpe/Sortino ratio to 0-1 scale"""
        # Good strategies: 1-2 = 0.6-0.8, Excellent: >2 = 0.8-1.0
        # Acceptable: 0-1 = 0.4-0.6, Bad: <0 = 0-0.4
        if value > 3.0:
            return 1.0
        elif value > 2.0:
            return 0.8 + (value - 2.0) * 0.2
        elif value > 1.0:
            return 0.6 + (value - 1.0) * 0.2
        elif value > 0.0:
            return 0.4 + value * 0.2
        elif value > -1.0:
            return 0.2 + (value + 1.0) * 0.2
        else:
            return max(0.0, 0.2 + (value + 1.0) * 0.1)  # Cap at 0
    
    @staticmethod
    def _calculate_return_score(total_return: float) -> float:
        """Calculate normalized return score"""
        # 100% return = 1.0, 0% = 0.5, -50% = 0.0
        if total_return > 5.0:  # 500% return
            return 1.0
        elif total_return > 1.0:  # 100%+ return
            return 0.8 + min(0.2, (total_return - 1.0) / 20.0)
        elif total_return > 0.0:
            return 0.5 + total_return * 0.3  # 0-100% return gives 0.5-0.8
        else:
            return max(0.0, 0.5 + total_return)  # Negative returns reduce score
    
    @staticmethod
    def _calculate_risk_score(max_drawdown: float) -> float:
        """Calculate risk management score based on drawdown"""
        drawdown_abs = abs(max_drawdown)
        
        if drawdown_abs < 0.05:  # <5% drawdown
            return 1.0
        elif drawdown_abs < 0.10:  # <10% drawdown
            return 0.9
        elif drawdown_abs < 0.20:  # <20% drawdown
            return 0.7
        elif drawdown_abs < 0.30:  # <30% drawdown
            return 0.5
        elif drawdown_abs < 0.40:  # <40% drawdown
            return 0.3
        elif drawdown_abs < 0.50:  # <50% drawdown
            return 0.1
        else:
            return 0.0  # >50% drawdown = unacceptable
    
    @staticmethod
    def _calculate_trade_quality_score(win_rate: float, profit_factor: float, avg_win_loss_ratio: float) -> float:
        """Calculate trade quality score"""
        # Win rate component (0-7.5 points)
        win_rate_score = 0.0
        if win_rate > 0.65:  # >65% win rate
            win_rate_score = 7.5
        elif win_rate > 0.55:  # 55-65% win rate
            win_rate_score = 5.0 + (win_rate - 0.55) * 25.0
        elif win_rate > 0.45:  # 45-55% win rate
            win_rate_score = 2.5 + (win_rate - 0.45) * 25.0
        else:
            win_rate_score = win_rate * 5.0
        
        # Profit factor component (0-7.5 points)
        profit_factor_score = 0.0
        if profit_factor > 2.0:  # Excellent
            profit_factor_score = 7.5
        elif profit_factor > 1.5:  # Good
            profit_factor_score = 5.0 + (profit_factor - 1.5) * 5.0
        elif profit_factor > 1.2:  # Acceptable
            profit_factor_score = 2.5 + (profit_factor - 1.2) * 8.33
        elif profit_factor > 1.0:  # Barely profitable
            profit_factor_score = (profit_factor - 1.0) * 12.5
        else:
            profit_factor_score = 0.0
        
        return (win_rate_score + profit_factor_score) / 15.0  # Normalize to 0-1
    
    @staticmethod
    def _calculate_statistical_score(total_trades: int) -> float:
        """Calculate statistical significance score"""
        # More trades = more reliable, but don't reward excessive trading
        if total_trades >= 50:  # Excellent sample size
            return 1.0
        elif total_trades >= 30:
            return 0.8
        elif total_trades >= 20:
            return 0.6
        elif total_trades >= 10:
            return 0.4
        elif total_trades >= 5:
            return 0.2
        else:
            return 0.0
    
    @staticmethod
    def _apply_modifiers(score: float, metrics: StrategyMetrics) -> float:
        """Apply bonus/penalty modifiers based on additional criteria"""
        final_score = score
        
        # Bonus for consistency (high win rate AND high profit factor)
        if metrics.win_rate > 0.55 and metrics.profit_factor > 1.5:
            final_score *= 1.1
        
        # Penalty for high volatility without corresponding returns
        if hasattr(metrics, 'volatility') and metrics.volatility > 0.05 and metrics.total_return < 0.1:
            final_score *= 0.8
        
        # Bonus for low drawdown with decent returns
        if abs(metrics.max_drawdown) < 0.1 and metrics.total_return > 0.2:
            final_score *= 1.15
        
        # Cap score at reasonable range
        return max(-100.0, min(100.0, final_score * 100.0))  # Scale to 0-100 range

# ==============================================================================
# SYSTEME D'AFFICHAGE PROFESSIONNEL
# ==============================================================================

class Color:
    """Codes couleur ANSI pour un terminal moderne"""
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    GRAY = '\033[90m'
    RESET = '\033[0m'

class PhoenixOptimizer:
    def __init__(self, config_path='config.json'):
        self.config = self._load_config(config_path)
        self.strategy_map = {
            "MeanReversion": MeanReversion,
            "MA_Enhanced": MA_Enhanced,
            "Momentum_Enhanced": Momentum_Enhanced,
            "MeanReversion_Pro": MeanReversion_Pro,
            "MA_Momentum_Hybrid": MA_Momentum_Hybrid,
            "Volatility_Regime_Adaptive": Volatility_Regime_Adaptive
        }
        self.scoring_system = ScoringSystem()

    def _load_config(self, path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except:
            return {}

    def _extract_metrics(self, metrics_dict: Dict[str, Any]) -> StrategyMetrics:
        """Extract metrics from backtest results into StrategyMetrics object"""
        return StrategyMetrics(
            sharpe_ratio=metrics_dict.get('sharpe_ratio', 0.0),
            total_return=metrics_dict.get('total_return', 0.0),
            max_drawdown=metrics_dict.get('max_drawdown', 0.0),
            win_rate=metrics_dict.get('win_rate', 0.0),
            profit_factor=metrics_dict.get('profit_factor', 0.0),
            total_trades=metrics_dict.get('total_trades', 0),
            avg_win_loss_ratio=metrics_dict.get('avg_win_loss_ratio', 0.0),
            calmar_ratio=metrics_dict.get('calmar_ratio', 0.0),
            sortino_ratio=metrics_dict.get('sortino_ratio', 0.0),
            volatility=metrics_dict.get('volatility', 0.0)
        )

    def objective(self, trial, strategy_cls, pairs, fast_mode=False):
        """
        Fonction objectif pour Optuna avec scoring amélioré.
        """
        # 1. Génération des hyperparamètres via la méthode statique de la stratégie
        try:
            params = strategy_cls.get_optuna_params(trial)
        except NotImplementedError:
            return -float('inf')
        
        # 2. Validation supplémentaire des paramètres
        try:
            bounds = strategy_cls.get_param_bounds()
            for param_name, (min_val, max_val, _) in bounds.items():
                if param_name in params:
                    # Clip values to bounds
                    if isinstance(params[param_name], (int, float)):
                        params[param_name] = max(min_val, min(params[param_name], max_val))
        except:
            pass
        
        # 3. Créer une nouvelle instance de backtester pour chaque essai
        try:
            backtester = AdaptiveBacktester()
        except Exception as e:
            print(f"{Color.RED}❌ Failed to create backtester: {e}{Color.RESET}")
            return -float('inf')
        
        total_score = 0.0
        valid_runs = 0
        all_metrics = []
        
        # 4. Boucle sur les paires
        test_pairs = pairs[:1] if fast_mode else pairs  # Limit pairs in fast mode
        
        for pair in test_pairs:
            try:
                # Run backtest asynchronously
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Use minimal days for fast mode
                days = "14" if fast_mode else "90"  # More data for better evaluation
                
                metrics = loop.run_until_complete(
                    backtester.run_backtest(strategy_cls.__name__, pair, days, params)
                )
                
                loop.close()
                
                if 'error' in metrics:
                    continue
                
                # 5. Extraire les métriques et calculer le score
                strategy_metrics = self._extract_metrics(metrics)
                
                # Skip if insufficient data
                if not strategy_metrics.has_sufficient_data:
                    continue
                
                # Calculate comprehensive score
                score = self.scoring_system.calculate_score(strategy_metrics)
                
                # Store metrics for potential analysis
                all_metrics.append(strategy_metrics)
                
                total_score += score
                valid_runs += 1
                
                # Early stopping for terrible parameters
                if score < -50 and valid_runs >= 2:
                    return -float('inf')
                
            except Exception as e:
                print(f"{Color.YELLOW}⚠️ Error testing {pair}: {e}{Color.RESET}")
                continue
        
        # 6. Return average score across pairs
        if valid_runs == 0:
            return -float('inf')
        
        # Calculate average score
        avg_score = total_score / valid_runs
        
        # Additional penalty for high variability across pairs (if multiple pairs tested)
        if len(all_metrics) >= 3:
            # Check consistency across pairs
            returns = [m.total_return for m in all_metrics]
            if np.std(returns) > 0.2:  # High variability in returns
                avg_score *= 0.8
        
        return avg_score

    def run_optimization(self, strategy_name: str, n_trials=100, fast_mode=False):
        """Lance l'optimisation Optuna avec améliorations."""
        if strategy_name not in self.strategy_map:
            print(f"{Color.RED}❌ Stratégie inconnue: {strategy_name}{Color.RESET}")
            return

        strategy_cls = self.strategy_map[strategy_name]
        
        # Sélection des paires
        if fast_mode:
            pairs = ['BTC/USD', 'ETH/USD']  # Two pairs even in fast mode for better evaluation
        else:
            pairs = self.config.get('trading', {}).get('pairs', 
                     ["BTC/USD","ETH/USD","SOL/USD","BNB/USD","XRP/USD","ADA/USD","DOGE/USD"])

        print(f"\n{Color.BOLD}{Color.PURPLE}🚀 Démarrage Optimisation: {strategy_name}{Color.RESET}")
        print(f"{Color.GRAY}Target: {len(pairs)} paires | Trials: {n_trials} | Fast Mode: {fast_mode}{Color.RESET}")
        print(f"{Color.GRAY}Scoring: Risk-Adjusted Return (30%) + Profitability (25%) + Risk Mgmt (20%) + Trade Quality (15%) + Stats (10%){Color.RESET}")

        # Create study with storage for persistence
        study_name = f"{strategy_name}_optimization"
        storage_name = f"sqlite:///optuna_{study_name}.db"
        
        # Configure sampler with more exploration early on
        sampler = optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=min(20, n_trials // 5),  # Exploration phase
            multivariate=True,
            group=True
        )
        
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
            direction="maximize",
            sampler=sampler
        )
        
        try:
            # Track progress and best score
            best_score = -float('inf')
            
            def progress_callback(study, trial):
                nonlocal best_score
                
                if trial.number % 5 == 0:
                    current_best = study.best_value
                    
                    if current_best > best_score:
                        improvement = current_best - best_score if best_score != -float('inf') else 0
                        print(f"{Color.GREEN}   ✓ Trial {trial.number}/{n_trials} - New Best: {current_best:.2f} (+{improvement:.2f}){Color.RESET}")
                        best_score = current_best
                    else:
                        print(f"{Color.GRAY}   Trial {trial.number}/{n_trials} - Best: {current_best:.2f}{Color.RESET}")
            
            # Run optimization
            study.optimize(
                lambda trial: self.objective(trial, strategy_cls, pairs, fast_mode), 
                n_trials=n_trials,
                callbacks=[progress_callback],
                show_progress_bar=True,
                gc_after_trial=True  # Clean up memory
            )
        except KeyboardInterrupt:
            print(f"\n{Color.YELLOW}⚠️ Optimisation interrompue.{Color.RESET}")
            return self._analyze_partial_results(study, strategy_name)
        except Exception as e:
            print(f"\n{Color.RED}❌ Optimization error: {e}{Color.RESET}")
            return

        # Display comprehensive results
        self._display_optimization_results(study, strategy_name)

    def _display_optimization_results(self, study, strategy_name):
        """Display comprehensive optimization results"""
        print(f"\n{Color.GREEN}{Color.BOLD}🏆 OPTIMIZATION COMPLETE{Color.RESET}")
        print(f"{Color.GREEN}Best Score: {study.best_value:.2f}/100{Color.RESET}")
        
        # Score interpretation
        if study.best_value >= 80:
            score_emoji = "🎯"
            score_desc = "EXCELLENT"
        elif study.best_value >= 60:
            score_emoji = "⭐"
            score_desc = "GOOD"
        elif study.best_value >= 40:
            score_emoji = "📊"
            score_desc = "FAIR"
        elif study.best_value >= 20:
            score_emoji = "⚠️"
            score_desc = "WEAK"
        else:
            score_emoji = "❌"
            score_desc = "POOR"
        
        print(f"{Color.CYAN}Rating: {score_emoji} {score_desc}{Color.RESET}")
        print(f"\n{Color.CYAN}Best Parameters:{Color.RESET}")
        
        for param, value in study.best_params.items():
            print(f"  {param}: {value}")
        
        # Display trial statistics
        trials_df = study.trials_dataframe()
        if not trials_df.empty:
            print(f"\n{Color.GRAY}Optimization Statistics:{Color.RESET}")
            print(f"  Completed trials: {len(trials_df)}")
            print(f"  Mean score: {trials_df['value'].mean():.2f}")
            print(f"  Std score: {trials_df['value'].std():.2f}")
            print(f"  Best score percentile: {(study.best_value - trials_df['value'].min()) / (trials_df['value'].max() - trials_df['value'].min()) * 100:.1f}%")
            
            # Count successful trials (score > 0)
            successful = len(trials_df[trials_df['value'] > 0])
            print(f"  Successful trials (score > 0): {successful}/{len(trials_df)} ({successful/len(trials_df)*100:.1f}%)")
        
        # Save results
        self.save_best_params(strategy_name, study.best_params, study.best_value)

    def _analyze_partial_results(self, study, strategy_name):
        """Analyze partial results when optimization is interrupted"""
        if len(study.trials) == 0:
            print(f"{Color.YELLOW}No trials completed.{Color.RESET}")
            return
        
        best_trial = study.best_trial
        print(f"\n{Color.YELLOW}📊 PARTIAL RESULTS (Interrupted){Color.RESET}")
        print(f"{Color.GRAY}Best score from {len(study.trials)} trials: {best_trial.value:.2f}{Color.RESET}")
        
        if best_trial.value > 0:
            self.save_best_params(strategy_name, best_trial.params, best_trial.value)
            print(f"{Color.GREEN}Partial results saved.{Color.RESET}")

    def run_strategy_test(self, strategy_name: str, params: Dict = None):
        """
        Exécute un test unitaire rapide de la stratégie avec analyse détaillée.
        """
        if strategy_name not in self.strategy_map:
            print(f"{Color.RED}❌ Stratégie inconnue: {strategy_name}{Color.RESET}")
            return

        print(f"\n{Color.CYAN}{Color.BOLD}🔬 STRATEGY TEST: {strategy_name}{Color.RESET}")
        
        try:
            # Create backtester
            backtester = AdaptiveBacktester()
            
            # Run test
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Test with first pair
            test_pair = backtester.config['trading']['pairs'][0] if backtester.config['trading']['pairs'] else "BTC/USD"
            print(f"{Color.GRAY}Testing on {test_pair} with 30 days of data...{Color.RESET}")
            
            # Use provided params or default
            test_params = params or {}
            
            metrics = loop.run_until_complete(
                backtester.run_backtest(strategy_name, test_pair, "30", test_params)
            )
            
            loop.close()
            
            if 'error' in metrics:
                print(f"{Color.RED}❌ Test failed: {metrics['error']}{Color.RESET}")
                return
            
            # Extract metrics
            strategy_metrics = self._extract_metrics(metrics)
            
            # Calculate score
            score = self.scoring_system.calculate_score(strategy_metrics)
            
            # Display comprehensive results
            self._display_test_results(strategy_metrics, score)
            
        except Exception as e:
            print(f"{Color.RED}❌ Test error: {e}{Color.RESET}")

    def _display_test_results(self, metrics: StrategyMetrics, score: float):
        """Display detailed test results"""
        print(f"\n{Color.GREEN}✅ TEST RESULTS (Score: {score:.2f}/100):{Color.RESET}")
        print(f"{Color.GRAY}{'─'*50}{Color.RESET}")
        
        # Profitability metrics
        print(f"\n{Color.BOLD}📈 Profitability:{Color.RESET}")
        print(f"  Total Return:    {metrics.total_return*100:+.2f}%")
        print(f"  Win Rate:        {metrics.win_rate*100:.1f}%")
        print(f"  Profit Factor:   {metrics.profit_factor:.2f}")
        print(f"  Avg Win/Loss:    {metrics.avg_win_loss_ratio:.2f}")
        
        # Risk metrics
        print(f"\n{Color.BOLD}⚠️  Risk Metrics:{Color.RESET}")
        print(f"  Max Drawdown:    {abs(metrics.max_drawdown)*100:.2f}%")
        print(f"  Sharpe Ratio:    {metrics.sharpe_ratio:.3f}")
        if hasattr(metrics, 'sortino_ratio') and metrics.sortino_ratio != 0:
            print(f"  Sortino Ratio:   {metrics.sortino_ratio:.3f}")
        if hasattr(metrics, 'calmar_ratio') and metrics.calmar_ratio != 0:
            print(f"  Calmar Ratio:    {metrics.calmar_ratio:.3f}")
        
        # Activity metrics
        print(f"\n{Color.BOLD}📊 Activity:{Color.RESET}")
        print(f"  Total Trades:    {metrics.total_trades}")
        
        # Evaluation
        print(f"\n{Color.BOLD}📋 Evaluation:{Color.RESET}")
        if score >= 80:
            print(f"{Color.GREEN}  🎯 EXCELLENT: Strategy shows strong potential{Color.RESET}")
        elif score >= 60:
            print(f"{Color.GREEN}  ⭐ GOOD: Strategy shows promise{Color.RESET}")
        elif score >= 40:
            print(f"{Color.YELLOW}  📊 FAIR: Strategy needs improvement{Color.RESET}")
        elif score >= 20:
            print(f"{Color.YELLOW}  ⚠️  WEAK: Strategy has significant issues{Color.RESET}")
        else:
            print(f"{Color.RED}  ❌ POOR: Strategy is not viable{Color.RESET}")
        
        # Recommendations
        print(f"\n{Color.BOLD}💡 Recommendations:{Color.RESET}")
        if metrics.total_trades < 10:
            print(f"  • Need more trades for reliable evaluation")
        if abs(metrics.max_drawdown) > 0.2:
            print(f"  • Reduce maximum drawdown (currently {abs(metrics.max_drawdown)*100:.1f}%)")
        if metrics.sharpe_ratio < 1.0:
            print(f"  • Improve risk-adjusted returns (Sharpe: {metrics.sharpe_ratio:.2f})")
        if metrics.win_rate < 0.4:
            print(f"  • Improve win rate (currently {metrics.win_rate*100:.1f}%)")

    def save_best_params(self, strategy_name: str, params: Dict, score: float):
        """Sauvegarde les meilleurs paramètres dans config.json."""
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
            
            if 'strategies' not in config:
                config['strategies'] = {}
            if 'parameters' not in config['strategies']:
                config['strategies']['parameters'] = {}
            if 'optimized' not in config['strategies']:
                config['strategies']['optimized'] = {}
            
            # Save to parameters section
            config['strategies']['parameters'][strategy_name] = {
                **config['strategies']['parameters'].get(strategy_name, {}),
                **params
            }
            
            # Save to optimized section with metadata
            safe_score = score if score != float('-inf') else -999.0
            config['strategies']['optimized'][strategy_name] = {
                "parameters": params,
                "score": safe_score,
                "optimized_at": datetime.now().isoformat(),
                "score_components": {
                    "total": safe_score,
                    "rating": self._get_score_rating(safe_score)
                }
            }
            
            with open('config.json', 'w') as f:
                json.dump(config, f, indent=4)
                
            print(f"{Color.BLUE}💾 Parameters saved to config.json{Color.RESET}")
            
            # Also save to database if available
            try:
                from database import Database
                db = Database()
                
                # Find strategy ID
                strategies_res = db.client.table("strategies").select("id").eq("name", strategy_name).execute()
                if strategies_res.data:
                    strategy_id = strategies_res.data[0]['id']
                    
                    # Save to strategy_parameters table
                    db.save_strategy_parameters(
                        strategy_id=strategy_id,
                        parameters=params,
                        performance={"optimization_score": score}
                    )
                    print(f"{Color.BLUE}💾 Parameters saved to database{Color.RESET}")
            except:
                pass  # Database save is optional
            
        except Exception as e:
            print(f"{Color.RED}❌ Save error: {e}{Color.RESET}")

    def _get_score_rating(self, score: float) -> str:
        """Convert score to rating"""
        if score >= 80:
            return "EXCELLENT"
        elif score >= 60:
            return "GOOD"
        elif score >= 40:
            return "FAIR"
        elif score >= 20:
            return "WEAK"
        else:
            return "POOR"

    def interactive_menu(self):
        while True:
            print(f"\n{Color.BOLD}{Color.PURPLE}🛡️  PHOENIX OPTIMIZER (Advanced Scoring){Color.RESET}")
            print(f"{Color.GRAY}═══════════════════════════════════════════════════════{Color.RESET}")
            print("1. Optimize a strategy")
            print("2. Optimize all strategies (sequential)")
            print("3. Test a strategy (Detailed Analysis)")
            print("4. View optimization history")
            print("5. Exit")
            
            choice = input(f"\n{Color.CYAN}Choice > {Color.RESET}").strip()
            
            if choice == '1':
                print(f"\n{Color.CYAN}Available strategies:{Color.RESET}")
                for i, s_name in enumerate(self.strategy_map.keys(), 1):
                    print(f"  {i}. {s_name}")
                
                s_choice = input(f"\n{Color.CYAN}Strategy name or number > {Color.RESET}").strip()
                
                # Handle numeric input
                if s_choice.isdigit():
                    idx = int(s_choice) - 1
                    s_names = list(self.strategy_map.keys())
                    if 0 <= idx < len(s_names):
                        s_name = s_names[idx]
                    else:
                        print(f"{Color.RED}❌ Invalid number{Color.RESET}")
                        continue
                else:
                    s_name = s_choice
                
                if s_name in self.strategy_map:
                    trials = input(f"Trials (default: 100) > ").strip()
                    trials = int(trials) if trials.isdigit() else 100
                    
                    fast = input("Fast mode? (y/n, default: n) > ").strip().lower()
                    fast_mode = fast == 'y'
                    
                    self.run_optimization(s_name, n_trials=trials, fast_mode=fast_mode)
                else:
                    print(f"{Color.RED}❌ Unknown strategy: {s_name}{Color.RESET}")
                    
            elif choice == '2':
                print(f"\n{Color.YELLOW}⚠️ Sequential optimization of ALL strategies{Color.RESET}")
                print(f"{Color.GRAY}Each strategy will be optimized with 50 trials in fast mode{Color.RESET}")
                confirm = input("Continue? (y/n) > ").strip().lower()
                if confirm == 'y':
                    for s_name in self.strategy_map.keys():
                        print(f"\n{Color.PURPLE}{'='*60}{Color.RESET}")
                        self.run_optimization(s_name, n_trials=50, fast_mode=True)
                        time.sleep(2)  # Brief pause between strategies
            elif choice == '3':
                s_name = input("Strategy name > ").strip()
                if s_name in self.strategy_map:
                    # Check if we have optimized parameters
                    try:
                        with open('config.json', 'r') as f:
                            config = json.load(f)
                        optimized = config.get('strategies', {}).get('optimized', {}).get(s_name, {})
                        if optimized and 'parameters' in optimized:
                            use_optimized = input(f"Use optimized parameters? (y/n, default: y) > ").strip().lower()
                            if use_optimized != 'n':
                                params = optimized['parameters']
                                print(f"{Color.GRAY}Using optimized parameters (score: {optimized.get('score', 0):.2f}){Color.RESET}")
                                self.run_strategy_test(s_name, params)
                                continue
                    except:
                        pass
                    self.run_strategy_test(s_name)
                else:
                    print(f"{Color.RED}❌ Unknown strategy{Color.RESET}")
            elif choice == '4':
                self.view_optimization_history()
            elif choice == '5':
                print(f"\n{Color.GREEN}👋 Goodbye!{Color.RESET}")
                break
            else:
                print(f"{Color.RED}❌ Invalid choice{Color.RESET}")

    def view_optimization_history(self):
        """View past optimization results with detailed scoring."""
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
            
            optimized = config.get('strategies', {}).get('optimized', {})
            
            if not optimized:
                print(f"\n{Color.YELLOW}No optimization history found.{Color.RESET}")
                return
            
            print(f"\n{Color.BOLD}{Color.CYAN}📊 OPTIMIZATION HISTORY{Color.RESET}")
            print(f"{Color.GRAY}{'─'*60}{Color.RESET}")
            
            for strategy, data in sorted(optimized.items(), key=lambda x: x[1].get('score', 0), reverse=True):
                score = data.get('score', 0)
                rating = data.get('score_components', {}).get('rating', 'UNKNOWN')
                date = data.get('optimized_at', 'Unknown')
                
                # Color code based on score
                if score >= 60:
                    score_color = Color.GREEN
                elif score >= 40:
                    score_color = Color.YELLOW
                else:
                    score_color = Color.RED
                
                print(f"\n{Color.BOLD}{strategy}{Color.RESET}")
                print(f"  {score_color}Score: {score:.2f}/100 ({rating}){Color.RESET}")
                print(f"  {Color.GRAY}Date: {date}{Color.RESET}")
                
                params = data.get('parameters', {})
                if params:
                    print(f"  {Color.GRAY}Parameters:{Color.RESET}")
                    for param, value in params.items():
                        print(f"    {param}: {value}")
            
        except Exception as e:
            print(f"{Color.RED}❌ Error reading history: {e}{Color.RESET}")

def main():
    parser = argparse.ArgumentParser(description="Phoenix Strategy Optimizer (Advanced Scoring)")
    parser.add_argument('--strategy', type=str, help='Strategy name')
    parser.add_argument('--trials', type=int, default=100, help='Number of trials (default: 100)')
    parser.add_argument('--fast-mode', action='store_true', help='Fast optimization (less data)')
    parser.add_argument('--test', action='store_true', help='Run a single test instead of optimization')
    parser.add_argument('--list', action='store_true', help='List available strategies')
    parser.add_argument('--verbose', action='store_true', help='Detailed logs')
    
    args = parser.parse_args()
    
    optimizer = PhoenixOptimizer()
    
    if args.list:
        print(f"\n{Color.BOLD}{Color.PURPLE}Available Strategies:{Color.RESET}")
        for strategy in optimizer.strategy_map.keys():
            print(f"  • {strategy}")
        return
    
    if args.strategy:
        if args.test:
            optimizer.run_strategy_test(args.strategy)
        else:
            optimizer.run_optimization(args.strategy, n_trials=args.trials, fast_mode=args.fast_mode)
    else:
        optimizer.interactive_menu()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Color.YELLOW}⚠️ Interrupted by user{Color.RESET}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Color.RED}❌ Fatal error: {e}{Color.RESET}")
        sys.exit(1)