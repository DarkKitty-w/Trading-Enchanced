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

# Import du Backtester et des Stratégies
from backtest import Backtester
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

    def _load_config(self, path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except:
            return {}

    def objective(self, trial, strategy_cls, pairs, fast_mode=False):
        """
        Fonction objectif pour Optuna.
        """
        # 1. Génération des hyperparamètres via la méthode statique de la stratégie
        try:
            params = strategy_cls.get_optuna_params(trial)
        except NotImplementedError:
            return -float('inf')
        
        # 2. Créer une nouvelle instance de backtester pour chaque essai
        try:
            backtester = Backtester()
        except Exception as e:
            print(f"{Color.RED}❌ Failed to create backtester: {e}{Color.RESET}")
            return -float('inf')
        
        total_score = 0.0
        valid_runs = 0
        
        # 3. Boucle sur les paires
        for pair in pairs[:2] if fast_mode else pairs:  # Limit pairs in fast mode
            try:
                # Run backtest asynchronously
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Use minimal days for fast mode
                days = "7" if fast_mode else "30"
                
                metrics = loop.run_until_complete(
                    backtester.run_backtest(strategy_cls.__name__, pair, days, params)
                )
                
                loop.close()
                
                if 'error' in metrics:
                    continue
                
                # 4. Calcul du Score (Sharpe * Return) avec pénalités
                sharpe = metrics.get('sharpe_ratio', 0)
                total_ret = metrics.get('total_return', 0)
                max_dd = abs(metrics.get('max_drawdown', 0))
                win_rate = metrics.get('win_rate', 0)
                
                # Score components
                sharpe_component = max(sharpe, -2)  # Cap negative sharpe
                return_component = 1 + total_ret
                dd_penalty = 1 - (max_dd * 2)  # Heavy penalty for drawdowns
                win_rate_bonus = 1 + (win_rate - 0.5)  # Bonus for >50% win rate
                
                # Combined score
                score = sharpe_component * return_component * dd_penalty * win_rate_bonus
                
                # Additional penalty for very low trades
                total_trades = metrics.get('total_trades', 0)
                if total_trades < 5:
                    score *= 0.5
                
                total_score += score
                valid_runs += 1
                
            except Exception as e:
                print(f"{Color.YELLOW}⚠️ Error testing {pair}: {e}{Color.RESET}")
                continue
        
        # Moyenne des scores sur les paires
        if valid_runs == 0:
            return -float('inf')
            
        final_score = total_score / valid_runs
        return final_score

    def run_optimization(self, strategy_name: str, n_trials=50, fast_mode=False):
        """Lance l'optimisation Optuna."""
        if strategy_name not in self.strategy_map:
            print(f"{Color.RED}❌ Stratégie inconnue: {strategy_name}{Color.RESET}")
            return

        strategy_cls = self.strategy_map[strategy_name]
        
        # Sélection des paires
        if fast_mode:
            pairs = ['BTC/USD']  # Une seule paire pour aller vite
        else:
            pairs = self.config.get('trading', {}).get('pairs', ['BTC/USD', 'ETH/USD', 'SOL/USD'])

        print(f"\n{Color.BOLD}{Color.PURPLE}🚀 Démarrage Optimisation: {strategy_name}{Color.RESET}")
        print(f"{Color.GRAY}Target: {len(pairs)} paires | Trials: {n_trials} | Fast Mode: {fast_mode}{Color.RESET}")

        # Create study with storage for persistence
        study_name = f"{strategy_name}_optimization"
        storage_name = f"sqlite:///optuna_{study_name}.db"
        
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        try:
            # Add stream callback for progress
            def progress_callback(study, trial):
                if trial.number % 5 == 0:
                    print(f"{Color.GRAY}   Trial {trial.number}/{n_trials} - Best: {study.best_value:.4f}{Color.RESET}")
            
            study.optimize(
                lambda trial: self.objective(trial, strategy_cls, pairs, fast_mode), 
                n_trials=n_trials,
                callbacks=[progress_callback],
                show_progress_bar=True
            )
        except KeyboardInterrupt:
            print(f"\n{Color.YELLOW}⚠️ Optimisation interrompue.{Color.RESET}")
            return
        except Exception as e:
            print(f"\n{Color.RED}❌ Optimization error: {e}{Color.RESET}")
            return

        print(f"\n{Color.GREEN}{Color.BOLD}🏆 OPTIMIZATION COMPLETE{Color.RESET}")
        print(f"{Color.GREEN}Best Score: {study.best_value:.4f}{Color.RESET}")
        print(f"{Color.CYAN}Best Parameters:{Color.RESET}")
        
        for param, value in study.best_params.items():
            print(f"  {param}: {value}")
        
        # Display trial statistics
        trials_df = study.trials_dataframe()
        if not trials_df.empty:
            print(f"\n{Color.GRAY}Trial Statistics:{Color.RESET}")
            print(f"  Completed trials: {len(trials_df)}")
            print(f"  Mean score: {trials_df['value'].mean():.4f}")
            print(f"  Std score: {trials_df['value'].std():.4f}")
        
        # Sauvegarde
        self.save_best_params(strategy_name, study.best_params, study.best_value)

    def run_strategy_test(self, strategy_name: str):
        """
        Exécute un test unitaire rapide de la stratégie.
        """
        if strategy_name not in self.strategy_map:
            print(f"{Color.RED}❌ Stratégie inconnue: {strategy_name}{Color.RESET}")
            return

        print(f"\n{Color.CYAN}{Color.BOLD}🔬 STRATEGY TEST: {strategy_name}{Color.RESET}")
        
        try:
            # Create backtester
            backtester = Backtester()
            
            # Run test
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Test with first pair
            test_pair = backtester.config['trading']['pairs'][0] if backtester.config['trading']['pairs'] else "BTC/USD"
            print(f"{Color.GRAY}Testing on {test_pair} with 7 days of data...{Color.RESET}")
            
            metrics = loop.run_until_complete(
                backtester.run_backtest(strategy_name, test_pair, days="7")
            )
            
            loop.close()
            
            if 'error' in metrics:
                print(f"{Color.RED}❌ Test failed: {metrics['error']}{Color.RESET}")
                return
            
            # Display results
            print(f"\n{Color.GREEN}✅ TEST RESULTS:{Color.RESET}")
            print(f"  Sharpe Ratio:    {metrics['sharpe_ratio']:.3f}")
            print(f"  Total Return:    {metrics['total_return_pct']:+.2f}%")
            print(f"  Max Drawdown:    {metrics['max_drawdown']*100:.2f}%")
            print(f"  Win Rate:        {metrics['win_rate']*100:.1f}%")
            print(f"  Total Trades:    {metrics['total_trades']}")
            print(f"  Profit Factor:   {metrics['profit_factor']:.2f}")
            
            if metrics['sharpe_ratio'] > 1.0 and metrics['total_return_pct'] > 0:
                print(f"\n{Color.GREEN}🎯 Strategy shows promise!{Color.RESET}")
            elif metrics['sharpe_ratio'] < 0 or metrics['total_return_pct'] < -10:
                print(f"\n{Color.RED}⚠️ Strategy needs improvement{Color.RESET}")
            else:
                print(f"\n{Color.YELLOW}📊 Strategy shows mixed results{Color.RESET}")
                
        except Exception as e:
            print(f"{Color.RED}❌ Test error: {e}{Color.RESET}")

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
            # 1. Sanitize the score (Convert -inf to a real number)
            safe_score = score
            if safe_score == float('-inf') or safe_score == -float('inf'):
                safe_score = -999.0  # Use a safe negative number instead of Infinity
            config['strategies']['optimized'][strategy_name] = {
                "parameters": params,
                "score": safe_score,
                "optimized_at": datetime.now().isoformat()
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

    def interactive_menu(self):
        while True:
            print(f"\n{Color.BOLD}{Color.PURPLE}🛡️  PHOENIX OPTIMIZER{Color.RESET}")
            print(f"{Color.GRAY}═══════════════════════════════════════════════════════{Color.RESET}")
            print("1. Optimize a strategy")
            print("2. Optimize all strategies (fast mode)")
            print("3. Test a strategy (Single Run)")
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
                    trials = input(f"Trials (default: 50) > ").strip()
                    trials = int(trials) if trials.isdigit() else 50
                    
                    fast = input("Fast mode? (y/n, default: n) > ").strip().lower()
                    fast_mode = fast == 'y'
                    
                    self.run_optimization(s_name, n_trials=trials, fast_mode=fast_mode)
                else:
                    print(f"{Color.RED}❌ Unknown strategy: {s_name}{Color.RESET}")
                    
            elif choice == '2':
                print(f"\n{Color.YELLOW}⚠️ This will optimize ALL strategies in fast mode{Color.RESET}")
                confirm = input("Continue? (y/n) > ").strip().lower()
                if confirm == 'y':
                    for s_name in self.strategy_map.keys():
                        self.run_optimization(s_name, n_trials=20, fast_mode=True)
                        print(f"\n{Color.GRAY}{'─'*50}{Color.RESET}")
            elif choice == '3':
                s_name = input("Strategy name > ").strip()
                self.run_strategy_test(s_name)
            elif choice == '4':
                self.view_optimization_history()
            elif choice == '5':
                print(f"\n{Color.GREEN}👋 Goodbye!{Color.RESET}")
                break
            else:
                print(f"{Color.RED}❌ Invalid choice{Color.RESET}")

    def view_optimization_history(self):
        """View past optimization results."""
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
            
            optimized = config.get('strategies', {}).get('optimized', {})
            
            if not optimized:
                print(f"\n{Color.YELLOW}No optimization history found.{Color.RESET}")
                return
            
            print(f"\n{Color.BOLD}{Color.CYAN}📊 OPTIMIZATION HISTORY{Color.RESET}")
            print(f"{Color.GRAY}{'─'*50}{Color.RESET}")
            
            for strategy, data in optimized.items():
                print(f"\n{Color.BOLD}{strategy}{Color.RESET}")
                print(f"  Score: {data.get('score', 0):.4f}")
                print(f"  Date: {data.get('optimized_at', 'Unknown')}")
                
                params = data.get('parameters', {})
                if params:
                    print(f"  Parameters:")
                    for param, value in params.items():
                        print(f"    {param}: {value}")
            
        except Exception as e:
            print(f"{Color.RED}❌ Error reading history: {e}{Color.RESET}")

def main():
    parser = argparse.ArgumentParser(description="Phoenix Strategy Optimizer")
    parser.add_argument('--strategy', type=str, help='Strategy name')
    parser.add_argument('--trials', type=int, default=50, help='Number of trials')
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