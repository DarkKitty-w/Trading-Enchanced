import optuna
import json
import logging
import os

from backtest import Backtester
from strategies import (
    MeanReversion, 
    MA_Enhanced, 
    Momentum_Enhanced, 
    MeanReversion_Pro, 
    MA_Momentum_Hybrid, 
    Volatility_Regime_Adaptive
)

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("PhoenixOptimizer")
optuna.logging.set_verbosity(optuna.logging.WARNING)

class PhoenixOptimizer:
    def __init__(self):
        self.config_path = 'config.json'

    def _get_strategy_class(self, strategy_name):
        mapping = {
            "MeanReversion": MeanReversion,
            "MA_Enhanced": MA_Enhanced,
            "Momentum_Enhanced": Momentum_Enhanced,
            "MeanReversion_Pro": MeanReversion_Pro,
            "MA_Momentum_Hybrid": MA_Momentum_Hybrid,
            "Volatility_Regime_Adaptive": Volatility_Regime_Adaptive
        }
        return mapping.get(strategy_name)

    def load_config(self):
        with open(self.config_path, 'r') as f:
            return json.load(f)

    def optimize_strategy(self, strat_name, n_trials=50):
        print(f"\n🧠 OPTIMISATION CIBLÉE : {strat_name}")
        print(f"   🎯 Objectif : Maximiser le SQN multi-coins")

        strat_class = self._get_strategy_class(strat_name)
        if not strat_class:
            print(f"   ❌ Stratégie '{strat_name}' introuvable.")
            return

        def objective(trial):
            params = strat_class.get_optuna_params(trial)

            # Forcer SL/TP dans des bornes réalistes
            for k in params:
                if "sl_pct" in k:
                    params[k] = min(max(params[k], 0.001), 0.015)  # 0.1%-1.5%
                if "tp_pct" in k:
                    params[k] = min(max(params[k], 0.002), 0.03)   # 0.2%-3%

            bt = Backtester()
            coins = bt.config['trading']['pairs']
            total_score = 0
            total_trades = 0

            for coin in coins:
                result = bt.run_backtest(strat_name, override_params=params, pair=coin)
                trades = result.get('total_trades', 0)
                total_trades += trades

                # Pénalité si trop peu de trades
                if trades < 10:
                    total_score += -10
                    continue

                ret = result.get('total_return', 0)
                max_dd = result.get('max_drawdown', 0.0)

                # Score basé sur SQN approximatif
                score = ret * 100 if max_dd < 0.0001 else ret / max_dd
                total_score += score

            return total_score / len(coins) if len(coins) > 0 else total_score

        # Étude Optuna
        study = optuna.create_study(direction="maximize")
        try:
            study.optimize(objective, n_trials=n_trials)
            print(f"   🏆 MEILLEUR SCORE : {study.best_value:.4f}")
            print("   ⚙️  PARAMÈTRES GAGNANTS :")
            for k, v in study.best_params.items():
                print(f"       - {k}: {v}")
            self.save_params(strat_name, study.best_params)
        except KeyboardInterrupt:
            print("   🛑 Optimisation stoppée par l'utilisateur.")
        except Exception as e:
            print(f"   ❌ Erreur durant l'optimisation : {e}")

    def save_params(self, strat_name, new_params):
        config = self.load_config()
        if strat_name not in config['strategies']['parameters']:
            config['strategies']['parameters'][strat_name] = {}
        config['strategies']['parameters'][strat_name].update(new_params)
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print("   💾 Configuration mise à jour avec succès.")

    def run_all(self, n_trials_per_strat=50):
        config = self.load_config()
        strategies = config['strategies']['active_strategies']
        print("="*60)
        print(f"🔥 DÉMARRAGE OPTIMISATION MASSIVE ({len(strategies)} Stratégies)")
        print(f"⏱️  Données : Bybit 1 Minute (10000 bougies)")
        print(f"🔄 Essais par stratégie : {n_trials_per_strat}")
        print("="*60)

        for strat in strategies:
            self.optimize_strategy(strat, n_trials=n_trials_per_strat)

        print("\n✨ TERMINÉ ! Toutes les stratégies ont été calibrées.")

if __name__ == "__main__":
    optimizer = PhoenixOptimizer()
    optimizer.run_all(n_trials_per_strat=100)
