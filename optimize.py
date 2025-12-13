import optuna
import json
import logging
import numpy as np
from typing import Dict, Any
import pandas as pd
import time
import sys
from datetime import datetime
import textwrap

# Import du Backtester et des Stratégies
from backtest import Backtester
from strategies import (
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
    LIGHTGRAY = '\033[37m'
    WHITE = '\033[97m'
    RESET = '\033[0m'

class TerminalDisplay:
    """Gestionnaire d'affichage pour un terminal propre et organisé"""
    
    @staticmethod
    def clear_screen():
        """Efface l'écran du terminal"""
        print("\033[2J\033[H", end="")
    
    @staticmethod
    def print_header(title, width=80):
        """Affiche un en-tête stylisé"""
        print(f"\n{Color.PURPLE}{'='*width}{Color.RESET}")
        print(f"{Color.BOLD}{Color.CYAN}{title.center(width)}{Color.RESET}")
        print(f"{Color.PURPLE}{'='*width}{Color.RESET}")
    
    @staticmethod
    def print_section(title, width=60):
        """Affiche une section"""
        print(f"\n{Color.BOLD}{Color.YELLOW}{title}{Color.RESET}")
        print(f"{Color.GRAY}{'-'*len(title)}{Color.RESET}")
    
    @staticmethod
    def print_subsection(title):
        """Affiche une sous-section"""
        print(f"\n{Color.BOLD}{Color.WHITE}› {title}{Color.RESET}")
    
    @staticmethod
    def print_info(message):
        """Affiche un message d'information"""
        print(f"{Color.LIGHTGRAY}[i] {message}{Color.RESET}")
    
    @staticmethod
    def print_success(message):
        """Affiche un message de succès"""
        print(f"{Color.GREEN}[✓] {message}{Color.RESET}")
    
    @staticmethod
    def print_warning(message):
        """Affiche un message d'avertissement"""
        print(f"{Color.YELLOW}[!] {message}{Color.RESET}")
    
    @staticmethod
    def print_error(message):
        """Affiche un message d'erreur"""
        print(f"{Color.RED}[✗] {message}{Color.RESET}")
    
    @staticmethod
    def print_progress(current, total, prefix="", suffix="", length=50):
        """Affiche une barre de progression"""
        percent = current / total
        filled = int(length * percent)
        bar = "█" * filled + "░" * (length - filled)
        
        # Couleur en fonction du pourcentage
        if percent < 0.3:
            color = Color.RED
        elif percent < 0.7:
            color = Color.YELLOW
        else:
            color = Color.GREEN
        
        print(f"\r{prefix} {color}{bar}{Color.RESET} {percent:.1%} {suffix}", end="")
        if current == total:
            print()
    
    @staticmethod
    def print_table(headers, rows, col_widths=None):
        """Affiche un tableau formaté"""
        if not rows:
            return
        
        # Déterminer les largeurs de colonnes
        if col_widths is None:
            col_widths = [len(str(h)) for h in headers]
            for row in rows:
                for i, cell in enumerate(row):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Afficher l'en-tête
        header_line = "┌" + "┬".join(["─" * (w + 2) for w in col_widths]) + "┐"
        print(f"{Color.GRAY}{header_line}{Color.RESET}")
        
        header_cells = []
        for i, header in enumerate(headers):
            header_cells.append(f" {Color.BOLD}{header:<{col_widths[i]}}{Color.RESET} ")
        print(f"{Color.GRAY}│{Color.RESET}" + f"{Color.GRAY}│{Color.RESET}".join(header_cells) + f"{Color.GRAY}│{Color.RESET}")
        
        separator_line = "├" + "┼".join(["─" * (w + 2) for w in col_widths]) + "┤"
        print(f"{Color.GRAY}{separator_line}{Color.RESET}")
        
        # Afficher les lignes
        for row in rows:
            row_cells = []
            for i, cell in enumerate(row):
                row_cells.append(f" {str(cell):<{col_widths[i]}} ")
            print(f"{Color.GRAY}│{Color.RESET}" + f"{Color.GRAY}│{Color.RESET}".join(row_cells) + f"{Color.GRAY}│{Color.RESET}")
        
        footer_line = "└" + "┴".join(["─" * (w + 2) for w in col_widths]) + "┘"
        print(f"{Color.GRAY}{footer_line}{Color.RESET}")
    
    @staticmethod
    def print_param_box(params, title="Paramètres"):
        """Affiche une boîte de paramètres stylisée"""
        if not params:
            return
        
        max_key_len = max(len(str(k)) for k in params.keys())
        max_val_len = max(len(f"{v:.6f}" if isinstance(v, float) else str(v)) for v in params.values())
        
        width = max_key_len + max_val_len + 7
        width = max(width, len(title) + 4)
        
        print(f"\n{Color.BLUE}╔{'═'*(width+2)}╗{Color.RESET}")
        print(f"{Color.BLUE}║{Color.RESET} {Color.BOLD}{Color.CYAN}{title.center(width)}{Color.RESET} {Color.BLUE}║{Color.RESET}")
        print(f"{Color.BLUE}╠{'═'*(width+2)}╣{Color.RESET}")
        
        for key, value in params.items():
            if isinstance(value, float):
                value_str = f"{value:.6f}"
            else:
                value_str = str(value)
            
            key_display = f"{Color.LIGHTGRAY}{key}{Color.RESET}"
            val_display = f"{Color.YELLOW}{value_str}{Color.RESET}"
            
            spacing = " " * (width - len(key) - len(value_str))
            print(f"{Color.BLUE}║{Color.RESET} {key_display}:{spacing}{val_display} {Color.BLUE}║{Color.RESET}")
        
        print(f"{Color.BLUE}╚{'═'*(width+2)}╝{Color.RESET}")
    
    @staticmethod
    def get_user_input(prompt, default=None, input_type=str):
        """Obtient une entrée utilisateur avec valeur par défaut"""
        prompt_text = f"{Color.CYAN}{prompt}{Color.RESET}"
        if default is not None:
            prompt_text += f" [{default}]: "
        else:
            prompt_text += ": "
        
        user_input = input(prompt_text).strip()
        
        if not user_input and default is not None:
            return default
        
        if input_type == int:
            try:
                return int(user_input)
            except ValueError:
                return default
        elif input_type == float:
            try:
                return float(user_input)
            except ValueError:
                return default
        
        return user_input

# ==============================================================================
# ESPACE DE RECHERCHE DES STRATÉGIES
# ==============================================================================

STRATEGY_SEARCH_SPACE = {
    "MeanReversion": {
        "period": {"type": "int", "low": 5, "high": 200},
        "buy_threshold": {"type": "float", "low": 0.85, "high": 0.99},
        "sell_threshold": {"type": "float", "low": 1.01, "high": 1.20},
        "stop_loss_pct": {"type": "float", "low": 0.005, "high": 0.15},
        "take_profit_pct": {"type": "float", "low": 0.01, "high": 0.40},
        "min_volatility_filter": {"type": "float", "low": 0.0, "high": 0.05}
    },
    "MA_Enhanced": {
        "ma_short": {"type": "int", "low": 5, "high": 50},
        "ma_long": {"type": "int", "low": 20, "high": 200},
        "rsi_period": {"type": "int", "low": 5, "high": 30},
        "rsi_oversold": {"type": "int", "low": 15, "high": 45},
        "rsi_overbought": {"type": "int", "low": 55, "high": 90},
        "stop_loss_pct": {"type": "float", "low": 0.005, "high": 0.15},
        "take_profit_pct": {"type": "float", "low": 0.01, "high": 0.40}
    },
    "Momentum_Enhanced": {
        "momentum_period": {"type": "int", "low": 5, "high": 50},
        "rsi_period": {"type": "int", "low": 5, "high": 30},
        "rsi_oversold": {"type": "int", "low": 15, "high": 45},
        "rsi_overbought": {"type": "int", "low": 55, "high": 90},
        "stop_loss_pct": {"type": "float", "low": 0.005, "high": 0.15},
        "take_profit_pct": {"type": "float", "low": 0.01, "high": 0.40}
    },
    "MeanReversion_Pro": {
        "period": {"type": "int", "low": 5, "high": 200},
        "zscore_threshold": {"type": "float", "low": 1.0, "high": 3.0},
        "stop_loss_pct": {"type": "float", "low": 0.005, "high": 0.15},
        "take_profit_pct": {"type": "float", "low": 0.01, "high": 0.40},
        "min_volatility_filter": {"type": "float", "low": 0.0, "high": 0.05}
    },
    "MA_Momentum_Hybrid": {
        "ma_short": {"type": "int", "low": 5, "high": 50},
        "ma_long": {"type": "int", "low": 20, "high": 200},
        "momentum_period": {"type": "int", "low": 5, "high": 50},
        "stop_loss_pct": {"type": "float", "low": 0.005, "high": 0.15},
        "take_profit_pct": {"type": "float", "low": 0.01, "high": 0.40}
    },
    "Volatility_Regime_Adaptive": {
        "regime_period": {"type": "int", "low": 10, "high": 100},
        "regime_low_threshold": {"type": "float", "low": 0.3, "high": 1.0},
        "regime_high_threshold": {"type": "float", "low": 1.0, "high": 3.0},
        "regime_low_sl_pct": {"type": "float", "low": 0.005, "high": 0.10},
        "regime_low_tp_pct": {"type": "float", "low": 0.01, "high": 0.20},
        "regime_high_sl_pct": {"type": "float", "low": 0.01, "high": 0.15},
        "regime_high_tp_pct": {"type": "float", "low": 0.02, "high": 0.30}
    }
}

# ==============================================================================
# OPTIMISEUR PRINCIPAL
# ==============================================================================

class PhoenixOptimizer:
    def __init__(self, verbose=False, fast_mode=False):
        self.config_path = 'config.json'
        self.config = self.load_config()
        self.display = TerminalDisplay()
        self.verbose = verbose
        self.fast_mode = fast_mode  # Nouveau: mode rapide pour accélérer
        self.start_time = None
        
        # Statistiques
        self.stats = {
            'total_trials': 0,
            'successful_trials': 0,
            'failed_trials': 0,
            'best_score': -float('inf'),
            'optimization_time': 0
        }

    def _get_strategy_class(self, strategy_name: str):
        """Mappe le nom de la stratégie à sa classe Python"""
        mapping = {
            "MeanReversion": MeanReversion,
            "MA_Enhanced": MA_Enhanced,
            "Momentum_Enhanced": Momentum_Enhanced,
            "MeanReversion_Pro": MeanReversion_Pro,
            "MA_Momentum_Hybrid": MA_Momentum_Hybrid,
            "Volatility_Regime_Adaptive": Volatility_Regime_Adaptive
        }
        return mapping.get(strategy_name)

    def get_optuna_params(self, trial, strategy_name: str) -> Dict:
        """Génère les paramètres pour Optuna à partir de l'espace de recherche"""
        if strategy_name not in STRATEGY_SEARCH_SPACE:
            raise ValueError(f"Stratégie {strategy_name} non trouvée dans l'espace de recherche")
        
        params = {}
        param_space = STRATEGY_SEARCH_SPACE[strategy_name]
        
        for param_name, param_config in param_space.items():
            param_type = param_config["type"]
            
            if param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name, 
                    param_config["low"], 
                    param_config["high"]
                )
            elif param_type == "float":
                params[param_name] = trial.suggest_float(
                    param_name, 
                    param_config["low"], 
                    param_config["high"]
                )
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name, 
                    param_config["choices"]
                )
            else:
                raise ValueError(f"Type de paramètre inconnu: {param_type}")
        
        return params

    def load_config(self) -> dict:
        """Charge la configuration depuis le fichier JSON"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.display.print_error(f"Impossible de charger config.json: {e}")
            return {}

    def show_banner(self):
        """Affiche la bannière d'introduction"""
        self.display.clear_screen()
        
        banner = """
╔══════════════════════════════════════════════════════════╗
║                  PHOENIX TRADING OPTIMIZER               ║
║                 Système d'Optimisation IA                ║
╚══════════════════════════════════════════════════════════╝
        """
        print(f"{Color.PURPLE}{banner}{Color.RESET}")
        
        # Informations système
        print(f"{Color.GRAY}▶ Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"▶ Version: 2.0.0")
        print(f"▶ Mode: {'Verbose' if self.verbose else 'Standard'}")
        print(f"▶ Fichier config: {self.config_path}")
        print(f"▶ Mode rapide: {'Activé' if self.fast_mode else 'Désactivé'}{Color.RESET}")
        print()

    def adjust_and_validate_params(self, strat_name: str, raw_params: Dict) -> Dict:
        """Ajuste et valide les paramètres pour qu'ils soient réalistes"""
        adjusted = raw_params.copy()
        
        # Validation automatique basée sur l'espace de recherche
        if strat_name in STRATEGY_SEARCH_SPACE:
            for param_name, param_config in STRATEGY_SEARCH_SPACE[strat_name].items():
                if param_name in adjusted:
                    low = param_config.get("low", None)
                    high = param_config.get("high", None)
                    
                    if low is not None and high is not None:
                        adjusted[param_name] = np.clip(adjusted[param_name], low, high)
        
        # Logique de sécurité supplémentaire
        safety_rules = {
            'ma_short': ('ma_long', lambda x: x + 10),
            'short_window': ('long_window', lambda x: x + 10),
        }
        
        for param, (dep_param, adjust_func) in safety_rules.items():
            if param in adjusted and dep_param in adjusted:
                if adjusted[param] >= adjusted[dep_param]:
                    adjusted[dep_param] = adjust_func(adjusted[param])
        
        # Assurer que Take Profit > Stop Loss
        for base in ['', 'regime_low_', 'regime_high_']:
            sl_key = f'{base}stop_loss_pct'
            tp_key = f'{base}take_profit_pct'
            
            if sl_key in adjusted and tp_key in adjusted:
                if adjusted[tp_key] <= adjusted[sl_key]:
                    adjusted[tp_key] = adjusted[sl_key] * 1.5
        
        return adjusted

    def calculate_strategy_score(self, results: Dict[str, Any], total_trades: int, pairs_tested: int) -> float:
        """
        Calcule un score intelligent avec la nouvelle formule:
        Score = (Gain * 0.5) + (Sharpe * 0.3) - (Drawdown * 2.0)
        """
        if total_trades == 0:
            return -5.0
        
        total_return = results.get('total_return', 0)
        sharpe_ratio = results.get('sharpe_ratio', 0)
        max_drawdown = results.get('max_drawdown', 0)
        
        # NOUVELLE FORMULE: Score = (Gain * 0.5) + (Sharpe * 0.3) - (Drawdown * 2.0)
        base_score = (total_return * 0.5) + (sharpe_ratio * 0.3) - (max_drawdown * 2.0)
        
        # Facteur de confiance basé sur le nombre de trades
        if total_trades < 5:
            confidence_factor = 0.3  # Très peu de trades = peu de confiance
        elif total_trades < 20:
            confidence_factor = 0.6  # Nombre modéré de trades
        else:
            confidence_factor = 1.0  # Assez de trades pour être confiant
        
        # Pénalité pour très faible win_rate
        win_rate = results.get('win_rate', 0)
        if win_rate < 0.3:
            base_score -= (0.3 - win_rate) * 2.0
        
        # Bonus pour très bon win_rate
        if win_rate > 0.7:
            base_score += (win_rate - 0.7) * 1.0
        
        # Ajuster selon le nombre de paires testées
        pair_factor = min(pairs_tested / 3.0, 1.0)  # Normaliser par 3 paires
        
        final_score = base_score * confidence_factor * pair_factor
        
        return max(final_score, -10)

    def format_duration(self, seconds):
        """Formate une durée en secondes en chaîne lisible"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}min"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

    def optimize_strategy(self, strat_name: str, n_trials: int = 50) -> Dict:
        """Optimisation pour une stratégie spécifique avec affichage propre"""
        
        self.display.print_header(f"OPTIMISATION : {strat_name}")
        
        strat_class = self._get_strategy_class(strat_name)
        if not strat_class:
            self.display.print_error(f"Stratégie '{strat_name}' introuvable")
            return {}
        
        # Initialisation du backtester en mode approprié
        bt = Backtester(verbose=False)
        pairs = self.config['trading']['pairs']
        
        # MODIFICATION IMPORTANTE: Utiliser 3 paires pour l'optimisation
        if len(pairs) >= 3:
            test_pairs = pairs[:3]  # Les 3 premières paires
            if self.verbose:
                self.display.print_info(f"Utilisation de 3 paires pour l'optimisation: {', '.join(test_pairs)}")
        elif len(pairs) == 2:
            test_pairs = pairs  # Les 2 paires disponibles
            if self.verbose:
                self.display.print_info(f"Utilisation de 2 paires pour l'optimisation: {', '.join(test_pairs)}")
        else:
            test_pairs = pairs  # Une seule paire disponible
            if self.verbose:
                self.display.print_warning(f"Seulement 1 paire disponible: {test_pairs[0]}")
        
        self.display.print_section("Configuration")
        self.display.print_info(f"Paires de test: {', '.join(test_pairs)}")
        self.display.print_info(f"Essais d'optimisation: {n_trials}")
        self.display.print_info(f"Mode détaillé: {'Activé' if self.verbose else 'Désactivé'}")
        self.display.print_info(f"Mode rapide: {'Activé' if self.fast_mode else 'Désactivé'}")
        
        # Variables de suivi
        trial_scores = []
        best_score = -float('inf')
        best_params = {}
        
        def objective(trial):
            # Générer les paramètres avec le nouvel espace de recherche
            raw_params = self.get_optuna_params(trial, strat_name)
            params = self.adjust_and_validate_params(strat_name, raw_params)
            
            # Backtests sur les 3 paires
            all_results = []
            total_trades_all = 0
            
            for pair in test_pairs:
                try:
                    # Utiliser le mode optimisation rapide si activé
                    result = bt.run_backtest(
                        strat_name, 
                        override_params=params, 
                        pair=pair,
                        # optimization_mode=self.fast_mode  # Mode optimisation rapide si fast_mode=True
                    )
                    all_results.append(result)
                    total_trades_all += result.get('total_trades', 0)
                    
                    if self.verbose and trial.number % 10 == 0:
                        self.display.print_info(
                            f"  {pair}: {result.get('total_trades', 0)} trades, "
                            f"Return={result.get('total_return', 0)*100:.2f}%"
                        )
                        
                except Exception as e:
                    if self.verbose:
                        self.display.print_warning(f"Erreur sur {pair}: {str(e)[:50]}...")
                    continue
            
            if not all_results:
                if self.verbose:
                    self.display.print_warning("Aucun résultat valide pour cet essai")
                return -5.0
            
            # Agrégation et score
            aggregated = {
                'total_return': np.mean([r.get('total_return', 0) for r in all_results]),
                'sharpe_ratio': np.mean([r.get('sharpe_ratio', 0) for r in all_results]),
                'max_drawdown': np.mean([r.get('max_drawdown', 0) for r in all_results]),
                'win_rate': np.mean([r.get('win_rate', 0) for r in all_results]),
                'total_trades': total_trades_all
            }
            
            # Afficher les détails si verbose
            if self.verbose and trial.number % 5 == 0:
                self.display.print_info(
                    f"Essai {trial.number}: "
                    f"Return={aggregated['total_return']*100:.2f}%, "
                    f"Sharpe={aggregated['sharpe_ratio']:.2f}, "
                    f"Drawdown={aggregated['max_drawdown']*100:.2f}%, "
                    f"Trades={aggregated['total_trades']}"
                )
            
            score = self.calculate_strategy_score(aggregated, total_trades_all, len(test_pairs))
            trial_scores.append(score)
            
            # Mettre à jour le meilleur score
            nonlocal best_score, best_params
            if score > best_score:
                best_score = score
                best_params = params.copy()
            
            # Affichage de progression
            if trial.number % 2 == 0 or trial.number == n_trials - 1:  # Plus fréquent pour 3 paires
                elapsed = time.time() - self.start_time
                avg_time = elapsed / (trial.number + 1)
                remaining = avg_time * (n_trials - trial.number - 1)
                
                # Calcul des stats par paire pour l'affichage
                avg_return = aggregated['total_return'] * 100
                avg_sharpe = aggregated['sharpe_ratio']
                avg_drawdown = aggregated['max_drawdown'] * 100
                
                self.display.print_progress(
                    trial.number + 1,
                    n_trials,
                    prefix=f"Essai {trial.number + 1:3d}/{n_trials}",
                    suffix=f"Score: {score:6.2f} | Trades: {total_trades_all:3d} | Ret: {avg_return:+.1f}% | Reste: {self.format_duration(remaining)}"
                )
            
            return score
        
        # Lancement de l'optimisation
        self.display.print_section("Lancement de l'optimisation")
        self.display.print_info(f"Début de l'optimisation sur {len(test_pairs)} paires...")
        self.start_time = time.time()
        
        try:
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=10),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
            )
            
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            
            # Résultats
            elapsed_time = time.time() - self.start_time
            
            self.display.print_section("Résultats de l'optimisation")
            
            # Tableau des statistiques
            stats_rows = [
                ["Score optimal", f"{study.best_value:.2f}"],
                ["Essais effectués", f"{len(study.trials)}"],
                ["Temps total", f"{self.format_duration(elapsed_time)}"],
                ["Score moyen", f"{np.mean(trial_scores):.2f}"],
                ["Meilleur essai", f"#{study.best_trial.number}"],
                ["Paires utilisées", f"{len(test_pairs)}"]
            ]
            
            self.display.print_table(
                ["Métrique", "Valeur"],
                stats_rows,
                col_widths=[25, 15]
            )
            
            # Détails du meilleur score
            if study.best_params:
                self.display.print_section("Détails du meilleur score")
                best_details = [
                    ["Score Composite", f"{study.best_value:.2f}"],
                    ["Formule:", ""],
                    ["  - Gain × 0.5", f"{(study.best_value * 0.5):.2f}"],
                    ["  - Sharpe × 0.3", f"{(study.best_value * 0.3):.2f}"],
                    ["  - Drawdown × -2.0", f"{(study.best_value * -2.0):.2f}"]
                ]
                
                self.display.print_table(
                    ["Composant", "Valeur"],
                    best_details,
                    col_widths=[25, 15]
                )
            
            # Affichage des meilleurs paramètres
            if study.best_params:
                try:
                    self.display.print_param_box(study.best_params, "Paramètres optimaux")
                except Exception as e:
                    self.display.print_error(f"Erreur lors de l'affichage des paramètres: {e}")
            
            # Sauvegarde des paramètres optimaux
            if study.best_params:
                try:
                    self.save_optimized_params(strat_name, study.best_params)
                    self.display.print_success(f"Paramètres sauvegardés dans config.json")
                except Exception as e:
                    self.display.print_error(f"Erreur lors de la sauvegarde: {e}")
            
            # Validation finale (avec toutes les paires)
            self.display.print_section(f"Validation finale sur {len(pairs)} paires")
            
            final_results = []
            total_final_trades = 0
            
            for pair in pairs:
                try:
                    # Validation avec backtest complet (pas en mode optimisation)
                    result = bt.run_backtest(strat_name, override_params=study.best_params, pair=pair)
                    final_results.append((pair, result))
                    total_final_trades += result.get('total_trades', 0)
                except Exception as e:
                    self.display.print_warning(f"Erreur sur {pair}: {str(e)[:50]}...")
            
            # Tableau des résultats par paire
            if final_results:
                result_rows = []
                for pair, result in final_results:
                    trades = result.get('total_trades', 0)
                    ret = result.get('total_return', 0) * 100
                    sharpe = result.get('sharpe_ratio', 0)
                    drawdown = result.get('max_drawdown', 0) * 100
                    
                    # Calcul du score de validation
                    val_score = (result.get('total_return', 0) * 0.5) + \
                               (sharpe * 0.3) - \
                               (result.get('max_drawdown', 0) * 2.0)
                    
                    # Couleur pour le retour
                    ret_color = Color.GREEN if ret > 0 else Color.RED
                    ret_str = f"{ret_color}{ret:+.2f}%{Color.RESET}"
                    
                    # Couleur pour le score
                    score_color = Color.GREEN if val_score > 0 else Color.RED
                    score_str = f"{score_color}{val_score:.2f}{Color.RESET}"
                    
                    result_rows.append([pair, trades, ret_str, f"{sharpe:.2f}", f"{drawdown:.2f}%", score_str])
                
                self.display.print_table(
                    ["Paire", "Trades", "Retour %", "Sharpe", "Drawdown", "Score"],
                    result_rows,
                    col_widths=[10, 8, 12, 10, 12, 10]
                )
            
            # Afficher le total des trades
            self.display.print_info(f"Total trades sur validation: {total_final_trades}")
            
            # Afficher les moyennes
            if final_results:
                avg_return = np.mean([r[1].get('total_return', 0) * 100 for r in final_results])
                avg_sharpe = np.mean([r[1].get('sharpe_ratio', 0) for r in final_results])
                avg_drawdown = np.mean([r[1].get('max_drawdown', 0) * 100 for r in final_results])
                
                self.display.print_info(
                    f"Moyennes: Retour={avg_return:+.2f}%, "
                    f"Sharpe={avg_sharpe:.2f}, "
                    f"Drawdown={avg_drawdown:.2f}%"
                )
            
            return study.best_params
            
        except KeyboardInterrupt:
            self.display.print_warning("Optimisation interrompue par l'utilisateur")
            return {}
        except Exception as e:
            self.display.print_error(f"Erreur: {e}")
            return {}

    def save_optimized_params(self, strat_name: str, optimized_params: Dict):
        """Écrit les paramètres optimisés dans config.json"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            if 'strategies' not in config:
                config['strategies'] = {}
            if 'parameters' not in config['strategies']:
                config['strategies']['parameters'] = {}
            
            if strat_name not in config['strategies']['parameters']:
                config['strategies']['parameters'][strat_name] = {}
            
            config['strategies']['parameters'][strat_name].update(optimized_params)
            
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            # Backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"config_backup_{timestamp}.json"
            with open(backup_file, 'w') as f:
                json.dump(config, f, indent=4)
            
        except Exception as e:
            self.display.print_error(f"Erreur sauvegarde: {e}")

    def run_multi_strategy_optimization(self, strategy_list: list = None, n_trials_per_strat: int = 30):
        """Optimisation de plusieurs stratégies avec affichage organisé"""
        
        self.show_banner()
        
        if strategy_list is None:
            strategy_list = self.config['strategies'].get('active_strategies', [])
        
        if not strategy_list:
            strategy_list = ['MeanReversion', 'MA_Enhanced']
        
        # Informations générales
        self.display.print_header("OPTIMISATION MULTI-STRATÉGIES")
        
        summary_table = [
            ["Stratégies à optimiser", f"{len(strategy_list)}"],
            ["Essais par stratégie", f"{n_trials_per_strat}"],
            ["Date de début", datetime.now().strftime("%H:%M:%S")],
            ["Mode", "Complet"],
            ["Paires par optimisation", "3 (ou moins si <3)"],
            ["Score Formula", "(Gain*0.5)+(Sharpe*0.3)-(Drawdown*2.0)"]
        ]
        
        self.display.print_table(
            ["Paramètre", "Valeur"],
            summary_table,
            col_widths=[25, 30]
        )
        
        # Optimisation séquentielle
        results = {}
        total_start_time = time.time()
        
        for i, strat in enumerate(strategy_list):
            strat_start_time = time.time()
            
            self.display.print_header(f"STRATÉGIE {i+1}/{len(strategy_list)}: {strat}")
            
            optimized_params = self.optimize_strategy(strat, n_trials=n_trials_per_strat)
            results[strat] = optimized_params
            
            strat_time = time.time() - strat_start_time
            
            # Pause entre les stratégies
            if i < len(strategy_list) - 1:
                pause_time = 5 if self.fast_mode else 2
                self.display.print_info(f"Pause de {pause_time} secondes avant la prochaine stratégie...")
                time.sleep(pause_time)
        
        # Résumé final
        total_time = time.time() - total_start_time
        
        self.display.print_header("RÉSUMÉ FINAL")
        
        # Tableau des résultats
        result_rows = []
        for strat, params in results.items():
            status = "✓" if params else "✗"
            status_color = Color.GREEN if params else Color.RED
            param_count = len(params) if params else 0
            
            result_rows.append([
                f"{status_color}{status}{Color.RESET}",
                strat,
                f"{param_count} params" if params else "Échec"
            ])
        
        self.display.print_table(
            ["Status", "Stratégie", "Résultat"],
            result_rows,
            col_widths=[8, 25, 20]
        )
        
        # Statistiques finales
        self.display.print_section("Statistiques globales")
        stats_rows = [
            ["Temps total", f"{self.format_duration(total_time)}"],
            ["Stratégies réussies", f"{sum(1 for p in results.values() if p)}/{len(results)}"],
            ["Paramètres optimisés", f"{sum(len(p) for p in results.values() if p)}"]
        ]
        
        self.display.print_table(
            ["Métrique", "Valeur"],
            stats_rows,
            col_widths=[25, 15]
        )
        
        # Instructions
        self.display.print_section("Prochaines étapes")
        tips = [
            "Lancer 'python backtest.py' pour tester les performances",
            "Consulter 'config_backup_*.json' pour les sauvegardes",
            "Utiliser '--verbose' pour plus de détails",
            "Exécuter '--check-data' pour vérifier les données",
            "Nouveau score: (Gain*0.5)+(Sharpe*0.3)-(Drawdown*2.0)",
            "Optimisation effectuée sur 3 paires (plus robuste)"
        ]
        
        for i, tip in enumerate(tips, 1):
            print(f"{Color.GRAY}{i}. {tip}{Color.RESET}")
        
        print()
        return results

    def run_quick_test(self, strategy_name: str):
        """Test rapide d'une stratégie avec affichage propre"""
        self.display.print_header(f"TEST RAPIDE : {strategy_name}")
        
        try:
            bt = Backtester(verbose=self.verbose)
            result = bt.run_backtest(strategy_name)
            
            if result:
                self.display.print_section("Résultats du test")
                
                # Calcul du nouveau score
                total_return = result.get('total_return', 0)
                sharpe_ratio = result.get('sharpe_ratio', 0)
                max_drawdown = result.get('max_drawdown', 0)
                
                score = (total_return * 0.5) + (sharpe_ratio * 0.3) - (max_drawdown * 2.0)
                
                # Tableau des résultats
                results_rows = [
                    ["Score Composite", f"{score:.2f}"],
                    ["Breakdown:", ""],
                    ["  - Gain (x0.5)", f"{total_return * 0.5:.3f}"],
                    ["  - Sharpe (x0.3)", f"{sharpe_ratio * 0.3:.3f}"],
                    ["  - Drawdown Penalty (x2.0)", f"{max_drawdown * -2.0:.3f}"],
                    ["", ""],
                    ["Retour total", f"{total_return*100:+.2f}%"],
                    ["Ratio de Sharpe", f"{sharpe_ratio:.3f}"],
                    ["Max Drawdown", f"{max_drawdown*100:.2f}%"],
                    ["Win Rate", f"{result.get('win_rate', 0)*100:.1f}%"],
                    ["Nombre de trades", f"{result.get('total_trades', 0)}"],
                    ["Profit Factor", f"{result.get('profit_factor', 0):.2f}"]
                ]
                
                self.display.print_table(
                    ["Métrique", "Valeur"],
                    results_rows,
                    col_widths=[30, 20]
                )
                
                # Afficher les paramètres utilisés
                if strategy_name in self.config.get('strategies', {}).get('parameters', {}):
                    self.display.print_param_box(
                        self.config['strategies']['parameters'][strategy_name],
                        "Paramètres utilisés"
                    )
            else:
                self.display.print_error("Aucun résultat obtenu")
            
            return result
        except Exception as e:
            self.display.print_error(f"Erreur lors du test: {e}")
            return None

    def check_data_quality(self):
        """Vérifie la qualité des données avec affichage organisé"""
        self.display.print_header("VÉRIFICATION DES DONNÉES")
        
        bt = Backtester(verbose=True)  # Activer verbose pour voir la progression
        pairs = self.config['trading']['pairs'][:5]  # Limiter à 5 paires
        
        self.display.print_section("Analyse des données disponibles")
        
        data_rows = []
        for pair in pairs:
            try:
                df = bt.get_cached_data(pair, interval='1', total_candles=10000)
                if df is not None and len(df) > 0:
                    days = len(df) / (24 * 60)  # Convertir bougies 1min en jours
                    volatility = df['close'].pct_change().std() * 100
                    
                    status = "✓"
                    status_color = Color.GREEN
                    
                    data_rows.append([
                        f"{status_color}{status}{Color.RESET}",
                        pair,
                        f"{len(df):,}",
                        f"{days:.1f}j",
                        f"{volatility:.2f}%"
                    ])
                else:
                    data_rows.append([
                        f"{Color.RED}✗{Color.RESET}",
                        pair,
                        "N/A",
                        "N/A",
                        "N/A"
                    ])
            except Exception as e:
                data_rows.append([
                    f"{Color.RED}✗{Color.RESET}",
                    pair,
                    f"Erreur",
                    "N/A",
                    "N/A"
                ])
        
        self.display.print_table(
            ["Status", "Paire", "Bougies", "Période", "Volatilité"],
            data_rows,
            col_widths=[8, 10, 12, 10, 12]
        )
        
        # Recommandations
        self.display.print_section("Recommandations")
        
        recommendations = [
            "Minimum recommandé: 10,000 bougies (≈7 jours en 1min)",
            "Volatilité idéale: 0.5% - 5% par bougie",
            "Assurez-vous d'avoir des données récentes",
            "Vérifiez les gaps dans les données"
        ]
        
        for rec in recommendations:
            print(f"{Color.GRAY}• {rec}{Color.RESET}")
        
        print()

    def interactive_menu(self):
        """Menu interactif amélioré avec plus d'options"""
        self.show_banner()
        
        while True:
            print(f"{Color.BOLD}Menu Principal:{Color.RESET}")
            print(f"{Color.GRAY}1. Optimisation complète (toutes stratégies)")
            print(f"2. Optimisation d'une stratégie spécifique")
            print(f"3. Test rapide d'une stratégie")
            print(f"4. Vérification des données")
            print(f"5. Quitter{Color.RESET}")
            
            choice = self.display.get_user_input("Votre choix [1-5]", "1", int)
            
            if choice == 1:
                # Demander quelles stratégies
                available_strategies = list(self.config['strategies'].get('parameters', {}).keys())
                if not available_strategies:
                    available_strategies = ['MeanReversion', 'MA_Enhanced', 'Momentum_Enhanced']
                
                print(f"\n{Color.BOLD}Stratégies disponibles:{Color.RESET}")
                for i, strat in enumerate(available_strategies, 1):
                    print(f"{Color.GRAY}  {i}. {strat}{Color.RESET}")
                
                strat_choice = self.display.get_user_input(
                    f"Stratégies à optimiser (numéros séparés par des virgules, ou 'all')",
                    "all"
                )
                
                if strat_choice.lower() == 'all':
                    strategies_to_optimize = available_strategies
                else:
                    try:
                        indices = [int(i.strip()) - 1 for i in strat_choice.split(',')]
                        strategies_to_optimize = [available_strategies[i] for i in indices if 0 <= i < len(available_strategies)]
                    except:
                        strategies_to_optimize = available_strategies[:2]
                
                # Demander le nombre d'essais
                trials = self.display.get_user_input(
                    "Nombre d'essais par stratégie",
                    30,  # Valeur par défaut
                    int
                )
                
                # Demander si mode rapide
                fast_mode_choice = self.display.get_user_input(
                    "Mode rapide? (oui/non) [oui]",
                    "oui"
                )
                fast_mode = fast_mode_choice.lower() in ['oui', 'yes', 'y', 'o']
                
                # Configurer le mode rapide
                self.fast_mode = fast_mode
                
                # Lancer l'optimisation
                self.run_multi_strategy_optimization(strategies_to_optimize, n_trials_per_strat=trials)
                
                input(f"\n{Color.GRAY}Appuyez sur Entrée pour continuer...{Color.RESET}")
                
            elif choice == 2:
                # Demander la stratégie
                available_strategies = list(self.config['strategies'].get('parameters', {}).keys())
                if not available_strategies:
                    available_strategies = ['MeanReversion', 'MA_Enhanced', 'Momentum_Enhanced']
                
                print(f"\n{Color.BOLD}Stratégies disponibles:{Color.RESET}")
                for i, strat in enumerate(available_strategies, 1):
                    print(f"{Color.GRAY}  {i}. {strat}{Color.RESET}")
                
                strat_idx = self.display.get_user_input(
                    "Sélectionnez une stratégie (numéro)",
                    1,  # Valeur par défaut
                    int
                ) - 1
                
                if 0 <= strat_idx < len(available_strategies):
                    strategy = available_strategies[strat_idx]
                    
                    # Demander le nombre d'essais
                    trials = self.display.get_user_input(
                        f"Nombre d'essais pour {strategy}",
                        50,  # Valeur par défaut
                        int
                    )
                    
                    # Demander si mode rapide
                    fast_mode_choice = self.display.get_user_input(
                        "Mode rapide? (oui/non) [oui]",
                        "oui"
                    )
                    fast_mode = fast_mode_choice.lower() in ['oui', 'yes', 'y', 'o']
                    
                    # Configurer le mode rapide
                    self.fast_mode = fast_mode
                    
                    # Lancer l'optimisation
                    self.optimize_strategy(strategy, n_trials=trials)
                else:
                    self.display.print_error("Sélection invalide")
                
                input(f"\n{Color.GRAY}Appuyez sur Entrée pour continuer...{Color.RESET}")
                
            elif choice == 3:
                # Demander la stratégie pour le test
                available_strategies = list(self.config['strategies'].get('parameters', {}).keys())
                if not available_strategies:
                    available_strategies = ['MeanReversion', 'MA_Enhanced', 'Momentum_Enhanced']
                
                print(f"\n{Color.BOLD}Stratégies disponibles:{Color.RESET}")
                for i, strat in enumerate(available_strategies, 1):
                    print(f"{Color.GRAY}  {i}. {strat}{Color.RESET}")
                
                strat_idx = self.display.get_user_input(
                    "Sélectionnez une stratégie à tester (numéro)",
                    1,  # Valeur par défaut
                    int
                ) - 1
                
                if 0 <= strat_idx < len(available_strategies):
                    strategy = available_strategies[strat_idx]
                    self.run_quick_test(strategy)
                else:
                    self.display.print_error("Sélection invalide")
                
                input(f"\n{Color.GRAY}Appuyez sur Entrée pour continuer...{Color.RESET}")
                
            elif choice == 4:
                self.check_data_quality()
                input(f"\n{Color.GRAY}Appuyez sur Entrée pour continuer...{Color.RESET}")
                
            elif choice == 5:
                print(f"\n{Color.GREEN}Merci d'avoir utilisé Phoenix Optimizer !{Color.RESET}")
                break
            else:
                self.display.print_error("Choix invalide. Veuillez réessayer.")


# ==============================================================================
# EXÉCUTION PRINCIPALE
# ==============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Phoenix Trading Optimizer - Système d\'optimisation IA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  %(prog)s --strategy MeanReversion --trials 30
  %(prog)s --all --trials 50
  %(prog)s --strategy MA_Enhanced --test
  %(prog)s --check-data
  %(prog)s --interactive
  
Pour plus d'informations:
  https://github.com/votre-repo/phoenix-trading
        """
    )
    
    parser.add_argument('--strategy', '-s', type=str, 
                       help='Nom de la stratégie à optimiser')
    parser.add_argument('--all', '-a', action='store_true', 
                       help='Optimiser toutes les stratégies actives')
    parser.add_argument('--trials', '-t', type=int, default=50, 
                       help='Nombre d\'essais par stratégie (défaut: 50)')
    parser.add_argument('--test', action='store_true', 
                       help='Test rapide sans optimisation')
    parser.add_argument('--check-data', action='store_true', 
                       help='Vérifier la qualité des données')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Mode verbeux avec plus de détails')
    parser.add_argument('--quick', '-q', action='store_true', 
                       help='Mode rapide avec moins d\'essais')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Lancer le mode interactif (menu)')
    parser.add_argument('--fast-mode', action='store_true',
                       help='Mode optimisation rapide (moins de données)')
    
    args = parser.parse_args()
    
    # Ajustements selon les arguments
    if args.quick and not args.trials:
        args.trials = 20
    
    # Créer l'optimiseur
    optimizer = PhoenixOptimizer(verbose=args.verbose, fast_mode=args.fast_mode)
    
    # Exécuter la commande appropriée
    if args.interactive:
        optimizer.interactive_menu()
        
    elif args.check_data:
        optimizer.check_data_quality()
    
    elif args.test and args.strategy:
        optimizer.run_quick_test(args.strategy)
    
    elif args.strategy:
        optimizer.show_banner()
        optimizer.optimize_strategy(args.strategy, n_trials=args.trials)
    
    elif args.all:
        strategies = optimizer.config['strategies'].get('active_strategies', [])
        if not strategies:
            strategies = ['MeanReversion', 'MA_Enhanced', 'Momentum_Enhanced']
        optimizer.run_multi_strategy_optimization(strategies, n_trials_per_strat=args.trials)
    
    else:
        # Mode interactif par défaut
        optimizer.interactive_menu()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Color.YELLOW}Interrompu par l'utilisateur.{Color.RESET}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Color.RED}Erreur inattendue: {e}{Color.RESET}")
        sys.exit(1)


"""""""""""""""""""""""
Nouveau paramètre --fast-mode
"""""""""""""""""""""""
"""
# Optimisation rapide (3 paires, données réduites)
python optimize.py --strategy MA_Enhanced --trials 30 --fast-mode --verbose

# Optimisation complète (3 paires, données complètes)
python optimize.py --strategy MA_Enhanced --trials 30 --verbose
"""
"""""""""""""""""""""""""""""""""
Commandes recommandées maintenant :
""""""""""""""""""""""""""""""
"
# Test rapide sur 1 paire
python optimize.py --strategy MA_Enhanced --test --verbose

# Optimisation rapide (3 paires, mode rapide)
python optimize.py --strategy MA_Enhanced --trials 30 --fast-mode --verbose

# Optimisation complète (3 paires, données complètes)
python optimize.py --strategy MA_Enhanced --trials 50 --verbose

# Toutes les stratégies en mode rapide
python optimize.py --all --trials 30 --fast-mode --verbose
"""
