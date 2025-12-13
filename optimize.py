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
    WHITE = '\033[97m'  # AJOUTÉ: Couleur WHITE manquante
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
            
            # CORRIGÉ: Remplacement de Color.WHITE par Color.LIGHTGRAY
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
# OPTIMISEUR PRINCIPAL
# ==============================================================================

class PhoenixOptimizer:
    def __init__(self, verbose=False):
        self.config_path = 'config.json'
        self.config = self.load_config()
        self.display = TerminalDisplay()
        self.verbose = verbose
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
        print(f"▶ Fichier config: {self.config_path}{Color.RESET}")
        print()

    def adjust_and_validate_params(self, strat_name: str, raw_params: Dict) -> Dict:
        """Ajuste et valide les paramètres pour qu'ils soient réalistes"""
        adjusted = raw_params.copy()
        
        # Règles de sécurité communes (élargies pour plus de permissivité)
        safety_rules = {
            'stop_loss_pct': (0.005, 0.15),
            'take_profit_pct': (0.01, 0.40),
            'regime_low_sl_pct': (0.005, 0.10),
            'regime_low_tp_pct': (0.01, 0.20),
            'regime_high_sl_pct': (0.01, 0.15),
            'regime_high_tp_pct': (0.02, 0.30),
            'rsi_oversold': (15, 45),
            'rsi_overbought': (55, 90),
            'period': (5, 200),  # Élargi de 100 à 200
            'short_window': (3, 50),  # Élargi de 30 à 50
            'long_window': (10, 200),  # Élargi de 100 à 200
            'buy_threshold': (0.85, 0.99),  # AJOUTÉ: Pour MeanReversion
            'sell_threshold': (1.01, 1.20),  # AJOUTÉ: Pour MeanReversion
            'ma_short': (5, 50),  # AJOUTÉ: Pour MA_Enhanced
            'ma_long': (20, 200),  # AJOUTÉ: Pour MA_Enhanced
            'min_volatility_filter': (0.0, 0.05),  # AJOUTÉ: Filtre de volatilité
        }
        
        for param, (min_val, max_val) in safety_rules.items():
            if param in adjusted:
                try:
                    if isinstance(adjusted[param], (int, float)):
                        adjusted[param] = np.clip(adjusted[param], min_val, max_val)
                except:
                    pass
        
        # Assurer que Take Profit > Stop Loss
        for base in ['', 'regime_low_', 'regime_high_']:
            sl_key = f'{base}stop_loss_pct'
            tp_key = f'{base}take_profit_pct'
            
            if sl_key in adjusted and tp_key in adjusted:
                if adjusted[tp_key] <= adjusted[sl_key]:
                    adjusted[tp_key] = adjusted[sl_key] * 1.5  # Augmenté de 1.3 à 1.5
        
        # Assurer que short_window < long_window
        if 'short_window' in adjusted and 'long_window' in adjusted:
            if adjusted['short_window'] >= adjusted['long_window']:
                adjusted['long_window'] = adjusted['short_window'] + 10
        
        # Assurer que ma_short < ma_long
        if 'ma_short' in adjusted and 'ma_long' in adjusted:
            if adjusted['ma_short'] >= adjusted['ma_long']:
                adjusted['ma_long'] = adjusted['ma_short'] + 15
        
        return adjusted

    def calculate_strategy_score(self, results: Dict[str, Any], total_trades: int, pairs_tested: int) -> float:
        """Calcule un score intelligent pour évaluer la stratégie"""
        if total_trades == 0:
            return -5.0
        
        total_return = results.get('total_return', 0)
        sharpe_ratio = results.get('sharpe_ratio', 0)
        max_drawdown = max(results.get('max_drawdown', 1.0), 0.001)
        win_rate = results.get('win_rate', 0)
        
        # Score de base
        base_score = (sharpe_ratio * 8) + (total_return * 60)
        
        # Facteur de confiance
        confidence_factor = min(np.sqrt(total_trades / 10), 3.0) if total_trades >= 5 else 0.5  # Réduit de 20 à 10
        
        # Bonus/Pénalités
        bonus = 0
        penalty = 0
        
        if total_return > 0:
            bonus += total_return * 100  # Augmenté de 80 à 100
        
        if sharpe_ratio > 0:
            bonus += sharpe_ratio * 30  # Augmenté de 25 à 30
        
        if max_drawdown > 0.30:
            penalty -= 8 * (max_drawdown - 0.30)  # Réduit de 10 à 8
        
        # Bonus pour win_rate
        if win_rate > 0.5:  # Win rate > 50%
            bonus += (win_rate - 0.5) * 20
        
        final_score = (base_score * confidence_factor) + bonus + penalty
        return max(final_score, -10)  # Réduit de -20 à -10

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
        
        # Initialisation
        bt = Backtester()
        pairs = self.config['trading']['pairs']
        test_pairs = pairs[:3] if len(pairs) > 3 else pairs
        
        self.display.print_section("Configuration")
        self.display.print_info(f"Paires de test: {', '.join(test_pairs)}")
        self.display.print_info(f"Essais d'optimisation: {n_trials}")
        self.display.print_info(f"Mode détaillé: {'Activé' if self.verbose else 'Désactivé'}")
        
        # Variables de suivi
        trial_scores = []
        best_score = -float('inf')
        best_params = {}
        
        def objective(trial):
            # Générer les paramètres
            raw_params = strat_class.get_optuna_params(trial)
            params = self.adjust_and_validate_params(strat_name, raw_params)
            
            # Backtests
            all_results = []
            total_trades_all = 0
            
            for pair in test_pairs:
                try:
                    result = bt.run_backtest(strat_name, override_params=params, pair=pair)
                    all_results.append(result)
                    total_trades_all += result.get('total_trades', 0)
                except Exception as e:
                    if self.verbose:
                        self.display.print_warning(f"Erreur sur {pair}: {str(e)[:50]}...")
                    continue
            
            if not all_results:
                return -5.0
            
            # Agrégation et score
            aggregated = {
                'total_return': np.mean([r.get('total_return', 0) for r in all_results]),
                'sharpe_ratio': np.mean([r.get('sharpe_ratio', 0) for r in all_results]),
                'max_drawdown': np.mean([r.get('max_drawdown', 0) for r in all_results]),
                'win_rate': np.mean([r.get('win_rate', 0) for r in all_results]),
                'total_trades': total_trades_all
            }
            
            score = self.calculate_strategy_score(aggregated, total_trades_all, len(test_pairs))
            trial_scores.append(score)
            
            # Mettre à jour le meilleur score
            nonlocal best_score, best_params
            if score > best_score:
                best_score = score
                best_params = params.copy()
            
            # Affichage de progression
            if trial.number % 5 == 0 or trial.number == n_trials - 1:
                elapsed = time.time() - self.start_time
                avg_time = elapsed / (trial.number + 1)
                remaining = avg_time * (n_trials - trial.number - 1)
                
                self.display.print_progress(
                    trial.number + 1,
                    n_trials,
                    prefix=f"Essai {trial.number + 1:3d}/{n_trials}",
                    suffix=f"Score: {score:6.2f} | Trades: {total_trades_all:3d} | Temps restant: {self.format_duration(remaining)}"
                )
            
            return score
        
        # Lancement de l'optimisation
        self.display.print_section("Lancement de l'optimisation")
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
                ["Meilleur essai", f"#{study.best_trial.number}"]
            ]
            
            self.display.print_table(
                ["Métrique", "Valeur"],
                stats_rows,
                col_widths=[25, 15]
            )
            
            # Affichage des meilleurs paramètres (avec try-except)
            if study.best_params:
                try:
                    self.display.print_param_box(study.best_params, "Paramètres optimaux")
                except Exception as e:
                    self.display.print_error(f"Erreur lors de l'affichage des paramètres: {e}")
            
            # Sauvegarde des paramètres optimaux (déplacé avant la validation)
            if study.best_params:
                try:
                    self.save_optimized_params(strat_name, study.best_params)
                    self.display.print_success(f"Paramètres sauvegardés dans config.json")
                except Exception as e:
                    self.display.print_error(f"Erreur lors de la sauvegarde: {e}")
            
            # Validation finale
            self.display.print_section("Validation finale")
            
            final_results = []
            total_final_trades = 0
            
            for pair in pairs:
                try:
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
                    
                    # Couleur pour le retour
                    ret_color = Color.GREEN if ret > 0 else Color.RED
                    ret_str = f"{ret_color}{ret:+.2f}%{Color.RESET}"
                    
                    result_rows.append([pair, trades, ret_str, f"{sharpe:.2f}"])
                
                self.display.print_table(
                    ["Paire", "Trades", "Retour %", "Sharpe"],
                    result_rows,
                    col_widths=[10, 8, 12, 10]
                )
            
            # Afficher le total des trades
            self.display.print_info(f"Total trades sur validation: {total_final_trades}")
            
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
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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
            ["Mode", "Complet"]
        ]
        
        self.display.print_table(
            ["Paramètre", "Valeur"],
            summary_table,
            col_widths=[25, 20]
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
                self.display.print_info(f"Pause de 2 secondes avant la prochaine stratégie...")
                time.sleep(2)
        
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
            "Exécuter '--check-data' pour vérifier les données"
        ]
        
        for i, tip in enumerate(tips, 1):
            print(f"{Color.GRAY}{i}. {tip}{Color.RESET}")
        
        print()
        return results

    def run_quick_test(self, strategy_name: str):
        """Test rapide d'une stratégie avec affichage propre"""
        self.display.print_header(f"TEST RAPIDE : {strategy_name}")
        
        try:
            bt = Backtester()
            result = bt.run_backtest(strategy_name)
            
            if result:
                self.display.print_section("Résultats du test")
                
                # Tableau des résultats
                results_rows = [
                    ["Retour total", f"{result.get('total_return', 0)*100:+.2f}%"],
                    ["Ratio de Sharpe", f"{result.get('sharpe_ratio', 0):.3f}"],
                    ["Max Drawdown", f"{result.get('max_drawdown', 0)*100:.2f}%"],
                    ["Win Rate", f"{result.get('win_rate', 0)*100:.1f}%"],
                    ["Nombre de trades", f"{result.get('total_trades', 0)}"],
                    ["Profit Factor", f"{result.get('profit_factor', 0):.2f}"]
                ]
                
                self.display.print_table(
                    ["Métrique", "Valeur"],
                    results_rows,
                    col_widths=[20, 15]
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
        
        bt = Backtester()
        pairs = self.config['trading']['pairs'][:5]  # Limiter à 5 paires
        
        self.display.print_section("Analyse des données disponibles")
        
        data_rows = []
        for pair in pairs:
            try:
                df = bt.load_data(pair)
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
    
    args = parser.parse_args()
    
    # Ajustements selon les arguments
    if args.quick and not args.trials:
        args.trials = 20
    
    # Créer l'optimiseur
    optimizer = PhoenixOptimizer(verbose=args.verbose)
    
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
"""
# Mode interactif (demande tout)
python optimize.py

# Mode interactif explicite
python optimize.py --interactive

# Vérification données
python optimize.py --check-data

# Test rapide
python optimize.py --strategy MeanReversion --test --verbose

# Optimisation simple
python optimize.py --strategy MA_Enhanced --trials 40

# Optimisation complète
python optimize.py --all --trials 50 --verbose
"""
