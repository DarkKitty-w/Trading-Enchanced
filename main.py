import asyncio
import logging
import signal
import json
import os
from typing import Dict, List, Optional
from datetime import datetime, timezone

# Imports Tiers
import ccxt.async_support as ccxt
from dotenv import load_dotenv

# Imports Projet (Architecture Propre)
from models import (
    Portfolio, 
    Position, 
    MarketCandle, 
    Signal, 
    SignalType, 
    Trade,
    PortfolioItem,  # AJOUT: Import manquant
    TradeRecord     # AJOUT: Import manquant
)
from market_data import MarketDataManager
from database import DatabaseHandler
from execution import ExecutionManager
from analytics import AdvancedChartGenerator
import strategies  # Module dynamique

# Configuration Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("phoenix_core.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PhoenixOrchestrator")

class PhoenixBot:
    """
    Contrôleur Principal (Orchestrator).
    
    Responsabilités :
    1. Initialiser les services (Data, DB, Exec, Stratégies).
    2. Boucle d'événements (Tick).
    3. Router les données : Exchange -> MarketData -> Strategy -> Execution -> DB.
    
    Ne contient AUCUNE logique de calcul financier ou statistique.
    """

    def __init__(self, config_path: str = "config.json"):
        load_dotenv()
        self.is_running = False
        self.config = self._load_config(config_path)
        
        # --- 1. Injection des Services ---
        
        # Base de données (Persistance)
        self.db = DatabaseHandler()
        
        # Gestionnaire de Données Marché (Mémoire Tampon)
        # Remplace la logique pd.concat lourde
        self.market_data = MarketDataManager(
            max_history_size=self.config['system'].get('max_history_size', 1000)
        )
        
        # Gestionnaire d'Exécution (Calculs de risque, Ordres)
        self.execution = ExecutionManager(self.config)
        
        # Moteur d'Analyse (Reporting)
        self.analytics = AdvancedChartGenerator(
            output_dir=self.config['system'].get('output_dir', 'logs')
        )
        
        # --- 2. État Interne (Modèles Typés) ---
        
        # Le Portfolio est la "Source de Vérité" de l'état financier
        self.portfolio = self._initialize_portfolio()
        
        # Stratégies actives (Mappage Nom -> Instance)
        self.active_strategies = strategies.get_active_strategies(self.config)
        
        # Connecteur Exchange (Initié dans setup)
        self.exchange: Optional[ccxt.Exchange] = None
        
        logger.info(f"🤖 Phoenix Bot initialisé avec {len(self.active_strategies)} stratégies.")

    def _load_config(self, path: str) -> dict:
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.critical(f"❌ Configuration illisible : {e}")
            raise

    def _initialize_portfolio(self) -> Portfolio:
        """Charge l'état depuis la DB ou crée un nouveau portfolio."""
        try:
            # Charge les items de portfolio depuis la DB
            portfolio_items = self.db.load_portfolio()
            if portfolio_items and len(portfolio_items) > 0:
                # Reconstruit le portfolio à partir des items
                return self._load_portfolio_from_items(portfolio_items)
        except Exception as e:
            logger.warning(f"⚠️ Impossible de charger l'état précédent ({e}). Démarrage à neuf.")
        
        # Portfolio vierge
        return Portfolio(
            initial_capital=self.config['portfolio']['initial_capital_per_strategy'],
            current_cash=self.config['portfolio']['initial_capital_per_strategy'],
            currency=self.config['portfolio']['currency']
        )

    def _load_portfolio_from_items(self, items: List[PortfolioItem]) -> Portfolio:
        """Reconstruit un objet Portfolio à partir des items de la base de données."""
        # Trouve l'item le plus récent pour les métadonnées
        latest_item = max(items, key=lambda x: x.timestamp)
        
        # Récupère toutes les positions ouvertes
        positions = []
        for item in items:
            if item.position_id and item.status == "OPEN":
                positions.append(
                    Position(
                        symbol=item.symbol,
                        strategy_name=item.strategy_name,
                        quantity=item.quantity,
                        entry_price=item.entry_price,
                        current_price=item.current_price,
                        entry_time=item.entry_time
                    )
                )
        
        # Crée et retourne le portfolio
        portfolio = Portfolio(
            initial_capital=latest_item.initial_capital,
            current_cash=latest_item.current_cash,
            currency=latest_item.currency,
            positions=positions
        )
        
        # Restaure l'historique des snapshots si disponible
        for item in items:
            if item.snapshot_data:
                portfolio.history_snapshots.append(item.snapshot_data)
        
        return portfolio

    def _convert_portfolio_to_items(self) -> List[PortfolioItem]:
        """Convertit l'état actuel du portfolio en items pour la base de données."""
        items = []
        timestamp = datetime.now(timezone.utc)
        
        # Item principal avec l'état global
        main_item = PortfolioItem(
            timestamp=timestamp,
            initial_capital=self.portfolio.initial_capital,
            current_cash=self.portfolio.current_cash,
            currency=self.portfolio.currency,
            total_equity=self.portfolio.total_equity,
            symbol="GLOBAL",
            position_id=None,
            status="SUMMARY"
        )
        items.append(main_item)
        
        # Items pour chaque position ouverte
        for pos in self.portfolio.positions:
            pos_item = PortfolioItem(
                timestamp=timestamp,
                initial_capital=self.portfolio.initial_capital,
                current_cash=self.portfolio.current_cash,
                currency=self.portfolio.currency,
                total_equity=self.portfolio.total_equity,
                symbol=pos.symbol,
                strategy_name=pos.strategy_name,
                position_id=id(pos),  # Identifiant unique
                quantity=pos.quantity,
                entry_price=pos.entry_price,
                current_price=pos.current_price,
                entry_time=pos.entry_time,
                status="OPEN"
            )
            items.append(pos_item)
        
        return items

    async def setup(self):
        """Configuration asynchrone (Connexions API)."""
        exchange_id = 'binance'  # Configurable
        exchange_class = getattr(ccxt, exchange_id)
        
        self.exchange = exchange_class({
            'apiKey': os.environ.get('BINANCE_API_KEY'),
            'secret': os.environ.get('BINANCE_SECRET_KEY'),
            'timeout': 30000,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'} 
        })
        
        # Chargement des marchés (nécessaire pour les précisions)
        await self.exchange.load_markets()
        logger.info("✅ Connexion Exchange établie.")

    async def shutdown(self):
        """Arrêt propre."""
        self.is_running = False
        if self.exchange:
            await self.exchange.close()
        
        # Sauvegarde finale de l'état
        if self.portfolio:
            portfolio_items = self._convert_portfolio_to_items()
            self.db.save_portfolio(portfolio_items)
            
        logger.info("👋 Arrêt complet du système.")

    async def run(self):
        """Boucle principale (Event Loop)."""
        await self.setup()
        self.is_running = True
        
        pairs = self.config['trading']['pairs']
        timeframe = self.config['trading']['timeframe']
        
        logger.info(f"🚀 Démarrage de la boucle de trading sur {len(pairs)} paires.")
        
        while self.is_running:
            start_time = datetime.now()
            
            # Traitement parallèle des paires avec gestion d'erreurs
            tasks = []
            for pair in pairs:
                task = asyncio.create_task(self._process_pair(pair, timeframe))
                task.add_done_callback(self._handle_task_exception)
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            
            # Synchronisation & Reporting périodique
            await self._periodic_sync()
            
            # Respect du Rate Limit global
            elapsed = (datetime.now() - start_time).total_seconds()
            sleep_time = max(1.0, 60.0 - elapsed)  # Attend la prochaine minute environ
            await asyncio.sleep(sleep_time)

    def _handle_task_exception(self, task: asyncio.Task):
        """Gère les exceptions des tâches asynchrones."""
        if task.exception():
            logger.error(f"❌ Erreur dans tâche asynchrone: {task.exception()}")

    async def _process_pair(self, symbol: str, timeframe: str):
        """
        Logique atomique pour une paire.
        1. Fetch Market Data
        2. Update Model
        3. Run Strategies
        4. Execute Signals
        """
        try:
            # 1. Acquisition de données (IO Bound)
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=100)  # Augmenté pour avoir suffisamment d'historique
            if not ohlcv or len(ohlcv) < 50:  # Au moins 50 bougies nécessaires
                logger.debug(f"⏳ Données insuffisantes pour {symbol}")
                return

            # On prend la dernière bougie clôturée (avant-dernière liste)
            last_closed = ohlcv[-2] if len(ohlcv) > 1 else ohlcv[-1]
            current_candle = MarketCandle(
                timestamp=last_closed[0],
                symbol=symbol,
                open=last_closed[1],
                high=last_closed[2],
                low=last_closed[3],
                close=last_closed[4],
                volume=last_closed[5]
            )

            # 2. Mise à jour du MarketDataManager
            self.market_data.add_candle(current_candle)
            
            # Mise à jour du prix courant dans le portfolio
            self.portfolio.update_market_prices({symbol: current_candle.close})
            
            # 3. Récupération de l'historique pour les stratégies
            df_history = self.market_data.get_history_dataframe(symbol, required_rows=50)
            
            if df_history is None or df_history.empty:
                logger.debug(f"⏳ Historique insuffisant pour {symbol}")
                return

            # 4. Exécution des Stratégies
            for strategy in self.active_strategies:
                try:
                    signal_obj: Signal = strategy.analyze(df_history, self.portfolio)
                    
                    if signal_obj.signal_type != SignalType.HOLD:
                        logger.info(f"💡 Signal détecté: {signal_obj}")
                        await self._execute_signal(signal_obj, current_candle.close)
                        
                except Exception as e:
                    logger.error(f"⚠️ Erreur stratégie {strategy.name} sur {symbol}: {e}")

        except ccxt.NetworkError as e:
            logger.warning(f"📡 Erreur réseau sur {symbol}: {e}")
        except Exception as e:
            logger.error(f"❌ Erreur critique boucle {symbol}: {e}", exc_info=True)

    async def _execute_signal(self, signal: Signal, current_price: float):
        """Délègue l'exécution et met à jour le Portfolio."""
        
        # 1. Calculs pré-trade
        execution_plan = self.execution.plan_trade(
            signal=signal, 
            portfolio=self.portfolio, 
            current_price=current_price
        )
        
        if not execution_plan:
            return

        # 2. Envoi Ordre Exchange
        try:
            logger.info(f"⚡ Exécution ordre {signal.signal_type} sur {signal.symbol}")
            
            # 3. Mise à jour du Modèle Portfolio
            if signal.signal_type == SignalType.BUY:
                new_pos = Position(
                    symbol=signal.symbol,
                    strategy_name=signal.strategy_name,
                    quantity=execution_plan['quantity'],
                    entry_price=current_price,
                    current_price=current_price
                )
                self.portfolio.add_position(new_pos)
                self.portfolio.current_cash -= (execution_plan['quantity'] * current_price)
                
            elif signal.signal_type == SignalType.SELL:
                # Clôture Position
                trade: Trade = self.portfolio.close_position(signal.symbol, current_price)
                
                # Convertir et sauvegarder le trade
                trade_record = TradeRecord(
                    timestamp=datetime.now(timezone.utc),
                    symbol=trade.symbol,
                    strategy_name=trade.strategy_name,
                    side="SELL",
                    quantity=trade.quantity,
                    entry_price=trade.entry_price,
                    exit_price=trade.exit_price,
                    pnl=trade.pnl,
                    pnl_percent=trade.pnl_percent
                )
                self.db.record_trade(trade_record)

            # 4. Sauvegarde État Portfolio
            portfolio_items = self._convert_portfolio_to_items()
            self.db.save_portfolio(portfolio_items)

        except Exception as e:
            logger.error(f"💥 Échec exécution ordre: {e}")

    async def _periodic_sync(self):
        """Tâches de fond périodiques."""
        try:
            # Snapshot des performances
            snapshot = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_equity": float(self.portfolio.total_equity),
                "cash": float(self.portfolio.current_cash),
                "positions_count": len(self.portfolio.positions),
                "unrealized_pnl": float(self.portfolio.unrealized_pnl),
                "realized_pnl": float(self.portfolio.realized_pnl)
            }
            self.portfolio.history_snapshots.append(snapshot)
            self.db.save_portfolio_history(snapshot)
            
        except Exception as e:
            logger.error(f"⚠️ Erreur sync périodique: {e}")

# ==============================================================================
# Point d'entrée
# ==============================================================================

if __name__ == "__main__":
    bot = PhoenixBot()
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Gestion propre des signaux (CTRL+C)
    def handle_exit():
        logger.info("🛑 Signal d'arrêt reçu...")
        asyncio.create_task(bot.shutdown())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_exit)
        
    try:
        loop.run_until_complete(bot.run())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()
