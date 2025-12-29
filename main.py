import asyncio
import logging
import signal
import sys
import json
from typing import Dict, Any, Optional, List

# Third-party imports
from dotenv import load_dotenv
import pandas as pd

# Project imports
from core.database import Database
from core.models import Signal, SignalType, PositionSide
from core.market_data import MarketDataManager
from core.execution import ExecutionManager 
import strategies.strategies as strategies
from datetime import datetime

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("PhoenixOrchestrator")

class PhoenixBot:
    """
    Cloud-Native Orchestrator.
    Manages the lifecycle of strategies, efficient data fetching, and persists state to Supabase.
    """

    def __init__(self):
        load_dotenv()
        self.is_running = False
        
        # 1. Load Configuration
        self.config = self._load_config()
        
        # 2. Connect to Supabase
        try:
            self.db = Database()
            logger.info("✅ Database connected successfully.")
        except Exception as e:
            logger.critical(f"❌ Database connection failed: {e}")
            sys.exit(1)

        # 3. Initialize Services
        self.market_data = MarketDataManager(self.config)
        self.execution = ExecutionManager(self.db, self.config)
        
        # 4. Strategy Registry
        self.strategy_db_ids: Dict[str, str] = {} 
        self.strategy_instances: Dict[str, Any] = {}
        
        self.active_strategies = self.config['strategies']['active_strategies']
        self.trading_pairs = self.config['trading']['pairs']
        
        # 5. Data caching for API optimization
        self.data_cache: Dict[str, Dict] = {
            'market_data': {},
            'last_fetch': {}
        }
        self.cache_ttl = self._get_cache_ttl(self.config['trading'].get('timeframe', '5m'))  # Cache TTL in seconds

    def _load_config(self) -> Dict[str, Any]:
        """Loads configuration from JSON file."""
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
                logger.info("✅ Configuration loaded.")
                return config
        except Exception as e:
            logger.critical(f"❌ Failed to load config.json: {e}")
            sys.exit(1)

    async def initialize(self):
        """Registers strategies and warms up the engine."""
        logger.info("🚀 Initializing Phoenix Engine...")
        
        initial_capital = self.config['portfolio']['initial_capital_per_strategy']
        
        for strat_name in self.active_strategies:
            try:
                # 1. Check if strategy Class exists in strategies.py
                if not hasattr(strategies, strat_name):
                    raise ValueError(f"Strategy Class '{strat_name}' not found in strategies.py")
                
                # 2. Instantiate the Strategy Class
                StrategyClass = getattr(strategies, strat_name)
                strategy_instance = StrategyClass(self.config)
                self.strategy_instances[strat_name] = strategy_instance

                # 3. Register in Database (Get unique ID)
                s_id = self.db.register_strategy(strat_name, initial_capital=initial_capital)
                self.strategy_db_ids[strat_name] = s_id
                
                logger.info(f"   ✅ Registered: {strat_name} ({s_id})")
                
            except Exception as e:
                logger.error(f"   ❌ Failed to register {strat_name}: {e}")
                # Remove from active strategies if registration fails
                self.active_strategies.remove(strat_name)

    async def run(self):
        """Main Event Loop."""
        await self.initialize()
        self.is_running = True

        # ADD: Start time for max runtime check
        from datetime import datetime
        start_time = datetime.now()
        max_runtime_seconds = self.config['system']['max_runtime_minutes'] * 60

        logger.info(f"🟢 System Online. Tracking {len(self.trading_pairs)} pairs for {len(self.strategy_instances)} strategies.")
        logger.info(f"⏰ Max runtime: {self.config['system']['max_runtime_minutes']} minutes")

        while self.is_running:
            # ADD: Check max runtime
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > max_runtime_seconds:
                logger.info(f"🕒 Max runtime of {self.config['system']['max_runtime_minutes']} minutes reached. Shutting down.")
                await self.shutdown()
                break
            
            try:
                await self._process_cycle()
                
                # Sleep interval based on timeframe
                timeframe = self.config['trading'].get('timeframe', '1m')
                sleep_seconds = self._get_sleep_interval(timeframe)
                
                logger.debug(f"   ... Cycle complete. Sleeping {sleep_seconds}s.")
                await asyncio.sleep(sleep_seconds)

            except KeyboardInterrupt:
                await self.shutdown()
                break
            except Exception as e:
                logger.error(f"⚠️ Unexpected error in main loop: {e}", exc_info=True)
                await asyncio.sleep(10)

    def _get_sleep_interval(self, timeframe: str) -> int:
        """Determines sleep interval based on trading timeframe."""
        timeframe_map = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400
        }
        return timeframe_map.get(timeframe, 60)

    def _get_cache_ttl(self, timeframe: str) -> int:
        """Determines cache TTL based on trading timeframe."""
        timeframe_map = {
            '1m': 30,      # 30 seconds for 1m data
            '5m': 150,     # 2.5 minutes for 5m data
            '15m': 450,    # 7.5 minutes for 15m data
            '1h': 1800,    # 30 minutes for 1h data
            '4h': 7200,    # 2 hours for 4h data
            '1d': 86400,   # 1 day for daily data
        }
        return timeframe_map.get(timeframe, 150)  # Default 2.5 minutes

    async def _process_cycle(self):
        """
        One complete trading cycle:
        1. Fetch Data Once (Optimization)
        2. Run Strategies (Logic)
        3. Execute Signals (Risk & Order Mgmt)
        4. Log Heartbeat
        """
        timeframe = self.config['trading'].get('timeframe', '1m')
        # --- 1. Batch Data Fetching with Caching ---
        market_snapshot = await self.market_data.fetch_latest_candles(
            pairs=self.trading_pairs,
            timeframe=timeframe,
            limit=500  # Ensure enough data for strategy indicators
        )

        if not market_snapshot:
            logger.warning("⚠️ No market data received this cycle.")
            return

        # --- 2. Strategy Execution Loop ---
        for strat_name, strat_instance in self.strategy_instances.items():
            strat_id = self.strategy_db_ids.get(strat_name)
            if not strat_id:
                logger.warning(f"⚠️ No DB ID found for strategy: {strat_name}")
                continue
            
            # Fetch all open positions for this strategy
            open_positions = self.db.get_strategy_positions(strat_id, status="OPEN")
            
            # Iterate through all available market data
            for pair, df in market_snapshot.items():

                # FIX: Check if we have enough data for this strategy
                if len(df) < strat_instance.min_data_required():
                    logger.debug(f"Skipping {strat_name} on {pair}: insufficient data ({len(df)} < {strat_instance.min_data_required()})")
                    continue
                
                # Get current price (last close) for Execution/Logging
                current_price = df.iloc[-1]['close']
                current_time = df.iloc[-1]['timestamp']
                
                # Check if we already have a position for this pair
                current_pos = next((p for p in open_positions if p['symbol'] == pair), None)
                
                # --- Heartbeat: Log Unrealized PnL ---
                if current_pos:
                    await self._log_heartbeat(strat_id, current_pos, current_price, current_time)

                # --- Run Strategy Logic ---
                try:
                    # Pass the DataFrame to the strategy instance
                    signal = strat_instance.generate_signal(df, pair)
                    
                    if signal and signal.signal_type != SignalType.HOLD:
                        # --- 3. Execute Signal ---
                        logger.info(f"🔔 {strat_name} on {pair}: {signal.signal_type}")
                        await self.execution.process_signal(strat_id, signal, current_price)
                        
                except Exception as e:
                    logger.error(f"❌ Error running {strat_name} on {pair}: {e}", exc_info=True)

    async def _get_cached_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fetches market data with caching to reduce API calls.
        """
        cache_key = f"{symbol}_{self.config['trading'].get('timeframe', '1m')}"
        current_time = datetime.now()
        
        # Check cache
        if cache_key in self.data_cache['market_data']:
            cache_time = self.data_cache['last_fetch'].get(cache_key)
            if cache_time and (current_time - cache_time).seconds < self.cache_ttl:
                return self.data_cache['market_data'][cache_key]
        
        # Fetch new data
        try:
            df = await self.market_data.get_latest_candle(symbol)
            if df is not None and not df.empty:
                self.data_cache['market_data'][cache_key] = df
                self.data_cache['last_fetch'][cache_key] = current_time
            return df
        except Exception as e:
            logger.error(f"❌ Error fetching data for {symbol}: {e}")
            return None

    async def _log_heartbeat(self, strategy_id: str, position: Dict, current_price: float, current_time: datetime):
        """
        Calculates and logs unrealized PnL.
        """
        try:
            entry_price = float(position['entry_price'])
            quantity = float(position['quantity'])
            side = position['side']
            
            if side == PositionSide.LONG:
                unrealized_pnl = (current_price - entry_price) * quantity
            else:  # SHORT
                unrealized_pnl = (entry_price - current_price) * quantity
                
            # Calculate return percentage
            position_value = entry_price * quantity
            return_pct = (unrealized_pnl / position_value) * 100 if position_value > 0 else 0
            
            # Log to database
            self.db.log_performance(strategy_id, {
                "unrealized_pnl": unrealized_pnl,
                "asset": position['symbol'],
                "price": current_price,
                "total_return_pct": return_pct,
                "timestamp": current_time.isoformat()
            })
            
            # Log to console (optional, for debugging)
            logger.debug(f"💓 {position['symbol']}: Unrealized PnL: ${unrealized_pnl:.2f} ({return_pct:.2f}%)")
            
        except Exception as e:
            logger.error(f"❌ Failed to log heartbeat for {strategy_id}: {e}")

    async def shutdown(self):
        """Graceful Shutdown."""
        self.is_running = False
        logger.info("🛑 Shutting down Phoenix Orchestrator...")
        
        # Close any open positions (optional)
        for strat_name, strat_id in self.strategy_db_ids.items():
            positions = self.db.get_strategy_positions(strat_id, status="OPEN")
            if positions:
                logger.info(f"⚠️ {strat_name} has {len(positions)} open position(s) on shutdown")
        
        logger.info("👋 Goodbye.")
        sys.exit(0)

if __name__ == "__main__":
    bot = PhoenixBot()
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    def handle_exit(signum, frame):
        print(f"\n🛑 Received signal {signum}, shutting down...")
        asyncio.create_task(bot.shutdown())
        
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    try:
        loop.run_until_complete(bot.run())
    except KeyboardInterrupt:
        print("\n🛑 Manual shutdown requested")
        loop.run_until_complete(bot.shutdown())
    finally:
        loop.close()