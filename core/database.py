import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

class Database:
    """
    Authoritative Supabase Connector.
    Enforces strict strategy isolation by requiring strategy_id for all operations.
    Replaces the legacy SQLite DatabaseHandler.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
            cls._instance._init_connection()
        return cls._instance

    def _init_connection(self):
        """Initializes the connection to Supabase."""
        url: str = os.environ.get("SUPABASE_URL")
        key: str = os.environ.get("SUPABASE_KEY")

        if not url or not key:
            raise ValueError("❌ Missing SUPABASE_URL or SUPABASE_KEY in environment variables.")

        try:
            self.client: Client = create_client(url, key)
            print(f"✅ Connected to Supabase: {url}")
        except Exception as e:
            print(f"❌ Failed to connect to Supabase: {e}")
            raise e

    # ------------------------------------------------------------------
    # 1. STRATEGY MANAGEMENT
    # ------------------------------------------------------------------

    def register_strategy(self, name: str, initial_capital: float = 10000.0) -> str:
        """
        Registers a strategy. If it exists, returns its ID.
        If it's new, creates it and initializes its cash wallet.
        """
        # 1. Try to fetch existing strategy by name
        res = self.client.table("strategies").select("id").eq("name", name).execute()
        
        if res.data:
            strategy_id = res.data[0]['id']
            print(f"ℹ️ Strategy '{name}' found: {strategy_id}")
            return strategy_id

        # 2. Create new strategy
        try:
            new_strat = self.client.table("strategies").insert({"name": name}).execute()
            strategy_id = new_strat.data[0]['id']

            # 3. Initialize Cash
            self.client.table("strategy_cash").insert({
                "strategy_id": strategy_id,
                "initial_cash": initial_capital,
                "available_cash": initial_capital
            }).execute()

            print(f"🆕 Strategy '{name}' created: {strategy_id}")
            return strategy_id

        except Exception as e:
            print(f"❌ Error registering strategy '{name}': {e}")
            raise e

    # ------------------------------------------------------------------
    # 2. CASH MANAGEMENT
    # ------------------------------------------------------------------

    def get_strategy_cash(self, strategy_id: str) -> float:
        """Fetch available cash for a specific strategy."""
        res = self.client.table("strategy_cash")\
            .select("available_cash")\
            .eq("strategy_id", strategy_id)\
            .execute()
        
        if res.data and res.data[0]['available_cash'] is not None:
            return float(res.data[0]['available_cash'])
        return 0.0

    def update_cash(self, strategy_id: str, amount: float):
        """
        Update the available cash. 
        NOTE: 'amount' is the NEW total amount, not a delta.
        """
        self.client.table("strategy_cash")\
            .update({
                "available_cash": amount, 
                "updated_at": datetime.utcnow().isoformat()
            })\
            .eq("strategy_id", strategy_id)\
            .execute()

    # ------------------------------------------------------------------
    # 3. TRADE LOGGING
    # ------------------------------------------------------------------

    def log_trade(self, strategy_id: str, symbol: str, side: str, price: float, quantity: float, fees: float = 0.0, timestamp: datetime = None):
        """Logs a paper trade execution."""
        if timestamp is None:
            timestamp = datetime.utcnow()
            
        payload = {
            "strategy_id": strategy_id,
            "symbol": symbol,
            "side": side.upper(),  # BUY or SELL
            "price": price,
            "quantity": quantity,
            "fees": fees,
            "executed_at": timestamp.isoformat()
        }
        
        result = self.client.table("trades").insert(payload).execute()
        if result.data:
            return result.data[0]['id']
        return None

    def get_strategy_trades(self, strategy_id: str) -> List[Dict]:
        """Fetch trade history strictly for this strategy."""
        res = self.client.table("trades")\
            .select("*")\
            .eq("strategy_id", strategy_id)\
            .order("executed_at", desc=True)\
            .execute()
        return res.data

    # ------------------------------------------------------------------
    # 4. POSITION MANAGEMENT
    # ------------------------------------------------------------------

    def open_position(self, strategy_id: str, symbol: str, side: str, quantity: float, entry_price: float, timestamp: datetime = None):
        """Creates a new OPEN position record."""
        if timestamp is None:
            timestamp = datetime.utcnow()
            
        payload = {
            "strategy_id": strategy_id,
            "symbol": symbol,
            "side": side.upper(),  # LONG or SHORT
            "quantity": quantity,
            "entry_price": entry_price,
            "status": "OPEN",
            "opened_at": timestamp.isoformat()
        }
        
        result = self.client.table("positions").insert(payload).execute()
        if result.data:
            return result.data[0]['id']
        return None

    def close_position(self, position_id: str, exit_price: float, timestamp: datetime = None):
        """
        Marks a specific position as CLOSED.
        Returns the closed position for PnL calculation.
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
            
        # First get the position to calculate PnL
        position_res = self.client.table("positions")\
            .select("*")\
            .eq("id", position_id)\
            .execute()
        
        if not position_res.data:
            print(f"⚠️ Position {position_id} not found")
            return None
            
        position = position_res.data[0]
        
        # Calculate realized PnL
        entry_price = position.get('entry_price')
        quantity = position.get('quantity')
        side = position.get('side')
        
        if entry_price is None or quantity is None:
            print(f"⚠️ Position {position_id} has null entry_price or quantity")
            realized_pnl = 0.0
        else:
            entry_price = float(entry_price)
            quantity = float(quantity)
            
            if side == "LONG":
                realized_pnl = (exit_price - entry_price) * quantity
            else:  # SHORT
                realized_pnl = (entry_price - exit_price) * quantity
        
        # Update the position
        self.client.table("positions")\
            .update({
                "status": "CLOSED", 
                "exit_price": exit_price,
                "realized_pnl": realized_pnl,
                "closed_at": timestamp.isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            })\
            .eq("id", position_id)\
            .execute()
        
        return position

    def get_strategy_positions(self, strategy_id: str, status: str = "OPEN") -> List[Dict]:
        """Fetch positions filtered by strategy and status."""
        res = self.client.table("positions")\
            .select("*")\
            .eq("strategy_id", strategy_id)\
            .eq("status", status)\
            .execute()
        
        # Ensure all numeric fields are floats, not None
        positions = []
        for pos in res.data:
            # Convert numeric fields, handling None values
            if pos.get('quantity') is not None:
                pos['quantity'] = float(pos['quantity'])
            if pos.get('entry_price') is not None:
                pos['entry_price'] = float(pos['entry_price'])
            if pos.get('exit_price') is not None:
                pos['exit_price'] = float(pos['exit_price'])
            positions.append(pos)
        
        return positions

    # ------------------------------------------------------------------
    # 5. PERFORMANCE METRICS & HEARTBEAT
    # ------------------------------------------------------------------

    def log_performance(self, strategy_id: str, metrics: Dict[str, Any]):
        """
        Logs a snapshot of performance metrics.
        """
        # Map 'pnl_percentage' to 'total_return_pct' (column name in database)
        if 'pnl_percentage' in metrics:
            metrics['total_return_pct'] = metrics.pop('pnl_percentage')
        
        # Filter metrics to ensure we only send what the DB expects
        valid_keys = {"total_pnl", "sharpe_ratio", "max_drawdown", "win_rate", 
                     "trades_count", "unrealized_pnl", "asset", "price", 
                     "total_return_pct", "win", "entry_price", "exit_price", 
                     "holding_period", "calculated_at"}
        
        payload = {k: v for k, v in metrics.items() if k in valid_keys}
        payload["strategy_id"] = strategy_id
        
        # Convert holding_period to days if present
        if "holding_period" in payload and payload["holding_period"] is not None:
            # Ensure holding_period is a float representing days
            if not isinstance(payload["holding_period"], (int, float)):
                try:
                    payload["holding_period"] = float(payload["holding_period"])
                except (ValueError, TypeError):
                    payload["holding_period"] = 0.0
        
        # Use current time for calculation if not provided
        if "calculated_at" not in payload:
            payload["calculated_at"] = datetime.utcnow().isoformat()
        
        try:
            self.client.table("performance_metrics").insert(payload).execute()
        except Exception as e:
            print(f"⚠️ Error logging performance for strategy {strategy_id}: {e}")

    # ------------------------------------------------------------------
    # 6. SYSTEM LOGS (replaces execution_metadata)
    # ------------------------------------------------------------------

    def log_system_event(self, level: str, module: str, message: str, details: Dict[str, Any] = None):
        """
        Logs system events to the system_logs table.
        Use this instead of the non-existent execution_metadata table.
        """
        payload = {
            "level": level.upper(),  # INFO, WARNING, ERROR, DEBUG
            "module": module,  # e.g., "execution", "strategy", "market_data"
            "message": message,
            "details": details or {},
            "created_at": datetime.utcnow().isoformat()
        }
        
        try:
            self.client.table("system_logs").insert(payload).execute()
        except Exception as e:
            print(f"⚠️ Error logging system event: {e}")

    # ------------------------------------------------------------------
    # 7. PORTFOLIO HISTORY (for backtesting and analytics)
    # ------------------------------------------------------------------

    def log_portfolio_history(self, strategy_id: str, timestamp: datetime, equity: float, 
                            cash: float, positions_value: float):
        """
        Logs portfolio snapshot for equity curve reconstruction.
        """
        payload = {
            "strategy_id": strategy_id,
            "timestamp": timestamp.isoformat(),
            "total_equity": equity,
            "cash": cash,
            "positions_value": positions_value,
            "logged_at": datetime.utcnow().isoformat()
        }
        self.client.table("portfolio_history").insert(payload).execute()

    def get_portfolio_history(self, strategy_id: str, limit: int = 1000) -> List[Dict]:
        """Fetch portfolio history for a strategy."""
        res = self.client.table("portfolio_history")\
            .select("*")\
            .eq("strategy_id", strategy_id)\
            .order("timestamp")\
            .limit(limit)\
            .execute()
        return res.data

    # ------------------------------------------------------------------
    # 8. MARKET DATA CACHE
    # ------------------------------------------------------------------

    def cache_market_data(self, symbol: str, timeframe: str, data: Dict, ttl_minutes: int = 5):
        """
        Caches market data to reduce API calls.
        """
        expires_at = datetime.utcnow().timestamp() + (ttl_minutes * 60)
        
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "data": data,
            "expires_at": expires_at,
            "cached_at": datetime.utcnow().isoformat()
        }
        
        # Delete old cache for this symbol/timeframe
        self.client.table("market_data_cache")\
            .delete()\
            .eq("symbol", symbol)\
            .eq("timeframe", timeframe)\
            .execute()
        
        # Insert new cache
        self.client.table("market_data_cache").insert(payload).execute()

    def get_cached_market_data(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """
        Retrieves cached market data if not expired.
        """
        current_time = datetime.utcnow().timestamp()
        
        res = self.client.table("market_data_cache")\
            .select("*")\
            .eq("symbol", symbol)\
            .eq("timeframe", timeframe)\
            .gt("expires_at", current_time)\
            .execute()
        
        if res.data:
            return res.data[0]["data"]
        return None

    # ------------------------------------------------------------------
    # 9. STRATEGY PARAMETERS (for optimization)
    # ------------------------------------------------------------------

    def save_strategy_parameters(self, strategy_id: str, parameters: Dict[str, Any], 
                               performance: Dict[str, Any]):
        """
        Saves optimized parameters for a strategy.
        """
        payload = {
            "strategy_id": strategy_id,
            "parameters": parameters,
            "performance": performance,
            "saved_at": datetime.utcnow().isoformat()
        }
        self.client.table("strategy_parameters").insert(payload).execute()

    def get_best_strategy_parameters(self, strategy_id: str) -> Optional[Dict]:
        """
        Retrieves the best performing parameters for a strategy.
        """
        res = self.client.table("strategy_parameters")\
            .select("*")\
            .eq("strategy_id", strategy_id)\
            .order("performance->'sharpe_ratio'", desc=True)\
            .limit(1)\
            .execute()
        
        if res.data:
            return res.data[0]
        return None

    # ------------------------------------------------------------------
    # 10. STRATEGY STATS
    # ------------------------------------------------------------------

    def get_strategy_stats(self, strategy_id: str) -> Dict[str, Any]:
        """Get basic statistics for a strategy."""
        stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'open_positions': 0
        }
        
        try:
            # Get trade count
            trades = self.get_strategy_trades(strategy_id)
            stats['total_trades'] = len(trades)
            
            # Get open positions count
            open_positions = self.get_strategy_positions(strategy_id, status="OPEN")
            stats['open_positions'] = len(open_positions)
            
            # Get performance metrics (most recent)
            perf_res = self.client.table("performance_metrics")\
                .select("*")\
                .eq("strategy_id", strategy_id)\
                .order("calculated_at", desc=True)\
                .limit(1)\
                .execute()
                
            if perf_res.data:
                stats['total_pnl'] = perf_res.data[0].get('total_pnl', 0.0)
                stats['winning_trades'] = perf_res.data[0].get('winning_trades', 0)
                
        except Exception as e:
            print(f"⚠️ Error getting strategy stats: {e}")
            
        return stats

    # ------------------------------------------------------------------
    # 11. TRANSACTION SUPPORT
    # ------------------------------------------------------------------

    def execute_transaction(self, operations: List[Dict]):
        """
        Executes multiple database operations in a transaction.
        Each operation should be a dict with keys: 'table', 'operation', 'data'
        Example: {'table': 'trades', 'operation': 'insert', 'data': {...}}
        """
        # Note: Supabase doesn't support multi-table transactions in the Python client.
        # This is a simulated transaction that executes sequentially.
        # For true transactions, consider using Supabase's RPC functions.
        
        results = []
        for op in operations:
            try:
                table = self.client.table(op['table'])
                operation = op['operation']
                data = op['data']
                
                if operation == 'insert':
                    result = table.insert(data).execute()
                elif operation == 'update':
                    result = table.update(data).execute()
                elif operation == 'delete':
                    result = table.delete().execute()
                else:
                    raise ValueError(f"Unknown operation: {operation}")
                
                results.append(result)
                
            except Exception as e:
                # If any operation fails, we can't rollback easily
                # Log error and raise
                print(f"❌ Transaction failed at operation {op}: {e}")
                raise e
        
        return results