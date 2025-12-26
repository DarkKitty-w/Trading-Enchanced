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
        
        if res.data:
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

    def log_trade(self, strategy_id: str, symbol: str, side: str, price: float, quantity: float, fees: float = 0.0):
        """Logs a paper trade execution."""
        payload = {
            "strategy_id": strategy_id,
            "symbol": symbol,
            "side": side.upper(),  # BUY or SELL
            "price": price,
            "quantity": quantity,
            "fees": fees,
            "executed_at": datetime.utcnow().isoformat()
        }
        self.client.table("trades").insert(payload).execute()

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

    def open_position(self, strategy_id: str, symbol: str, side: str, quantity: float, entry_price: float):
        """Creates a new OPEN position record."""
        payload = {
            "strategy_id": strategy_id,
            "symbol": symbol,
            "side": side.upper(),  # LONG or SHORT
            "quantity": quantity,
            "entry_price": entry_price,
            "status": "OPEN",
            "opened_at": datetime.utcnow().isoformat()
        }
        self.client.table("positions").insert(payload).execute()

    def close_position(self, strategy_id: str, symbol: str, exit_price: float):
        """
        Marks OPEN positions for this strategy/symbol as CLOSED.
        Returns the closed position for PnL calculation.
        """
        # Find open positions for this specific strategy & symbol
        open_positions = self.client.table("positions")\
            .select("*")\
            .eq("strategy_id", strategy_id)\
            .eq("symbol", symbol)\
            .eq("status", "OPEN")\
            .execute()

        if not open_positions.data:
            print(f"⚠️ No open position found to close for {symbol} (Strategy ID: {strategy_id})")
            return None

        # Update them to CLOSED
        ids_to_close = [p['id'] for p in open_positions.data]
        
        self.client.table("positions")\
            .update({
                "status": "CLOSED", 
                "exit_price": exit_price,
                "closed_at": datetime.utcnow().isoformat()
            })\
            .in_("id", ids_to_close)\
            .execute()
        
        return open_positions.data[0]  # Return first closed position

    def get_strategy_positions(self, strategy_id: str, status: str = "OPEN") -> List[Dict]:
        """Fetch positions filtered by strategy and status."""
        res = self.client.table("positions")\
            .select("*")\
            .eq("strategy_id", strategy_id)\
            .eq("status", status)\
            .execute()
        return res.data

    # ------------------------------------------------------------------
    # 5. PERFORMANCE METRICS & HEARTBEAT
    # ------------------------------------------------------------------

    def log_performance(self, strategy_id: str, metrics: Dict[str, Any]):
        """
        Logs a snapshot of performance metrics.
        """
        # Filter metrics to ensure we only send what the DB expects
        valid_keys = {"total_pnl", "sharpe_ratio", "max_drawdown", "win_rate", 
                     "trades_count", "unrealized_pnl", "asset", "price", 
                     "return_pct", "timestamp", "win"}
        payload = {k: v for k, v in metrics.items() if k in valid_keys}
        
        payload["strategy_id"] = strategy_id
        
        # Use provided timestamp or current time
        if "timestamp" not in payload:
            payload["calculated_at"] = datetime.utcnow().isoformat()
        else:
            payload["calculated_at"] = payload.pop("timestamp")
        
        self.client.table("performance_metrics").insert(payload).execute()

    # ------------------------------------------------------------------
    # 6. PORTFOLIO HISTORY (for backtesting and analytics)
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
    # 7. MARKET DATA CACHE
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
    # 8. STRATEGY PARAMETERS (for optimization)
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
    # 9. TRANSACTION SUPPORT
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
