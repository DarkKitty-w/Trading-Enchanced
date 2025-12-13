import os
import logging
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from supabase import create_client, Client

class DatabaseHandler:
    def __init__(self):
        """
        Supabase Database Handler.
        Updated to match PhoenixBot's main.py data structures (Lists).
        """
        self.logger = logging.getLogger("PhoenixDB")
        
        self.url: str = os.environ.get("SUPABASE_URL")
        self.key: str = os.environ.get("SUPABASE_KEY")
        self.client: Optional[Client] = None
        
        self._connect()

    def _connect(self):
        """Robust connection with error handling."""
        if not self.url or not self.key:
            self.logger.warning("⚠️ Supabase credentials missing. Running in 'VOLATILE' mode (No Save).")
            return

        try:
            self.client = create_client(self.url, self.key)
            # Lightweight ping test
            self.client.table("portfolio_state").select("symbol").limit(1).execute()
            self.logger.info("✅ Supabase connection established.")
        except Exception as e:
            self.logger.error(f"❌ Supabase connection failed: {e}")
            self.client = None

    def get_client(self) -> Optional[Client]:
        return self.client

    # --- PORTFOLIO METHODS ---

    def load_portfolio(self) -> List[Dict[str, Any]]:
        """
        Loads the current portfolio state.
        Returns a LIST of dictionaries to match main.py expectations.
        """
        if not self.client:
            return []

        try:
            response = self.client.table("portfolio_state").select("*").execute()
            data = response.data
            
            if data:
                self.logger.info(f"📥 Portfolio loaded: {len(data)} positions.")
                return data # Returns list directly [symbol, strategy, quantity...]
            return []
            
        except Exception as e:
            self.logger.error(f"❌ Error Loading Portfolio: {e}")
            return []

    def save_portfolio(self, portfolio: List[Dict[str, Any]]):
        """
        Saves the portfolio list to Supabase (Upsert).
        """
        if not self.client or not portfolio:
            return

        try:
            # Upsert the list directly. 
            # Ensure your Supabase table has a composite primary key (symbol, strategy)
            self.client.table("portfolio_state").upsert(portfolio).execute()
        except Exception as e:
            self.logger.error(f"❌ Error Saving Portfolio: {e}")

    # --- HISTORY METHODS (Added to fix your error) ---

    def load_portfolio_history(self) -> List[Dict[str, Any]]:
        """Loads historical portfolio snapshots."""
        if not self.client: return []
        try:
            # Assumes table 'portfolio_history' exists
            response = self.client.table("portfolio_history").select("*").order("timestamp", desc=False).execute()
            return response.data if response.data else []
        except Exception as e:
            self.logger.error(f"⚠️ Could not load portfolio history: {e}")
            return []

    def load_trades_history(self) -> List[Dict[str, Any]]:
        """Loads historical trades."""
        if not self.client: return []
        try:
            response = self.client.table("trades").select("*").order("timestamp", desc=False).execute()
            return response.data if response.data else []
        except Exception as e:
            self.logger.error(f"⚠️ Could not load trades history: {e}")
            return []

    # --- TRADE METHODS ---

    def save_trades(self, trades_list: List[Dict[str, Any]]):
        """
        Saves a list of trades. 
        Used by main.py: self.db.save_trades(self.trades_history)
        """
        if not self.client or not trades_list:
            return

        try:
            # We take the last trade to avoid re-upserting the whole history every cycle
            # Or use upsert if you have a unique ID per trade.
            # Here we try to upsert the whole list (efficient for small lists, safer for data integrity)
            self.client.table("trades").upsert(trades_list).execute()
        except Exception as e:
            self.logger.error(f"❌ Error Saving Trades: {e}")

    def log_system_event(self, level: str, message: str, details: Dict = None):
        """System logging."""
        if not self.client: return

        try:
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": str(level),
                "message": str(message),
                "details": json.dumps(details) if details else None
            }
            self.client.table("system_logs").insert(payload).execute()
        except Exception:
            pass 

if __name__ == "__main__":
    db = DatabaseHandler()
    if db.client:
        print("✅ Connection Test OK")
        pf = db.load_portfolio()
        print(f"📦 Current Portfolio (List): {pf}")
