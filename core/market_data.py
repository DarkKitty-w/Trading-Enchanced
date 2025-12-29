import asyncio
import pandas as pd
import logging
import time
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import numpy as np
import ccxt.async_support as ccxt

logger = logging.getLogger("PhoenixMarketData")

class MarketDataManager:
    """
    Manages market data fetching from KuCoin using ccxt library.
    
    Features:
    - Multi-timeframe support
    - Intelligent caching (memory + disk)
    - Rate limiting handled by ccxt
    - Data stitching for long historical periods
    - Real volume data
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize KuCoin exchange instance
        exchange_config = {
            'enableRateLimit': True,
            'rateLimit': 1000,  # KuCoin rate limit
            'timeout': 30000,
        }
        
        # Add API credentials if provided in config
        if 'exchange' in self.config:
            exchange_config.update(self.config['exchange'])
        
        self.exchange = ccxt.kucoin(exchange_config)
        
        # Caching
        self._memory_cache: Dict[str, Dict] = {}
        self._cache_dir = "data_cache"
        self._cache_ttl = self.config.get('data', {}).get('cache_ttl_minutes', 5) * 60
        
        # Create cache directory
        os.makedirs(self._cache_dir, exist_ok=True)
        
        # Timeframe mapping (ccxt standard timeframes)
        self.timeframe_map = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d'
        }
        
        # Symbol mapping - ccxt uses standard format like "BTC/USDT"
        self.symbol_map = self._build_symbol_map()
        
        logger.info(f"✅ MarketDataManager initialized for KuCoin with {self._cache_ttl}s cache TTL")

    def _build_symbol_map(self) -> Dict[str, str]:
        """Build symbol mapping for common pairs."""
        mapping = {}
        
        # Common trading pairs on KuCoin
        common_pairs = [
            "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
            "ADA/USDT", "DOGE/USDT", "DOT/USDT", "LINK/USDT", "MATIC/USDT",
            "AVAX/USDT", "ATOM/USDT", "UNI/USDT", "AAVE/USDT", "ALGO/USDT"
        ]
        
        for pair in common_pairs:
            # Map various formats to standard pair
            base_symbol = pair.split('/')[0]
            mapping[base_symbol] = pair
            mapping[base_symbol + "/USD"] = pair
            mapping[base_symbol + "USDT"] = pair
            mapping[pair.replace('/', '')] = pair
        
        return mapping

    def _map_symbol_to_pair(self, symbol: str) -> str:
        """
        Maps trading pair symbols to KuCoin format.
        Returns standard format like "BTC/USDT"
        """
        # Try exact match first
        if symbol in self.symbol_map:
            return self.symbol_map[symbol]
        
        # Clean the symbol
        clean_symbol = symbol.upper().replace('/', '').replace('-', '')
        
        # Try mapping without quote
        for key in [clean_symbol, symbol.upper()]:
            if key in self.symbol_map:
                return self.symbol_map[key]
        
        # Check if it already has USDT
        if symbol.upper().endswith('USDT'):
            base = symbol.upper()[:-4]
            return f"{base}/USDT"
        
        # Last resort: assume USDT pair
        logger.warning(f"⚠️ Could not map symbol '{symbol}', using as-is with USDT")
        if '/' not in symbol:
            return f"{symbol.upper()}/USDT"
        return symbol.upper()

    def _get_cache_key(self, symbol: str, timeframe: str, days: str) -> str:
        """Generate cache key for data."""
        return f"{symbol.replace('/', '_')}_{timeframe}_{days}"

    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from cache (memory + disk)."""
        # Check memory cache first
        if cache_key in self._memory_cache:
            cached_data = self._memory_cache[cache_key]
            if time.time() - cached_data['timestamp'] < self._cache_ttl:
                logger.debug(f"📂 Loaded from memory cache: {cache_key}")
                return cached_data['data']
        
        # Check disk cache
        cache_file = os.path.join(self._cache_dir, f"{cache_key}.parquet")
        if os.path.exists(cache_file):
            file_age = time.time() - os.path.getmtime(cache_file)
            if file_age < self._cache_ttl:
                try:
                    df = pd.read_parquet(cache_file)
                    # Update memory cache
                    self._memory_cache[cache_key] = {
                        'data': df,
                        'timestamp': time.time()
                    }
                    logger.debug(f"📂 Loaded from disk cache: {cache_key}")
                    return df
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load cache file {cache_file}: {e}")
        
        return None

    def _save_to_cache(self, cache_key: str, df: pd.DataFrame):
        """Save data to cache (memory + disk)."""
        # Save to memory cache
        self._memory_cache[cache_key] = {
            'data': df,
            'timestamp': time.time()
        }
        
        # Save to disk cache
        try:
            cache_file = os.path.join(self._cache_dir, f"{cache_key}.parquet")
            df.to_parquet(cache_file)
            logger.debug(f"💾 Saved to cache: {cache_key}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to save cache file: {e}")
    async def fetch_latest_candles(self, pairs: List[str], timeframe: str, limit: int = 500) -> Dict[str, pd.DataFrame]:
        """
        Fetches the latest OHLCV data for multiple pairs in a batch.
        """
        market_snapshot = {}
        tasks = [self.get_latest_candle(pair, timeframe) for pair in pairs]
        results = await asyncio.gather(*tasks)

        for pair, df in zip(pairs, results):
            if df is not None and not df.empty:
                market_snapshot[pair] = df
        
        return market_snapshot

    async def get_historical_data(self, symbol: str, days: str = "30", 
                                 timeframe: str = None) -> Optional[pd.DataFrame]:
        """
        Fetches OHLCV data from KuCoin with intelligent caching and data stitching.
        
        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            days: Data range (e.g., "1", "14", "30", "90", "max")
            timeframe: Candle timeframe (e.g., "1h", "4h", "1d")
        """
        if timeframe is None:
            timeframe = self.config.get('trading', {}).get('timeframe', '1h')
        
        # Map symbol to KuCoin pair
        pair = self._map_symbol_to_pair(symbol)
        
        # Check cache first
        cache_key = self._get_cache_key(pair, timeframe, days)
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        # Get ccxt timeframe
        ccxt_timeframe = self.timeframe_map.get(timeframe, '1h')
        
        # For large time ranges, use data stitching
        if days in ['max', '365', '180', '90']:
            df = await self._get_stitched_data(pair, ccxt_timeframe, days)
        else:
            df = await self._fetch_from_api(pair, ccxt_timeframe, days)
        
        if df is not None:
            self._save_to_cache(cache_key, df)
        
        return df

    async def _fetch_from_api(self, pair: str, timeframe: str, 
                             days: str) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from KuCoin using ccxt."""
        try:
            # Calculate since parameter
            since = None
            if days != 'max':
                days_int = int(days)
                since = int((datetime.now() - timedelta(days=days_int)).timestamp() * 1000)
            
            # Fetch OHLCV data
            ohlcv = await self.exchange.fetch_ohlcv(
                pair,
                timeframe=timeframe,
                since=since,
                limit=1000  # Maximum candles per request
            )
            
            if not ohlcv:
                logger.warning(f"⚠️ Received empty data for {pair}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Sort & Clean
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"📥 Fetched {len(df)} {timeframe} candles for {pair}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error fetching {pair}: {e}")
            return None

    async def _get_stitched_data(self, pair: str, timeframe: str, 
                                days: str) -> Optional[pd.DataFrame]:
        """
        Stitch together multiple API calls for long historical data.
        """
        logger.info(f"🧵 Stitching data for {pair} ({days} days, {timeframe})")
        
        all_data = []
        current_time = int(datetime.now().timestamp() * 1000)
        
        try:
            # Calculate target time based on days
            if days == 'max':
                # For max, we'll fetch until we can't get more data
                target_time = 0
            else:
                target_time = int((datetime.now() - timedelta(days=int(days))).timestamp() * 1000)
            
            # KuCoin OHLCV limit is 1500 candles per request
            limit = 1500
            
            # Get candle duration in milliseconds
            timeframe_to_ms = {
                '1m': 60 * 1000,
                '5m': 5 * 60 * 1000,
                '15m': 15 * 60 * 1000,
                '1h': 60 * 60 * 1000,
                '4h': 4 * 60 * 60 * 1000,
                '1d': 24 * 60 * 60 * 1000,
            }
            
            candle_duration = timeframe_to_ms.get(timeframe, 60 * 60 * 1000)
            
            while True:
                # Calculate since for this batch
                batch_since = current_time - (limit * candle_duration)
                
                if batch_since < target_time and target_time > 0:
                    batch_since = target_time
                
                # Fetch batch
                ohlcv = await self.exchange.fetch_ohlcv(
                    pair,
                    timeframe=timeframe,
                    since=batch_since,
                    limit=limit
                )
                
                if not ohlcv:
                    break
                
                # Convert to DataFrame and add to list
                batch_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                batch_df['timestamp'] = pd.to_datetime(batch_df['timestamp'], unit='ms')
                all_data.append(batch_df)
                
                # Update current time for next batch
                current_time = batch_since
                
                # Check if we've reached target time
                if (target_time > 0 and batch_since <= target_time) or len(ohlcv) < limit:
                    break
                
                # Be nice to the API
                await asyncio.sleep(1)
            
            if not all_data:
                return None
            
            # Combine all batches
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['timestamp'])
            combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
            
            # Filter by target time if needed
            if target_time > 0:
                combined_df = combined_df[combined_df['timestamp'] >= pd.to_datetime(target_time, unit='ms')]
            
            logger.info(f"✅ Stitched {len(combined_df)} candles for {pair}")
            return combined_df
            
        except Exception as e:
            logger.error(f"❌ Error stitching data for {pair}: {e}")
            return None
        finally:
            await self.exchange.close()

    async def get_latest_candle(self, symbol: str, timeframe: str = None) -> Optional[pd.DataFrame]:
        """
        Retrieves the latest market data.
        Optimized for frequent polling with caching.
        """
        if timeframe is None:
            timeframe = self.config.get('trading', {}).get('timeframe', '1h')
        
        # Map symbol to pair
        pair = self._map_symbol_to_pair(symbol)
        
        # For live trading, fetch minimal data
        cache_key = f"{pair.replace('/', '_')}_{timeframe}_latest"
        cached_data = self._load_from_cache(cache_key)
        
        if cached_data is not None:
            # Check if cache is fresh (last 5 minutes)
            cache_age = time.time() - self._memory_cache.get(cache_key, {}).get('timestamp', 0)
            if cache_age < 300:  # 5 minutes
                return cached_data
        
        # Fetch fresh data (last 1 day)
        ccxt_timeframe = self.timeframe_map.get(timeframe, '1h')
        df = await self._fetch_from_api(pair, ccxt_timeframe, "1")
        
        if df is not None:
            self._save_to_cache(cache_key, df)
        
        await self.exchange.close()
        return df

    def get_available_timeframes(self) -> List[str]:
        """Get list of available timeframes."""
        return list(self.timeframe_map.keys())

    async def get_available_pairs(self) -> List[str]:
        """Get available trading pairs on KuCoin."""
        try:
            await self.exchange.load_markets()
            pairs = list(self.exchange.markets.keys())
            await self.exchange.close()
            return pairs
        except Exception as e:
            logger.error(f"❌ Error fetching available pairs: {e}")
            return []

    def clear_cache(self, older_than_hours: int = 24):
        """Clear old cache files."""
        try:
            cutoff_time = time.time() - (older_than_hours * 3600)
            cleared = 0
            
            for filename in os.listdir(self._cache_dir):
                filepath = os.path.join(self._cache_dir, filename)
                if os.path.getmtime(filepath) < cutoff_time:
                    os.remove(filepath)
                    cleared += 1
            
            # Clear memory cache
            old_keys = []
            for key, data in self._memory_cache.items():
                if data['timestamp'] < cutoff_time:
                    old_keys.append(key)
            
            for key in old_keys:
                del self._memory_cache[key]
            
            logger.info(f"🧹 Cleared {cleared} cache files and {len(old_keys)} memory entries")
            
        except Exception as e:
            logger.error(f"❌ Error clearing cache: {e}")

    async def close(self):
        """Close the exchange connection."""
        await self.exchange.close()