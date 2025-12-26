import aiohttp
import asyncio
import pandas as pd
import logging
import time
import json
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

logger = logging.getLogger("PhoenixMarketData")

class MarketDataManager:
    """
    Manages market data fetching from CoinGecko with advanced features.
    
    Features:
    - Multi-timeframe support
    - Intelligent caching (memory + disk)
    - Rate limiting with backoff
    - Data stitching for long historical periods
    - Volume data from alternative sources
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.base_url = "https://api.coingecko.com/api/v3"
        
        # Rate Limiting: CoinGecko Free Tier
        self._last_call_time = 0.0
        self._call_delay = 7.0  # Increased for safety
        self._rate_limit_hits = 0
        
        # Caching
        self._memory_cache: Dict[str, Dict] = {}
        self._cache_dir = "data_cache"
        self._cache_ttl = self.config.get('data', {}).get('cache_ttl_minutes', 5) * 60
        
        # Create cache directory
        os.makedirs(self._cache_dir, exist_ok=True)
        
        # Timeframe mapping
        self.timeframe_map = {
            '1m': {'days': '1', 'interval': 'minute'},
            '5m': {'days': '1', 'interval': 'minute'},
            '15m': {'days': '1', 'interval': 'minute'},
            '1h': {'days': '90', 'interval': 'hourly'},
            '4h': {'days': '90', 'interval': 'hourly'},
            '1d': {'days': 'max', 'interval': 'daily'}
        }
        
        # Symbol mapping with fallbacks
        self.symbol_map = self._build_symbol_map()
        
        logger.info(f"✅ MarketDataManager initialized with {self._cache_ttl}s cache TTL")

    def _build_symbol_map(self) -> Dict[str, str]:
        """Build comprehensive symbol to CoinGecko ID mapping."""
        mapping = {
            # Bitcoin
            "BTC": "bitcoin", "BTC/USD": "bitcoin", "BTCUSDT": "bitcoin",
            # Ethereum
            "ETH": "ethereum", "ETH/USD": "ethereum", "ETHUSDT": "ethereum",
            # Solana
            "SOL": "solana", "SOL/USD": "solana", "SOLUSDT": "solana",
            # Binance Coin
            "BNB": "binancecoin", "BNB/USD": "binancecoin", "BNBUSDT": "binancecoin",
            # Ripple
            "XRP": "ripple", "XRP/USD": "ripple", "XRPUSDT": "ripple",
            # Cardano
            "ADA": "cardano", "ADA/USD": "cardano", "ADAUSDT": "cardano",
            # Dogecoin
            "DOGE": "dogecoin", "DOGE/USD": "dogecoin", "DOGEUSDT": "dogecoin",
            # Polkadot
            "DOT": "polkadot", "DOT/USD": "polkadot", "DOTUSDT": "polkadot",
            # Chainlink
            "LINK": "chainlink", "LINK/USD": "chainlink", "LINKUSDT": "chainlink",
            # Polygon
            "MATIC": "matic-network", "MATIC/USD": "matic-network", "MATICUSDT": "matic-network",
            # Avalanche
            "AVAX": "avalanche-2", "AVAX/USD": "avalanche-2", "AVAXUSDT": "avalanche-2"
        }
        
        # Add lowercase versions
        additional = {}
        for key, value in mapping.items():
            additional[key.lower()] = value
            additional[key.replace('/', '').lower()] = value
            
        mapping.update(additional)
        return mapping

    def _map_symbol_to_id(self, symbol: str) -> Optional[str]:
        """
        Maps trading pair symbols to CoinGecko API IDs with multiple fallback strategies.
        """
        # Clean the symbol
        clean_symbol = symbol.upper().replace('/', '').replace('-', '')
        
        # Try exact match first
        for key in [symbol, clean_symbol, symbol.replace('/', '')]:
            if key in self.symbol_map:
                return self.symbol_map[key]
        
        # Try removing quote currency
        quote_currencies = ['USD', 'USDT', 'USDC', 'EUR', 'GBP']
        for quote in quote_currencies:
            if symbol.endswith(f"/{quote}"):
                base = symbol[:-len(quote)-1]
                if base in self.symbol_map:
                    return self.symbol_map[base]
                # Try uppercase
                if base.upper() in self.symbol_map:
                    return self.symbol_map[base.upper()]
        
        # Try lowercase
        if symbol.lower() in self.symbol_map:
            return self.symbol_map[symbol.lower()]
        
        # Last resort: use the symbol as-is (might work for major coins)
        logger.warning(f"⚠️ Could not map symbol '{symbol}' to CoinGecko ID, using as-is")
        return symbol.lower().split('/')[0]

    def _get_cache_key(self, symbol: str, timeframe: str, days: str) -> str:
        """Generate cache key for data."""
        return f"{symbol}_{timeframe}_{days}"

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

    async def _enforce_rate_limit(self):
        """Intelligent rate limiting with backoff."""
        now = time.time()
        elapsed = now - self._last_call_time
        
        # Dynamic delay based on rate limit hits
        base_delay = self._call_delay
        if self._rate_limit_hits > 0:
            base_delay = min(base_delay * (2 ** self._rate_limit_hits), 60)
        
        if elapsed < base_delay:
            wait_time = base_delay - elapsed
            logger.debug(f"⏳ Rate limiting: waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
        
        self._last_call_time = time.time()

    async def get_historical_data(self, symbol: str, days: str = "30", 
                                 timeframe: str = None) -> Optional[pd.DataFrame]:
        """
        Fetches OHLC data with intelligent caching and data stitching.
        
        Args:
            symbol: Trading pair (e.g., "BTC/USD")
            days: Data range (e.g., "1", "14", "30", "90", "max")
            timeframe: Candle timeframe (e.g., "1h", "4h", "1d")
        """
        if timeframe is None:
            timeframe = self.config.get('trading', {}).get('timeframe', '1h')
        
        coin_id = self._map_symbol_to_id(symbol)
        if not coin_id:
            logger.error(f"❌ Could not map symbol: {symbol}")
            return None
        
        # Check cache first
        cache_key = self._get_cache_key(symbol, timeframe, days)
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        # Determine API parameters based on timeframe
        timeframe_config = self.timeframe_map.get(timeframe, {'days': '30', 'interval': 'hourly'})
        
        # For large time ranges, use data stitching
        if days in ['max', '365', '180', '90'] and timeframe_config['interval'] != 'daily':
            df = await self._get_stitched_data(coin_id, symbol, days, timeframe)
        else:
            df = await self._fetch_from_api(coin_id, symbol, days, timeframe_config)
        
        if df is not None:
            self._save_to_cache(cache_key, df)
        
        return df

    async def _fetch_from_api(self, coin_id: str, symbol: str, days: str, 
                             config: Dict) -> Optional[pd.DataFrame]:
        """Fetch data directly from API."""
        await self._enforce_rate_limit()
        
        url = f"{self.base_url}/coins/{coin_id}/ohlc"
        params = {
            "vs_currency": "usd",
            "days": days,
            "precision": "full"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as response:
                    
                    # Handle rate limits
                    if response.status == 429:
                        self._rate_limit_hits += 1
                        wait_time = 60 * (2 ** min(self._rate_limit_hits, 3))
                        logger.warning(f"⚠️ Rate limit hit. Waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        return None
                    
                    if response.status != 200:
                        logger.error(f"❌ API Error for {symbol}: {response.status}")
                        return None
                    
                    # Reset rate limit hits on successful call
                    if self._rate_limit_hits > 0:
                        self._rate_limit_hits = max(0, self._rate_limit_hits - 1)
                    
                    data = await response.json()
                    
                    if not data or not isinstance(data, list):
                        logger.warning(f"⚠️ Received empty data for {symbol}")
                        return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Add synthetic volume (CoinGecko free doesn't provide volume)
            # Use average volume based on price movement
            price_range = df['high'] - df['low']
            avg_price = (df['high'] + df['low'] + df['close']) / 3
            df['volume'] = (price_range / avg_price) * 1000000  # Synthetic
            
            # Sort & Clean
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Resample to desired timeframe if needed
            if config['interval'] == 'hourly' and len(df) > 0:
                df = self._resample_to_timeframe(df, '1h')
            elif config['interval'] == 'daily' and len(df) > 0:
                df = self._resample_to_timeframe(df, '1D')
            
            logger.info(f"📥 Fetched {len(df)} {config['interval']} candles for {symbol}")
            return df
            
        except asyncio.TimeoutError:
            logger.error(f"❌ Timeout fetching {symbol}")
            return None
        except Exception as e:
            logger.error(f"❌ Error fetching {symbol}: {e}")
            return None

    async def _get_stitched_data(self, coin_id: str, symbol: str, days: str, 
                                timeframe: str) -> Optional[pd.DataFrame]:
        """
        Stitch together multiple API calls for long historical data.
        """
        logger.info(f"🧵 Stitching data for {symbol} ({days} days, {timeframe})")
        
        # Determine chunk size based on timeframe
        if timeframe in ['1h', '4h']:
            chunk_days = 90  # CoinGecko max for hourly
        else:
            chunk_days = 365  # Daily data
        
        all_data = []
        current_date = datetime.now()
        
        try:
            # Calculate number of chunks needed
            target_days = 365 if days == 'max' else int(days)
            num_chunks = (target_days + chunk_days - 1) // chunk_days
            
            for i in range(num_chunks):
                chunk_start = current_date - timedelta(days=chunk_days * (i + 1))
                chunk_end = current_date - timedelta(days=chunk_days * i)
                
                # Use smaller chunks for the most recent data
                chunk_size = min(chunk_days, target_days - (i * chunk_days))
                
                logger.debug(f"   Fetching chunk {i+1}/{num_chunks}: {chunk_size} days")
                
                # Fetch chunk
                chunk_df = await self._fetch_from_api(
                    coin_id, 
                    symbol, 
                    str(chunk_size),
                    {'days': str(chunk_size), 'interval': 'daily' if timeframe == '1d' else 'hourly'}
                )
                
                if chunk_df is not None:
                    all_data.append(chunk_df)
                
                # Be nice to the API
                if i < num_chunks - 1:
                    await asyncio.sleep(2)
            
            if not all_data:
                return None
            
            # Combine all chunks
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['timestamp'])
            combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
            
            # Resample to desired timeframe
            if timeframe == '4h':
                combined_df = self._resample_to_timeframe(combined_df, '4h')
            elif timeframe == '1h':
                combined_df = self._resample_to_timeframe(combined_df, '1h')
            
            logger.info(f"✅ Stitched {len(combined_df)} candles for {symbol}")
            return combined_df
            
        except Exception as e:
            logger.error(f"❌ Error stitching data for {symbol}: {e}")
            return None

    def _resample_to_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample DataFrame to desired timeframe.
        """
        if len(df) < 2:
            return df
        
        # Set timestamp as index
        df = df.copy()
        df.set_index('timestamp', inplace=True)
        
        # Resample
        resampled = df.resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        resampled.reset_index(inplace=True)
        return resampled

    async def get_latest_candle(self, symbol: str, timeframe: str = None) -> Optional[pd.DataFrame]:
        """
        Retrieves the latest market data.
        Optimized for frequent polling with caching.
        """
        if timeframe is None:
            timeframe = self.config.get('trading', {}).get('timeframe', '1h')
        
        # For live trading, fetch minimal data
        cache_key = f"{symbol}_{timeframe}_latest"
        cached_data = self._load_from_cache(cache_key)
        
        if cached_data is not None:
            # Check if cache is fresh (last 5 minutes)
            cache_age = time.time() - self._memory_cache.get(cache_key, {}).get('timestamp', 0)
            if cache_age < 300:  # 5 minutes
                return cached_data
        
        # Fetch fresh data
        df = await self.get_historical_data(symbol, days="1", timeframe=timeframe)
        
        if df is not None:
            self._save_to_cache(cache_key, df)
        
        return df

    def get_available_timeframes(self) -> List[str]:
        """Get list of available timeframes."""
        return list(self.timeframe_map.keys())

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