from typing import Dict, Any, Optional, Union, List, Callable, Set
from pathlib import Path
import json
import time
import shutil
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor
import threading
import zlib

class CacheLevel(Enum):
    """Cache storage levels"""
    MEMORY = "memory"    # Fast, volatile memory cache
    DISK = "disk"       # Persistent disk cache
    HYBRID = "hybrid"   # Both memory and disk

class CacheStrategy(Enum):
    """Cache eviction strategies"""
    LRU = "lru"    # Least Recently Used
    LFU = "lfu"    # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"    # Time To Live

class CacheEvent(Enum):
    """Cache event types for callbacks"""
    ITEM_ADDED = "added"
    ITEM_REMOVED = "removed"
    ITEM_EXPIRED = "expired"
    CACHE_CLEARED = "cleared"

@dataclass
class CacheConfig:
    """Cache configuration settings"""
    level: CacheLevel
    strategy: CacheStrategy
    max_memory_size: int            # Maximum memory cache size in bytes
    max_disk_size: int             # Maximum disk cache size in bytes
    ttl: Optional[int] = 3600      # Time to live in seconds (for TTL strategy)
    max_entries: Optional[int] = None  # Maximum number of cache entries
    compression: bool = True        # Whether to compress disk cache
    cleanup_interval: int = 300     # Cleanup interval in seconds

class CacheEntry:
    """Represents a cached item"""
    def __init__(
        self,
        key: str,
        value: Any,
        timestamp: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.key = key
        self.value = value
        self.timestamp = timestamp
        self.metadata = metadata or {}
        self.access_count = 0
        self.last_access = timestamp

    def update_access(self):
        """Update access statistics"""
        self.access_count += 1
        self.last_access = time.time()

@dataclass
class CacheMetrics:
    """Detailed cache metrics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_get_time: float = 0.0
    total_set_time: float = 0.0
    
class CacheManager:
    """Manages caching for video processing results"""
    
    def __init__(
        self,
        config: CacheConfig,
        cache_dir: Optional[Path] = None
    ):
        self.config = config
        self.cache_dir = cache_dir or Path.home() / ".video_processor_cache"
        self.logger = logging.getLogger(__name__)
        
        # Initialize cache storage
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.cache_lock = threading.RLock()
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        
        # Create cache directory
        if self.config.level in [CacheLevel.DISK, CacheLevel.HYBRID]:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Start cleanup thread
        self._start_cleanup_thread()
        
        self.metrics = CacheMetrics()
        self._event_handlers: Dict[CacheEvent, List[Callable]] = {
            event: [] for event in CacheEvent
        }
        self._prefetch_keys: Set[str] = set()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="CacheWorker")
        
    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        def cleanup_task():
            while not self._stop_cleanup.is_set():
                try:
                    self._cleanup_cache()
                except Exception as e:
                    self.logger.error(f"Cache cleanup error: {str(e)}")
                self._stop_cleanup.wait(self.config.cleanup_interval)
                
        self._cleanup_thread = threading.Thread(
            target=cleanup_task,
            daemon=True,
            name="CacheCleanupThread"
        )
        self._cleanup_thread.start()
        
    def _generate_key(self, data: Any) -> str:
        """Generate cache key from data"""
        if isinstance(data, (str, bytes)):
            key_data = data if isinstance(data, bytes) else data.encode()
        else:
            key_data = json.dumps(data, sort_keys=True).encode()
            
        return hashlib.sha256(key_data).hexdigest()
        
    def _get_cache_path(self, key: str) -> Path:
        """Get disk cache path for key"""
        return self.cache_dir / f"{key}.cache"
        
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for disk storage"""
        data = pickle.dumps(value)
        if self.config.compression:
            data = zlib.compress(data)
        return data
        
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from disk storage"""
        if self.config.compression:
            data = zlib.decompress(data)
        return pickle.loads(data)
        
    def _enforce_memory_limit(self):
        """Enforce memory cache size limit"""
        if not self.config.max_memory_size:
            return
            
        with self.cache_lock:
            # Calculate current size
            current_size = sum(len(pickle.dumps(entry.value)) 
                             for entry in self.memory_cache.values())
            
            while current_size > self.config.max_memory_size and self.memory_cache:
                # Remove entry based on strategy
                key_to_remove = self._select_entry_to_remove()
                if key_to_remove:
                    removed_size = len(pickle.dumps(
                        self.memory_cache[key_to_remove].value
                    ))
                    del self.memory_cache[key_to_remove]
                    current_size -= removed_size
                    
    def _enforce_disk_limit(self):
        """Enforce disk cache size limit"""
        if not self.config.max_disk_size or not self.cache_dir.exists():
            return
            
        # Get cache files sorted by last modified time
        cache_files = sorted(
            self.cache_dir.glob("*.cache"),
            key=lambda p: p.stat().st_mtime
        )
        
        total_size = sum(p.stat().st_size for p in cache_files)
        
        while total_size > self.config.max_disk_size and cache_files:
            file_to_remove = cache_files.pop(0)  # Remove oldest
            total_size -= file_to_remove.stat().st_size
            file_to_remove.unlink()
            
    def _select_entry_to_remove(self) -> Optional[str]:
        """Select cache entry to remove based on strategy"""
        if not self.memory_cache:
            return None
            
        if self.config.strategy == CacheStrategy.LRU:
            return min(
                self.memory_cache.items(),
                key=lambda x: x[1].last_access
            )[0]
        elif self.config.strategy == CacheStrategy.LFU:
            return min(
                self.memory_cache.items(),
                key=lambda x: x[1].access_count
            )[0]
        elif self.config.strategy == CacheStrategy.FIFO:
            return min(
                self.memory_cache.items(),
                key=lambda x: x[1].timestamp
            )[0]
        elif self.config.strategy == CacheStrategy.TTL:
            current_time = time.time()
            expired = [
                key for key, entry in self.memory_cache.items()
                if current_time - entry.timestamp > self.config.ttl
            ]
            return expired[0] if expired else None
            
    def _cleanup_cache(self):
        """Clean up expired and excess cache entries"""
        try:
            # Clean memory cache
            if self.config.level in [CacheLevel.MEMORY, CacheLevel.HYBRID]:
                self._enforce_memory_limit()
                
                if self.config.strategy == CacheStrategy.TTL:
                    current_time = time.time()
                    with self.cache_lock:
                        expired = [
                            key for key, entry in self.memory_cache.items()
                            if current_time - entry.timestamp > self.config.ttl
                        ]
                        for key in expired:
                            del self.memory_cache[key]
                            
            # Clean disk cache
            if self.config.level in [CacheLevel.DISK, CacheLevel.HYBRID]:
                self._enforce_disk_limit()
                
                if self.config.strategy == CacheStrategy.TTL:
                    current_time = time.time()
                    for cache_file in self.cache_dir.glob("*.cache"):
                        if current_time - cache_file.stat().st_mtime > self.config.ttl:
                            cache_file.unlink()
                            
        except Exception as e:
            self.logger.error(f"Error during cache cleanup: {str(e)}")
            
    def add_event_handler(self, event: CacheEvent, handler: Callable) -> None:
        """Add an event handler for cache events"""
        self._event_handlers[event].append(handler)
        
    def remove_event_handler(self, event: CacheEvent, handler: Callable) -> None:
        """Remove an event handler"""
        if handler in self._event_handlers[event]:
            self._event_handlers[event].remove(handler)
            
    def _trigger_event(self, event: CacheEvent, key: str, value: Any = None) -> None:
        """Trigger event handlers"""
        for handler in self._event_handlers[event]:
            try:
                handler(key, value)
            except Exception as e:
                self.logger.error(f"Error in event handler: {str(e)}")
                
    def prefetch(self, keys: List[str]) -> None:
        """Prefetch items into memory cache"""
        if self.config.level not in [CacheLevel.MEMORY, CacheLevel.HYBRID]:
            return
            
        def _prefetch_item(key: str) -> None:
            if key not in self.memory_cache:
                value = self.get(key)
                if value is not None:
                    self._prefetch_keys.add(key)
                    
        for key in keys:
            self._executor.submit(_prefetch_item, key)
            
    def get_batch(self, keys: List[str], default: Any = None) -> Dict[str, Any]:
        """Get multiple items from cache"""
        results = {}
        for key in keys:
            results[key] = self.get(key, default)
        return results
        
    def set_batch(
        self,
        items: Dict[str, Any],
        metadata: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> bool:
        """Set multiple items in cache"""
        success = True
        for key, value in items.items():
            item_metadata = metadata.get(key) if metadata else None
            if not self.set(key, value, item_metadata):
                success = False
        return success
        
    def save_state(self, state_file: Path) -> bool:
        """Save cache state to file"""
        try:
            state = {
                'config': asdict(self.config),
                'metrics': asdict(self.metrics),
                'prefetch_keys': list(self._prefetch_keys)
            }
            
            # Save memory cache if using memory/hybrid
            if self.config.level in [CacheLevel.MEMORY, CacheLevel.HYBRID]:
                with self.cache_lock:
                    state['memory_cache'] = {
                        key: {
                            'value': entry.value,
                            'timestamp': entry.timestamp,
                            'metadata': entry.metadata,
                            'access_count': entry.access_count,
                            'last_access': entry.last_access
                        }
                        for key, entry in self.memory_cache.items()
                    }
                    
            with open(state_file, 'wb') as f:
                pickle.dump(state, f)
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving cache state: {str(e)}")
            return False
            
    def load_state(self, state_file: Path) -> bool:
        """Load cache state from file"""
        try:
            with open(state_file, 'rb') as f:
                state = pickle.load(f)
                
            # Restore configuration
            self.config = CacheConfig(**state['config'])
            self.metrics = CacheMetrics(**state['metrics'])
            self._prefetch_keys = set(state['prefetch_keys'])
            
            # Restore memory cache if using memory/hybrid
            if self.config.level in [CacheLevel.MEMORY, CacheLevel.HYBRID]:
                with self.cache_lock:
                    self.memory_cache.clear()
                    for key, entry_data in state.get('memory_cache', {}).items():
                        self.memory_cache[key] = CacheEntry(
                            key=key,
                            value=entry_data['value'],
                            timestamp=entry_data['timestamp'],
                            metadata=entry_data['metadata']
                        )
                        self.memory_cache[key].access_count = entry_data['access_count']
                        self.memory_cache[key].last_access = entry_data['last_access']
                        
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading cache state: {str(e)}")
            return False
            
    def get_stats(self) -> Dict[str, Any]:
        """Get basic cache statistics"""
        stats = {
            'level': self.config.level.value,
            'strategy': self.config.strategy.value,
            'memory_entries': 0,
            'memory_size': 0,
            'disk_entries': 0,
            'disk_size': 0
        }
        
        try:
            # Get memory stats
            if self.config.level in [CacheLevel.MEMORY, CacheLevel.HYBRID]:
                with self.cache_lock:
                    stats['memory_entries'] = len(self.memory_cache)
                    stats['memory_size'] = sum(
                        len(pickle.dumps(entry.value))
                        for entry in self.memory_cache.values()
                    )
                    
            # Get disk stats
            if self.config.level in [CacheLevel.DISK, CacheLevel.HYBRID]:
                cache_files = list(self.cache_dir.glob("*.cache"))
                stats['disk_entries'] = len(cache_files)
                stats['disk_size'] = sum(p.stat().st_size for p in cache_files)
                
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {str(e)}")
            
        return stats
        
    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics and metrics"""
        stats = self.get_stats()  # Get basic stats
        
        total_ops = self.metrics.hits + self.metrics.misses
        
        # Add metrics
        stats.update({
            'hit_rate': (self.metrics.hits / total_ops if total_ops > 0 else 0),
            'miss_rate': (self.metrics.misses / total_ops if total_ops > 0 else 0),
            'eviction_count': self.metrics.evictions,
            'avg_get_time': (self.metrics.total_get_time / total_ops 
                           if total_ops > 0 else 0),
            'avg_set_time': (self.metrics.total_set_time / total_ops 
                           if total_ops > 0 else 0),
            'prefetched_keys': len(self._prefetch_keys)
        })
        
        return stats
        
    def get(self, key: Union[str, Any], default: Any = None) -> Optional[Any]:
        """Get value from cache with metrics"""
        start_time = time.time()
        try:
            if not isinstance(key, str):
                key = self._generate_key(key)
                
            # Check memory cache
            if self.config.level in [CacheLevel.MEMORY, CacheLevel.HYBRID]:
                with self.cache_lock:
                    if key in self.memory_cache:
                        entry = self.memory_cache[key]
                        
                        # Check TTL
                        if (self.config.strategy == CacheStrategy.TTL and
                            time.time() - entry.timestamp > self.config.ttl):
                            del self.memory_cache[key]
                            self.metrics.misses += 1
                        else:
                            entry.update_access()
                            self.metrics.hits += 1
                            return entry.value
                            
            # Check disk cache
            if self.config.level in [CacheLevel.DISK, CacheLevel.HYBRID]:
                cache_path = self._get_cache_path(key)
                if cache_path.exists():
                    # Check TTL
                    if (self.config.strategy == CacheStrategy.TTL and
                        time.time() - cache_path.stat().st_mtime > self.config.ttl):
                        cache_path.unlink()
                        self.metrics.misses += 1
                        return default
                        
                    with open(cache_path, 'rb') as f:
                        value = self._deserialize_value(f.read())
                        
                    # Store in memory if hybrid
                    if self.config.level == CacheLevel.HYBRID:
                        self.set(key, value)
                        
                    self.metrics.hits += 1
                    return value
                    
            self.metrics.misses += 1
            return default
            
        except Exception as e:
            self.logger.error(f"Error reading from cache: {str(e)}")
            self.metrics.misses += 1
            return default
        finally:
            self.metrics.total_get_time += time.time() - start_time
            
    def set(
        self,
        key: Union[str, Any],
        value: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Set value in cache with metrics"""
        start_time = time.time()
        try:
            if not isinstance(key, str):
                key = self._generate_key(key)
                
            timestamp = time.time()
            
            # Store in memory
            if self.config.level in [CacheLevel.MEMORY, CacheLevel.HYBRID]:
                with self.cache_lock:
                    self.memory_cache[key] = CacheEntry(
                        key=key,
                        value=value,
                        timestamp=timestamp,
                        metadata=metadata
                    )
                    self._enforce_memory_limit()
                    
            # Store on disk
            if self.config.level in [CacheLevel.DISK, CacheLevel.HYBRID]:
                cache_path = self._get_cache_path(key)
                with open(cache_path, 'wb') as f:
                    f.write(self._serialize_value(value))
                self._enforce_disk_limit()
                
            self._trigger_event(CacheEvent.ITEM_ADDED, key, value)
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing to cache: {str(e)}")
            return False
        finally:
            self.metrics.total_set_time += time.time() - start_time
            
    def delete(self, key: Union[str, Any]) -> bool:
        """Delete value from cache with event"""
        result = super().delete(key)
        if result:
            self._trigger_event(CacheEvent.ITEM_REMOVED, key)
        return result
        
    def clear(self) -> None:
        """Clear all cache entries with event"""
        try:
            # Clear memory cache
            if self.config.level in [CacheLevel.MEMORY, CacheLevel.HYBRID]:
                with self.cache_lock:
                    self.memory_cache.clear()
                    
            # Clear disk cache
            if self.config.level in [CacheLevel.DISK, CacheLevel.HYBRID]:
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True)
                
            self._trigger_event(CacheEvent.CACHE_CLEARED, "all")
                
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")
        
    def __del__(self):
        """Cleanup when the cache manager is destroyed"""
        self._executor.shutdown(wait=False)
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._stop_cleanup.set()
            self._cleanup_thread.join(timeout=1.0)  # Wait up to 1 second for cleanup

def create_cache_manager(
    level: CacheLevel = CacheLevel.HYBRID,
    strategy: CacheStrategy = CacheStrategy.LRU,
    max_memory_mb: int = 1024,  # 1GB
    max_disk_mb: int = 5120,    # 5GB
    cache_dir: Optional[Path] = None
) -> CacheManager:
    """Create cache manager instance with common defaults"""
    config = CacheConfig(
        level=level,
        strategy=strategy,
        max_memory_size=max_memory_mb * 1024 * 1024,
        max_disk_size=max_disk_mb * 1024 * 1024,
        compression=True,
        cleanup_interval=300  # 5 minutes
    )
    return CacheManager(config, cache_dir) 