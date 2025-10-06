"""
Caching system for detection results to avoid redundant processing.
"""

import hashlib
import json
import logging
import pickle
import sqlite3
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import cv2
from datetime import datetime, timedelta
import threading
import queue

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry data structure."""
    key: str
    video_path: str
    frame_number: int
    timestamp: float
    detections: Dict[str, List]
    processing_time: float
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    file_size: int = 0


@dataclass
class CacheStats:
    """Cache statistics."""
    total_entries: int
    total_size_mb: float
    hit_count: int
    miss_count: int
    hit_rate: float
    oldest_entry: Optional[datetime]
    newest_entry: Optional[datetime]


class DetectionCache:
    """
    Caching system for detection results.
    """
    
    def __init__(self, 
                 cache_dir: str = "cache/detections",
                 max_size_mb: int = 1000,
                 max_age_hours: int = 24,
                 enable_database: bool = True,
                 db_path: str = "detection_cache.db"):
        """
        Initialize detection cache.
        
        Args:
            cache_dir: Directory for cache files
            max_size_mb: Maximum cache size in MB
            max_age_hours: Maximum age of cache entries in hours
            enable_database: Enable database storage
            db_path: Database file path
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_age = timedelta(hours=max_age_hours)
        self.enable_database = enable_database
        self.db_path = db_path
        
        # In-memory cache
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.cache_lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'cleanups': 0
        }
        
        # Database
        if self.enable_database:
            self._init_database()
        
        # Background cleanup
        self.cleanup_thread = None
        self.cleanup_interval = 3600  # 1 hour
        self.cleanup_running = False
        
        logger.info(f"DetectionCache initialized: {cache_dir}, max {max_size_mb}MB")
    
    def _init_database(self):
        """Initialize SQLite database for cache metadata."""
        try:
            self.db_connection = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = self.db_connection.cursor()
            
            # Create cache table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    video_path TEXT NOT NULL,
                    frame_number INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    detections TEXT NOT NULL,
                    processing_time REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    accessed_at TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    file_size INTEGER DEFAULT 0
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_video_path ON cache_entries(video_path)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_frame_number ON cache_entries(frame_number)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON cache_entries(created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_accessed_at ON cache_entries(accessed_at)')
            
            self.db_connection.commit()
            logger.info(f"Cache database initialized: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            self.enable_database = False
    
    def _generate_cache_key(self, video_path: str, frame_number: int, config_hash: str = "") -> str:
        """Generate cache key for video frame."""
        # Include video path, frame number, and config hash
        key_data = f"{video_path}:{frame_number}:{config_hash}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_frame_hash(self, frame: np.ndarray) -> str:
        """Generate hash for frame content."""
        # Resize frame to reduce hash computation time
        small_frame = cv2.resize(frame, (64, 64))
        frame_bytes = small_frame.tobytes()
        return hashlib.md5(frame_bytes).hexdigest()
    
    def get(self, video_path: str, frame_number: int, frame: np.ndarray, config_hash: str = "") -> Optional[Dict[str, List]]:
        """
        Get cached detection results.
        
        Args:
            video_path: Path to video file
            frame_number: Frame number
            frame: Frame data for validation
            config_hash: Configuration hash
            
        Returns:
            Cached detections or None if not found
        """
        cache_key = self._generate_cache_key(video_path, frame_number, config_hash)
        
        with self.cache_lock:
            # Check memory cache first
            if cache_key in self.memory_cache:
                entry = self.memory_cache[cache_key]
                
                # Validate frame content
                frame_hash = self._get_frame_hash(frame)
                expected_hash = self._get_frame_hash_from_entry(entry)
                
                if frame_hash == expected_hash:
                    # Update access statistics
                    entry.accessed_at = datetime.now()
                    entry.access_count += 1
                    
                    # Update database
                    if self.enable_database:
                        self._update_cache_entry(entry)
                    
                    self.stats['hits'] += 1
                    logger.debug(f"Cache hit: {cache_key}")
                    return entry.detections
                else:
                    # Frame content changed, remove from cache
                    del self.memory_cache[cache_key]
                    if self.enable_database:
                        self._remove_cache_entry(cache_key)
            
            # Check database cache
            if self.enable_database:
                entry = self._load_cache_entry(cache_key)
                if entry:
                    # Validate frame content
                    frame_hash = self._get_frame_hash(frame)
                    expected_hash = self._get_frame_hash_from_entry(entry)
                    
                    if frame_hash == expected_hash:
                        # Load into memory cache
                        self.memory_cache[cache_key] = entry
                        entry.accessed_at = datetime.now()
                        entry.access_count += 1
                        
                        # Update database
                        self._update_cache_entry(entry)
                        
                        self.stats['hits'] += 1
                        logger.debug(f"Cache hit (from DB): {cache_key}")
                        return entry.detections
                    else:
                        # Frame content changed, remove from database
                        self._remove_cache_entry(cache_key)
        
        self.stats['misses'] += 1
        logger.debug(f"Cache miss: {cache_key}")
        return None
    
    def put(self, 
            video_path: str, 
            frame_number: int, 
            frame: np.ndarray,
            detections: Dict[str, List],
            processing_time: float,
            config_hash: str = "") -> str:
        """
        Store detection results in cache.
        
        Args:
            video_path: Path to video file
            frame_number: Frame number
            frame: Frame data
            detections: Detection results
            config_hash: Configuration hash
            
        Returns:
            Cache key
        """
        cache_key = self._generate_cache_key(video_path, frame_number, config_hash)
        frame_hash = self._get_frame_hash(frame)
        
        # Create cache entry
        entry = CacheEntry(
            key=cache_key,
            video_path=video_path,
            frame_number=frame_number,
            timestamp=frame_number / 30.0,  # Assume 30 FPS
            detections=detections,
            processing_time=processing_time,
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            access_count=1
        )
        
        with self.cache_lock:
            # Store in memory cache
            self.memory_cache[cache_key] = entry
            
            # Store in database
            if self.enable_database:
                self._store_cache_entry(entry)
            
            # Check cache size and evict if necessary
            self._check_cache_size()
        
        logger.debug(f"Cached detection results: {cache_key}")
        return cache_key
    
    def _get_frame_hash_from_entry(self, entry: CacheEntry) -> str:
        """Get frame hash from cache entry (stored in metadata)."""
        # For now, we'll use a simple approach
        # In a real implementation, you might store the hash in the entry
        return hashlib.md5(f"{entry.video_path}:{entry.frame_number}".encode()).hexdigest()
    
    def _store_cache_entry(self, entry: CacheEntry):
        """Store cache entry in database."""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO cache_entries 
                (key, video_path, frame_number, timestamp, detections, processing_time,
                 created_at, accessed_at, access_count, file_size)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry.key,
                entry.video_path,
                entry.frame_number,
                entry.timestamp,
                json.dumps(entry.detections),
                entry.processing_time,
                entry.created_at.isoformat(),
                entry.accessed_at.isoformat(),
                entry.access_count,
                entry.file_size
            ))
            self.db_connection.commit()
        except Exception as e:
            logger.error(f"Failed to store cache entry: {e}")
    
    def _load_cache_entry(self, cache_key: str) -> Optional[CacheEntry]:
        """Load cache entry from database."""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute('SELECT * FROM cache_entries WHERE key = ?', (cache_key,))
            row = cursor.fetchone()
            
            if row:
                return CacheEntry(
                    key=row[0],
                    video_path=row[1],
                    frame_number=row[2],
                    timestamp=row[3],
                    detections=json.loads(row[4]),
                    processing_time=row[5],
                    created_at=datetime.fromisoformat(row[6]),
                    accessed_at=datetime.fromisoformat(row[7]),
                    access_count=row[8],
                    file_size=row[9]
                )
        except Exception as e:
            logger.error(f"Failed to load cache entry: {e}")
        
        return None
    
    def _update_cache_entry(self, entry: CacheEntry):
        """Update cache entry in database."""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute('''
                UPDATE cache_entries 
                SET accessed_at = ?, access_count = ?
                WHERE key = ?
            ''', (
                entry.accessed_at.isoformat(),
                entry.access_count,
                entry.key
            ))
            self.db_connection.commit()
        except Exception as e:
            logger.error(f"Failed to update cache entry: {e}")
    
    def _remove_cache_entry(self, cache_key: str):
        """Remove cache entry from database."""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute('DELETE FROM cache_entries WHERE key = ?', (cache_key,))
            self.db_connection.commit()
        except Exception as e:
            logger.error(f"Failed to remove cache entry: {e}")
    
    def _check_cache_size(self):
        """Check cache size and evict entries if necessary."""
        if not self.enable_database:
            return
        
        try:
            # Calculate current cache size
            cursor = self.db_connection.cursor()
            cursor.execute('SELECT SUM(file_size) FROM cache_entries')
            total_size = cursor.fetchone()[0] or 0
            
            if total_size > self.max_size_bytes:
                logger.info(f"Cache size exceeded: {total_size / 1024**2:.1f}MB > {self.max_size_bytes / 1024**2:.1f}MB")
                self._evict_oldest_entries()
        except Exception as e:
            logger.error(f"Failed to check cache size: {e}")
    
    def _evict_oldest_entries(self):
        """Evict oldest cache entries."""
        try:
            cursor = self.db_connection.cursor()
            
            # Get oldest entries
            cursor.execute('''
                SELECT key FROM cache_entries 
                ORDER BY accessed_at ASC, access_count ASC
                LIMIT 100
            ''')
            old_keys = [row[0] for row in cursor.fetchall()]
            
            # Remove from database
            for key in old_keys:
                cursor.execute('DELETE FROM cache_entries WHERE key = ?', (key,))
                if key in self.memory_cache:
                    del self.memory_cache[key]
            
            self.db_connection.commit()
            self.stats['evictions'] += len(old_keys)
            
            logger.info(f"Evicted {len(old_keys)} cache entries")
            
        except Exception as e:
            logger.error(f"Failed to evict cache entries: {e}")
    
    def cleanup_expired_entries(self):
        """Clean up expired cache entries."""
        cutoff_time = datetime.now() - self.max_age
        
        with self.cache_lock:
            # Clean memory cache
            expired_keys = []
            for key, entry in self.memory_cache.items():
                if entry.created_at < cutoff_time:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.memory_cache[key]
            
            # Clean database cache
            if self.enable_database:
                try:
                    cursor = self.db_connection.cursor()
                    cursor.execute('DELETE FROM cache_entries WHERE created_at < ?', (cutoff_time.isoformat(),))
                    deleted_count = cursor.rowcount
                    self.db_connection.commit()
                    
                    self.stats['cleanups'] += deleted_count
                    logger.info(f"Cleaned up {deleted_count} expired cache entries")
                    
                except Exception as e:
                    logger.error(f"Failed to cleanup expired entries: {e}")
    
    def start_background_cleanup(self):
        """Start background cleanup thread."""
        if self.cleanup_running:
            return
        
        self.cleanup_running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        logger.info("Background cache cleanup started")
    
    def stop_background_cleanup(self):
        """Stop background cleanup thread."""
        self.cleanup_running = False
        if self.cleanup_thread:
            self.cleanup_thread.join()
        logger.info("Background cache cleanup stopped")
    
    def _cleanup_loop(self):
        """Background cleanup loop."""
        while self.cleanup_running:
            try:
                self.cleanup_expired_entries()
                time.sleep(self.cleanup_interval)
            except Exception as e:
                logger.error(f"Background cleanup error: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self.cache_lock:
            total_entries = len(self.memory_cache)
            
            if self.enable_database:
                try:
                    cursor = self.db_connection.cursor()
                    cursor.execute('SELECT COUNT(*) FROM cache_entries')
                    total_entries = cursor.fetchone()[0]
                    
                    cursor.execute('SELECT SUM(file_size) FROM cache_entries')
                    total_size = cursor.fetchone()[0] or 0
                except Exception as e:
                    logger.error(f"Failed to get cache stats: {e}")
                    total_size = 0
            else:
                total_size = 0
            
            hit_rate = 0.0
            if self.stats['hits'] + self.stats['misses'] > 0:
                hit_rate = self.stats['hits'] / (self.stats['hits'] + self.stats['misses'])
            
            # Get oldest and newest entries
            oldest_entry = None
            newest_entry = None
            
            if self.memory_cache:
                entries = list(self.memory_cache.values())
                oldest_entry = min(entries, key=lambda e: e.created_at).created_at
                newest_entry = max(entries, key=lambda e: e.created_at).created_at
            
            return CacheStats(
                total_entries=total_entries,
                total_size_mb=total_size / 1024**2,
                hit_count=self.stats['hits'],
                miss_count=self.stats['misses'],
                hit_rate=hit_rate,
                oldest_entry=oldest_entry,
                newest_entry=newest_entry
            )
    
    def clear_cache(self):
        """Clear all cache entries."""
        with self.cache_lock:
            # Clear memory cache
            self.memory_cache.clear()
            
            # Clear database cache
            if self.enable_database:
                try:
                    cursor = self.db_connection.cursor()
                    cursor.execute('DELETE FROM cache_entries')
                    self.db_connection.commit()
                    logger.info("Cache cleared")
                except Exception as e:
                    logger.error(f"Failed to clear cache: {e}")
            
            # Reset statistics
            self.stats = {'hits': 0, 'misses': 0, 'evictions': 0, 'cleanups': 0}
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information."""
        stats = self.get_stats()
        
        return {
            'cache_dir': str(self.cache_dir),
            'max_size_mb': self.max_size_bytes / 1024**2,
            'max_age_hours': self.max_age.total_seconds() / 3600,
            'database_enabled': self.enable_database,
            'background_cleanup': self.cleanup_running,
            'stats': {
                'total_entries': stats.total_entries,
                'total_size_mb': stats.total_size_mb,
                'hit_count': stats.hit_count,
                'miss_count': stats.miss_count,
                'hit_rate': stats.hit_rate,
                'evictions': self.stats['evictions'],
                'cleanups': self.stats['cleanups']
            },
            'oldest_entry': stats.oldest_entry.isoformat() if stats.oldest_entry else None,
            'newest_entry': stats.newest_entry.isoformat() if stats.newest_entry else None
        }
    
    def close(self):
        """Close cache and cleanup resources."""
        self.stop_background_cleanup()
        
        if self.enable_database and hasattr(self, 'db_connection'):
            self.db_connection.close()
        
        logger.info("Detection cache closed")


class CachedDetectionProcessor:
    """
    Detection processor with caching support.
    """
    
    def __init__(self, 
                 detection_callback: callable,
                 cache: DetectionCache,
                 config_hash: str = ""):
        """
        Initialize cached detection processor.
        
        Args:
            detection_callback: Function to perform detection
            cache: Detection cache instance
            config_hash: Configuration hash for cache key
        """
        self.detection_callback = detection_callback
        self.cache = cache
        self.config_hash = config_hash
        
        # Statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_processing_time': 0.0,
            'cached_processing_time': 0.0
        }
    
    def detect_with_cache(self, 
                         video_path: str, 
                         frame_number: int, 
                         frame: np.ndarray) -> Dict[str, List]:
        """
        Perform detection with caching.
        
        Args:
            video_path: Path to video file
            frame_number: Frame number
            frame: Frame data
            
        Returns:
            Detection results
        """
        # Try to get from cache
        cached_result = self.cache.get(video_path, frame_number, frame, self.config_hash)
        
        if cached_result is not None:
            self.stats['cache_hits'] += 1
            return cached_result
        
        # Perform detection
        start_time = time.time()
        detections = self.detection_callback(frame)
        processing_time = time.time() - start_time
        
        # Store in cache
        self.cache.put(
            video_path=video_path,
            frame_number=frame_number,
            frame=frame,
            detections=detections,
            processing_time=processing_time,
            config_hash=self.config_hash
        )
        
        self.stats['cache_misses'] += 1
        self.stats['total_processing_time'] += processing_time
        self.stats['cached_processing_time'] += processing_time
        
        return detections
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        hit_rate = self.stats['cache_hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'hit_rate': hit_rate,
            'total_processing_time': self.stats['total_processing_time'],
            'cached_processing_time': self.stats['cached_processing_time'],
            'time_saved': self.stats['total_processing_time'] - self.stats['cached_processing_time']
        }
