"""
Database optimization utilities for improved performance and storage efficiency.
"""

import sqlite3
import logging
import time
import json
import pickle
import threading
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime, timedelta
import queue
import hashlib
import zlib
import numpy as np
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class DatabaseStats:
    """Database performance statistics."""
    total_queries: int
    avg_query_time: float
    slow_queries: int
    cache_hits: int
    cache_misses: int
    connection_pool_size: int
    active_connections: int
    database_size_mb: float
    index_count: int
    table_count: int


@dataclass
class QueryResult:
    """Database query result."""
    data: List[Dict[str, Any]]
    execution_time: float
    rows_returned: int
    cache_hit: bool = False


class ConnectionPool:
    """
    Database connection pool for efficient connection management.
    """
    
    def __init__(self, 
                 db_path: str,
                 max_connections: int = 10,
                 min_connections: int = 2,
                 connection_timeout: float = 30.0):
        """
        Initialize connection pool.
        
        Args:
            db_path: Database file path
            max_connections: Maximum number of connections
            min_connections: Minimum number of connections
            connection_timeout: Connection timeout in seconds
        """
        self.db_path = db_path
        self.max_connections = max_connections
        self.min_connections = min_connections
        self.connection_timeout = connection_timeout
        
        # Connection management
        self.connections = queue.Queue(maxsize=max_connections)
        self.active_connections = 0
        self.connection_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'connections_created': 0,
            'connections_reused': 0,
            'connections_closed': 0,
            'pool_hits': 0,
            'pool_misses': 0
        }
        
        # Initialize minimum connections
        self._initialize_connections()
        
        logger.info(f"ConnectionPool initialized: {db_path}, max {max_connections} connections")
    
    def _initialize_connections(self):
        """Initialize minimum number of connections."""
        for _ in range(self.min_connections):
            conn = self._create_connection()
            self.connections.put(conn)
            self.stats['connections_created'] += 1
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection."""
        conn = sqlite3.connect(
            self.db_path,
            timeout=self.connection_timeout,
            check_same_thread=False
        )
        
        # Configure connection
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA mmap_size=268435456")  # 256MB
        
        # Enable row factory for dict-like access
        conn.row_factory = sqlite3.Row
        
        return conn
    
    @contextmanager
    def get_connection(self):
        """Get a database connection from the pool."""
        conn = None
        try:
            # Try to get existing connection
            try:
                conn = self.connections.get(timeout=5.0)
                self.stats['pool_hits'] += 1
                self.stats['connections_reused'] += 1
            except queue.Empty:
                # Create new connection if pool is empty
                with self.connection_lock:
                    if self.active_connections < self.max_connections:
                        conn = self._create_connection()
                        self.active_connections += 1
                        self.stats['connections_created'] += 1
                        self.stats['pool_misses'] += 1
                    else:
                        # Wait for connection to become available
                        conn = self.connections.get(timeout=10.0)
                        self.stats['pool_hits'] += 1
            
            yield conn
            
        finally:
            # Return connection to pool
            if conn:
                try:
                    self.connections.put(conn, timeout=1.0)
                except queue.Full:
                    # Pool is full, close connection
                    conn.close()
                    with self.connection_lock:
                        self.active_connections -= 1
                    self.stats['connections_closed'] += 1
    
    def close_all(self):
        """Close all connections in the pool."""
        while not self.connections.empty():
            try:
                conn = self.connections.get_nowait()
                conn.close()
                self.stats['connections_closed'] += 1
            except queue.Empty:
                break
        
        with self.connection_lock:
            self.active_connections = 0
        
        logger.info("All connections closed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return {
            'pool_size': self.connections.qsize(),
            'active_connections': self.active_connections,
            'max_connections': self.max_connections,
            'min_connections': self.min_connections,
            'stats': self.stats.copy()
        }


class QueryCache:
    """
    Query result cache for improved performance.
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        """
        Initialize query cache.
        
        Args:
            max_size: Maximum number of cached queries
            ttl_seconds: Time to live for cached results
        """
        self.max_size = max_size
        self.ttl = timedelta(seconds=ttl_seconds)
        
        # Cache storage
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.cache_lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size': 0
        }
        
        logger.info(f"QueryCache initialized: max {max_size} entries, TTL {ttl_seconds}s")
    
    def _generate_key(self, query: str, params: Tuple) -> str:
        """Generate cache key for query and parameters."""
        key_data = f"{query}:{params}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, query: str, params: Tuple = ()) -> Optional[Any]:
        """Get cached query result."""
        key = self._generate_key(query, params)
        
        with self.cache_lock:
            if key in self.cache:
                result, timestamp = self.cache[key]
                
                # Check if result is still valid
                if datetime.now() - timestamp < self.ttl:
                    self.stats['hits'] += 1
                    return result
                else:
                    # Expired, remove from cache
                    del self.cache[key]
                    self.stats['evictions'] += 1
            
            self.stats['misses'] += 1
            return None
    
    def put(self, query: str, params: Tuple, result: Any):
        """Store query result in cache."""
        key = self._generate_key(query, params)
        
        with self.cache_lock:
            # Check cache size
            if len(self.cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]
                self.stats['evictions'] += 1
            
            # Store result
            self.cache[key] = (result, datetime.now())
            self.stats['size'] = len(self.cache)
    
    def clear(self):
        """Clear all cached results."""
        with self.cache_lock:
            self.cache.clear()
            self.stats['size'] = 0
            self.stats['evictions'] += len(self.cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.cache_lock:
            hit_rate = 0.0
            if self.stats['hits'] + self.stats['misses'] > 0:
                hit_rate = self.stats['hits'] / (self.stats['hits'] + self.stats['misses'])
            
            return {
                'size': self.stats['size'],
                'max_size': self.max_size,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'evictions': self.stats['evictions'],
                'hit_rate': hit_rate
            }


class DatabaseOptimizer:
    """
    Database optimization manager.
    """
    
    def __init__(self, 
                 db_path: str,
                 enable_connection_pool: bool = True,
                 enable_query_cache: bool = True,
                 enable_compression: bool = True,
                 max_connections: int = 10):
        """
        Initialize database optimizer.
        
        Args:
            db_path: Database file path
            enable_connection_pool: Enable connection pooling
            enable_query_cache: Enable query result caching
            enable_compression: Enable data compression
            max_connections: Maximum number of connections
        """
        self.db_path = db_path
        self.enable_connection_pool = enable_connection_pool
        self.enable_query_cache = enable_query_cache
        self.enable_compression = enable_compression
        
        # Initialize components
        if self.enable_connection_pool:
            self.connection_pool = ConnectionPool(db_path, max_connections)
        else:
            self.connection_pool = None
        
        if self.enable_query_cache:
            self.query_cache = QueryCache()
        else:
            self.query_cache = None
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'total_query_time': 0.0,
            'slow_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Performance monitoring
        self.slow_query_threshold = 1.0  # 1 second
        
        logger.info(f"DatabaseOptimizer initialized: {db_path}")
    
    def execute_query(self, 
                     query: str, 
                     params: Tuple = (),
                     fetch_all: bool = True,
                     use_cache: bool = True) -> QueryResult:
        """
        Execute a database query with optimization.
        
        Args:
            query: SQL query string
            params: Query parameters
            fetch_all: Whether to fetch all results
            use_cache: Whether to use query cache
            
        Returns:
            Query result
        """
        start_time = time.time()
        cache_hit = False
        
        # Try to get from cache
        if self.enable_query_cache and use_cache:
            cached_result = self.query_cache.get(query, params)
            if cached_result is not None:
                cache_hit = True
                self.stats['cache_hits'] += 1
                execution_time = time.time() - start_time
                
                return QueryResult(
                    data=cached_result,
                    execution_time=execution_time,
                    rows_returned=len(cached_result),
                    cache_hit=True
                )
        
        # Execute query
        if self.connection_pool:
            with self.connection_pool.get_connection() as conn:
                result = self._execute_with_connection(conn, query, params, fetch_all)
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                result = self._execute_with_connection(conn, query, params, fetch_all)
            finally:
                conn.close()
        
        execution_time = time.time() - start_time
        
        # Update statistics
        self.stats['total_queries'] += 1
        self.stats['total_query_time'] += execution_time
        
        if execution_time > self.slow_query_threshold:
            self.stats['slow_queries'] += 1
            logger.warning(f"Slow query detected: {execution_time:.2f}s - {query[:100]}...")
        
        # Cache result
        if self.enable_query_cache and use_cache and not cache_hit:
            self.query_cache.put(query, params, result)
            self.stats['cache_misses'] += 1
        
        return QueryResult(
            data=result,
            execution_time=execution_time,
            rows_returned=len(result),
            cache_hit=cache_hit
        )
    
    def _execute_with_connection(self, 
                                conn: sqlite3.Connection, 
                                query: str, 
                                params: Tuple, 
                                fetch_all: bool) -> List[Dict[str, Any]]:
        """Execute query with given connection."""
        cursor = conn.cursor()
        
        try:
            cursor.execute(query, params)
            
            if fetch_all:
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
            else:
                return []
                
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
        finally:
            cursor.close()
    
    def execute_batch(self, 
                     queries: List[Tuple[str, Tuple]],
                     use_transaction: bool = True) -> List[QueryResult]:
        """
        Execute multiple queries in batch.
        
        Args:
            queries: List of (query, params) tuples
            use_transaction: Whether to use transaction
            
        Returns:
            List of query results
        """
        start_time = time.time()
        results = []
        
        if self.connection_pool:
            with self.connection_pool.get_connection() as conn:
                if use_transaction:
                    conn.execute("BEGIN TRANSACTION")
                
                try:
                    for query, params in queries:
                        result = self._execute_with_connection(conn, query, params, True)
                        results.append(QueryResult(
                            data=result,
                            execution_time=0.0,  # Will be calculated for batch
                            rows_returned=len(result)
                        ))
                    
                    if use_transaction:
                        conn.commit()
                        
                except Exception as e:
                    if use_transaction:
                        conn.rollback()
                    raise
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                if use_transaction:
                    conn.execute("BEGIN TRANSACTION")
                
                for query, params in queries:
                    result = self._execute_with_connection(conn, query, params, True)
                    results.append(QueryResult(
                        data=result,
                        execution_time=0.0,
                        rows_returned=len(result)
                    ))
                
                if use_transaction:
                    conn.commit()
                    
            except Exception as e:
                if use_transaction:
                    conn.rollback()
                raise
            finally:
                conn.close()
        
        total_time = time.time() - start_time
        
        # Update statistics
        self.stats['total_queries'] += len(queries)
        self.stats['total_query_time'] += total_time
        
        return results
    
    def optimize_database(self):
        """Optimize database performance."""
        logger.info("Starting database optimization...")
        
        optimization_queries = [
            "ANALYZE",
            "VACUUM",
            "PRAGMA optimize",
            "PRAGMA wal_checkpoint(TRUNCATE)"
        ]
        
        for query in optimization_queries:
            try:
                self.execute_query(query, use_cache=False)
                logger.info(f"Executed optimization: {query}")
            except Exception as e:
                logger.error(f"Optimization failed: {query} - {e}")
        
        logger.info("Database optimization completed")
    
    def create_indexes(self, table_indexes: Dict[str, List[str]]):
        """
        Create database indexes for improved performance.
        
        Args:
            table_indexes: Dictionary mapping table names to list of column names
        """
        logger.info("Creating database indexes...")
        
        for table_name, columns in table_indexes.items():
            for column in columns:
                index_name = f"idx_{table_name}_{column}"
                query = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({column})"
                
                try:
                    self.execute_query(query, use_cache=False)
                    logger.info(f"Created index: {index_name}")
                except Exception as e:
                    logger.error(f"Index creation failed: {index_name} - {e}")
        
        logger.info("Index creation completed")
    
    def get_database_stats(self) -> DatabaseStats:
        """Get database performance statistics."""
        # Get database size
        db_size = 0
        if Path(self.db_path).exists():
            db_size = Path(self.db_path).stat().st_size / 1024**2
        
        # Get table and index counts
        table_count = 0
        index_count = 0
        
        try:
            result = self.execute_query(
                "SELECT COUNT(*) as count FROM sqlite_master WHERE type='table'",
                use_cache=False
            )
            table_count = result.data[0]['count'] if result.data else 0
            
            result = self.execute_query(
                "SELECT COUNT(*) as count FROM sqlite_master WHERE type='index'",
                use_cache=False
            )
            index_count = result.data[0]['count'] if result.data else 0
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
        
        # Calculate average query time
        avg_query_time = 0.0
        if self.stats['total_queries'] > 0:
            avg_query_time = self.stats['total_query_time'] / self.stats['total_queries']
        
        # Get connection pool stats
        connection_pool_size = 0
        active_connections = 0
        
        if self.connection_pool:
            pool_stats = self.connection_pool.get_stats()
            connection_pool_size = pool_stats['pool_size']
            active_connections = pool_stats['active_connections']
        
        return DatabaseStats(
            total_queries=self.stats['total_queries'],
            avg_query_time=avg_query_time,
            slow_queries=self.stats['slow_queries'],
            cache_hits=self.stats['cache_hits'],
            cache_misses=self.stats['cache_misses'],
            connection_pool_size=connection_pool_size,
            active_connections=active_connections,
            database_size_mb=db_size,
            index_count=index_count,
            table_count=table_count
        )
    
    def export_database(self, export_path: str):
        """Export database to file."""
        logger.info(f"Exporting database to {export_path}")
        
        try:
            if self.connection_pool:
                with self.connection_pool.get_connection() as conn:
                    with open(export_path, 'w') as f:
                        for line in conn.iterdump():
                            f.write(f"{line}\n")
            else:
                conn = sqlite3.connect(self.db_path)
                with open(export_path, 'w') as f:
                    for line in conn.iterdump():
                        f.write(f"{line}\n")
                conn.close()
            
            logger.info(f"Database exported successfully: {export_path}")
            
        except Exception as e:
            logger.error(f"Database export failed: {e}")
            raise
    
    def import_database(self, import_path: str):
        """Import database from file."""
        logger.info(f"Importing database from {import_path}")
        
        try:
            if self.connection_pool:
                with self.connection_pool.get_connection() as conn:
                    with open(import_path, 'r') as f:
                        sql_script = f.read()
                    conn.executescript(sql_script)
            else:
                conn = sqlite3.connect(self.db_path)
                with open(import_path, 'r') as f:
                    sql_script = f.read()
                conn.executescript(sql_script)
                conn.close()
            
            logger.info(f"Database imported successfully: {import_path}")
            
        except Exception as e:
            logger.error(f"Database import failed: {e}")
            raise
    
    def close(self):
        """Close database optimizer and cleanup resources."""
        if self.connection_pool:
            self.connection_pool.close_all()
        
        if self.query_cache:
            self.query_cache.clear()
        
        logger.info("Database optimizer closed")


class CompressedDataStore:
    """
    Compressed data storage for large objects.
    """
    
    def __init__(self, db_path: str, compression_level: int = 6):
        """
        Initialize compressed data store.
        
        Args:
            db_path: Database file path
            compression_level: Compression level (1-9)
        """
        self.db_path = db_path
        self.compression_level = compression_level
        
        # Initialize database
        self._init_database()
        
        logger.info(f"CompressedDataStore initialized: {db_path}")
    
    def _init_database(self):
        """Initialize database for compressed storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create compressed data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS compressed_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                data BLOB NOT NULL,
                compressed_size INTEGER NOT NULL,
                original_size INTEGER NOT NULL,
                compression_ratio REAL NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                accessed_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_compressed_key ON compressed_data(key)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_compressed_accessed ON compressed_data(accessed_at)')
        
        conn.commit()
        conn.close()
    
    def store(self, key: str, data: Any) -> bool:
        """
        Store compressed data.
        
        Args:
            key: Unique key for the data
            data: Data to store (must be serializable)
            
        Returns:
            True if successful
        """
        try:
            # Serialize data
            serialized_data = pickle.dumps(data)
            original_size = len(serialized_data)
            
            # Compress data
            compressed_data = zlib.compress(serialized_data, self.compression_level)
            compressed_size = len(compressed_data)
            compression_ratio = compressed_size / original_size if original_size > 0 else 0
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO compressed_data 
                (key, data, compressed_size, original_size, compression_ratio, accessed_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (key, compressed_data, compressed_size, original_size, compression_ratio))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Stored compressed data: {key}, ratio: {compression_ratio:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store compressed data: {e}")
            return False
    
    def retrieve(self, key: str) -> Optional[Any]:
        """
        Retrieve and decompress data.
        
        Args:
            key: Key for the data
            
        Returns:
            Decompressed data or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT data FROM compressed_data WHERE key = ?
            ''', (key,))
            
            row = cursor.fetchone()
            if row is None:
                conn.close()
                return None
            
            compressed_data = row[0]
            
            # Update access time
            cursor.execute('''
                UPDATE compressed_data SET accessed_at = CURRENT_TIMESTAMP WHERE key = ?
            ''', (key,))
            
            conn.commit()
            conn.close()
            
            # Decompress data
            serialized_data = zlib.decompress(compressed_data)
            data = pickle.loads(serialized_data)
            
            logger.debug(f"Retrieved compressed data: {key}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to retrieve compressed data: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """
        Delete compressed data.
        
        Args:
            key: Key for the data
            
        Returns:
            True if successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM compressed_data WHERE key = ?', (key,))
            deleted = cursor.rowcount > 0
            
            conn.commit()
            conn.close()
            
            if deleted:
                logger.debug(f"Deleted compressed data: {key}")
            
            return deleted
            
        except Exception as e:
            logger.error(f"Failed to delete compressed data: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get compressed data store statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get total statistics
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_entries,
                    SUM(compressed_size) as total_compressed_size,
                    SUM(original_size) as total_original_size,
                    AVG(compression_ratio) as avg_compression_ratio
                FROM compressed_data
            ''')
            
            row = cursor.fetchone()
            if row:
                total_entries, total_compressed, total_original, avg_ratio = row
                
                # Calculate space savings
                space_saved = total_original - total_compressed if total_original else 0
                space_saved_percent = (space_saved / total_original * 100) if total_original else 0
                
                stats = {
                    'total_entries': total_entries,
                    'total_compressed_size_mb': total_compressed / 1024**2,
                    'total_original_size_mb': total_original / 1024**2,
                    'space_saved_mb': space_saved / 1024**2,
                    'space_saved_percent': space_saved_percent,
                    'avg_compression_ratio': avg_ratio
                }
            else:
                stats = {
                    'total_entries': 0,
                    'total_compressed_size_mb': 0,
                    'total_original_size_mb': 0,
                    'space_saved_mb': 0,
                    'space_saved_percent': 0,
                    'avg_compression_ratio': 0
                }
            
            conn.close()
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get compressed data stats: {e}")
            return {}
    
    def cleanup_old_data(self, max_age_days: int = 30):
        """
        Clean up old compressed data.
        
        Args:
            max_age_days: Maximum age in days
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                DELETE FROM compressed_data 
                WHERE accessed_at < datetime('now', '-{} days')
            '''.format(max_age_days))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old compressed data entries")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old compressed data: {e}")
