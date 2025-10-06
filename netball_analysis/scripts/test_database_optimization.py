"""
Test script for database optimization performance.
"""

import time
import logging
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.performance import DatabaseOptimizer, ConnectionPool, QueryCache, CompressedDataStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_connection_pool():
    """Test connection pool functionality."""
    logger.info("=== Connection Pool Test ===")
    
    # Create temporary database
    temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_db.close()
    
    try:
        # Initialize connection pool
        pool = ConnectionPool(
            db_path=temp_db.name,
            max_connections=5,
            min_connections=2
        )
        
        # Create test table
        with pool.get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS test_table (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    value REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
        
        # Test concurrent connections
        print("Testing concurrent connections...")
        start_time = time.time()
        
        def insert_data(conn_id):
            with pool.get_connection() as conn:
                for i in range(10):
                    conn.execute(
                        'INSERT INTO test_table (name, value) VALUES (?, ?)',
                        (f'conn_{conn_id}_item_{i}', i * 1.5)
                    )
                conn.commit()
        
        # Simulate concurrent access
        import threading
        threads = []
        for i in range(3):
            thread = threading.Thread(target=insert_data, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        connection_time = time.time() - start_time
        
        # Get pool statistics
        stats = pool.get_stats()
        print(f"Connection Pool Statistics:")
        print(f"  Pool size: {stats['pool_size']}")
        print(f"  Active connections: {stats['active_connections']}")
        print(f"  Connections created: {stats['stats']['connections_created']}")
        print(f"  Connections reused: {stats['stats']['connections_reused']}")
        print(f"  Pool hits: {stats['stats']['pool_hits']}")
        print(f"  Pool misses: {stats['stats']['pool_misses']}")
        print(f"  Connection time: {connection_time:.2f}s")
        
        # Verify data
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM test_table')
            count = cursor.fetchone()[0]
            print(f"  Total records: {count}")
        
        pool.close_all()
        
    finally:
        # Cleanup
        if Path(temp_db.name).exists():
            Path(temp_db.name).unlink()


def test_query_cache():
    """Test query cache functionality."""
    logger.info("=== Query Cache Test ===")
    
    # Create temporary database
    temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_db.close()
    
    try:
        # Initialize query cache
        cache = QueryCache(max_size=100, ttl_seconds=5)
        
        # Create test table
        conn = sqlite3.connect(temp_db.name)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS cache_test (
                id INTEGER PRIMARY KEY,
                name TEXT,
                value REAL
            )
        ''')
        
        # Insert test data
        for i in range(20):
            conn.execute(
                'INSERT INTO cache_test (name, value) VALUES (?, ?)',
                (f'item_{i}', i * 2.5)
            )
        conn.commit()
        conn.close()
        
        # Test cache functionality
        query = "SELECT * FROM cache_test WHERE value > ?"
        params = (10.0,)
        
        # First query - should miss cache
        print("First query (cache miss):")
        start_time = time.time()
        result1 = cache.get(query, params)
        get_time1 = time.time() - start_time
        
        if result1 is None:
            # Execute query and cache result
            conn = sqlite3.connect(temp_db.name)
            cursor = conn.cursor()
            cursor.execute(query, params)
            result1 = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            cache.put(query, params, result1)
            print(f"  Cache miss, stored {len(result1)} rows in {get_time1*1000:.2f}ms")
        
        # Second query - should hit cache
        print("Second query (cache hit):")
        start_time = time.time()
        result2 = cache.get(query, params)
        get_time2 = time.time() - start_time
        
        if result2 is not None:
            print(f"  Cache hit, retrieved {len(result2)} rows in {get_time2*1000:.2f}ms")
        
        # Test cache expiration
        print("Testing cache expiration...")
        time.sleep(6)  # Wait for expiration
        
        start_time = time.time()
        result3 = cache.get(query, params)
        get_time3 = time.time() - start_time
        
        if result3 is None:
            print(f"  Cache expired, miss in {get_time3*1000:.2f}ms")
        
        # Get cache statistics
        stats = cache.get_stats()
        print(f"Cache Statistics:")
        print(f"  Size: {stats['size']}")
        print(f"  Hits: {stats['hits']}")
        print(f"  Misses: {stats['misses']}")
        print(f"  Evictions: {stats['evictions']}")
        print(f"  Hit rate: {stats['hit_rate']*100:.1f}%")
        
        cache.clear()
        
    finally:
        # Cleanup
        if Path(temp_db.name).exists():
            Path(temp_db.name).unlink()


def test_database_optimizer():
    """Test database optimizer functionality."""
    logger.info("=== Database Optimizer Test ===")
    
    # Create temporary database
    temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_db.close()
    
    try:
        # Initialize database optimizer
        optimizer = DatabaseOptimizer(
            db_path=temp_db.name,
            enable_connection_pool=True,
            enable_query_cache=True,
            max_connections=5
        )
        
        # Create test tables
        create_queries = [
            '''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE,
                    age INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''',
            '''
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER,
                    product TEXT,
                    amount REAL,
                    order_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            '''
        ]
        
        for query in create_queries:
            optimizer.execute_query(query, use_cache=False)
        
        # Insert test data
        print("Inserting test data...")
        insert_queries = []
        
        for i in range(100):
            insert_queries.append((
                'INSERT INTO users (name, email, age) VALUES (?, ?, ?)',
                (f'user_{i}', f'user_{i}@example.com', 20 + (i % 50))
            ))
        
        for i in range(200):
            insert_queries.append((
                'INSERT INTO orders (user_id, product, amount) VALUES (?, ?, ?)',
                (i % 100 + 1, f'product_{i % 10}', 10.0 + (i % 100))
            ))
        
        # Execute batch insert
        start_time = time.time()
        results = optimizer.execute_batch(insert_queries, use_transaction=True)
        batch_time = time.time() - start_time
        
        print(f"Batch insert completed in {batch_time:.2f}s")
        
        # Test query performance
        print("Testing query performance...")
        
        # Test 1: Simple query
        start_time = time.time()
        result1 = optimizer.execute_query(
            'SELECT COUNT(*) as count FROM users WHERE age > ?',
            (30,)
        )
        query1_time = time.time() - start_time
        
        print(f"Simple query: {result1.rows_returned} rows in {query1_time*1000:.2f}ms")
        print(f"  Cache hit: {result1.cache_hit}")
        
        # Test 2: Complex query
        start_time = time.time()
        result2 = optimizer.execute_query('''
            SELECT u.name, COUNT(o.id) as order_count, SUM(o.amount) as total_amount
            FROM users u
            LEFT JOIN orders o ON u.id = o.user_id
            WHERE u.age > ?
            GROUP BY u.id, u.name
            HAVING COUNT(o.id) > ?
            ORDER BY total_amount DESC
            LIMIT ?
        ''', (25, 1, 10))
        query2_time = time.time() - start_time
        
        print(f"Complex query: {result2.rows_returned} rows in {query2_time*1000:.2f}ms")
        print(f"  Cache hit: {result2.cache_hit}")
        
        # Test 3: Cached query (should be faster)
        start_time = time.time()
        result3 = optimizer.execute_query(
            'SELECT COUNT(*) as count FROM users WHERE age > ?',
            (30,)
        )
        query3_time = time.time() - start_time
        
        print(f"Cached query: {result3.rows_returned} rows in {query3_time*1000:.2f}ms")
        print(f"  Cache hit: {result3.cache_hit}")
        
        # Create indexes for optimization
        print("Creating indexes...")
        table_indexes = {
            'users': ['name', 'email', 'age'],
            'orders': ['user_id', 'product', 'amount']
        }
        optimizer.create_indexes(table_indexes)
        
        # Test query after indexing
        start_time = time.time()
        result4 = optimizer.execute_query('''
            SELECT u.name, COUNT(o.id) as order_count
            FROM users u
            LEFT JOIN orders o ON u.id = o.user_id
            WHERE u.age > ?
            GROUP BY u.id, u.name
            ORDER BY order_count DESC
            LIMIT ?
        ''', (30, 5))
        query4_time = time.time() - start_time
        
        print(f"Indexed query: {result4.rows_returned} rows in {query4_time*1000:.2f}ms")
        
        # Get database statistics
        stats = optimizer.get_database_stats()
        print(f"Database Statistics:")
        print(f"  Total queries: {stats.total_queries}")
        print(f"  Average query time: {stats.avg_query_time*1000:.2f}ms")
        print(f"  Slow queries: {stats.slow_queries}")
        print(f"  Cache hits: {stats.cache_hits}")
        print(f"  Cache misses: {stats.cache_misses}")
        print(f"  Database size: {stats.database_size_mb:.1f} MB")
        print(f"  Table count: {stats.table_count}")
        print(f"  Index count: {stats.index_count}")
        
        # Optimize database
        print("Optimizing database...")
        optimizer.optimize_database()
        
        optimizer.close()
        
    finally:
        # Cleanup
        if Path(temp_db.name).exists():
            Path(temp_db.name).unlink()


def test_compressed_data_store():
    """Test compressed data store functionality."""
    logger.info("=== Compressed Data Store Test ===")
    
    # Create temporary database
    temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_db.close()
    
    try:
        # Initialize compressed data store
        store = CompressedDataStore(temp_db.name, compression_level=6)
        
        # Test storing different types of data
        test_data = {
            'small_data': {'key': 'value', 'number': 42},
            'medium_data': list(range(1000)),
            'large_data': np.random.rand(100, 100).tolist(),
            'text_data': 'This is a test string with some content. ' * 100
        }
        
        print("Storing test data...")
        for key, data in test_data.items():
            start_time = time.time()
            success = store.store(key, data)
            store_time = time.time() - start_time
            
            if success:
                print(f"  Stored {key} in {store_time*1000:.2f}ms")
            else:
                print(f"  Failed to store {key}")
        
        # Test retrieving data
        print("Retrieving test data...")
        for key in test_data.keys():
            start_time = time.time()
            retrieved_data = store.retrieve(key)
            retrieve_time = time.time() - start_time
            
            if retrieved_data is not None:
                print(f"  Retrieved {key} in {retrieve_time*1000:.2f}ms")
                
                # Verify data integrity
                if isinstance(retrieved_data, dict):
                    print(f"    Dict keys: {len(retrieved_data)}")
                elif isinstance(retrieved_data, list):
                    print(f"    List length: {len(retrieved_data)}")
                elif isinstance(retrieved_data, str):
                    print(f"    String length: {len(retrieved_data)}")
            else:
                print(f"  Failed to retrieve {key}")
        
        # Test data that doesn't exist
        non_existent = store.retrieve('non_existent_key')
        if non_existent is None:
            print("  Correctly returned None for non-existent key")
        
        # Get compression statistics
        stats = store.get_stats()
        print(f"Compression Statistics:")
        print(f"  Total entries: {stats['total_entries']}")
        print(f"  Original size: {stats['total_original_size_mb']:.2f} MB")
        print(f"  Compressed size: {stats['total_compressed_size_mb']:.2f} MB")
        print(f"  Space saved: {stats['space_saved_mb']:.2f} MB")
        print(f"  Space saved %: {stats['space_saved_percent']:.1f}%")
        print(f"  Average compression ratio: {stats['avg_compression_ratio']:.2f}")
        
        # Test deletion
        print("Testing deletion...")
        deleted = store.delete('small_data')
        if deleted:
            print("  Successfully deleted 'small_data'")
        
        # Try to retrieve deleted data
        retrieved = store.retrieve('small_data')
        if retrieved is None:
            print("  Correctly returned None for deleted data")
        
        # Test cleanup
        print("Testing cleanup...")
        store.cleanup_old_data(max_age_days=0)  # Clean up all data
        
        # Check final statistics
        final_stats = store.get_stats()
        print(f"Final statistics: {final_stats['total_entries']} entries")
        
    finally:
        # Cleanup
        if Path(temp_db.name).exists():
            Path(temp_db.name).unlink()


def test_performance_comparison():
    """Compare performance with and without optimization."""
    logger.info("=== Performance Comparison Test ===")
    
    # Create temporary databases
    temp_db1 = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_db1.close()
    temp_db2 = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_db2.close()
    
    try:
        # Test 1: Without optimization
        print("1. Testing without optimization...")
        start_time = time.time()
        
        conn = sqlite3.connect(temp_db1.name)
        conn.execute('''
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY,
                name TEXT,
                value REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Insert data
        for i in range(1000):
            conn.execute(
                'INSERT INTO test_table (name, value) VALUES (?, ?)',
                (f'item_{i}', i * 1.5)
            )
        conn.commit()
        
        # Query data
        for i in range(100):
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM test_table WHERE value > ? ORDER BY value DESC LIMIT ?',
                (i * 10, 10)
            )
            results = cursor.fetchall()
        
        conn.close()
        unoptimized_time = time.time() - start_time
        
        print(f"   Unoptimized time: {unoptimized_time:.2f}s")
        
        # Test 2: With optimization
        print("2. Testing with optimization...")
        start_time = time.time()
        
        optimizer = DatabaseOptimizer(
            db_path=temp_db2.name,
            enable_connection_pool=True,
            enable_query_cache=True,
            max_connections=5
        )
        
        # Create table
        optimizer.execute_query('''
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY,
                name TEXT,
                value REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''', use_cache=False)
        
        # Insert data in batch
        insert_queries = []
        for i in range(1000):
            insert_queries.append((
                'INSERT INTO test_table (name, value) VALUES (?, ?)',
                (f'item_{i}', i * 1.5)
            ))
        
        optimizer.execute_batch(insert_queries, use_transaction=True)
        
        # Create index
        optimizer.create_indexes({'test_table': ['value']})
        
        # Query data
        for i in range(100):
            result = optimizer.execute_query(
                'SELECT * FROM test_table WHERE value > ? ORDER BY value DESC LIMIT ?',
                (i * 10, 10)
            )
        
        optimizer.close()
        optimized_time = time.time() - start_time
        
        print(f"   Optimized time: {optimized_time:.2f}s")
        
        # Calculate speedup
        speedup = unoptimized_time / optimized_time
        print(f"   Speedup: {speedup:.2f}x")
        
        # Get final statistics
        final_stats = optimizer.get_database_stats()
        print(f"   Final database size: {final_stats.database_size_mb:.1f} MB")
        print(f"   Total queries: {final_stats.total_queries}")
        print(f"   Average query time: {final_stats.avg_query_time*1000:.2f}ms")
        
    finally:
        # Cleanup
        if Path(temp_db1.name).exists():
            Path(temp_db1.name).unlink()
        if Path(temp_db2.name).exists():
            Path(temp_db2.name).unlink()


def main():
    """Main test function."""
    logger.info("Starting database optimization tests...")
    
    # Test connection pool
    test_connection_pool()
    print()
    
    # Test query cache
    test_query_cache()
    print()
    
    # Test database optimizer
    test_database_optimizer()
    print()
    
    # Test compressed data store
    test_compressed_data_store()
    print()
    
    # Test performance comparison
    test_performance_comparison()
    
    logger.info("Database optimization tests completed!")


if __name__ == "__main__":
    main()
