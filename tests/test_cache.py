"""Tests for TTL LRU cache."""

import time

import pytest

from mlx_serve.core.model_manager import TTLLRUCache


class TestTTLLRUCache:
    """Test TTLLRUCache class."""

    def test_set_and_get(self):
        """Test basic set and get operations."""
        cache = TTLLRUCache(maxsize=3, ttl=60)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_nonexistent(self):
        """Test getting non-existent key."""
        cache = TTLLRUCache(maxsize=3, ttl=60)
        assert cache.get("nonexistent") is None

    def test_contains(self):
        """Test __contains__ method."""
        cache = TTLLRUCache(maxsize=3, ttl=60)
        cache.set("key1", "value1")
        assert "key1" in cache
        assert "key2" not in cache

    def test_len(self):
        """Test __len__ method."""
        cache = TTLLRUCache(maxsize=3, ttl=60)
        assert len(cache) == 0
        cache.set("key1", "value1")
        assert len(cache) == 1
        cache.set("key2", "value2")
        assert len(cache) == 2

    def test_keys(self):
        """Test keys method."""
        cache = TTLLRUCache(maxsize=3, ttl=60)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        keys = cache.keys()
        assert "key1" in keys
        assert "key2" in keys

    def test_remove(self):
        """Test remove method."""
        cache = TTLLRUCache(maxsize=3, ttl=60)
        cache.set("key1", "value1")
        assert cache.remove("key1") is True
        assert cache.get("key1") is None
        assert cache.remove("key1") is False

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = TTLLRUCache(maxsize=2, ttl=60)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        # Access key1 to make it recently used
        cache.get("key1")
        # Add key3, should evict key2 (least recently used)
        cache.set("key3", "value3")

        assert cache.get("key1") == "value1"
        assert cache.get("key3") == "value3"
        # key2 should have been evicted
        assert len(cache) == 2

    def test_ttl_expiration(self):
        """Test TTL expiration."""
        cache = TTLLRUCache(maxsize=3, ttl=1)  # 1 second TTL
        cache.set("key1", "value1")

        # Should be available immediately
        assert cache.get("key1") == "value1"

        # Wait for TTL to expire
        time.sleep(1.5)

        # Should be expired
        assert cache.get("key1") is None

    def test_ttl_refresh_on_access(self):
        """Test that TTL is refreshed on access."""
        cache = TTLLRUCache(maxsize=3, ttl=2)  # 2 second TTL
        cache.set("key1", "value1")

        # Wait 1 second
        time.sleep(1)

        # Access to refresh TTL
        assert cache.get("key1") == "value1"

        # Wait another 1.5 seconds (would be expired without refresh)
        time.sleep(1.5)

        # Should still be available because we refreshed
        assert cache.get("key1") == "value1"

    def test_cleanup_expired(self):
        """Test cleanup of expired entries."""
        cache = TTLLRUCache(maxsize=3, ttl=1)
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # Wait for expiration
        time.sleep(1.5)

        # Cleanup should remove expired entries
        expired = cache.cleanup_expired()
        assert "key1" in expired
        assert "key2" in expired
        assert len(cache) == 0

    def test_thread_safety(self):
        """Test thread safety of cache operations."""
        import threading

        cache = TTLLRUCache(maxsize=100, ttl=60)
        errors = []

        def writer(start, count):
            try:
                for i in range(count):
                    cache.set(f"key-{start + i}", f"value-{start + i}")
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    for key in cache.keys():
                        cache.get(key)
            except Exception as e:
                errors.append(e)

        threads = []
        # Start multiple writers
        for i in range(5):
            t = threading.Thread(target=writer, args=(i * 20, 20))
            threads.append(t)

        # Start multiple readers
        for _ in range(3):
            t = threading.Thread(target=reader)
            threads.append(t)

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
