from pathlib import Path
import time
from cache_manager import create_cache_manager, CacheLevel, CacheStrategy

def main():
    try:
        # Create a cache manager with default settings
        cache_manager = create_cache_manager()

        # Or customize the configuration
        custom_cache = create_cache_manager(
            level=CacheLevel.HYBRID,  # Uses both memory and disk
            strategy=CacheStrategy.LRU,  # Least Recently Used eviction
            max_memory_mb=2048,  # 2GB memory cache
            max_disk_mb=10240,  # 10GB disk cache
            cache_dir=Path("./cache")  # Custom cache directory
        )

        # Example data to cache
        video_id = "12345"
        video_key = f"video_{video_id}"
        processing_result = {
            'frames': [1, 2, 3, 4, 5],  # Example frames
            'metadata': {'duration': 120, 'fps': 30},
            'timestamp': 1234567890
        }

        print("Storing data in cache...")
        # Store processing results
        custom_cache.set(video_key, processing_result)

        print("Retrieving data from cache...")
        # Retrieve cached results
        cached_result = custom_cache.get(video_key)
        if cached_result:
            print("Found cached result:", cached_result)
        else:
            print("No cached result found")

        # Get cache statistics
        stats = custom_cache.get_stats()
        print("\nCache Statistics:")
        print(f"Memory entries: {stats['memory_entries']}")
        print(f"Memory size: {stats['memory_size']} bytes")
        print(f"Disk entries: {stats['disk_entries']}")
        print(f"Disk size: {stats['disk_size']} bytes")

        print("\nDeleting cache entry...")
        # Delete a cache entry
        custom_cache.delete(video_key)

        print("Clearing all cache entries...")
        # Clear all cache entries
        custom_cache.clear()

    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        # The cache manager's __del__ method will handle cleanup
        print("Cleanup complete")

if __name__ == "__main__":
    main() 