from pathlib import Path
import time
from src.core.cache import (
    create_cache_manager,
    CacheLevel,
    CacheStrategy,
    CacheEvent
)

def on_item_added(key: str, value: any) -> None:
    """Event handler for when items are added to cache"""
    print(f"Cache item added: {key}")

def on_item_removed(key: str, value: any) -> None:
    """Event handler for when items are removed from cache"""
    print(f"Cache item removed: {key}")

def main():
    state_file = Path("cache_state.pkl")
    try:
        # Create cache manager with custom settings
        cache_manager = create_cache_manager(
            level=CacheLevel.HYBRID,
            strategy=CacheStrategy.LRU,
            max_memory_mb=1024,
            max_disk_mb=5120,
            cache_dir=Path("./cache")
        )

        # Register event handlers
        cache_manager.add_event_handler(CacheEvent.ITEM_ADDED, on_item_added)
        cache_manager.add_event_handler(CacheEvent.ITEM_REMOVED, on_item_removed)

        # Example batch data
        batch_data = {
            'video_1': {'frames': [1, 2, 3], 'metadata': {'duration': 30}},
            'video_2': {'frames': [4, 5, 6], 'metadata': {'duration': 45}},
            'video_3': {'frames': [7, 8, 9], 'metadata': {'duration': 60}}
        }

        # Store batch data
        print("\nStoring batch data...")
        cache_manager.set_batch(batch_data)

        # Prefetch some items
        print("\nPrefetching items...")
        cache_manager.prefetch(['video_1', 'video_2'])

        # Get batch of items
        print("\nRetrieving batch data...")
        results = cache_manager.get_batch(['video_1', 'video_2', 'video_3'])
        for key, value in results.items():
            print(f"{key}: {value}")

        # Save cache state
        print("\nSaving cache state...")
        if cache_manager.save_state(state_file):
            print("Cache state saved successfully")
        else:
            print("Failed to save cache state")

        # Clear cache
        print("\nClearing cache...")
        cache_manager.clear()

        # Load cache state
        print("\nLoading cache state...")
        if cache_manager.load_state(state_file):
            print("Cache state loaded successfully")
        else:
            print("Failed to load cache state")

        # Get detailed statistics
        stats = cache_manager.get_detailed_stats()
        print("\nDetailed Cache Statistics:")
        print(f"Hit rate: {stats['hit_rate']:.2%}")
        print(f"Miss rate: {stats['miss_rate']:.2%}")
        print(f"Average get time: {stats['avg_get_time']*1000:.2f}ms")
        print(f"Average set time: {stats['avg_set_time']*1000:.2f}ms")
        print(f"Memory entries: {stats['memory_entries']}")
        print(f"Memory size: {stats['memory_size']} bytes")
        print(f"Disk entries: {stats['disk_entries']}")
        print(f"Disk size: {stats['disk_size']} bytes")
        print(f"Prefetched keys: {stats['prefetched_keys']}")
        print(f"Eviction count: {stats['eviction_count']}")

    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise  # Re-raise the exception for debugging
    finally:
        # Clean up the state file
        if state_file.exists():
            try:
                state_file.unlink()
                print("State file cleaned up")
            except Exception as e:
                print(f"Error cleaning up state file: {str(e)}")
        print("Cleanup complete")

if __name__ == "__main__":
    main() 