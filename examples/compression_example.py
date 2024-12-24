from src.compression import create_compression_middleware, CompressionMethod, StreamingCompressor
import json

def main():
    # Create middleware with default configuration
    middleware = create_compression_middleware()

    # Create middleware with custom configuration
    custom_config = {
        'enabled': True,
        'min_size': 2048,  # 2KB minimum
        'level': 7,        # Higher compression
        'excluded_paths': ['/api/stream', '/api/download'],
        'excluded_types': ['image/', 'video/']
    }
    custom_middleware = create_compression_middleware(custom_config)

    # Example response data
    response_data = "Hello, World!" * 1000  # Create some sample data
    headers = {
        'Content-Type': 'text/plain',
        'Accept-Encoding': 'gzip, deflate, br',
        'X-Original-Path': '/api/text'
    }

    # Compress response
    compressed_data, new_headers = middleware.compress_response(
        response_data,
        headers
    )

    print(f"Original size: {len(response_data.encode('utf-8'))} bytes")
    print(f"Compressed size: {len(compressed_data)} bytes")
    print(f"Compression method: {new_headers.get('Content-Encoding', 'none')}")

    # Example of streaming compression
    streamer = StreamingCompressor(
        middleware,
        CompressionMethod.GZIP
    )

    # Simulate streaming data
    chunks = [b"Hello", b" ", b"World", b"!" * 1000]  # Make last chunk bigger
    compressed_chunks = []

    for chunk in chunks:
        compressed_chunk = streamer.compress_chunk(chunk)
        if compressed_chunk:
            compressed_chunks.append(compressed_chunk)

    # Get final chunk
    final_chunk = streamer.flush()
    if final_chunk:
        compressed_chunks.append(final_chunk)

    print("\nStreaming compression example:")
    total_input = sum(len(chunk) for chunk in chunks)
    total_output = sum(len(chunk) for chunk in compressed_chunks)
    print(f"Total input size: {total_input} bytes")
    print(f"Total compressed size: {total_output} bytes")
    print(f"Streaming compression ratio: {round(total_output/total_input, 4)}")

    # Print compression statistics
    print("\nCompression Statistics:")
    stats = middleware.get_compression_stats()
    print(json.dumps(stats, indent=2))

    # Test with different content types and paths
    test_cases = [
        {
            'data': 'Small',  # Too small to compress
            'headers': {
                'Content-Type': 'text/plain',
                'Accept-Encoding': 'gzip',
                'X-Original-Path': '/api/text'
            }
        },
        {
            'data': 'A' * 5000,  # Large enough to compress
            'headers': {
                'Content-Type': 'text/html',
                'Accept-Encoding': 'gzip',
                'X-Original-Path': '/api/html'
            }
        },
        {
            'data': 'B' * 5000,
            'headers': {
                'Content-Type': 'image/jpeg',  # Excluded type
                'Accept-Encoding': 'gzip',
                'X-Original-Path': '/api/image'
            }
        },
        {
            'data': 'C' * 5000,
            'headers': {
                'Content-Type': 'text/plain',
                'Accept-Encoding': 'gzip',
                'X-Original-Path': '/api/stream'  # Excluded path
            }
        }
    ]

    print("\nTesting different scenarios:")
    for case in test_cases:
        _, headers = middleware.compress_response(case['data'], case['headers'])
        print(f"\nPath: {case['headers']['X-Original-Path']}")
        print(f"Content-Type: {case['headers']['Content-Type']}")
        print(f"Compressed: {'Content-Encoding' in headers}")

    # Print final statistics
    print("\nFinal Compression Statistics:")
    stats = middleware.get_compression_stats()
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main() 