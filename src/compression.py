from typing import Dict, Any, Optional, List, Union
import gzip
import zlib
import brotli
from enum import Enum
import logging
from dataclasses import dataclass
import json
from pathlib import Path

class CompressionMethod(Enum):
    """Supported compression methods"""
    GZIP = "gzip"
    DEFLATE = "deflate"
    BROTLI = "br"
    NONE = "none"

@dataclass
class CompressionConfig:
    """Configuration for compression middleware"""
    enabled: bool = True
    preferred_methods: List[CompressionMethod] = None
    min_size: int = 1024  # Minimum size in bytes to compress
    level: int = 6  # Compression level (1-9)
    excluded_paths: List[str] = None
    excluded_types: List[str] = None
    buffer_size: int = 8192  # Buffer size for streaming compression

    def __post_init__(self):
        if self.preferred_methods is None:
            self.preferred_methods = [
                CompressionMethod.BROTLI,
                CompressionMethod.GZIP,
                CompressionMethod.DEFLATE
            ]
        if self.excluded_paths is None:
            self.excluded_paths = []
        if self.excluded_types is None:
            self.excluded_types = ['image/', 'video/', 'audio/']

class CompressionMiddleware:
    """Middleware for handling HTTP compression"""
    
    def __init__(self, config: Optional[CompressionConfig] = None):
        self.config = config or CompressionConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize compressors
        self.compressors = {
            CompressionMethod.GZIP: self._create_gzip_compressor,
            CompressionMethod.DEFLATE: self._create_deflate_compressor,
            CompressionMethod.BROTLI: self._create_brotli_compressor
        }
        
        # Initialize metrics
        self.metrics = {
            'total_bytes_before': 0,
            'total_bytes_after': 0,
            'requests_compressed': 0,
            'requests_skipped': 0,
            'compression_methods': {method.value: 0 for method in CompressionMethod}
        }
        
    def _create_gzip_compressor(self, level: int):
        return lambda data: gzip.compress(data, compresslevel=level)
        
    def _create_deflate_compressor(self, level: int):
        return lambda data: zlib.compress(data, level=level)
        
    def _create_brotli_compressor(self, level: int):
        quality = int((level / 9) * 11)  # Convert to brotli quality (0-11)
        return lambda data: brotli.compress(data, quality=quality)
        
    def _should_compress(
        self,
        path: str,
        content_type: str,
        content_length: int
    ) -> bool:
        """Check if content should be compressed"""
        if not self.config.enabled:
            return False
            
        # Check content length
        if content_length < self.config.min_size:
            return False
            
        # Check excluded paths
        if any(path.startswith(excluded) for excluded in self.config.excluded_paths):
            return False
            
        # Check excluded content types
        if any(content_type.startswith(excluded) for excluded in self.config.excluded_types):
            return False
            
        return True
        
    def _get_accepted_encoding(self, headers: Dict[str, str]) -> List[CompressionMethod]:
        """Get list of accepted encoding methods from headers"""
        accept_encoding = headers.get('Accept-Encoding', '')
        accepted = []
        
        # Parse accept-encoding header
        for encoding in accept_encoding.split(','):
            encoding = encoding.strip().lower()
            if encoding == 'gzip':
                accepted.append(CompressionMethod.GZIP)
            elif encoding == 'deflate':
                accepted.append(CompressionMethod.DEFLATE)
            elif encoding == 'br':
                accepted.append(CompressionMethod.BROTLI)
                
        return accepted
        
    def _select_compression_method(
        self,
        accepted_encodings: List[CompressionMethod]
    ) -> CompressionMethod:
        """Select best compression method based on preferences and support"""
        # Find first preferred method that is accepted
        for method in self.config.preferred_methods:
            if method in accepted_encodings:
                return method
        return CompressionMethod.NONE
        
    def compress_response(
        self,
        data: Union[bytes, str],
        headers: Dict[str, str]
    ) -> tuple[bytes, Dict[str, str]]:
        """Compress response data if appropriate"""
        try:
            # Convert string to bytes if needed
            if isinstance(data, str):
                data = data.encode('utf-8')
                
            content_type = headers.get('Content-Type', '')
            content_length = len(data)
            path = headers.get('X-Original-Path', '')
            
            # Update metrics for original size
            self.metrics['total_bytes_before'] += content_length
            
            # Check if we should compress
            if not self._should_compress(path, content_type, content_length):
                self.metrics['requests_skipped'] += 1
                return data, headers
                
            # Get accepted encodings
            accepted_encodings = self._get_accepted_encoding(headers)
            if not accepted_encodings:
                self.metrics['requests_skipped'] += 1
                return data, headers
                
            # Select compression method
            method = self._select_compression_method(accepted_encodings)
            if method == CompressionMethod.NONE:
                self.metrics['requests_skipped'] += 1
                return data, headers
                
            # Compress data
            compressor = self.compressors[method](self.config.level)
            compressed_data = compressor(data)
            
            # Update metrics
            self.metrics['requests_compressed'] += 1
            self.metrics['total_bytes_after'] += len(compressed_data)
            self.metrics['compression_methods'][method.value] += 1
            
            # Update headers
            new_headers = headers.copy()
            new_headers['Content-Encoding'] = method.value
            new_headers['Content-Length'] = str(len(compressed_data))
            new_headers.pop('Accept-Ranges', None)  # Remove accept-ranges for compressed content
            
            return compressed_data, new_headers
            
        except Exception as e:
            self.logger.error(f"Compression error: {str(e)}")
            self.metrics['requests_skipped'] += 1
            return data, headers
            
    def compress_stream(self, data: bytes, method: CompressionMethod) -> bytes:
        """Compress chunk of streaming data"""
        try:
            if method == CompressionMethod.NONE:
                return data
                
            compressor = self.compressors[method](self.config.level)
            return compressor(data)
            
        except Exception as e:
            self.logger.error(f"Stream compression error: {str(e)}")
            return data
            
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics and metrics"""
        stats = {
            'enabled': self.config.enabled,
            'preferred_methods': [m.value for m in self.config.preferred_methods],
            'min_size': self.config.min_size,
            'level': self.config.level,
            'excluded_paths': self.config.excluded_paths,
            'excluded_types': self.config.excluded_types,
            'metrics': {
                **self.metrics,
                'compression_ratio': (
                    round(self.metrics['total_bytes_after'] / self.metrics['total_bytes_before'], 4)
                    if self.metrics['total_bytes_before'] > 0 else 0
                ),
                'bytes_saved': self.metrics['total_bytes_before'] - self.metrics['total_bytes_after'],
                'compression_rate': (
                    round(self.metrics['requests_compressed'] / 
                        (self.metrics['requests_compressed'] + self.metrics['requests_skipped']), 4)
                    if (self.metrics['requests_compressed'] + self.metrics['requests_skipped']) > 0
                    else 0
                )
            }
        }
        return stats

class StreamingCompressor:
    """Handles streaming compression for large responses"""
    
    def __init__(
        self,
        middleware: CompressionMiddleware,
        method: CompressionMethod,
        buffer_size: Optional[int] = None
    ):
        self.middleware = middleware
        self.method = method
        self.buffer_size = buffer_size or middleware.config.buffer_size
        self.buffer = bytearray()
        
        # Initialize the appropriate compressor
        if method == CompressionMethod.GZIP:
            self.compressor = gzip.GzipFile(mode='wb', fileobj=bytearray())
        elif method == CompressionMethod.DEFLATE:
            self.compressor = zlib.compressobj(level=middleware.config.level)
        elif method == CompressionMethod.BROTLI:
            quality = int((middleware.config.level / 9) * 11)
            self.compressor = brotli.Compressor(quality=quality)
        else:
            self.compressor = None
        
    def compress_chunk(self, chunk: bytes) -> bytes:
        """Compress a chunk of data"""
        if self.method == CompressionMethod.NONE or self.compressor is None:
            return chunk
            
        self.buffer.extend(chunk)
        
        if len(self.buffer) >= self.buffer_size:
            # Compress buffered data
            to_compress = bytes(self.buffer)
            self.buffer.clear()
            
            if self.method == CompressionMethod.GZIP:
                self.compressor.write(to_compress)
                return self.compressor._buffer.getvalue()
            elif self.method == CompressionMethod.DEFLATE:
                return self.compressor.compress(to_compress)
            elif self.method == CompressionMethod.BROTLI:
                return self.compressor.process(to_compress)
            
        return b''
        
    def flush(self) -> bytes:
        """Flush any remaining data in buffer and finalize compression"""
        if self.method == CompressionMethod.NONE or self.compressor is None:
            data = bytes(self.buffer)
            self.buffer.clear()
            return data
            
        result = bytearray()
        
        # Compress any remaining buffered data
        if self.buffer:
            if self.method == CompressionMethod.GZIP:
                self.compressor.write(bytes(self.buffer))
            elif self.method == CompressionMethod.DEFLATE:
                result.extend(self.compressor.compress(bytes(self.buffer)))
            elif self.method == CompressionMethod.BROTLI:
                result.extend(self.compressor.process(bytes(self.buffer)))
            self.buffer.clear()
        
        # Finalize compression
        if self.method == CompressionMethod.GZIP:
            self.compressor.close()
            result.extend(self.compressor._buffer.getvalue())
        elif self.method == CompressionMethod.DEFLATE:
            result.extend(self.compressor.flush())
        elif self.method == CompressionMethod.BROTLI:
            result.extend(self.compressor.finish())
            
        return bytes(result)

def create_compression_middleware(
    config: Optional[Dict[str, Any]] = None
) -> CompressionMiddleware:
    """Create compression middleware instance"""
    compression_config = CompressionConfig(
        **config if config else {}
    )
    return CompressionMiddleware(compression_config) 