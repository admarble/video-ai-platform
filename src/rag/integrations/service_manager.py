from typing import Dict, Any, Optional
import asyncio
from pathlib import Path

class ServiceManagerIntegration:
    def __init__(
        self,
        service_manager: 'ServiceManager',
        rag_service: 'RAGService'
    ):
        self.service_manager = service_manager
        self.rag_service = rag_service
        self._initialized = False
        
    async def initialize_rag_services(self):
        """Initialize RAG-related services"""
        if self._initialized:
            return
            
        try:
            # Initialize vector store
            vector_store = await self.service_manager.initialize_service(
                'vector_store',
                config=self.rag_service.config.vector_store
            )
            
            # Initialize LLM
            llm = await self.service_manager.initialize_service(
                'mistral_llm',
                config=self.rag_service.config.llm
            )
            
            # Initialize clip generator
            clip_generator = await self.service_manager.initialize_service(
                'clip_generator',
                config=self.rag_service.config.clip_generator
            )
            
            # Set up RAG service dependencies
            await self.rag_service.setup_dependencies(
                vector_store=vector_store,
                llm=llm,
                clip_generator=clip_generator
            )
            
            # Register health checks
            self._register_health_checks()
            
            # Register metrics
            self._register_metrics()
            
            self._initialized = True
            
        except Exception as e:
            await self.service_manager.log_error(
                "Failed to initialize RAG services",
                error=str(e)
            )
            raise
            
    def _register_health_checks(self):
        """Register health checks for RAG services"""
        self.service_manager.add_health_check(
            'vector_store',
            self.rag_service.check_vector_store_health
        )
        self.service_manager.add_health_check(
            'llm',
            self.rag_service.check_llm_health
        )
        self.service_manager.add_health_check(
            'clip_generator',
            self.rag_service.check_clip_generator_health
        )
        
    def _register_metrics(self):
        """Register metrics for RAG services"""
        self.service_manager.register_metric(
            'rag_search_latency',
            'histogram',
            'RAG search latency in seconds'
        )
        self.service_manager.register_metric(
            'rag_index_count',
            'counter',
            'Number of scenes indexed in RAG'
        )
        self.service_manager.register_metric(
            'clip_generation_duration',
            'histogram',
            'Clip generation duration in seconds'
        )
        
    async def cleanup(self):
        """Cleanup RAG services"""
        if not self._initialized:
            return
            
        try:
            # Cleanup vector store
            await self.service_manager.cleanup_service('vector_store')
            
            # Cleanup LLM
            await self.service_manager.cleanup_service('mistral_llm')
            
            # Cleanup clip generator
            await self.service_manager.cleanup_service('clip_generator')
            
            self._initialized = False
            
        except Exception as e:
            await self.service_manager.log_error(
                "Failed to cleanup RAG services",
                error=str(e)
            )
