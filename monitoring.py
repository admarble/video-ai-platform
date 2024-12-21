from typing import Dict, Any, Optional
import logging
from pathlib import Path
from retry import RetryHandler, RetryError
from retry_configs import (
    METRIC_COLLECTION_RETRY,
    ALERT_NOTIFICATION_RETRY,
    MONITORING_SERVICE_RETRY
)
from metrics_collector import MetricsCollector
from alert_manager import AlertManager

class RetryableMonitoringSystem:
    """Extends the base monitoring system with retry capabilities"""
    
    def __init__(self, base_dir: Path, config: Optional[Dict[str, Any]] = None):
        self.metrics_collector = MetricsCollector(base_dir / "metrics")
        self.alert_manager = AlertManager(config or {})
        self.logger = logging.getLogger(__name__)
        
        # Add retry handlers to critical methods
        self._add_retry_handlers()
        
    def _add_retry_handlers(self):
        """Add retry handlers to critical monitoring methods"""
        # Metric collection retries
        self.collect_system_metrics = RetryHandler(METRIC_COLLECTION_RETRY)(
            self.metrics_collector.collect_system_metrics
        )
        
        # Alert notification retries
        self.alert_manager._send_email_alert = RetryHandler(ALERT_NOTIFICATION_RETRY)(
            self.alert_manager._send_email_alert
        )
        self.alert_manager._send_slack_alert = RetryHandler(ALERT_NOTIFICATION_RETRY)(
            self.alert_manager._send_slack_alert
        )
        self.alert_manager._send_webhook_alert = RetryHandler(ALERT_NOTIFICATION_RETRY)(
            self.alert_manager._send_webhook_alert
        )
        
        # Service monitoring retries
        self.check_service_health = RetryHandler(MONITORING_SERVICE_RETRY)(
            self._check_service_health
        )
        
    def update_metrics(self, video_id: str, metrics: Dict[str, Any]) -> None:
        """Update metrics with retry capability"""
        @RetryHandler(METRIC_COLLECTION_RETRY)
        def update_with_retry():
            self.metrics_collector.update_processing_metrics(video_id, **metrics)
            
        try:
            update_with_retry()
        except RetryError as e:
            self.logger.error(f"Failed to update metrics after retries: {str(e)}")
            
    def process_alerts(self, metrics: Dict[str, Any]) -> None:
        """Process alerts with retry capability"""
        @RetryHandler(ALERT_NOTIFICATION_RETRY)
        def check_and_send_alerts():
            self.alert_manager.check_metrics(metrics)
            
        try:
            check_and_send_alerts()
        except RetryError as e:
            self.logger.error(f"Failed to process alerts after retries: {str(e)}")
            
    def _check_service_health(self) -> Dict[str, bool]:
        """Check health of monitoring services"""
        return {
            'metrics_collector': self.metrics_collector.is_healthy(),
            'alert_manager': self.alert_manager.is_healthy()
        }
        
    def cleanup(self) -> None:
        """Cleanup with retry capability"""
        @RetryHandler(MONITORING_SERVICE_RETRY)
        def cleanup_with_retry():
            self.metrics_collector.cleanup_old_metrics()
            self.alert_manager.shutdown()
            
        try:
            cleanup_with_retry()
        except RetryError as e:
            self.logger.error(f"Failed to cleanup after retries: {str(e)}")

def create_retryable_monitoring(base_dir: Path, config: Dict[str, Any]) -> RetryableMonitoringSystem:
    """Create a monitoring system with retry capabilities"""
    return RetryableMonitoringSystem(base_dir, config) 