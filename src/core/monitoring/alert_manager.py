import json
import time
import queue
import threading
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from .types import AlertRule, Alert, AlertChannel, AlertSeverity
from .notifiers import EmailNotifier, SlackNotifier, WebhookNotifier

class AlertManager:
    """Manages alert rules and notifications"""
    
    def __init__(
        self,
        log_aggregator,
        config: Dict[str, Any],
        alert_history_path: Optional[Path] = None
    ):
        self.log_aggregator = log_aggregator
        self.config = config
        self.alert_history_path = alert_history_path
        
        self.rules: Dict[str, AlertRule] = {}
        self.alert_queue = queue.Queue()
        self.logger = logging.getLogger(__name__)
        
        # Initialize notifiers
        self.notifiers = {}
        if 'email' in config:
            self.notifiers[AlertChannel.EMAIL] = EmailNotifier(config['email'])
        if 'slack' in config:
            self.notifiers[AlertChannel.SLACK] = SlackNotifier(config['slack'])
        if 'webhook' in config:
            self.notifiers[AlertChannel.WEBHOOK] = WebhookNotifier(config['webhook'])
        
        # Start alert processor thread
        self._start_alert_processor()
        
        # Initialize default rules
        self._init_default_rules()
        
    def _init_default_rules(self):
        """Initialize default monitoring rules"""
        # High error rate alert
        self.add_rule(
            name="high_error_rate",
            description="High rate of errors detected",
            severity=AlertSeverity.CRITICAL,
            condition=self._check_error_rate,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            cooldown_minutes=30
        )
        
        # Processing bottleneck alert
        self.add_rule(
            name="processing_bottleneck",
            description="Video processing performance degraded",
            severity=AlertSeverity.WARNING,
            condition=self._check_processing_performance,
            channels=[AlertChannel.SLACK],
            cooldown_minutes=15
        )
        
        # System resource alert
        self.add_rule(
            name="system_resources",
            description="System resource usage high",
            severity=AlertSeverity.WARNING,
            condition=self._check_system_resources,
            channels=[AlertChannel.EMAIL],
            cooldown_minutes=60
        )
        
    def add_rule(
        self,
        name: str,
        description: str,
        severity: AlertSeverity,
        condition: callable,
        channels: list[AlertChannel],
        cooldown_minutes: int = 60
    ):
        """Add a new alert rule"""
        self.rules[name] = AlertRule(
            name=name,
            description=description,
            severity=severity,
            condition=condition,
            channels=channels,
            cooldown_minutes=cooldown_minutes
        )
        
    def _check_error_rate(self) -> Optional[Alert]:
        """Check for high error rates"""
        summary = self.log_aggregator.get_error_summary(hours=1)
        
        if summary['total_errors'] > self.config.get('max_errors_per_hour', 100):
            return Alert(
                rule_name="high_error_rate",
                severity=AlertSeverity.CRITICAL,
                message=f"High error rate detected: {summary['total_errors']} errors in the last hour",
                timestamp=time.time(),
                details={
                    'error_counts': summary['errors_by_component'],
                    'recent_errors': summary['recent_errors']
                }
            )
        return None
        
    def _check_processing_performance(self) -> Optional[Alert]:
        """Check for processing performance issues"""
        metrics = self.log_aggregator.get_processing_metrics(hours=1)
        
        if metrics['avg_processing_time'] > self.config.get('max_avg_processing_time', 300):
            return Alert(
                rule_name="processing_bottleneck",
                severity=AlertSeverity.WARNING,
                message=f"Processing performance degraded. Average time: {metrics['avg_processing_time']:.1f}s",
                timestamp=time.time(),
                details=metrics
            )
        return None
        
    def _check_system_resources(self) -> Optional[Alert]:
        """Check system resource usage"""
        metrics = self.log_aggregator.get_system_metrics()
        alerts = []
        
        if metrics['cpu_usage'] > self.config.get('max_cpu_percent', 90):
            alerts.append(f"CPU usage at {metrics['cpu_usage']}%")
        
        if metrics['memory_usage'] > self.config.get('max_memory_percent', 85):
            alerts.append(f"Memory usage at {metrics['memory_usage']}%")
            
        if metrics['disk_usage'] > self.config.get('max_disk_percent', 85):
            alerts.append(f"Disk usage at {metrics['disk_usage']}%")
            
        if alerts:
            return Alert(
                rule_name="system_resources",
                severity=AlertSeverity.WARNING,
                message="System resource alert: " + ", ".join(alerts),
                timestamp=time.time(),
                details=metrics
            )
        return None
        
    def _start_alert_processor(self):
        """Start background thread for processing alerts"""
        def process_alerts():
            while True:
                try:
                    alert = self.alert_queue.get()
                    if alert is None:  # Shutdown signal
                        break
                        
                    rule = self.rules[alert.rule_name]
                    
                    # Check cooldown period
                    if rule.last_triggered:
                        elapsed_minutes = (time.time() - rule.last_triggered) / 60
                        if elapsed_minutes < rule.cooldown_minutes:
                            continue
                    
                    # Send notifications
                    self._send_alert_notifications(alert)
                    
                    # Update last triggered time
                    rule.last_triggered = time.time()
                    
                    # Store alert history
                    self._store_alert_history(alert)
                    
                except Exception as e:
                    self.logger.error(f"Error processing alert: {str(e)}")
                    
        self.processor_thread = threading.Thread(target=process_alerts, daemon=True)
        self.processor_thread.start()
        
    def _send_alert_notifications(self, alert: Alert):
        """Send alert notifications through configured channels"""
        rule = self.rules[alert.rule_name]
        
        for channel in rule.channels:
            if channel in self.notifiers:
                try:
                    self.notifiers[channel].send(alert)
                except Exception as e:
                    self.logger.error(f"Error sending {channel.value} notification: {str(e)}")
            
    def _store_alert_history(self, alert: Alert):
        """Store alert in history file"""
        if not self.alert_history_path:
            return
            
        history_entry = {
            "rule_name": alert.rule_name,
            "severity": alert.severity.value,
            "message": alert.message,
            "timestamp": alert.timestamp,
            "details": alert.details
        }
        
        try:
            if self.alert_history_path.exists():
                with open(self.alert_history_path) as f:
                    history = json.load(f)
            else:
                history = []
                
            history.append(history_entry)
            
            # Limit history size
            if len(history) > 1000:
                history = history[-1000:]
                
            with open(self.alert_history_path, 'w') as f:
                json.dump(history, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error storing alert history: {str(e)}")
            
    def check_rules(self):
        """Check all alert rules"""
        for rule in self.rules.values():
            try:
                # Check cooldown period
                if rule.last_triggered:
                    elapsed_minutes = (time.time() - rule.last_triggered) / 60
                    if elapsed_minutes < rule.cooldown_minutes:
                        continue
                
                alert = rule.condition()
                if alert:
                    self.alert_queue.put(alert)
            except Exception as e:
                self.logger.error(f"Error checking rule {rule.name}: {str(e)}")
                
    def start_monitoring(self, interval_seconds: int = 300):
        """Start periodic rule checking"""
        def monitor():
            while True:
                try:
                    self.check_rules()
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(interval_seconds)
                
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
        
    def shutdown(self):
        """Shutdown alert processing"""
        self.alert_queue.put(None)
        self.processor_thread.join() 