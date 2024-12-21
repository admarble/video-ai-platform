from typing import Dict, Any, List, Optional
import smtplib
import json
import requests
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

@dataclass
class AlertThreshold:
    """Threshold configuration for metrics"""
    metric_path: str  # dot notation path to metric (e.g., "cpu.percent")
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    def is_breached(self, metrics: Dict[str, Any]) -> bool:
        """Check if the metric breaches the threshold"""
        value = self._get_metric_value(metrics)
        if value is None:
            return False
            
        if self.min_value is not None and value < self.min_value:
            return True
        if self.max_value is not None and value > self.max_value:
            return True
            
        return False
        
    def _get_metric_value(self, metrics: Dict[str, Any]) -> Optional[float]:
        """Get metric value using dot notation path"""
        try:
            value = metrics
            for key in self.metric_path.split('.'):
                value = value[key]
            return float(value)
        except (KeyError, ValueError, TypeError):
            return None

class AlertManager:
    """Manages system alerts and notifications"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.thresholds: List[AlertThreshold] = self._load_thresholds()
        
    def _load_thresholds(self) -> List[AlertThreshold]:
        """Load alert thresholds from config"""
        thresholds = []
        for t in self.config.get('thresholds', []):
            thresholds.append(AlertThreshold(
                metric_path=t['metric_path'],
                min_value=t.get('min_value'),
                max_value=t.get('max_value')
            ))
        return thresholds
        
    def check_metrics(self, metrics: Dict[str, Any]) -> None:
        """Check metrics against thresholds and send alerts if needed"""
        breached_thresholds = []
        
        for threshold in self.thresholds:
            if threshold.is_breached(metrics):
                breached_thresholds.append(threshold)
                
        if breached_thresholds:
            self._send_alerts(metrics, breached_thresholds)
            
    def _send_alerts(self, metrics: Dict[str, Any], breached_thresholds: List[AlertThreshold]) -> None:
        """Send alerts through configured channels"""
        alert_message = self._format_alert_message(metrics, breached_thresholds)
        
        if 'email' in self.config:
            self._send_email_alert(alert_message)
        if 'slack' in self.config:
            self._send_slack_alert(alert_message)
        if 'webhook' in self.config:
            self._send_webhook_alert(alert_message)
            
    def _format_alert_message(self, metrics: Dict[str, Any], breached_thresholds: List[AlertThreshold]) -> str:
        """Format alert message with breached thresholds and current values"""
        lines = ["System Alert: Threshold Breach Detected"]
        lines.append("-" * 40)
        
        for threshold in breached_thresholds:
            value = threshold._get_metric_value(metrics)
            lines.append(f"Metric: {threshold.metric_path}")
            lines.append(f"Current Value: {value}")
            if threshold.min_value is not None:
                lines.append(f"Minimum Threshold: {threshold.min_value}")
            if threshold.max_value is not None:
                lines.append(f"Maximum Threshold: {threshold.max_value}")
            lines.append("-" * 40)
            
        return "\n".join(lines)
        
    def _send_email_alert(self, message: str) -> None:
        """Send alert via email"""
        email_config = self.config['email']
        
        msg = MIMEMultipart()
        msg['From'] = email_config['from']
        msg['To'] = email_config['to']
        msg['Subject'] = "System Alert"
        
        msg.attach(MIMEText(message, 'plain'))
        
        with smtplib.SMTP(email_config['smtp_host'], email_config['smtp_port']) as server:
            if email_config.get('use_tls', True):
                server.starttls()
            if 'username' in email_config:
                server.login(email_config['username'], email_config['password'])
            server.send_message(msg)
            
    def _send_slack_alert(self, message: str) -> None:
        """Send alert to Slack"""
        slack_config = self.config['slack']
        
        payload = {
            'text': f"```{message}```",
            'channel': slack_config.get('channel', '#alerts')
        }
        
        response = requests.post(
            slack_config['webhook_url'],
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()
        
    def _send_webhook_alert(self, message: str) -> None:
        """Send alert to custom webhook"""
        webhook_config = self.config['webhook']
        
        payload = {
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'severity': 'high'
        }
        
        response = requests.post(
            webhook_config['url'],
            json=payload,
            headers=webhook_config.get('headers', {})
        )
        response.raise_for_status()
        
    def shutdown(self) -> None:
        """Cleanup and close any open connections"""
        # Currently no cleanup needed, but method provided for future use
        pass
        
    def is_healthy(self) -> bool:
        """Check if alert manager is functioning properly"""
        try:
            # Try to load thresholds and check a test metric
            test_metrics = {'test': {'value': 0}}
            test_threshold = AlertThreshold(metric_path='test.value', max_value=1)
            test_threshold.is_breached(test_metrics)
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return False 