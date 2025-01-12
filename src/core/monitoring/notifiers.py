import json
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, Any
from .types import Alert, AlertSeverity

class EmailNotifier:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def send(self, alert: Alert):
        msg = MIMEMultipart()
        msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.rule_name}"
        msg['From'] = self.config['from']
        msg['To'] = ', '.join(self.config['to'])
        
        body = f"""
        Alert: {alert.rule_name}
        Severity: {alert.severity.value}
        Time: {datetime.fromtimestamp(alert.timestamp)}
        
        {alert.message}
        
        Details:
        {json.dumps(alert.details, indent=2) if alert.details else 'None'}
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP(self.config['smtp_host'], self.config['smtp_port']) as server:
            if self.config.get('use_tls'):
                server.starttls()
            if self.config.get('username'):
                server.login(self.config['username'], self.config['password'])
            server.send_message(msg)

class SlackNotifier:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def send(self, alert: Alert):
        color = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ff9400",
            AlertSeverity.CRITICAL: "#ff0000"
        }[alert.severity]
        
        payload = {
            "attachments": [{
                "color": color,
                "title": f"{alert.rule_name} Alert",
                "text": alert.message,
                "fields": [
                    {
                        "title": "Severity",
                        "value": alert.severity.value,
                        "short": True
                    },
                    {
                        "title": "Time",
                        "value": datetime.fromtimestamp(alert.timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                        "short": True
                    }
                ]
            }]
        }
        
        if alert.details:
            payload['attachments'][0]['fields'].append({
                "title": "Details",
                "value": f"```{json.dumps(alert.details, indent=2)}```",
                "short": False
            })
            
        requests.post(self.config['webhook_url'], json=payload)

class WebhookNotifier:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def send(self, alert: Alert):
        payload = {
            "alert_name": alert.rule_name,
            "severity": alert.severity.value,
            "message": alert.message,
            "timestamp": alert.timestamp,
            "details": alert.details
        }
        
        requests.post(
            self.config['url'],
            json=payload,
            headers=self.config.get('headers', {})
        ) 