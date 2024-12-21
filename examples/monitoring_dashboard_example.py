from pathlib import Path
import sys
import os

# Add the src directory to the Python path
src_dir = str(Path(__file__).parent.parent / 'src')
sys.path.append(src_dir)

from monitoring.monitoring_system import MonitoringSystem
from monitoring.monitoring_dashboard import create_monitoring_dashboard

def main():
    # Initialize the monitoring system
    config = {
        'metrics_dir': 'data/metrics',
        'alert_history_path': 'data/alert_history.json',
        'alert_config': {
            'email': {
                'enabled': False,
                'smtp_server': 'smtp.example.com',
                'smtp_port': 587,
                'username': 'your-email@example.com',
                'password': 'your-password',
                'from_addr': 'your-email@example.com',
                'to_addrs': ['admin@example.com']
            },
            'slack': {
                'enabled': False,
                'webhook_url': 'https://hooks.slack.com/services/your/webhook/url'
            }
        }
    }
    
    # Create data directory if it doesn't exist
    Path('data/metrics').mkdir(parents=True, exist_ok=True)
    
    # Initialize monitoring system
    monitoring_system = MonitoringSystem(None, config)
    
    # Create and run the dashboard
    dashboard = create_monitoring_dashboard(
        monitoring_system,
        port=8050,
        update_interval=5000  # Update every 5 seconds
    )

if __name__ == '__main__':
    main() 