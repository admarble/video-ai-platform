from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from pathlib import Path

from .monitoring_system import MonitoringSystem, MonitoringMetric, AlertSeverity

class MonitoringDashboard:
    def __init__(self, monitoring_system: MonitoringSystem, update_interval: int = 5000):
        self.monitoring_system = monitoring_system
        self.app = Dash(__name__)
        self.update_interval = update_interval
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        """Setup the dashboard layout"""
        self.app.layout = html.Div([
            html.H1("Video Processing Monitor", className="dashboard-title"),
            
            # System Metrics Section
            html.Div([
                html.H2("System Metrics"),
                dcc.Graph(id='system-metrics-graph'),
                dcc.Interval(
                    id='system-metrics-update',
                    interval=self.update_interval
                )
            ], className="dashboard-section"),
            
            # Processing Metrics Section
            html.Div([
                html.H2("Processing Performance"),
                dcc.Graph(id='processing-metrics-graph'),
                dcc.Interval(
                    id='processing-metrics-update',
                    interval=self.update_interval
                )
            ], className="dashboard-section"),
            
            # Alerts Section
            html.Div([
                html.H2("Recent Alerts"),
                html.Div(id='alerts-container'),
                dcc.Interval(
                    id='alerts-update',
                    interval=self.update_interval
                )
            ], className="dashboard-section")
        ])
        
    def setup_callbacks(self):
        """Setup the dashboard callbacks"""
        
        @self.app.callback(
            Output('system-metrics-graph', 'figure'),
            Input('system-metrics-update', 'n_intervals')
        )
        def update_system_metrics(_):
            # Get system metrics from monitoring system
            cpu_metrics = self.monitoring_system.get_metrics_summary(
                MonitoringMetric.CPU_USAGE, hours=1
            )
            memory_metrics = self.monitoring_system.get_metrics_summary(
                MonitoringMetric.MEMORY_USAGE, hours=1
            )
            gpu_metrics = self.monitoring_system.get_metrics_summary(
                MonitoringMetric.GPU_USAGE, hours=1
            )
            
            # Create time series figure
            fig = go.Figure()
            
            # Add traces for each metric
            fig.add_trace(go.Scatter(
                x=cpu_metrics['timestamps'],
                y=cpu_metrics['values'],
                name='CPU Usage %',
                line=dict(color='#1f77b4')
            ))
            
            fig.add_trace(go.Scatter(
                x=memory_metrics['timestamps'],
                y=memory_metrics['values'],
                name='Memory Usage %',
                line=dict(color='#2ca02c')
            ))
            
            if gpu_metrics:  # Add GPU metrics if available
                fig.add_trace(go.Scatter(
                    x=gpu_metrics['timestamps'],
                    y=gpu_metrics['values'],
                    name='GPU Usage %',
                    line=dict(color='#ff7f0e')
                ))
            
            fig.update_layout(
                title='System Resource Usage',
                xaxis_title='Time',
                yaxis_title='Usage %',
                hovermode='x unified'
            )
            
            return fig
            
        @self.app.callback(
            Output('processing-metrics-graph', 'figure'),
            Input('processing-metrics-update', 'n_intervals')
        )
        def update_processing_metrics(_):
            # Get processing metrics
            processing_metrics = self.monitoring_system.get_metrics_summary(
                MonitoringMetric.PROCESSING_TIME, hours=24
            )
            
            # Create figure with secondary y-axis
            fig = go.Figure()
            
            # Add processing time bars
            fig.add_trace(go.Bar(
                x=processing_metrics['video_ids'],
                y=processing_metrics['processing_times'],
                name='Processing Time (s)',
                marker_color='#1f77b4'
            ))
            
            # Add error rate line
            fig.add_trace(go.Scatter(
                x=processing_metrics['video_ids'],
                y=processing_metrics['error_rates'],
                name='Error Rate %',
                yaxis='y2',
                line=dict(color='#d62728')
            ))
            
            fig.update_layout(
                title='Processing Performance',
                xaxis_title='Video ID',
                yaxis_title='Processing Time (s)',
                yaxis2=dict(
                    title='Error Rate %',
                    overlaying='y',
                    side='right'
                ),
                barmode='group'
            )
            
            return fig
            
        @self.app.callback(
            Output('alerts-container', 'children'),
            Input('alerts-update', 'n_intervals')
        )
        def update_alerts(_):
            # Get recent alerts
            alerts = self.monitoring_system.get_recent_alerts(hours=24)
            
            alert_components = []
            for alert in alerts:
                severity_color = {
                    AlertSeverity.CRITICAL: 'red',
                    AlertSeverity.WARNING: 'orange',
                    AlertSeverity.INFO: 'blue'
                }[alert.severity]
                
                alert_components.append(html.Div([
                    html.H4([
                        html.Span('⚠ ', style={'color': severity_color}),
                        f"{alert.severity.value.upper()} Alert"
                    ]),
                    html.P(alert.message),
                    html.Small(
                        datetime.fromtimestamp(alert.timestamp).strftime(
                            '%Y-%m-%d %H:%M:%S'
                        )
                    )
                ], className='alert-item'))
                
            return html.Div(alert_components)
    
    def add_custom_chart(
        self,
        chart_id: str,
        title: str,
        metric: MonitoringMetric,
        chart_type: str = 'line',
        **kwargs
    ):
        """Add a custom chart to the dashboard"""
        new_section = html.Div([
            html.H2(title),
            dcc.Graph(id=chart_id),
            dcc.Interval(
                id=f'{chart_id}-update',
                interval=self.update_interval
            )
        ], className="dashboard-section")
        
        # Add to layout
        self.app.layout.children.append(new_section)
        
        # Add callback for the new chart
        @self.app.callback(
            Output(chart_id, 'figure'),
            Input(f'{chart_id}-update', 'n_intervals')
        )
        def update_custom_chart(_):
            metrics = self.monitoring_system.get_metrics_summary(metric, **kwargs)
            
            if chart_type == 'line':
                fig = px.line(
                    metrics['data'],
                    x='timestamp',
                    y='value',
                    title=title
                )
            elif chart_type == 'bar':
                fig = px.bar(
                    metrics['data'],
                    x='category',
                    y='value',
                    title=title
                )
            
            return fig
    
    def run(self, debug: bool = False, port: int = 8050):
        """Run the dashboard server"""
        self.app.run_server(debug=debug, port=port)

def create_monitoring_dashboard(
    monitoring_system: MonitoringSystem,
    port: int = 8050,
    update_interval: int = 5000
) -> MonitoringDashboard:
    """Create and run a monitoring dashboard"""
    dashboard = MonitoringDashboard(
        monitoring_system,
        update_interval=update_interval
    )
    
    # Add any custom charts
    dashboard.add_custom_chart(
        'queue-size-chart',
        'Processing Queue Size',
        MonitoringMetric.QUEUE_SIZE,
        chart_type='line',
        hours=6
    )
    
    # Run the dashboard
    dashboard.run(port=port)
    
    return dashboard 