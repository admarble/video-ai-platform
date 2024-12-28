from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from pathlib import Path
import io
import base64
import json
import zipfile
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
            # Header
            html.Div([
                html.H1("Video Processing Monitor", className="dashboard-title"),
                html.Div([
                    html.Label("Time Range:"),
                    dcc.Dropdown(
                        id='time-range',
                        options=[
                            {'label': 'Last Hour', 'value': '1'},
                            {'label': 'Last 6 Hours', 'value': '6'},
                            {'label': 'Last 24 Hours', 'value': '24'},
                            {'label': 'Last 7 Days', 'value': '168'}
                        ],
                        value='1',
                        className="time-range-selector"
                    )
                ], className="header-controls")
            ], className="dashboard-header"),
            
            # Export Controls Section
            html.Div([
                html.H2("Export Data"),
                html.Div([
                    # Data type selection
                    html.Div([
                        html.Label("Data Types:"),
                        dcc.Checklist(
                            id='export-data-types',
                            options=[
                                {'label': 'System Metrics', 'value': 'system_metrics'},
                                {'label': 'Processing Metrics', 'value': 'processing_metrics'},
                                {'label': 'Alerts', 'value': 'alerts'},
                                {'label': 'Error Logs', 'value': 'error_logs'}
                            ],
                            value=['system_metrics'],
                            className="export-checklist"
                        )
                    ], className="export-control-item"),
                    
                    # Date range selection
                    html.Div([
                        html.Label("Custom Date Range:"),
                        dcc.DatePickerRange(
                            id='export-date-range',
                            start_date=datetime.now() - timedelta(days=7),
                            end_date=datetime.now(),
                            display_format='YYYY-MM-DD'
                        )
                    ], className="export-control-item"),
                    
                    # Format selection
                    html.Div([
                        html.Label("Export Format:"),
                        dcc.Dropdown(
                            id='export-format',
                            options=[
                                {'label': 'CSV', 'value': 'csv'},
                                {'label': 'Excel', 'value': 'excel'},
                                {'label': 'JSON', 'value': 'json'}
                            ],
                            value='csv',
                            className="export-format-selector"
                        )
                    ], className="export-control-item"),
                    
                    # Export button and status
                    html.Div([
                        html.Button(
                            'Export Data',
                            id='export-button',
                            className='export-button'
                        ),
                        dcc.Download(id="export-download"),
                        html.Div(
                            id='export-status',
                            className='export-status'
                        )
                    ], className="export-control-item")
                ], className="export-controls-container")
            ], className="dashboard-section"),
            
            # System Metrics Section
            html.Div([
                html.H2("System Metrics"),
                dcc.Graph(
                    id='system-metrics-graph',
                    config={'displayModeBar': True}
                ),
                dcc.Interval(
                    id='system-metrics-update',
                    interval=self.update_interval
                )
            ], className="dashboard-section"),
            
            # Processing Performance Section
            html.Div([
                html.H2("Processing Performance"),
                html.Div([
                    dcc.Graph(
                        id='processing-metrics-graph',
                        className="performance-chart"
                    ),
                    dcc.Graph(
                        id='error-distribution-graph',
                        className="performance-chart"
                    )
                ], className="performance-charts-container"),
                dcc.Interval(
                    id='processing-metrics-update',
                    interval=self.update_interval
                )
            ], className="dashboard-section"),
            
            # Performance Trends Section
            html.Div([
                html.H2("Performance Trends"),
                dcc.Graph(id='performance-trends-graph'),
                dcc.Interval(
                    id='performance-trends-update',
                    interval=self.update_interval * 2
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
        ], className="dashboard-container")
        
    def setup_callbacks(self):
        """Setup the dashboard callbacks"""
        
        @self.app.callback(
            Output('system-metrics-graph', 'figure'),
            [Input('system-metrics-update', 'n_intervals'),
             Input('time-range', 'value')]
        )
        def update_system_metrics(_, time_range):
            hours = int(time_range)
            
            # Get system metrics from monitoring system
            cpu_metrics = self.monitoring_system.get_metrics_summary(
                MonitoringMetric.CPU_USAGE, hours=hours
            )
            memory_metrics = self.monitoring_system.get_metrics_summary(
                MonitoringMetric.MEMORY_USAGE, hours=hours
            )
            gpu_metrics = self.monitoring_system.get_metrics_summary(
                MonitoringMetric.GPU_USAGE, hours=hours
            )
            
            # Create time series figure
            fig = go.Figure()
            
            # Add traces for each metric with enhanced tooltips
            fig.add_trace(go.Scatter(
                x=cpu_metrics['timestamps'],
                y=cpu_metrics['values'],
                name='CPU Usage %',
                line=dict(color='#1f77b4'),
                hovertemplate="<b>CPU Usage</b><br>" +
                            "Time: %{x}<br>" +
                            "Usage: %{y:.1f}%<extra></extra>"
            ))
            
            fig.add_trace(go.Scatter(
                x=memory_metrics['timestamps'],
                y=memory_metrics['values'],
                name='Memory Usage %',
                line=dict(color='#2ca02c'),
                hovertemplate="<b>Memory Usage</b><br>" +
                            "Time: %{x}<br>" +
                            "Usage: %{y:.1f}%<extra></extra>"
            ))
            
            if gpu_metrics:
                fig.add_trace(go.Scatter(
                    x=gpu_metrics['timestamps'],
                    y=gpu_metrics['values'],
                    name='GPU Usage %',
                    line=dict(color='#ff7f0e'),
                    hovertemplate="<b>GPU Usage</b><br>" +
                            "Time: %{x}<br>" +
                            "Usage: %{y:.1f}%<extra></extra>"
                ))
            
            fig.update_layout(
                title='System Resource Usage',
                xaxis_title='Time',
                yaxis_title='Usage %',
                hovermode='x unified',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                ),
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            return fig
            
        @self.app.callback(
            [Output('processing-metrics-graph', 'figure'),
             Output('error-distribution-graph', 'figure')],
            [Input('processing-metrics-update', 'n_intervals'),
             Input('time-range', 'value')]
        )
        def update_processing_metrics(_, time_range):
            hours = int(time_range)
            
            # Get processing metrics
            processing_metrics = self.monitoring_system.get_metrics_summary(
                MonitoringMetric.PROCESSING_TIME, hours=hours
            )
            error_metrics = self.monitoring_system.get_metrics_summary(
                MonitoringMetric.ERROR_RATE, hours=hours
            )
            
            # Processing time and error rate figure
            fig1 = go.Figure()
            
            fig1.add_trace(go.Bar(
                x=processing_metrics['video_ids'],
                y=processing_metrics['processing_times'],
                name='Processing Time (s)',
                marker_color='#1f77b4',
                hovertemplate="<b>Processing Time</b><br>" +
                            "Video: %{x}<br>" +
                            "Time: %{y:.1f}s<extra></extra>"
            ))
            
            fig1.add_trace(go.Scatter(
                x=processing_metrics['video_ids'],
                y=processing_metrics['error_rates'],
                name='Error Rate %',
                yaxis='y2',
                line=dict(color='#d62728'),
                hovertemplate="<b>Error Rate</b><br>" +
                            "Video: %{x}<br>" +
                            "Rate: %{y:.1f}%<extra></extra>"
            ))
            
            fig1.update_layout(
                title='Processing Performance',
                xaxis_title='Video ID',
                yaxis_title='Processing Time (s)',
                yaxis2=dict(
                    title='Error Rate %',
                    overlaying='y',
                    side='right'
                ),
                barmode='group',
                hovermode='x unified'
            )
            
            # Error distribution figure
            fig2 = go.Figure()
            
            if 'error_types' in error_metrics:
                fig2.add_trace(go.Bar(
                    x=list(error_metrics['error_types'].keys()),
                    y=list(error_metrics['error_types'].values()),
                    marker_color='#d62728',
                    hovertemplate="<b>Error Type</b><br>" +
                                "%{x}<br>" +
                                "Count: %{y}<extra></extra>"
                ))
                
                fig2.update_layout(
                    title='Error Distribution',
                    xaxis_title='Error Type',
                    yaxis_title='Count',
                    hovermode='x unified'
                )
            
            return fig1, fig2
            
        @self.app.callback(
            Output('performance-trends-graph', 'figure'),
            [Input('performance-trends-update', 'n_intervals'),
             Input('time-range', 'value')]
        )
        def update_performance_trends(_, time_range):
            hours = int(time_range)
            
            # Get trend metrics
            processing_metrics = self.monitoring_system.get_metrics_summary(
                MonitoringMetric.PROCESSING_TIME, hours=hours
            )
            
            # Calculate moving averages
            df = pd.DataFrame({
                'timestamp': processing_metrics['timestamps'],
                'processing_time': processing_metrics['processing_times']
            })
            
            df['MA_5'] = df['processing_time'].rolling(window=5).mean()
            df['MA_20'] = df['processing_time'].rolling(window=20).mean()
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['processing_time'],
                name='Processing Time',
                mode='markers',
                marker=dict(size=6),
                hovertemplate="<b>Processing Time</b><br>" +
                            "Time: %{x}<br>" +
                            "Duration: %{y:.1f}s<extra></extra>"
            ))
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['MA_5'],
                name='5-point Moving Average',
                line=dict(width=2),
                hovertemplate="<b>5-point MA</b><br>" +
                            "Time: %{x}<br>" +
                            "Average: %{y:.1f}s<extra></extra>"
            ))
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['MA_20'],
                name='20-point Moving Average',
                line=dict(width=2),
                hovertemplate="<b>20-point MA</b><br>" +
                            "Time: %{x}<br>" +
                            "Average: %{y:.1f}s<extra></extra>"
            ))
            
            fig.update_layout(
                title='Processing Time Trends',
                xaxis_title='Time',
                yaxis_title='Processing Time (s)',
                hovermode='x unified',
                showlegend=True
            )
            
            return fig
            
        @self.app.callback(
            Output('alerts-container', 'children'),
            [Input('alerts-update', 'n_intervals'),
             Input('time-range', 'value')]
        )
        def update_alerts(_, time_range):
            hours = int(time_range)
            alerts = self.monitoring_system.get_recent_alerts(hours=hours)
            
            alert_components = []
            for alert in alerts:
                severity_color = {
                    AlertSeverity.CRITICAL: 'red',
                    AlertSeverity.WARNING: 'orange',
                    AlertSeverity.INFO: 'blue'
                }[alert.severity]
                
                alert_components.append(html.Div([
                    html.Div([
                        html.Span('⚠ ', style={'color': severity_color}),
                        html.Span(
                            f"{alert.severity.value.upper()} Alert",
                            className="alert-severity"
                        )
                    ], className="alert-header"),
                    html.P(alert.message, className="alert-message"),
                    html.Div([
                        html.Small(
                            datetime.fromtimestamp(alert.timestamp).strftime(
                                '%Y-%m-%d %H:%M:%S'
                            )
                        ),
                        html.Small(
                            f"Value: {alert.value:.2f} (Threshold: {alert.threshold:.2f})",
                            className="alert-values"
                        )
                    ], className="alert-details")
                ], className="alert-item"))
                
            return html.Div(alert_components)
            
        @self.app.callback(
            Output("export-download", "data"),
            Output("export-status", "children"),
            Input("export-button", "n_clicks"),
            State("export-data-types", "value"),
            State("export-date-range", "start_date"),
            State("export-date-range", "end_date"),
            State("export-format", "value"),
            prevent_initial_call=True
        )
        def handle_export(n_clicks, data_types, start_date, end_date, export_format):
            if n_clicks is None:
                return None, ""

            try:
                # Get data for each selected type
                export_data = {}
                start_dt = datetime.strptime(start_date.split('T')[0], '%Y-%m-%d')
                end_dt = datetime.strptime(end_date.split('T')[0], '%Y-%m-%d')

                for data_type in data_types:
                    if data_type == 'system_metrics':
                        export_data['system_metrics'] = self._get_system_metrics_data(start_dt, end_dt)
                    elif data_type == 'processing_metrics':
                        export_data['processing_metrics'] = self._get_processing_metrics_data(start_dt, end_dt)
                    elif data_type == 'alerts':
                        export_data['alerts'] = self._get_alerts_data(start_dt, end_dt)
                    elif data_type == 'error_logs':
                        export_data['error_logs'] = self._get_error_logs_data(start_dt, end_dt)

                # Export in selected format
                if export_format == 'csv':
                    return self._export_as_csv(export_data), html.Div("Export successful!", style={'color': 'green'})
                elif export_format == 'excel':
                    return self._export_as_excel(export_data), html.Div("Export successful!", style={'color': 'green'})
                elif export_format == 'json':
                    return self._export_as_json(export_data), html.Div("Export successful!", style={'color': 'green'})

            except Exception as e:
                return None, html.Div(f"Export failed: {str(e)}", style={'color': 'red'})

    def _get_system_metrics_data(self, start_dt: datetime, end_dt: datetime) -> List[Dict[str, Any]]:
        """Get system metrics data for export"""
        metrics = []
        for metric_type in [MonitoringMetric.CPU_USAGE, MonitoringMetric.MEMORY_USAGE, MonitoringMetric.GPU_USAGE]:
            summary = self.monitoring_system.get_metrics_summary(
                metric_type,
                start_time=start_dt,
                end_time=end_dt
            )
            metrics.extend([
                {
                    'timestamp': ts,
                    'metric_type': metric_type.value,
                    'value': val
                }
                for ts, val in zip(summary['timestamps'], summary['values'])
            ])
        return metrics

    def _get_processing_metrics_data(self, start_dt: datetime, end_dt: datetime) -> List[Dict[str, Any]]:
        """Get processing metrics data for export"""
        metrics = self.monitoring_system.get_metrics_summary(
            MonitoringMetric.PROCESSING_TIME,
            start_time=start_dt,
            end_time=end_dt
        )
        return [
            {
                'video_id': vid,
                'processing_time': time,
                'error_rate': err
            }
            for vid, time, err in zip(
                metrics['video_ids'],
                metrics['processing_times'],
                metrics['error_rates']
            )
        ]

    def _get_alerts_data(self, start_dt: datetime, end_dt: datetime) -> List[Dict[str, Any]]:
        """Get alerts data for export"""
        alerts = self.monitoring_system.get_recent_alerts(
            start_time=start_dt,
            end_time=end_dt
        )
        return [
            {
                'timestamp': alert.timestamp,
                'severity': alert.severity.value,
                'message': alert.message,
                'metric': alert.metric.value,
                'value': alert.value,
                'threshold': alert.threshold,
                'details': alert.details
            }
            for alert in alerts
        ]

    def _get_error_logs_data(self, start_dt: datetime, end_dt: datetime) -> List[Dict[str, Any]]:
        """Get error logs data for export"""
        return self.monitoring_system.get_error_logs(
            start_time=start_dt,
            end_time=end_dt
        )

    def _export_as_csv(self, data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Export data as CSV"""
        if len(data) == 1:
            # Single data type - export as single CSV
            data_type = list(data.keys())[0]
            df = pd.DataFrame(data[data_type])
            csv_string = df.to_csv(index=False)
            return dict(
                content=csv_string,
                filename=f"{data_type}_{datetime.now():%Y%m%d_%H%M%S}.csv",
                type='text/csv'
            )
        else:
            # Multiple data types - export as ZIP of CSVs
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                for data_type, items in data.items():
                    df = pd.DataFrame(items)
                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)
                    zf.writestr(f"{data_type}.csv", csv_buffer.getvalue())

            return dict(
                content=base64.b64encode(zip_buffer.getvalue()).decode(),
                filename=f"monitoring_data_{datetime.now():%Y%m%d_%H%M%S}.zip",
                type='application/zip',
                base64=True
            )

    def _export_as_excel(self, data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Export data as Excel"""
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            for data_type, items in data.items():
                df = pd.DataFrame(items)
                df.to_excel(writer, sheet_name=data_type, index=False)

        return dict(
            content=base64.b64encode(output.getvalue()).decode(),
            filename=f"monitoring_data_{datetime.now():%Y%m%d_%H%M%S}.xlsx",
            type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            base64=True
        )

    def _export_as_json(self, data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Export data as JSON"""
        # Convert datetime objects to strings
        def datetime_handler(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        json_string = json.dumps(data, default=datetime_handler, indent=2)
        return dict(
            content=json_string,
            filename=f"monitoring_data_{datetime.now():%Y%m%d_%H%M%S}.json",
            type='application/json'
        )

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
            [Input(f'{chart_id}-update', 'n_intervals'),
             Input('time-range', 'value')]
        )
        def update_custom_chart(_, time_range):
            hours = int(time_range)
            metrics = self.monitoring_system.get_metrics_summary(
                metric, hours=hours, **kwargs
            )
            
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
            
            fig.update_layout(hovermode='x unified')
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
    
    # Add default custom charts
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