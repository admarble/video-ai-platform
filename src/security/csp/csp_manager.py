from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import logging
import asyncio
from collections import defaultdict
import threading
import sqlite3
from pathlib import Path
from enum import Enum
from cors_config import CORSConfig

class SecurityLevel(Enum):
    """Security level configuration"""
    LOW = "low"      # Development mode
    MEDIUM = "medium"  # Production with some relaxed rules
    HIGH = "high"    # Strict production settings

@dataclass
class CSPConfig:
    """Content Security Policy configuration"""
    report_uri: str = ""
    report_only: bool = False
    directives: Dict[str, List[str]] = None

    def __post_init__(self):
        if self.directives is None:
            self.directives = {
                'default-src': ["'self'"],
                'script-src': ["'self'"],
                'style-src': ["'self'"],
                'img-src': ["'self'"],
                'connect-src': ["'self'"],
                'font-src': ["'self'"],
                'object-src': ["'none'"],
                'media-src': ["'self'"],
                'frame-src': ["'none'"]
            }

@dataclass
class SecurityHeaders:
    """Security headers configuration"""
    csp_config: CSPConfig
    frame_options: str = "DENY"
    content_type_options: str = "nosniff"
    xss_protection: str = "1; mode=block"
    referrer_policy: str = "strict-origin-when-cross-origin"

    def get_security_headers(self, include_cors: bool = False) -> Dict[str, str]:
        """Get all configured security headers"""
        headers = {
            'X-Frame-Options': self.frame_options,
            'X-Content-Type-Options': self.content_type_options,
            'X-XSS-Protection': self.xss_protection,
            'Referrer-Policy': self.referrer_policy
        }

        # Add CSP header
        csp_header = '; '.join(
            f"{key} {' '.join(values)}"
            for key, values in self.csp_config.directives.items()
        )
        
        if self.csp_config.report_uri:
            csp_header += f"; report-uri {self.csp_config.report_uri}"
            
        header_name = (
            'Content-Security-Policy-Report-Only'
            if self.csp_config.report_only
            else 'Content-Security-Policy'
        )
        headers[header_name] = csp_header

        return headers

def setup_security(
    cors_config: 'CORSConfig',
    security_level: SecurityLevel,
    config_path: Optional[Path] = None
) -> Tuple[SecurityHeaders, Dict[str, Any]]:
    """Setup security configuration"""
    # Load config if provided
    config: Dict[str, Any] = {}
    if config_path and config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    
    # Create CSP config based on security level
    csp_config = CSPConfig()
    if security_level == SecurityLevel.LOW:
        # Development settings - more relaxed
        csp_config.directives.update({
            'script-src': ["'self'", "'unsafe-inline'", "'unsafe-eval'"],
            'style-src': ["'self'", "'unsafe-inline'"],
            'img-src': ["'self'", "data:", "blob:"],
            'connect-src': ["'self'", "ws:", "wss:"],
        })
    elif security_level == SecurityLevel.MEDIUM:
        # Production settings with some flexibility
        csp_config.directives.update({
            'script-src': ["'self'", "'unsafe-inline'"],
            'style-src': ["'self'", "'unsafe-inline'"],
            'img-src': ["'self'", "data:"],
            'connect-src': ["'self'"],
        })
    # HIGH level uses default strict settings

    # Create security headers
    security_headers = SecurityHeaders(csp_config=csp_config)
    
    # Update config
    config['security_level'] = security_level.value
    
    return security_headers, config

@dataclass
class CSPViolation:
    """Represents a CSP violation report"""
    timestamp: datetime
    document_uri: str
    violated_directive: str
    blocked_uri: str
    source_file: Optional[str] = None
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    sample: Optional[str] = None

class CSPViolationStore:
    """Stores and manages CSP violation reports"""
    
    def __init__(self, db_path: str = "csp_violations.db"):
        self.db_path = db_path
        self.setup_database()
        
    def setup_database(self):
        """Initialize the SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS csp_violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    document_uri TEXT,
                    violated_directive TEXT,
                    blocked_uri TEXT,
                    source_file TEXT,
                    line_number INTEGER,
                    column_number INTEGER,
                    sample TEXT
                )
            """)
            
    def store_violation(self, violation: CSPViolation):
        """Store a CSP violation in the database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO csp_violations (
                    timestamp, document_uri, violated_directive, blocked_uri,
                    source_file, line_number, column_number, sample
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                violation.timestamp, violation.document_uri,
                violation.violated_directive, violation.blocked_uri,
                violation.source_file, violation.line_number,
                violation.column_number, violation.sample
            ))
            
    def get_recent_violations(
        self,
        hours: int = 24
    ) -> List[CSPViolation]:
        """Get violations from the last N hours"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM csp_violations
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            """, (cutoff,))
            
            return [
                CSPViolation(
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    document_uri=row['document_uri'],
                    violated_directive=row['violated_directive'],
                    blocked_uri=row['blocked_uri'],
                    source_file=row['source_file'],
                    line_number=row['line_number'],
                    column_number=row['column_number'],
                    sample=row['sample']
                )
                for row in cursor.fetchall()
            ]
            
    def get_violation_summary(
        self,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get summary statistics of recent violations"""
        violations = self.get_recent_violations(hours)
        
        if not violations:
            return {
                'total_violations': 0,
                'unique_sources': 0,
                'directive_counts': {},
                'blocked_uris': {}
            }
            
        directive_counts = defaultdict(int)
        blocked_uris = defaultdict(int)
        sources = set()
        
        for v in violations:
            directive_counts[v.violated_directive] += 1
            blocked_uris[v.blocked_uri] += 1
            if v.source_file:
                sources.add(v.source_file)
                
        return {
            'total_violations': len(violations),
            'unique_sources': len(sources),
            'directive_counts': dict(directive_counts),
            'blocked_uris': dict(blocked_uris)
        }

class CSPViolationHandler:
    """Handles and analyzes CSP violation reports"""
    
    def __init__(
        self,
        store: Optional[CSPViolationStore] = None,
        alert_threshold: int = 100,
        alert_interval: int = 3600,
        logger: Optional[logging.Logger] = None
    ):
        self.store = store or CSPViolationStore()
        self.alert_threshold = alert_threshold
        self.alert_interval = alert_interval
        self.logger = logger or logging.getLogger(__name__)
        
        # Setup violation tracking
        self.violation_counts = defaultdict(int)
        self.last_alert_time = datetime.utcnow()
        
        # Start monitoring thread
        self._start_monitor()
        
    def _start_monitor(self):
        """Start background monitoring thread"""
        def monitor():
            while True:
                try:
                    self._check_violation_threshold()
                    asyncio.sleep(60)  # Check every minute
                except Exception as e:
                    self.logger.error(f"Error in violation monitor: {str(e)}")
                    
        threading.Thread(target=monitor, daemon=True).start()
        
    def _check_violation_threshold(self):
        """Check if violations exceed threshold"""
        now = datetime.utcnow()
        if (now - self.last_alert_time).total_seconds() >= self.alert_interval:
            # Get violation summary
            summary = self.store.get_violation_summary(
                hours=self.alert_interval / 3600
            )
            
            if summary['total_violations'] >= self.alert_threshold:
                self._send_alert(summary)
                self.last_alert_time = now
                
    def _send_alert(self, summary: Dict[str, Any]):
        """Send alert about excessive CSP violations"""
        message = (
            f"CSP Violation Alert:\n"
            f"Total Violations: {summary['total_violations']}\n"
            f"Unique Sources: {summary['unique_sources']}\n\n"
            f"Most Violated Directives:\n"
        )
        
        # Add top violated directives
        for directive, count in sorted(
            summary['directive_counts'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]:
            message += f"- {directive}: {count}\n"
            
        self.logger.warning(message)
        
    async def handle_violation_report(
        self,
        report_data: Dict[str, Any]
    ) -> None:
        """Handle incoming CSP violation report"""
        try:
            # Extract violation details
            csp_report = report_data.get('csp-report', {})
            
            # Create violation object
            violation = CSPViolation(
                timestamp=datetime.utcnow(),
                document_uri=csp_report.get('document-uri', ''),
                violated_directive=csp_report.get('violated-directive', ''),
                blocked_uri=csp_report.get('blocked-uri', ''),
                source_file=csp_report.get('source-file'),
                line_number=csp_report.get('line-number'),
                column_number=csp_report.get('column-number'),
                sample=csp_report.get('script-sample')
            )
            
            # Store violation
            await asyncio.to_thread(self.store.store_violation, violation)
            
            # Update counts
            self.violation_counts[violation.violated_directive] += 1
            
            # Log violation
            self.logger.info(
                f"CSP Violation: {violation.violated_directive} "
                f"blocked {violation.blocked_uri}"
            )
            
        except Exception as e:
            self.logger.error(f"Error handling CSP violation report: {str(e)}")
            
    def get_violation_trends(
        self,
        days: int = 7
    ) -> Dict[str, Any]:
        """Get violation trends over time"""
        violations = self.store.get_recent_violations(hours=days * 24)
        
        # Group by day
        daily_counts = defaultdict(int)
        directive_trends = defaultdict(lambda: defaultdict(int))
        
        for violation in violations:
            day = violation.timestamp.date()
            daily_counts[day] += 1
            directive_trends[day][violation.violated_directive] += 1
            
        return {
            'daily_totals': dict(daily_counts),
            'directive_trends': {
                day.isoformat(): dict(counts)
                for day, counts in directive_trends.items()
            }
        }
        
    def analyze_violations(
        self,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Analyze recent violations for patterns"""
        violations = self.store.get_recent_violations(hours)
        
        # Group by source file
        source_violations = defaultdict(list)
        for v in violations:
            if v.source_file:
                source_violations[v.source_file].append(v)
                
        # Analyze patterns
        analysis = {
            'frequent_sources': [],
            'common_patterns': [],
            'recommendations': []
        }
        
        # Find frequent violation sources
        for source, source_violations in sorted(
            source_violations.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:5]:
            analysis['frequent_sources'].append({
                'source': source,
                'violations': len(source_violations),
                'directives': list(set(v.violated_directive for v in source_violations))
            })
            
        # Find common patterns
        directive_patterns = defaultdict(lambda: defaultdict(int))
        for v in violations:
            directive_patterns[v.violated_directive][v.blocked_uri] += 1
            
        for directive, patterns in directive_patterns.items():
            top_patterns = sorted(
                patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            if top_patterns:
                analysis['common_patterns'].append({
                    'directive': directive,
                    'patterns': [
                        {'uri': uri, 'count': count}
                        for uri, count in top_patterns
                    ]
                })
                
        # Generate recommendations
        recommendations = set()
        
        for pattern in analysis['common_patterns']:
            directive = pattern['directive']
            if directive.startswith('script-src'):
                if any('inline' in p['uri'] for p in pattern['patterns']):
                    recommendations.add(
                        "Consider using nonces or hashes for inline scripts "
                        "instead of 'unsafe-inline'"
                    )
            elif directive.startswith('style-src'):
                if any('inline' in p['uri'] for p in pattern['patterns']):
                    recommendations.add(
                        "Consider using nonces or hashes for inline styles "
                        "instead of 'unsafe-inline'"
                    )
                    
        analysis['recommendations'] = list(recommendations)
        
        return analysis

async def handle_csp_violation_endpoint(request) -> Dict[str, Any]:
    """Handle CSP violation report endpoint"""
    try:
        # Parse report data
        report_data = await request.json()
        
        # Create handler if needed
        handler = CSPViolationHandler()
        
        # Handle violation
        await handler.handle_violation_report(report_data)
        
        return {'status': 'success'}
        
    except Exception as e:
        logging.error(f"Error handling CSP violation: {str(e)}")
        return {'status': 'error', 'message': str(e)}

class SecureVideoProcessorWithCSP:
    """Video processor with integrated security and CSP reporting"""
    
    def __init__(
        self,
        cors_config: 'CORSConfig',
        security_level: 'SecurityLevel',
        config_path: Optional[Path] = None
    ):
        # Initialize base security
        self.security_headers, self.security_config = setup_security(
            cors_config,
            security_level,
            config_path
        )
        
        # Initialize CSP violation handling
        self.csp_handler = CSPViolationHandler(
            alert_threshold=50,  # Alert after 50 violations
            alert_interval=1800  # Check every 30 minutes
        )
        
        # Update CSP configuration with reporting
        self._setup_csp_reporting()
        
    def _setup_csp_reporting(self):
        """Configure CSP reporting"""
        # Add report-uri to CSP configuration
        self.security_headers.csp_config.report_uri = "/api/csp-report"
        
        # Enable report-only mode in development
        if self.security_config['security_level'] == SecurityLevel.LOW.value:
            self.security_headers.csp_config.report_only = True
            
    async def handle_csp_violation(self, request) -> Dict[str, Any]:
        """Handle incoming CSP violation report"""
        return await self.csp_handler.handle_violation_report(
            await request.json()
        )
        
    async def process_video(
        self,
        video_path: str,
        request_origin: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process video with security headers and CSP monitoring"""
        try:
            # Get response headers
            headers = self.security_headers.get_security_headers(
                include_cors=True
            )
            
            # Process video
            result = await super().process_video(video_path, request_origin)
            
            # Add security headers to response
            result['headers'] = headers
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing video: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'headers': headers
            }
            
    async def get_csp_analytics(self) -> Dict[str, Any]:
        """Get CSP violation analytics"""
        # Get basic violation summary
        summary = self.csp_handler.store.get_violation_summary(hours=24)
        
        # Get violation trends
        trends = self.csp_handler.get_violation_trends(days=7)
        
        # Get detailed analysis
        analysis = self.csp_handler.analyze_violations(hours=24)
        
        return {
            'summary': summary,
            'trends': trends,
            'analysis': analysis
        } 