"""
TransRapport V2.0 - Report Generation Pipeline
Integrates report generation with the main service for automated reporting.
"""

import os
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

from fastapi import BackgroundTasks

from .logging_framework import get_logger, with_error_handling
from reports.renderer import render_markdown


class ReportFormat(Enum):
    """Supported report formats."""
    MARKDOWN = "markdown"
    TXT = "txt"
    JSON = "json"
    PDF = "pdf"
    DOCX = "docx"


class ReportStatus(Enum):
    """Report generation status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ReportRequest:
    """Report generation request."""
    session_id: str
    format: ReportFormat
    template: Optional[str] = None
    include_telemetry: bool = True
    include_markers: bool = True
    include_transcript: bool = True
    include_stats: bool = True
    custom_sections: Optional[List[str]] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class ReportResult:
    """Report generation result."""
    request_id: str
    session_id: str
    status: ReportStatus
    format: ReportFormat
    file_path: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime = None
    completed_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class ReportGenerator:
    """Handles report generation for sessions."""
    
    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self.reports_dir = self.data_root / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger('report_generator')
        self.active_requests: Dict[str, ReportResult] = {}
        
    def _get_session_dir(self, session_id: str) -> Path:
        """Get session directory path."""
        return self.data_root / session_id
    
    def _get_report_dir(self, session_id: str) -> Path:
        """Get reports directory for a session."""
        report_dir = self._get_session_dir(session_id) / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        return report_dir
    
    @with_error_handling(get_logger('report_generator'), 'load_session_data')
    def _load_session_data(self, session_id: str) -> Dict[str, Any]:
        """Load all session data for report generation."""
        session_dir = self._get_session_dir(session_id)
        
        data = {
            'session_id': session_id,
            'segments': [],
            'stats': {},
            'markers': {},
            'telemetry': {},
            'meta': {}
        }
        
        # Load segments
        segments_file = session_dir / "segments.json"
        if segments_file.exists():
            try:
                with segments_file.open('r', encoding='utf-8') as f:
                    data['segments'] = json.load(f)
            except Exception as e:
                self.logger.handle_error(e, {'file': str(segments_file)})
        
        # Load stats
        stats_file = session_dir / "stats.json"
        if stats_file.exists():
            try:
                with stats_file.open('r', encoding='utf-8') as f:
                    data['stats'] = json.load(f)
            except Exception as e:
                self.logger.handle_error(e, {'file': str(stats_file)})
        
        # Load meta
        meta_file = session_dir / "meta.json"
        if meta_file.exists():
            try:
                with meta_file.open('r', encoding='utf-8') as f:
                    data['meta'] = json.load(f)
            except Exception as e:
                self.logger.handle_error(e, {'file': str(meta_file)})
        
        # Load marker events
        markers_file = session_dir / "markers.json"
        if markers_file.exists():
            try:
                with markers_file.open('r', encoding='utf-8') as f:
                    data['markers'] = json.load(f)
            except Exception as e:
                self.logger.handle_error(e, {'file': str(markers_file)})
        
        # Load telemetry
        telemetry_file = session_dir / "telemetry.json"
        if telemetry_file.exists():
            try:
                with telemetry_file.open('r', encoding='utf-8') as f:
                    data['telemetry'] = json.load(f)
            except Exception as e:
                self.logger.handle_error(e, {'file': str(telemetry_file)})
        
        return data
    
    @with_error_handling(get_logger('report_generator'), 'generate_markdown_report')
    def _generate_markdown_report(self, data: Dict[str, Any], request: ReportRequest) -> str:
        """Generate Markdown format report."""
        try:
            return render_markdown(data, request.template)
        except Exception:
            # Fallback to basic markdown generation
            return self._generate_basic_markdown(data, request)
    
    def _generate_basic_markdown(self, data: Dict[str, Any], request: ReportRequest) -> str:
        """Generate basic Markdown report as fallback."""
        session_id = data['session_id']
        lines = [
            f"# TransRapport Session Report",
            f"",
            f"**Session ID:** {session_id}",
            f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
            f"",
        ]
        
        # Session metadata
        meta = data.get('meta', {})
        if meta:
            lines.extend([
                "## Session Information",
                "",
                f"- **Language:** {meta.get('lang', 'auto')}",
                f"- **Created:** {meta.get('created_at', 'Unknown')}",
                "",
            ])
        
        # Statistics
        if request.include_stats and data.get('stats'):
            stats = data['stats']
            lines.extend([
                "## Statistics",
                "",
                f"- **Total Duration:** {stats.get('total_seconds', 0):.1f} seconds",
                f"- **Speakers:** {len(stats.get('speakers', []))}",
                "",
            ])
            
            if stats.get('speakers'):
                lines.append("### Speaker Distribution")
                lines.append("")
                for speaker in stats['speakers']:
                    share = speaker.get('share', 0) * 100
                    lines.append(f"- **{speaker.get('speaker', 'Unknown')}:** {share:.1f}% ({speaker.get('seconds', 0):.1f}s)")
                lines.append("")
        
        # Transcript
        if request.include_transcript and data.get('segments'):
            lines.extend([
                "## Transcript",
                "",
            ])
            
            for segment in data['segments']:
                start = segment.get('start', 0)
                end = segment.get('end', 0)
                text = segment.get('text', '').strip()
                speaker = segment.get('speaker', 'A')
                
                lines.append(f"**[{start:.1f}s - {end:.1f}s] {speaker}:** {text}")
                lines.append("")
        
        # Markers
        if request.include_markers and data.get('markers'):
            markers = data['markers']
            if markers:
                lines.extend([
                    "## Detected Markers",
                    "",
                ])
                
                marker_counts = {}
                for marker in markers:
                    name = marker.get('name', 'Unknown')
                    marker_counts[name] = marker_counts.get(name, 0) + 1
                
                for name, count in sorted(marker_counts.items()):
                    lines.append(f"- **{name}:** {count} instances")
                
                lines.append("")
        
        # Telemetry
        if request.include_telemetry and data.get('telemetry'):
            lines.extend([
                "## Telemetry Data",
                "",
                "```json",
                json.dumps(data['telemetry'], indent=2),
                "```",
                "",
            ])
        
        return "\n".join(lines)
    
    def _generate_txt_report(self, data: Dict[str, Any], request: ReportRequest) -> str:
        """Generate plain text report."""
        # Convert markdown to plain text (simplified)
        markdown = self._generate_markdown_report(data, request)
        # Remove markdown formatting
        txt = markdown.replace("**", "").replace("*", "").replace("#", "").replace("`", "")
        return txt
    
    def _generate_json_report(self, data: Dict[str, Any], request: ReportRequest) -> str:
        """Generate JSON format report."""
        report_data = {
            'session_id': data['session_id'],
            'generated_at': datetime.utcnow().isoformat(),
            'request': asdict(request)
        }
        
        if request.include_transcript:
            report_data['transcript'] = data.get('segments', [])
        
        if request.include_stats:
            report_data['statistics'] = data.get('stats', {})
        
        if request.include_markers:
            report_data['markers'] = data.get('markers', {})
        
        if request.include_telemetry:
            report_data['telemetry'] = data.get('telemetry', {})
        
        return json.dumps(report_data, indent=2, default=str)
    
    @with_error_handling(get_logger('report_generator'), 'generate_report')
    async def generate_report(self, request: ReportRequest) -> ReportResult:
        """Generate a report for a session."""
        request_id = f"{request.session_id}_{request.format.value}_{int(datetime.utcnow().timestamp())}"
        
        result = ReportResult(
            request_id=request_id,
            session_id=request.session_id,
            status=ReportStatus.PROCESSING,
            format=request.format
        )
        
        self.active_requests[request_id] = result
        
        try:
            # Load session data
            data = self._load_session_data(request.session_id)
            
            # Generate report content
            if request.format == ReportFormat.MARKDOWN:
                content = self._generate_markdown_report(data, request)
                file_ext = "md"
            elif request.format == ReportFormat.TXT:
                content = self._generate_txt_report(data, request)
                file_ext = "txt"
            elif request.format == ReportFormat.JSON:
                content = self._generate_json_report(data, request)
                file_ext = "json"
            else:
                raise ValueError(f"Unsupported format: {request.format}")
            
            # Save report file
            report_dir = self._get_report_dir(request.session_id)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{timestamp}.{file_ext}"
            file_path = report_dir / filename
            
            with file_path.open('w', encoding='utf-8') as f:
                f.write(content)
            
            # Update result
            result.status = ReportStatus.COMPLETED
            result.file_path = str(file_path)
            result.completed_at = datetime.utcnow()
            result.metadata = {
                'file_size': file_path.stat().st_size,
                'content_length': len(content)
            }
            
            self.logger.log_session_event('report_generated', request.session_id, {
                'format': request.format.value,
                'file_path': str(file_path)
            })
            
        except Exception as e:
            result.status = ReportStatus.FAILED
            result.error_message = str(e)
            result.completed_at = datetime.utcnow()
            
            self.logger.handle_error(e, {
                'request_id': request_id,
                'session_id': request.session_id,
                'format': request.format.value
            })
        
        return result
    
    def get_report_status(self, request_id: str) -> Optional[ReportResult]:
        """Get status of a report generation request."""
        return self.active_requests.get(request_id)
    
    def list_session_reports(self, session_id: str) -> List[Dict[str, Any]]:
        """List all reports for a session."""
        report_dir = self._get_report_dir(session_id)
        reports = []
        
        if report_dir.exists():
            for file_path in report_dir.glob("report_*.{md,txt,json,pdf,docx}"):
                stat = file_path.stat()
                reports.append({
                    'filename': file_path.name,
                    'path': str(file_path),
                    'size': stat.st_size,
                    'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        return sorted(reports, key=lambda x: x['created'], reverse=True)


# Background task function for FastAPI integration
async def generate_report_background(
    report_generator: ReportGenerator,
    request: ReportRequest
) -> ReportResult:
    """Background task for report generation."""
    return await report_generator.generate_report(request)