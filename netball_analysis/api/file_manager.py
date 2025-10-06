"""
File Manager for Netball Analysis API
Handles file operations, downloads, and storage management
"""

import os
import sys
import shutil
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import hashlib
import zipfile
import tempfile

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logging_utils import get_logger

logger = get_logger(__name__)

class FileManager:
    """Manages files for the netball analysis API."""
    
    def __init__(self, base_dir: str = "results"):
        """Initialize file manager."""
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.directories = {
            "results": self.base_dir,
            "uploads": self.base_dir.parent / "uploads",
            "temp": self.base_dir.parent / "temp",
            "archives": self.base_dir / "archives"
        }
        
        for dir_path in self.directories.values():
            dir_path.mkdir(exist_ok=True)
        
        # File type mappings
        self.file_types = {
            "video": ["mp4", "avi", "mov", "mkv", "webm"],
            "csv": ["csv"],
            "json": ["json"],
            "log": ["log", "txt"],
            "image": ["png", "jpg", "jpeg", "gif", "bmp"],
            "archive": ["zip", "tar", "gz"]
        }
        
        # MIME type mappings
        self.mime_types = {
            "video": "video/mp4",
            "csv": "text/csv",
            "json": "application/json",
            "log": "text/plain",
            "image": "image/png",
            "archive": "application/zip"
        }
        
        logger.info(f"FileManager initialized with base directory: {self.base_dir}")
    
    def get_job_directory(self, job_id: str) -> Path:
        """Get the directory for a specific job."""
        return self.directories["results"] / job_id
    
    def create_job_directory(self, job_id: str) -> Path:
        """Create a directory for a specific job."""
        job_dir = self.get_job_directory(job_id)
        job_dir.mkdir(exist_ok=True)
        return job_dir
    
    def get_file_path(self, job_id: str, file_type: str, filename: Optional[str] = None) -> Optional[Path]:
        """Get the path to a specific file for a job."""
        job_dir = self.get_job_directory(job_id)
        
        if not job_dir.exists():
            return None
        
        if filename:
            return job_dir / filename
        
        # Look for files with common extensions for the type
        extensions = self.file_types.get(file_type, [])
        
        for ext in extensions:
            pattern = f"*.{ext}"
            files = list(job_dir.glob(pattern))
            if files:
                return files[0]  # Return first match
        
        return None
    
    def list_job_files(self, job_id: str) -> List[Dict[str, Any]]:
        """List all files for a job."""
        job_dir = self.get_job_directory(job_id)
        
        if not job_dir.exists():
            return []
        
        files = []
        for file_path in job_dir.iterdir():
            if file_path.is_file():
                stat = file_path.stat()
                
                # Determine file type
                file_type = self._determine_file_type(file_path)
                
                files.append({
                    "filename": file_path.name,
                    "file_type": file_type,
                    "size_bytes": stat.st_size,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "download_url": f"/jobs/{job_id}/download/{file_type}"
                })
        
        return sorted(files, key=lambda x: x["created_at"], reverse=True)
    
    def get_file_info(self, job_id: str, file_type: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific file."""
        file_path = self.get_file_path(job_id, file_type)
        
        if not file_path or not file_path.exists():
            return None
        
        stat = file_path.stat()
        
        return {
            "filename": file_path.name,
            "file_type": file_type,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "download_url": f"/jobs/{job_id}/download/{file_type}",
            "mime_type": self.mime_types.get(file_type, "application/octet-stream")
        }
    
    def get_file_content(self, job_id: str, file_type: str) -> Optional[Tuple[Path, str]]:
        """Get file content and MIME type."""
        file_path = self.get_file_path(job_id, file_type)
        
        if not file_path or not file_path.exists():
            return None
        
        mime_type = self.mime_types.get(file_type, "application/octet-stream")
        
        return file_path, mime_type
    
    def create_archive(self, job_id: str, file_types: List[str] = None) -> Optional[Path]:
        """Create a ZIP archive of job files."""
        job_dir = self.get_job_directory(job_id)
        
        if not job_dir.exists():
            return None
        
        # Create archive filename
        archive_name = f"{job_id}_analysis_results.zip"
        archive_path = self.directories["archives"] / archive_name
        
        try:
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add files to archive
                for file_path in job_dir.iterdir():
                    if file_path.is_file():
                        # Check if file type is requested
                        if file_types:
                            file_type = self._determine_file_type(file_path)
                            if file_type not in file_types:
                                continue
                        
                        # Add file to archive
                        zipf.write(file_path, file_path.name)
                
                # Add metadata
                metadata = {
                    "job_id": job_id,
                    "created_at": datetime.now().isoformat(),
                    "file_count": len([f for f in job_dir.iterdir() if f.is_file()])
                }
                
                zipf.writestr("metadata.json", str(metadata))
            
            logger.info(f"Created archive for job {job_id}: {archive_path}")
            return archive_path
            
        except Exception as e:
            logger.error(f"Failed to create archive for job {job_id}: {e}")
            return None
    
    def cleanup_job_files(self, job_id: str, keep_archive: bool = True) -> bool:
        """Clean up files for a job."""
        job_dir = self.get_job_directory(job_id)
        
        if not job_dir.exists():
            return False
        
        try:
            # Create archive if requested
            if keep_archive:
                self.create_archive(job_id)
            
            # Remove job directory
            shutil.rmtree(job_dir)
            
            logger.info(f"Cleaned up files for job {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup files for job {job_id}: {e}")
            return False
    
    def cleanup_old_files(self, days: int = 7) -> int:
        """Clean up old files."""
        cutoff_time = datetime.now() - timedelta(days=days)
        cleaned_count = 0
        
        # Clean up old job directories
        for job_dir in self.directories["results"].iterdir():
            if job_dir.is_dir():
                try:
                    # Check if directory is old
                    if datetime.fromtimestamp(job_dir.stat().st_mtime) < cutoff_time:
                        self.cleanup_job_files(job_dir.name, keep_archive=True)
                        cleaned_count += 1
                except Exception as e:
                    logger.warning(f"Failed to cleanup old job directory {job_dir}: {e}")
        
        # Clean up old archives
        for archive_file in self.directories["archives"].iterdir():
            if archive_file.is_file():
                try:
                    if datetime.fromtimestamp(archive_file.stat().st_mtime) < cutoff_time:
                        archive_file.unlink()
                        cleaned_count += 1
                except Exception as e:
                    logger.warning(f"Failed to cleanup old archive {archive_file}: {e}")
        
        # Clean up old temp files
        for temp_file in self.directories["temp"].iterdir():
            if temp_file.is_file():
                try:
                    if datetime.fromtimestamp(temp_file.stat().st_mtime) < cutoff_time:
                        temp_file.unlink()
                        cleaned_count += 1
                except Exception as e:
                    logger.warning(f"Failed to cleanup old temp file {temp_file}: {e}")
        
        logger.info(f"Cleaned up {cleaned_count} old files")
        return cleaned_count
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {
            "total_size_mb": 0,
            "file_count": 0,
            "job_count": 0,
            "archive_count": 0,
            "by_type": {}
        }
        
        # Count files in results directory
        for job_dir in self.directories["results"].iterdir():
            if job_dir.is_dir():
                stats["job_count"] += 1
                
                for file_path in job_dir.iterdir():
                    if file_path.is_file():
                        stats["file_count"] += 1
                        size_mb = file_path.stat().st_size / (1024 * 1024)
                        stats["total_size_mb"] += size_mb
                        
                        # Count by type
                        file_type = self._determine_file_type(file_path)
                        if file_type not in stats["by_type"]:
                            stats["by_type"][file_type] = {"count": 0, "size_mb": 0}
                        stats["by_type"][file_type]["count"] += 1
                        stats["by_type"][file_type]["size_mb"] += size_mb
        
        # Count archives
        for archive_file in self.directories["archives"].iterdir():
            if archive_file.is_file():
                stats["archive_count"] += 1
                size_mb = archive_file.stat().st_size / (1024 * 1024)
                stats["total_size_mb"] += size_mb
        
        # Round sizes
        stats["total_size_mb"] = round(stats["total_size_mb"], 2)
        for file_type in stats["by_type"]:
            stats["by_type"][file_type]["size_mb"] = round(stats["by_type"][file_type]["size_mb"], 2)
        
        return stats
    
    def validate_file_upload(self, file_path: Path, max_size_mb: int = 500) -> Tuple[bool, str]:
        """Validate an uploaded file."""
        if not file_path.exists():
            return False, "File does not exist"
        
        # Check file size
        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            return False, f"File too large: {size_mb:.1f}MB (max: {max_size_mb}MB)"
        
        # Check file type
        file_type = self._determine_file_type(file_path)
        if file_type not in ["video", "image"]:
            return False, f"Unsupported file type: {file_type}"
        
        return True, "File is valid"
    
    def _determine_file_type(self, file_path: Path) -> str:
        """Determine the type of a file based on its extension."""
        extension = file_path.suffix.lower().lstrip('.')
        
        for file_type, extensions in self.file_types.items():
            if extension in extensions:
                return file_type
        
        return "unknown"
    
    def get_file_hash(self, file_path: Path) -> str:
        """Get the SHA256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate hash for {file_path}: {e}")
            return ""
    
    def create_temp_file(self, content: bytes, suffix: str = ".tmp") -> Path:
        """Create a temporary file with content."""
        temp_file = tempfile.NamedTemporaryFile(
            dir=self.directories["temp"],
            suffix=suffix,
            delete=False
        )
        
        try:
            temp_file.write(content)
            temp_file.close()
            return Path(temp_file.name)
        except Exception as e:
            logger.error(f"Failed to create temp file: {e}")
            return None

# Global file manager instance
_file_manager: Optional[FileManager] = None

def get_file_manager() -> FileManager:
    """Get the global file manager instance."""
    global _file_manager
    if _file_manager is None:
        _file_manager = FileManager()
    return _file_manager

