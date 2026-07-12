"""
Fast Google Drive transfer manager.

Two backends:
1. rclone CLI (preferred) — fastest for bulk transfers
2. Google Drive API (fallback) — works without rclone

Auto-detects rclone on startup. If not found, uses Google Drive API.
"""

from __future__ import annotations

import os
import shutil
import mimetypes
import subprocess
import concurrent.futures
from pathlib import Path
from typing import Optional

from loguru import logger


class FastDriveManager:
    """
    Fast file transfer to/from Google Drive.
    
    Uses rclone CLI if available (much faster for parallel transfers).
    Falls back to Google Drive API if rclone is not installed.
    """

    def __init__(self):
        self._rclone_path = self._find_rclone()
        self._use_rclone = self._rclone_path is not None
        
        if self._use_rclone:
            logger.info(f"🚀 Using rclone for fast transfers: {self._rclone_path}")
        else:
            logger.info("⚠️ rclone not found, using Google Drive API (slower)")
            self._api_manager = self._init_api_manager()

    def _find_rclone(self) -> str | None:
        """Find rclone binary."""
        path = shutil.which("rclone")
        if path:
            return path
        for candidate in ["/usr/bin/rclone", "/usr/local/bin/rclone", "/snap/bin/rclone"]:
            if os.path.isfile(candidate):
                return candidate
        return None

    def _init_api_manager(self):
        """Initialize Google Drive API fallback."""
        try:
            from server.config import settings
            from googleapiclient.discovery import build
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request

            if not all([settings.GOOGLE_CLIENT_ID, settings.GOOGLE_CLIENT_SECRET, settings.GOOGLE_REFRESH_TOKEN]):
                return None

            creds = Credentials(
                token=None,
                refresh_token=settings.GOOGLE_REFRESH_TOKEN,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=settings.GOOGLE_CLIENT_ID,
                client_secret=settings.GOOGLE_CLIENT_SECRET,
                scopes=["https://www.googleapis.com/auth/drive"]
            )
            creds.refresh(Request())
            service = build('drive', 'v3', credentials=creds)
            return service
        except Exception as e:
            logger.error(f"❌ Failed to init Google Drive API: {e}")
            return None

    def is_configured(self) -> bool:
        return self._use_rclone or self._api_manager is not None

    # ═══════════════════════════════════════════════════════════
    # Download
    # ═══════════════════════════════════════════════════════════

    def download_folder(self, folder_id: str, local_path: str) -> bool:
        """Download all files from a Drive folder."""
        os.makedirs(local_path, exist_ok=True)
        
        if self._use_rclone:
            return self._rclone_download_folder(folder_id, local_path)
        return self._api_download_folder(folder_id, local_path)

    def _rclone_download_folder(self, folder_id: str, local_path: str) -> bool:
        """Fast download via rclone."""
        cmd = [
            self._rclone_path, "copy",
            "gdrive:",
            local_path,
            "--drive-root-folder-id", folder_id,
            "--transfers", "8",
            "--checkers", "8",
            "--multi-thread-streams", "4",
            "--stats-one-line",
            "--log-level", "INFO",
        ]
        logger.info(f"⬇️ rclone download: folder {folder_id}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            count = len([f for f in os.listdir(local_path) if os.path.isfile(os.path.join(local_path, f))])
            logger.info(f"✅ Downloaded {count} files via rclone")
            return True
        logger.error(f"❌ rclone download failed: {result.stderr[:200]}")
        return False

    def _api_download_folder(self, folder_id: str, local_path: str) -> bool:
        """Fallback download via Google Drive API."""
        try:
            from googleapiclient.http import MediaIoBaseDownload
            
            files = self._api_list_files(folder_id)
            if not files:
                return True

            downloaded = 0
            for file_info in files:
                file_id = file_info['id']
                file_name = file_info['name']
                output_path = os.path.join(local_path, file_name)
                
                try:
                    request = self._api_manager.files().get_media(fileId=file_id)
                    with open(output_path, "wb") as f:
                        downloader = MediaIoBaseDownload(f, request, chunksize=10*1024*1024)
                        done = False
                        while not done:
                            _, done = downloader.next_chunk()
                    downloaded += 1
                except Exception as e:
                    logger.warning(f"⚠️ Download failed for {file_name}: {e}")

            logger.info(f"✅ Downloaded {downloaded}/{len(files)} files via API")
            return downloaded > 0
        except Exception as e:
            logger.error(f"❌ API download failed: {e}")
            return False

    # ═══════════════════════════════════════════════════════════
    # Upload
    # ═══════════════════════════════════════════════════════════

    def upload_file(self, file_path: str, folder_id: str) -> Optional[str]:
        """Upload a single file."""
        if not os.path.exists(file_path):
            logger.error(f"❌ File not found: {file_path}")
            return None
        
        if self._use_rclone:
            return self._rclone_upload_file(file_path, folder_id)
        return self._api_upload_file(file_path, folder_id)

    def upload_folder(self, local_path: str, folder_id: str) -> bool:
        """Upload all files in a folder."""
        if not os.path.isdir(local_path):
            return False
        
        if self._use_rclone:
            return self._rclone_upload_folder(local_path, folder_id)
        return self._api_upload_folder(local_path, folder_id)

    def _rclone_upload_file(self, file_path: str, folder_id: str) -> Optional[str]:
        """Fast upload via rclone."""
        cmd = [
            self._rclone_path, "copy",
            file_path,
            "gdrive:",
            "--drive-root-folder-id", folder_id,
            "--quiet",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            logger.info(f"✅ Uploaded: {os.path.basename(file_path)}")
            return folder_id
        logger.error(f"❌ rclone upload failed for {os.path.basename(file_path)}")
        return None

    def _rclone_upload_folder(self, local_path: str, folder_id: str) -> bool:
        """Fast bulk upload via rclone."""
        cmd = [
            self._rclone_path, "copy",
            local_path,
            "gdrive:",
            "--drive-root-folder-id", folder_id,
            "--transfers", "8",
            "--checkers", "8",
            "--multi-thread-streams", "4",
            "--stats-one-line",
            "--log-level", "INFO",
        ]
        logger.info(f"⬆️ rclone upload folder: {os.path.basename(local_path)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            count = len([f for f in os.listdir(local_path) if os.path.isfile(os.path.join(local_path, f))])
            logger.info(f"✅ Uploaded {count} files via rclone")
            return True
        logger.error(f"❌ rclone folder upload failed: {result.stderr[:200]}")
        return False

    def _api_upload_file(self, file_path: str, folder_id: str) -> Optional[str]:
        """Fallback upload via Google Drive API."""
        try:
            from googleapiclient.http import MediaFileUpload
            
            file_name = os.path.basename(file_path)
            mime_type, _ = mimetypes.guess_type(file_path)
            mime_type = mime_type or "application/octet-stream"

            metadata = {"name": file_name, "parents": [folder_id]}
            media = MediaFileUpload(file_path, mimetype=mime_type, resumable=True, chunksize=10*1024*1024)
            
            file = self._api_manager.files().create(
                body=metadata, media_body=media, fields="id"
            ).execute()
            
            file_id = file.get("id")
            logger.info(f"✅ Uploaded: {file_name}")
            return file_id
        except Exception as e:
            logger.error(f"❌ Upload failed: {e}")
            return None

    def _api_upload_folder(self, local_path: str, folder_id: str) -> bool:
        """Fallback parallel upload via Google Drive API."""
        files = [
            os.path.join(local_path, f)
            for f in os.listdir(local_path)
            if os.path.isfile(os.path.join(local_path, f))
        ]
        if not files:
            return True

        uploaded = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(self._api_upload_file, f, folder_id): f for f in files}
            for future in concurrent.futures.as_completed(futures):
                if future.result():
                    uploaded += 1

        logger.info(f"📊 Upload: {uploaded}/{len(files)} via API")
        return uploaded == len(files)

    # ═══════════════════════════════════════════════════════════
    # List / Delete
    # ═══════════════════════════════════════════════════════════

    def list_files_in_folder(self, folder_id: str) -> list[dict]:
        """List files in a Drive folder."""
        if self._use_rclone:
            return self._rclone_list(folder_id)
        return self._api_list_files(folder_id)

    def _rclone_list(self, folder_id: str) -> list[dict]:
        """List files via rclone lsjson."""
        cmd = [
            self._rclone_path, "lsjson",
            "gdrive:",
            "--drive-root-folder-id", folder_id,
            "--quiet",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            import json
            if result.returncode != 0:
                logger.error(f"❌ rclone lsjson failed: {result.stderr[:200]}")
                return []
            files = json.loads(result.stdout) if result.stdout.strip() else []
            return [{"id": f.get("ID", ""), "name": f.get("Name", "")} for f in files]
        except Exception as e:
            logger.error(f"❌ rclone list failed: {e}")
            return []

    def _api_list_files(self, folder_id: str) -> list[dict]:
        """List files via Google Drive API."""
        try:
            query = f"'{folder_id}' in parents and trashed=false"
            results = self._api_manager.files().list(
                q=query,
                fields="files(id, name, mimeType, size)",
                pageSize=1000
            ).execute()
            return results.get("files", [])
        except Exception as e:
            logger.error(f"❌ API list failed: {e}")
            return []

    def delete_file(self, file_id: str) -> bool:
        """Delete a file."""
        try:
            self._api_manager.files().delete(fileId=file_id).execute()
            return True
        except Exception as e:
            if "404" in str(e):
                return True
            return False

    def delete_folder(self, folder_id: str) -> bool:
        """Delete a folder."""
        return self.delete_file(folder_id)

    def get_folder_id_from_url(self, folder_url: str) -> Optional[str]:
        """Extract folder ID from URL."""
        try:
            if "folders/" in folder_url:
                return folder_url.split("folders/")[-1].split("?")[0]
            if "id=" in folder_url:
                return folder_url.split("id=")[-1].split("&")[0]
            return None
        except Exception:
            return None


# نسخة عالمية
_drive_manager = FastDriveManager()


def get_rclone() -> FastDriveManager:
    """الحصول على نسخة عالمية."""
    return _drive_manager
