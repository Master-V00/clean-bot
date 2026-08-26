"""
Google Drive transfer manager.

Uses Google Drive API for all file operations.
"""

from __future__ import annotations

import os
import mimetypes
from pathlib import Path
from typing import Optional

from loguru import logger


class FastDriveManager:
    """
    File transfer to/from Google Drive via the Drive API.
    """

    def __init__(self):
        self._api_manager = self._init_api_manager()
        if self._api_manager:
            logger.info("🚀 Google Drive API ready")
        else:
            logger.error("❌ Google Drive API not configured — check GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN in .env")

    def _init_api_manager(self):
        """Initialize Google Drive API."""
        try:
            from app.server_config import settings
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
        return self._api_manager is not None

    # ═══════════════════════════════════════════════════════════
    # Download
    # ═══════════════════════════════════════════════════════════

    def download_folder(self, folder_id: str, local_path: str) -> bool:
        """Download all files from a Drive folder."""
        os.makedirs(local_path, exist_ok=True)
        return self._api_download_folder(folder_id, local_path)

    def _api_download_folder(self, folder_id: str, local_path: str, retries: int = 3) -> bool:
        """Download files via Google Drive API with per-file retry."""
        import time
        try:
            from googleapiclient.http import MediaIoBaseDownload
            from google.auth.transport.requests import Request

            # Proactively refresh token if expired
            try:
                if self._api_manager._http.credentials.expired:
                    self._api_manager._http.credentials.refresh(Request())
                    logger.info("🔄 Google Drive token refreshed proactively")
            except Exception:
                pass

            files = self._api_list_files(folder_id)
            if not files:
                return True

            downloaded = 0
            for file_info in files:
                file_id = file_info['id']
                file_name = file_info['name']
                output_path = os.path.join(local_path, file_name)

                for attempt in range(retries):
                    try:
                        request = self._api_manager.files().get_media(fileId=file_id)
                        with open(output_path, "wb") as f:
                            downloader = MediaIoBaseDownload(f, request, chunksize=10*1024*1024)
                            done = False
                            while not done:
                                _, done = downloader.next_chunk()
                        downloaded += 1
                        break
                    except Exception as e:
                        if attempt < retries - 1:
                            wait = 2 ** attempt
                            logger.warning(f"⚠️ Download attempt {attempt+1}/{retries} failed for {file_name}: {e}, retrying in {wait}s...")
                            time.sleep(wait)
                        else:
                            logger.error(f"❌ Download failed for {file_name} after {retries} attempts: {e}")

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
        return self._api_upload_file(file_path, folder_id)

    def upload_folder(self, local_path: str, folder_id: str) -> bool:
        """Upload all files in a folder."""
        if not os.path.isdir(local_path):
            return False
        return self._api_upload_folder(local_path, folder_id)

    def _api_upload_file(self, file_path: str, folder_id: str, retries: int = 3) -> Optional[str]:
        """Upload a file via Google Drive API with retry."""
        file_name = os.path.basename(file_path)
        for attempt in range(retries):
            try:
                from googleapiclient.http import MediaFileUpload

                mime_type, _ = mimetypes.guess_type(file_path)
                mime_type = mime_type or "application/octet-stream"

                file_size = os.path.getsize(file_path)
                metadata = {"name": file_name, "parents": [folder_id]}

                if file_size < 5 * 1024 * 1024:
                    # Small file: simple upload (no resumable, no chunking issues)
                    media = MediaFileUpload(file_path, mimetype=mime_type, resumable=False)
                    file = self._api_manager.files().create(
                        body=metadata, media_body=media, fields="id"
                    ).execute()
                else:
                    # Large file: resumable upload
                    media = MediaFileUpload(file_path, mimetype=mime_type, resumable=True, chunksize=10*1024*1024)
                    file = self._api_manager.files().create(
                        body=metadata, media_body=media, fields="id"
                    ).execute()

                logger.info(f"✅ Uploaded: {file_name}")
                return file.get("id")
            except Exception as e:
                if attempt < retries - 1:
                    import time
                    wait = 2 ** attempt
                    logger.warning(f"⚠️ Upload attempt {attempt+1}/{retries} failed for {file_name}: {e}, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    logger.error(f"❌ Upload failed for {file_name} after {retries} attempts: {e}")
                    return None

    def _api_upload_folder(self, local_path: str, folder_id: str) -> bool:
        """Sequential upload via Google Drive API (avoids Content-Range conflicts)."""
        files = [
            os.path.join(local_path, f)
            for f in os.listdir(local_path)
            if os.path.isfile(os.path.join(local_path, f))
        ]
        if not files:
            return True

        uploaded = 0
        for file_path in files:
            if self._api_upload_file(file_path, folder_id):
                uploaded += 1

        logger.info(f"📊 Upload: {uploaded}/{len(files)} via API")
        return uploaded == len(files)

    # ═══════════════════════════════════════════════════════════
    # List / Delete
    # ═══════════════════════════════════════════════════════════

    def list_files_in_folder(self, folder_id: str) -> list[dict]:
        """List files in a Drive folder."""
        return self._api_list_files(folder_id)

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


# Singleton instance
_drive_manager = FastDriveManager()


def get_rclone() -> FastDriveManager:
    """Get the global drive manager instance."""
    return _drive_manager
