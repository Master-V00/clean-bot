"""
Google Drive API integration for uploading and managing files using User OAuth2 Refresh Token
"""
import os
import json
import mimetypes
from typing import List, Optional

from loguru import logger
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

from server.config import settings


class GoogleDriveManager:
    """Google Drive API client using User OAuth2 Refresh Token"""

    SCOPES = ["https://www.googleapis.com/auth/drive"]

    def __init__(self):
        self.service = None
        self.creds = None
        self._initialize()

    # ----------------------------
    # INIT
    # ----------------------------
    def _initialize(self):
        """Initialize Google Drive service"""
        try:
            # Validate settings
            if not all([
                settings.GOOGLE_CLIENT_ID,
                settings.GOOGLE_CLIENT_SECRET,
                settings.GOOGLE_REFRESH_TOKEN
            ]):
                logger.warning("⚠️ Missing Google OAuth credentials")
                return

            self.creds = Credentials(
                token=None,
                refresh_token=settings.GOOGLE_REFRESH_TOKEN,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=settings.GOOGLE_CLIENT_ID,
                client_secret=settings.GOOGLE_CLIENT_SECRET,
                scopes=self.SCOPES
            )

            # Refresh once at startup
            self._refresh_token_if_needed()

            self.service = build('drive', 'v3', credentials=self.creds)
            logger.info("👑 ✅ Google Drive initialized using User OAuth2")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Google Drive: {e}")
            self.service = None

    # ----------------------------
    # TOKEN HANDLING
    # ----------------------------
    def _refresh_token_if_needed(self):
        """Refresh token only if expired"""
        try:
            if not self.creds:
                return

            if not self.creds.valid or self.creds.expired:
                logger.info("🔄 Refreshing Google OAuth token...")
                self.creds.refresh(Request())
                logger.info("✅ Token refreshed successfully")

        except Exception as e:
            logger.error(f"❌ Failed to refresh token: {e}")

    # ----------------------------
    # UPLOAD FILE
    # ----------------------------
    def upload_file(self, file_path: str, folder_id: Optional[str] = None) -> Optional[str]:
        """Upload file to Google Drive"""

        try:
            if not self.service:
                logger.error("❌ Google Drive service not initialized")
                return None

            if not os.path.exists(file_path):
                logger.error(f"❌ File not found: {file_path}")
                return None

            self._refresh_token_if_needed()

            file_name = os.path.basename(file_path)

            mime_type, _ = mimetypes.guess_type(file_path)
            mime_type = mime_type or "application/octet-stream"

            metadata = {"name": file_name}

            if folder_id:
                metadata["parents"] = [folder_id]

            media = MediaFileUpload(
                file_path,
                mimetype=mime_type,
                resumable=True
            )

            file = self.service.files().create(
                body=metadata,
                media_body=media,
                fields="id"
            ).execute()

            file_id = file.get("id")
            logger.info(f"✅ Uploaded: {file_name} → ID: {file_id}")

            return file_id

        except Exception as e:
            logger.error(f"❌ Upload failed: {e}")
            return None

    # ----------------------------
    # LIST FILES
    # ----------------------------
    def list_files_in_folder(self, folder_id: str) -> List[dict]:
        """List files inside a Drive folder"""
        try:
            if not self.service:
                return []

            self._refresh_token_if_needed()

            query = f"'{folder_id}' in parents and trashed=false"

            results = self.service.files().list(
                q=query,
                fields="files(id, name, mimeType, createdTime)"
            ).execute()

            return results.get("files", [])

        except Exception as e:
            logger.error(f"❌ List files error: {e}")
            return []

    # ----------------------------
    # DOWNLOAD FILE (STREAM SAFE)
    # ----------------------------
    def download_file(self, file_id: str, output_path: str) -> bool:
        """Download file safely using streaming"""

        try:
            if not self.service:
                return False

            self._refresh_token_if_needed()

            request = self.service.files().get_media(fileId=file_id)

            with open(output_path, "wb") as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False

                while not done:
                    status, done = downloader.next_chunk()
                    if status:
                        logger.info(f"⬇️ Download {int(status.progress() * 100)}%")

            logger.info(f"✅ Downloaded → {output_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            return False

    # 🛠️ دالة التوافق مع الـ Worker (عشان الـ Worker بينادي الاسم ده وميضربش خطأ)
    def download_image(self, file_id: str, output_path: str) -> bool:
        """Alias for download_file to ensure compatibility with TaskWorker"""
        return self.download_file(file_id, output_path)

    # ----------------------------
    # DELETE FILE
    # ----------------------------
    def delete_file(self, file_id: str) -> bool:
        """Delete file from Drive"""
        try:
            if not self.service:
                return False

            self._refresh_token_if_needed()

            # 🛠️ التعديل هنا: استخراج الـ ID لو المبعوت رابط كامل
            if "drive.google.com" in str(file_id):
                file_id = self.get_folder_id_from_url(file_id) or file_id

            self.service.files().delete(fileId=file_id).execute()

            logger.info(f"🗑️ Deleted file: {file_id}")
            return True

        except Exception as e:
            # 404 means the file is already deleted — treat as success
            if "404" in str(e) or "notFound" in str(e) or "File not found" in str(e):
                logger.info(f"🗑️ File {file_id} already deleted, skipping.")
                return True
            logger.error(f"❌ Delete failed: {e}")
            return False

    # ----------------------------
    # EXTRACT FOLDER ID
    # ----------------------------
    def get_folder_id_from_url(self, folder_url: str) -> Optional[str]:
        """Extract folder ID from Google Drive URL"""
        try:
            if "folders/" in folder_url:
                return folder_url.split("folders/")[-1].split("?")[0]

            if "id=" in folder_url:
                return folder_url.split("id=")[-1].split("&")[0]

            return None

        except Exception as e:
            logger.error(f"❌ Folder ID extraction failed: {e}")
            return None


# Global instance
gdrive = GoogleDriveManager()