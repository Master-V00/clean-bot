"""
Cleanup task for removing expired files from Google Drive
Run this periodically to maintain storage space
"""
import asyncio
from server.database import db
from server.google_drive import gdrive
from loguru import logger


class CleanupScheduler:
    """Scheduled cleanup for expired outputs"""

    def __init__(self, interval_seconds: int = 60, expiration_minutes: int = 20):
        self.interval_seconds = interval_seconds
        self.expiration_minutes = expiration_minutes

    def cleanup_expired_outputs(self):
        """Cleanup expired outputs from Google Drive"""
        if gdrive.service is None:
            logger.warning("Google Drive not initialized – skipping cleanup")
            return
        try:
            logger.debug("Starting cleanup task...")

            expired_outputs = db.get_expired_outputs(self.expiration_minutes)

            if not expired_outputs:
                logger.debug("No expired outputs to clean")
                return

            logger.debug(f"Found {len(expired_outputs)} expired outputs")

            deleted_count = 0
            error_count = 0

            for output in expired_outputs:
                try:
                    request_id = output.get('id')
                    output_urls = output.get('output_url', '').split(',')

                    for file_id in output_urls:
                        if file_id.strip():
                            if gdrive.delete_file(file_id.strip()):
                                deleted_count += 1
                            else:
                                error_count += 1

                    db.mark_output_cleaned(request_id)

                except Exception as e:
                    logger.error(f"Error cleaning output: {e}")
                    error_count += 1

            logger.debug(f"Cleanup completed - Deleted: {deleted_count}, Errors: {error_count}")

        except Exception as e:
            logger.error(f"Cleanup task failed: {e}")


async def start_cleanup_task_async(interval_seconds: int = 60, expiration_minutes: int = 20):
    """Start cleanup task as a periodic background coroutine."""
    scheduler = CleanupScheduler(interval_seconds, expiration_minutes)
    while True:
        scheduler.cleanup_expired_outputs()
        await asyncio.sleep(interval_seconds)


if __name__ == "__main__":
    import asyncio
    asyncio.run(start_cleanup_task_async())
