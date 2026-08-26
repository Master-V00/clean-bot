"""
Supabase database client and utilities
"""
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List
from app.server_config import settings
from loguru import logger


class DatabaseClient:
    """Supabase database client wrapper"""
    
    def __init__(self):
        self.client = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Supabase client"""
        try:
            # Try to import supabase only if available
            try:
                from supabase import create_client
                if settings.SUPABASE_URL and settings.SUPABASE_KEY:
                    self.client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
                    logger.info("✅ Supabase database connected successfully")
                else:
                    logger.warning("⚠️  Supabase credentials not found in environment variables")
            except ImportError:
                logger.warning("⚠️  Supabase library not installed. Install with: pip install supabase")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Supabase: {e}")
            self.client = None
    
    def add_request(self, user_id: str, folder_url: str, status: str = "pending", channel_id: Optional[str] = None) -> Dict:
        """Add a new processing request to database"""
        try:
            if not self.client:
                raise RuntimeError("Database not initialized")
            
            data = {
                "discord_user_id": user_id,
                "google_drive_folder_url": folder_url,
                "status": status,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "pixel_count": 0,
                "output_url": None,
                "error_message": None,
                "channel_id": channel_id  # 🆕 حفظ معرف القناة هنا
            }
            
            response = self.client.table("requests").insert(data).execute()
            logger.info(f"✅ Request added for user {user_id} in channel {channel_id}")
            return response.data[0] if response.data else data
        except Exception as e:
            logger.error(f"❌ Failed to add request: {e}")
            raise
    
    def update_request_status(self, request_id: int, status: str, **kwargs) -> Dict:
        """Update request status and additional fields"""
        try:
            if not self.client:
                raise RuntimeError("Database not initialized")
            
            update_data = {"status": status, "updated_at": datetime.now(timezone.utc).isoformat()}
            update_data.update(kwargs)
            
            response = self.client.table("requests").update(update_data).eq("id", request_id).execute()
            logger.info(f"✅ Request {request_id} updated to {status}")
            return response.data[0] if response.data else {}
        except Exception as e:
            logger.error(f"❌ Failed to update request: {e}")
            raise
    
    def get_user_credits(self, user_id: str) -> int:
        """Get user credit balance"""
        try:
            if not self.client:
                raise RuntimeError("Database not initialized")
            
            response = self.client.table("users").select("credits").eq("discord_user_id", user_id).execute()
            
            if response.data:
                return response.data[0]["credits"]
            else:
                logger.warning(f"⚠️  User {user_id} not found, creating new user")
                self._create_user(user_id)
                return 0
        except Exception as e:
            logger.error(f"❌ Failed to get user credits: {e}")
            raise
    
    def deduct_credits(self, user_id: str, amount: int) -> bool:
        """Deduct credits from user account"""
        try:
            current_credits = self.get_user_credits(user_id)
            
            if current_credits < amount:
                logger.warning(f"⚠️  Insufficient credits for user {user_id}")
                return False
            
            new_balance = current_credits - amount
            self.client.table("users").update({"credits": new_balance}).eq("discord_user_id", user_id).execute()
            logger.info(f"✅ Deducted {amount} credits from user {user_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to deduct credits: {e}")
            raise
            
    def add_total_credits_used(self, user_id: str, amount: int) -> bool:
        """Add processed credits to user's total_credits_used balance"""
        try:
            if not self.client:
                raise RuntimeError("Database not initialized")
                
            # 1. جلب البيانات الحالية للمستخدم لمعرفة الـ total الحالي
            response = self.client.table("users").select("total_credits_used").eq("discord_user_id", user_id).execute()
            
            current_total = 0
            if response.data and response.data[0].get("total_credits_used") is not None:
                current_total = response.data[0]["total_credits_used"]
                
            # 2. حساب القيمة الجديدة وتحديثها
            new_total = current_total + amount
            self.client.table("users").update({"total_credits_used": new_total}).eq("discord_user_id", user_id).execute()
            logger.info(f"📊 Added {amount} to total_credits_used for user {user_id} (New total: {new_total})")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to update total_credits_used: {e}")
            return False
    
    def _create_user(self, user_id: str, initial_credits: int = 5):
        """Create a new user with initial credits"""
        try:
            if not self.client:
                raise RuntimeError("Database not initialized")
            
            data = {
                "discord_user_id": user_id,
                "credits": initial_credits,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            self.client.table("users").insert(data).execute()
            logger.info(f"✅ Created new user {user_id} with {initial_credits} credits")
        except Exception as e:
            logger.error(f"❌ Failed to create user: {e}")
    
    def get_next_pending_request(self) -> Optional[Dict]:
        """Get the next pending request and atomically claim it for processing"""
        try:
            if not self.client:
                raise RuntimeError("Database not initialized")

            # 1. Find the oldest pending request
            logger.info("📥 Polling database for pending requests...")
            response = (
                self.client.table("requests")
                .select("id, discord_user_id, google_drive_folder_url")
                .eq("status", "pending")
                .order("created_at", desc=False)
                .limit(1)
                .execute()
            )

            if not response.data:
                logger.info("😴 No pending requests found")
                return None

            request = response.data[0]
            request_id = request["id"]

            # 2. Atomically mark it as processing (only if still pending)
            update_response = (
                self.client.table("requests")
                .update({
                    "status": "processing",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                })
                .eq("id", request_id)
                .eq("status", "pending")
                .execute()
            )

            if not update_response.data:
                logger.warning(f"⚠️ Request #{request_id} was already claimed by another worker")
                return None

            logger.info(f"🚀 Found pending request #{request_id}")
            logger.info(f"✅ Request #{request_id} marked as processing")

            return {
                "id": request_id,
                "user_id": request["discord_user_id"],
                "folder_url": request["google_drive_folder_url"],
            }

        except Exception as e:
            logger.error(f"❌ Failed to get next pending request: {e}")
            return None

    def get_request(self, request_id: int) -> Optional[Dict]:
        """Get a request by ID"""
        try:
            if not self.client:
                return None
            response = self.client.table("requests").select("*").eq("id", request_id).maybe_single().execute()
            return response.data if response.data else None
        except Exception as e:
            logger.error(f"Failed to get request {request_id}: {e}")
            return None

    def get_queue_position(self, request_id: int) -> int:
        """Get the number of requests ahead of this one in the queue"""
        try:
            if not self.client:
                return 0

            # Count pending AND currently processing requests created before this one
            response = self.client.table("requests").select("created_at").eq("id", request_id).execute()
            if not response.data:
                return 0

            created_at = response.data[0]["created_at"]
            count_response = (
                self.client.table("requests")
                .select("id", count="exact")
                .in_("status", ["pending", "processing"])
                .lt("created_at", created_at)
                .execute()
            )
            return count_response.count if hasattr(count_response, 'count') else 0
        except Exception as e:
            logger.error(f"Failed to get queue position: {e}")
            return 0

    def get_settings_value(self, key: str) -> Optional[str]:
        """Get a setting value from the settings table"""
        try:
            if not self.client:
                return None

            response = (
                self.client.table("settings")
                .select("value")
                .eq("key", key)
                .maybe_single()
                .execute()
            )
            if response.data:
                return response.data.get("value")
            return None
        except Exception as e:
            logger.error(f"Failed to get settings value: {e}")
            return None

    def get_expired_outputs(self, expiration_minutes: int = 20) -> List[Dict]:
        """Get outputs that have expired and should be deleted"""
        try:
            if not self.client:
                raise RuntimeError("Database not initialized")
            
            expiration_time = (datetime.now(timezone.utc) - timedelta(minutes=expiration_minutes)).isoformat()
            
            # التعديل هنا: أضفنا شرط .neq لمنع جلب الطلبات المنظفة مسبقاً
            response = (
                self.client.table("requests")
                .select("*")
                .eq("status", "completed")
                .lt("updated_at", expiration_time)
                .neq("gdrive_file_cleaned", True) # 🌟 يتخطى أي طلب تم تنظيفه وعمل True له مسبقاً
                .execute()
            )
            
            return response.data if response.data else []
        except Exception as e:
            logger.error(f"❌ Failed to get expired outputs: {e}")
            return []
    
    def mark_output_cleaned(self, request_id: int):
        """Mark output as cleaned from Google Drive"""
        try:
            if not self.client:
                raise RuntimeError("Database not initialized")
            
            self.client.table("requests").update({"gdrive_file_cleaned": True}).eq("id", request_id).execute()
            logger.info(f"✅ Marked request {request_id} as cleaned")
        except Exception as e:
            logger.error(f"❌ Failed to mark output as cleaned: {e}")


# Global database instance
db = DatabaseClient()
