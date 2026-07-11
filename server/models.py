"""
Pydantic models for API requests and responses
"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ProcessingRequest(BaseModel):
    """Discord bot processing request model"""
    user_id: str = Field(..., description="Discord user ID")
    google_drive_folder_url: str = Field(..., description="Google Drive folder URL")
    channel_id: Optional[str] = Field(None, description="Discord channel ID for response delivery")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "123456789",
                "google_drive_folder_url": "https://drive.google.com/drive/folders/xxxxx",
                "channel_id": "123456789012345678"
            }
        }


class ProcessingResponse(BaseModel):
    """API response model"""
    success: bool
    message: str
    request_id: Optional[int] = None
    queue_position: Optional[int] = Field(None, description="Position in the processing queue")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Request queued successfully",
                "request_id": 1,
                "queue_position": 0
            }
        }


class RequestStatus(BaseModel):
    """Request status model"""
    id: int
    status: str  # pending, processing, completed, failed
    user_id: str
    pixel_count: int
    output_url: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "status": "completed",
                "user_id": "123456789",
                "pixel_count": 150000,
                "output_url": "https://drive.google.com/drive/folders/xxxxx",
                "error_message": None,
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:05:00"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str
    version: str = "1.0.0"
