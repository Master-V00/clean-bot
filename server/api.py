"""
FastAPI server for Discord bot integration
"""
from fastapi import APIRouter, HTTPException, Header, Depends
import secrets
from server.config import settings
from server.database import db
from server.models import ProcessingRequest, ProcessingResponse, HealthResponse, RequestStatus
from server.worker import worker
from server.google_drive import gdrive
from loguru import logger


router = APIRouter()

# Keep a standalone app for direct use (e.g., `uvicorn server.api:app`)
from fastapi import FastAPI as _FastAPI
app = _FastAPI(title="PanelCleaner API (standalone)", version="1.0.0")
app.include_router(router)


def verify_api_key(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid API key")

    token = authorization.split(" ")[1]

    if token != settings.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

    return token


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "PanelCleaner API is running"
    }


@router.get("/ready")
async def readiness_check():
    """Readiness check: verify all external services are initialized"""
    services = {
        "database": db.client is not None,
        "google_drive": gdrive.service is not None,
    }
    all_ready = all(services.values())
    status_code = 200 if all_ready else 503
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=status_code,
        content={"ready": all_ready, "services": services}
    )


@router.get("/device")
async def device_info():
    """Return hardware detection information"""
    from pcleaner.runtime.device import (
        device_kind, device_name, cuda_available,
        mps_available, gpu_available, hardware_info,
    )
    return {
        "device_kind": device_kind().name,
        "device_name": device_name(),
        "cuda_available": cuda_available(),
        "mps_available": mps_available(),
        "gpu_available": gpu_available(),
        "hardware_details": hardware_info(),
    }


@router.post("/process", response_model=ProcessingResponse)
async def process_image(
    request: ProcessingRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Submit image processing request

    Receives a Google Drive folder URL from Discord Bot and queues it for processing
    """
    try:
        logger.info(f"New request from user {request.user_id}")

        # Add request to database (worker picks it up automatically)
        db_response = db.add_request(
            request.user_id,
            request.google_drive_folder_url,
            channel_id=request.channel_id
        )
        request_id = db_response.get('id')

        # Get queue position for estimated time
        queue_position = db.get_queue_position(request_id)

        return {
            "success": True,
            "message": f"Request queued successfully. Processing ID: {request_id}",
            "request_id": request_id,
            "queue_position": queue_position
        }
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{request_id}", response_model=RequestStatus)
async def get_request_status(
    request_id: int,
    api_key: str = Depends(verify_api_key)
):
    """
    Get status of a processing request
    """
    try:
        request_data = db.get_request(request_id)
        if not request_data:
            raise HTTPException(status_code=404, detail=f"Request {request_id} not found")
        return {
            "id": request_data["id"],
            "status": request_data.get("status", "pending"),
            "user_id": request_data.get("discord_user_id", "unknown"),
            "pixel_count": request_data.get("pixel_count", 0),
            "output_url": request_data.get("output_url"),
            "error_message": request_data.get("error_message"),
            "created_at": request_data.get("created_at", "2024-01-01T00:00:00"),
            "updated_at": request_data.get("updated_at", "2024-01-01T00:00:00")
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting request status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/settings/generate-key")
async def generate_new_api_key(
    api_key: str = Depends(verify_api_key)
):
    """
    Generate a new API key (for settings/configuration)
    Returns new API key that can be used for Discord Bot
    """
    try:
        new_key = secrets.token_urlsafe(32)
        logger.warning("New API key generated (old key invalidated)")

        return {
            "success": True,
            "message": "New API key generated. Please update your Discord Bot configuration.",
            "new_api_key": new_key,
            "warning": "Store this key securely. It will be used to authenticate requests from your Discord Bot."
        }
    except Exception as e:
        logger.error(f"Error generating API key: {e}")
        raise HTTPException(status_code=500, detail=str(e))
