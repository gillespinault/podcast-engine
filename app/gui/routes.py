"""
Podcast Engine - GUI Routes
Web interface for podcast creation
"""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.config import settings, AUDIO_FORMATS

router = APIRouter(prefix="/gui", tags=["GUI"])

templates = Jinja2Templates(directory="app/gui/templates")


@router.get("/", response_class=HTMLResponse)
async def gui_home(request: Request):
    """
    Main GUI page

    Note: Voices are loaded dynamically by Alpine.js via /api/v1/voices endpoint.
    The GUI fetches all 67 voices on init() and filters by selected language.
    """
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "app_name": settings.app_name,
            "app_version": settings.app_version,
            # Voices loaded dynamically via Alpine.js fetch('/api/v1/voices')
            # See: app/gui/templates/index.html line 303-323
            "audio_formats": AUDIO_FORMATS,
            "default_voice": settings.default_voice,
            "default_speed": settings.default_speed,
            "default_bitrate": settings.default_bitrate,
        }
    )
