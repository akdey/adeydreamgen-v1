import requests
import os
from typing import List, Dict, Any

class PexelsClient:
    BASE_URL = "https://api.pexels.com/v1/videos"
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("PEXELS_API_KEY")
        if not self.api_key:
            raise ValueError("PEXELS_API_KEY is required")
        self.headers = {"Authorization": self.api_key}

    def search_videos(self, query: str, per_page: int = 15, page: int = 1) -> list[dict[str, any]]:
        """Search for videos on Pexels."""
        url = f"{self.BASE_URL}/search"
        params = {
            "query": query,
            "per_page": per_page,
            "page": page,
            "min_width": 1920, # Preference for high res
            "min_height": 1080
        }
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json().get("videos", [])

    def get_video_files(self, video_data: dict[str, any]) -> list[dict[str, any]]:
        """Extract high quality video files from video data."""
        return video_data.get("video_files", [])
