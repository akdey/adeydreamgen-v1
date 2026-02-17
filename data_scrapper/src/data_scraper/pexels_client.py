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
        """Search for videos on Pexels with exponential backoff."""
        import time
        url = f"{self.BASE_URL}/search"
        params = {
            "query": query,
            "per_page": per_page,
            "page": page,
            "min_width": 1920,
            "min_height": 1080
        }
        
        for attempt in range(3):
            try:
                response = requests.get(url, headers=self.headers, params=params, timeout=15)
                if response.status_code == 429:
                    wait = (attempt + 1) * 30
                    print(f"⚠️ Pexels Rate Limit (429). Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                return response.json().get("videos", [])
            except Exception as e:
                if attempt == 2: raise e
                time.sleep(5)
        return []

    def get_video_files(self, video_data: dict[str, any]) -> list[dict[str, any]]:
        """Extract high quality video files from video data."""
        return video_data.get("video_files", [])
