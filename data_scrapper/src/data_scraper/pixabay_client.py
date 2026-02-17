import requests
import os
from typing import List, Dict, Any

class PixabayClient:
    BASE_URL = "https://pixabay.com/api/videos/"
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("PIXABAY_API_KEY")
        if not self.api_key:
            raise ValueError("PIXABAY_API_KEY is required")

    def search_videos(self, query: str, per_page: int = 15, page: int = 1) -> list[dict[str, any]]:
        """Search for videos on Pixabay with exponential backoff."""
        import time
        params = {
            "key": self.api_key,
            "q": query,
            "per_page": per_page,
            "page": page,
            "video_type": "film",
            "min_width": 1920,
            "min_height": 1080
        }
        
        for attempt in range(3):
            try:
                response = requests.get(self.BASE_URL, params=params, timeout=15)
                if response.status_code == 429:
                    wait = (attempt + 1) * 30
                    print(f"⚠️ Pixabay Rate Limit (429). Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                return response.json().get("hits", [])
            except Exception as e:
                if attempt == 2: raise e
                time.sleep(5)
        return []

    def get_best_video_link(self, video_data: dict[str, any]) -> str:
        """Extract the best quality video link from Pixabay hit."""
        videos = video_data.get("videos", {})
        # Pixabay provides 'large', 'medium', 'small', 'tiny'
        # Large is usually 1920x1080 or 4k
        for size in ['large', 'medium']:
            if size in videos and videos[size].get('url'):
                return videos[size]['url']
        return ""
