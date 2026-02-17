import cv2
import os
import requests
import subprocess
import json
from typing import Tuple, Optional, Dict, Any

class VideoProcessor:
    @staticmethod
    def get_video_metadata(filepath: str) -> dict[str, any]:
        """Extract metadata from a video file using OpenCV."""
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return {}
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        # Check for audio stream using ffprobe
        has_audio = False
        try:
            cmd = [
                'ffprobe', 
                '-v', 'error', 
                '-show_entries', 'stream=codec_type', 
                '-of', 'json', 
                filepath
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            probe = json.loads(result.stdout)
            for stream in probe.get('streams', []):
                if stream.get('codec_type') == 'audio':
                    has_audio = True
                    break
        except Exception:
            # Fallback if ffprobe is missing or fails
            has_audio = False

        aspect_ratio = width / height
        # 16:9 is ~1.77, 9:16 is ~0.56
        if 1.7 <= aspect_ratio <= 1.8:
            orientation = "landscape"
        elif 0.5 <= aspect_ratio <= 0.6:
            orientation = "portrait"
        else:
            orientation = "other"
            
        return {
            "width": width,
            "height": height,
            "duration": duration,
            "orientation": orientation,
            "fps": fps,
            "has_audio": has_audio
        }

    @staticmethod
    def download_video(url: str, dest_path: str) -> bool:
        """Download video from URL to local path."""
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as e:
            print(f"Error downloading video: {e}")
            return False
